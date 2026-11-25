"""Window-based bitemporal/text fusion with internal or externally shared text.

Each window jointly attends over [pre, post, fusion queries, class text,
nuisance text]. Only the updated fusion queries form the attention correction.
Fusion queries start from a 1x1 convolution of concatenated pre/post features.
The convolutional feature also skips the attention branch: output = initial
fusion + reconstructed attention correction. There is no global token pooling.
Fixed 2-D positions distinguish spatial queries; a learned relative-position
bias is shared across the three spatial groups, with zero spatial bias for
all interactions involving text. By default each encoder layer alternates
ordinary and half-window-shifted joint attention. Shifted boundary windows
are padded separately, so opposite image edges never share a text state.
Fusion-query logits subtract log(valid token count) separately for visual
and text keys, removing the group-size prior without fixing their attention
budgets. Other query types retain their original joint attention.

Existing standalone configurations with type='TextGuidedFusionNeck' keep
working and own a TextBank. TextHTC supplies shared text explicitly instead.
With a boolean ``use_shifted_window``, encoder_layers controls the number of
ordinary/shifted attention pairs when shifting is enabled. A sequence
``use_shifted_window`` instead selects the window layout for each logical layer
and builds exactly one attention block per layer.
Text/source embeddings do not guarantee semantic alignment or noise suppression;
all trainable parts learn through the downstream detector losses.
"""

import math
from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from mm.cv.cnn import build_norm_layer
from mm.cv.cnn.bricks.transformer import FFN, MultiheadAttention
from mm.engine.model import BaseModule
from mm.det.registry import MODELS
from mm.det.structures.bbox import get_box_tensor
from mm.det.structures.mask import BitmapMasks

from .text_bank import TextBank, _prompt_groups
from .text_loss_normalization import TextLossNormalizer
from .instance_pooling import pool_instances, pool_log_instances
from .visual_semantic_head import VisualSemanticHead


def _position_encoding(height, width, dims, reference):
    """Normalized global 2-D sine/cosine positions, shape (1, H, W, D)."""
    y = torch.arange(height, device=reference.device, dtype=torch.float32)
    x = torch.arange(width, device=reference.device, dtype=torch.float32)
    y = (y + 0.5) / height * (2 * math.pi)
    x = (x + 0.5) / width * (2 * math.pi)
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    frequency = 10000 ** (
        torch.arange(dims // 4, device=reference.device, dtype=torch.float32)
        / (dims // 4))
    yy = yy.unsqueeze(-1) / frequency
    xx = xx.unsqueeze(-1) / frequency
    position = torch.cat((yy.sin(), yy.cos(), xx.sin(), xx.cos()), dim=-1)
    return position.unsqueeze(0).to(dtype=reference.dtype)


def _window_partition(x, window_size):
    """Padded BHWD -> (B*num_windows, window_size**2, D), row-major."""
    batch, height, width, dims = x.shape
    w = window_size
    return (x.reshape(batch, height // w, w, width // w, w, dims)
            .permute(0, 1, 3, 2, 4, 5)
            .reshape(-1, w * w, dims))


def _window_reverse(windows, window_size, batch, height, width):
    """Inverse of _window_partition; height/width are padded sizes."""
    w = window_size
    return (windows.reshape(batch, height // w, width // w, w, w, -1)
            .permute(0, 1, 3, 2, 4, 5)
            .reshape(batch, height, width, -1))


def _window_layout(height, width, window_size, shift, reference, batch):
    """Noncyclic shifted partition, with clipped boxes and padding-key mask.

    The offset grid implements Swin's shifted partition directly: pad the
    top/left by window_size - shift, then pad bottom/right to complete the grid.
    Unlike cyclic roll, disjoint boundary regions never share text tokens.
    Disable shifting independently on axes that already fit in one window.
    """
    w = window_size
    top = w - shift if shift and height > w else 0
    left = w - shift if shift and width > w else 0
    bottom, right = (-(height + top)) % w, (-(width + left)) % w
    padded_h, padded_w = height + top + bottom, width + left + right
    padding = (0, 0, left, right, top, bottom)
    valid = torch.ones((batch, height, width, 1),
                       device=reference.device, dtype=torch.bool)
    valid = _window_partition(F.pad(valid, padding, value=False), w).squeeze(-1)
    yy, xx = torch.meshgrid(
        torch.arange(padded_h // w, device=reference.device) * w - top,
        torch.arange(padded_w // w, device=reference.device) * w - left,
        indexing='ij')
    boxes = torch.stack((xx.clamp(min=0), yy.clamp(min=0),
                         (xx + w).clamp(max=width),
                         (yy + w).clamp(max=height)), dim=-1).reshape(-1, 4).float()
    # Group by padding geometry using Python shape metadata, not a GPU
    # unique/nonzero operation. Every group shares one attention bias.
    regions = {}
    columns = padded_w // w
    for row in range(padded_h // w):
        for column in range(columns):
            y0, x0 = row * w - top, column * w - left
            region = (max(0, -y0), min(w, height - y0),
                      max(0, -x0), min(w, width - x0))
            regions.setdefault(region, []).append(row * columns + column)
    window_count = len(boxes)
    groups = tuple((region, [b * window_count + i for b in range(batch) for i in indices])
                   for region, indices in regions.items())
    return dict(padding=padding, padded_shape=(padded_h, padded_w), groups=groups,
                offset=(top, left), valid=valid, boxes=boxes,
                has_padding=bool(top or left or bottom or right))


def _window_overlap_weights(source_boxes, target_boxes):
    """Read the same prompt's old states by overlap with each new window.

    Geometry is constant; weighted readout using these weights remains
    differentiable with respect to all class and nuisance token states.
    No mixing or averaging occurs between different prompt identities.
    """
    with torch.no_grad():
        extent = (torch.minimum(target_boxes[:, None, 2:], source_boxes[None, :, 2:])
                  - torch.maximum(target_boxes[:, None, :2], source_boxes[None, :, :2]))
        area = extent.clamp_min(0).prod(-1)
        return area / area.sum(-1, keepdim=True).clamp_min(1e-8)


@dataclass
class _WindowAttentionMask:
    """Broadcast interior bias, with per-window masks only at boundaries."""

    bias: torch.Tensor | None
    groups: tuple
    order: torch.Tensor | None
    inverse: torch.Tensor | None
    invalid_keys: torch.Tensor
    num_heads: int

    def dense(self, reference):
        """Legacy representation, only for explicitly requested attention maps."""
        count, length = self.invalid_keys.shape
        # Expand the same corrected group biases used by SDPA. Boundary
        # windows have their own valid-key counts, not the interior count.
        parts = [reference.new_zeros(size, self.num_heads, length, length)
                 if bias is None else bias.expand(size, -1, -1, -1)
                 for size, bias in self.groups]
        bias = parts[0] if len(parts) == 1 else torch.cat(parts, dim=0)
        if self.inverse is not None:
            bias = bias.index_select(0, self.inverse)
        return bias.reshape(count * self.num_heads, length, length)


class _JointAttentionBlock(BaseModule):
    """Pre-LN MMCV self-attention and FFN, both with internal residuals."""

    def __init__(self, dims, heads, expansion, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(dims)
        self.attention = MultiheadAttention(
            dims, heads, attn_drop=dropout, proj_drop=dropout,
            batch_first=True)
        self.norm2 = nn.LayerNorm(dims)
        self.ffn = FFN(
            embed_dims=dims, feedforward_channels=dims * expansion,
            ffn_drop=dropout, act_cfg=dict(type='GELU'))

    @staticmethod
    def _class_readout(q, k, bias, spatial_count, class_count, *, logarithmic,
                       return_visual=False):
        """Fusion -> text weights, normalized over ALL keys before slicing.

        Supervision keeps log probabilities from Q/K onward, including the
        arithmetic head mean. Reading pre-dropout probabilities also avoids
        random dropout zeros becoming impossible auxiliary targets.
        """
        begin, end = 2 * spatial_count, 3 * spatial_count
        query = q[:, :, begin:end]
        if logarithmic:
            query, k = query.float(), k.float()
        logits = (query * (q.shape[-1] ** -0.5)) @ k.transpose(-2, -1)
        if bias is not None:
            bias = bias[..., begin:end, :]
            logits = (logits.masked_fill(bias, float('-inf')) if bias.dtype == torch.bool
                      else logits + bias)
        if logarithmic:
            log_probability = F.log_softmax(logits, dim=-1)
            selected = log_probability[..., end:end + class_count]
            text = torch.logsumexp(selected, dim=1) - math.log(q.shape[1])
            if return_visual:
                # Read visual mass directly. 1 - sum(text) loses both precision
                # near unit text mass and the distinction from nuisance text.
                visual = torch.logsumexp(
                    torch.logsumexp(log_probability[..., :end], dim=-1),
                    dim=1) - math.log(q.shape[1])
                return text, visual.unsqueeze(-1)
            return text
        probability = logits.softmax(-1)
        text = probability[..., end:end + class_count].mean(dim=1)
        if return_visual:
            return text, probability[..., :end].sum(-1).mean(1).unsqueeze(-1)
        return text

    def forward(self, tokens, attention_mask, class_count=0, spatial_count=0,
                log_attention=False, return_visual_attention=False):
        normalized = self.norm1(tokens)
        mha = self.attention.attn
        class_attention = visual_attention = None
        if return_visual_attention and (not class_count or not log_attention):
            raise ValueError('Visual budget readout requires log class attention.')
        if (not isinstance(attention_mask, _WindowAttentionMask)
                or (class_count and self.training and mha.dropout > 0)):
            # Tensor-mask compatibility and nonzero training dropout retain
            # MHA's joint output/weight sampling. The configured dropout=0
            # path below reuses Q/K/V and reads only fusion-query rows.
            query = normalized.transpose(0, 1)
            dense_mask = (attention_mask.dense(tokens)
                          if isinstance(attention_mask, _WindowAttentionMask) else attention_mask)
            output, weights = self.attention.attn(
                query, query, query, attn_mask=dense_mask,
                need_weights=bool(class_count), average_attn_weights=True)
            output = output.transpose(0, 1)
            if class_count:
                if log_attention:
                    count, length, dims = normalized.shape
                    heads = mha.num_heads
                    projected = F.linear(normalized, mha.in_proj_weight, mha.in_proj_bias)
                    q, k, _ = projected.reshape(
                        count, length, 3, heads, dims // heads).permute(2, 0, 3, 1, 4).unbind(0)
                    bias = dense_mask
                    if bias is not None and bias.ndim == 3:
                        bias = bias.reshape(count, heads, length, length)
                    class_attention = self._class_readout(
                        q, k, bias, spatial_count, class_count, logarithmic=True,
                        return_visual=return_visual_attention)
                    if return_visual_attention:
                        class_attention, visual_attention = class_attention
                else:
                    class_attention = weights[
                        :, 2 * spatial_count:3 * spatial_count,
                        3 * spatial_count:3 * spatial_count + class_count]
        else:
            count, length, dims = normalized.shape
            heads = mha.num_heads
            projected = F.linear(normalized, mha.in_proj_weight, mha.in_proj_bias)
            if attention_mask.order is not None:
                projected = projected.index_select(0, attention_mask.order)
            q, k, v = projected.reshape(
                count, length, 3, heads, dims // heads).permute(2, 0, 3, 1, 4).unbind(0)
            outputs, readouts = [], []
            start = 0
            for size, bias in attention_mask.groups:
                stop = start + size
                outputs.append(F.scaled_dot_product_attention(
                    q[start:stop], k[start:stop], v[start:stop], attn_mask=bias,
                    dropout_p=mha.dropout if self.training else 0.))
                if class_count:
                    # Only fusion-query rows need materializing; supervised
                    # readouts never round tiny probabilities down to zero.
                    readouts.append(self._class_readout(
                        q[start:stop], k[start:stop], bias, spatial_count,
                        class_count, logarithmic=log_attention,
                        return_visual=return_visual_attention))
                start = stop
            output = outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=0)
            if class_count:
                if return_visual_attention:
                    visuals = [readout[1] for readout in readouts]
                    visual_attention = visuals[0] if len(visuals) == 1 else torch.cat(visuals)
                    readouts = [readout[0] for readout in readouts]
                class_attention = (readouts[0] if len(readouts) == 1
                                   else torch.cat(readouts, dim=0))
            if attention_mask.inverse is not None:
                output = output.index_select(0, attention_mask.inverse)
                if class_count:
                    class_attention = class_attention.index_select(0, attention_mask.inverse)
                    if return_visual_attention:
                        visual_attention = visual_attention.index_select(0, attention_mask.inverse)
            output = F.linear(output.transpose(1, 2).reshape(count, length, dims),
                              mha.out_proj.weight, mha.out_proj.bias)
        tokens = tokens + self.attention.dropout_layer(self.attention.proj_drop(output))
        tokens = self.ffn(self.norm2(tokens), identity=tokens)
        if return_visual_attention:
            return tokens, class_attention, visual_attention
        return (tokens, class_attention) if class_count else tokens


class _WindowFusionLevel(BaseModule):
    """Fuse one aligned pair of BCHW feature maps with dense fusion queries."""

    def __init__(self, channels, dims, window_size, heads, encoder_layers,
                 expansion, dropout, use_relative_position_bias,
                 fusion_norm_cfg=None, use_shifted_window=True, use_noise_text=True,
                 use_outer_residual=False, freeze_text_updates=False):
        super().__init__()
        self.window_size = window_size
        self.num_heads = heads
        if isinstance(use_shifted_window, bool):
            # Keep the original API: a shifted boolean enables an ordinary /
            # shifted pair for every logical encoder layer.
            self.shifted_window_schedule = None
            self._block_shifted = tuple(
                shifted
                for _ in range(encoder_layers)
                for shifted in ((False, True) if use_shifted_window else (False,)))
            self._block_depths = tuple(
                depth
                for depth in range(1, encoder_layers + 1)
                for _ in ((False, True) if use_shifted_window else (False,)))
        elif (isinstance(use_shifted_window, (list, tuple))
              and len(use_shifted_window) == encoder_layers
              and all(isinstance(value, bool) for value in use_shifted_window)):
            # A schedule represents one block per logical layer. This allows
            # arbitrary transitions, e.g. regular -> shifted -> regular.
            self.shifted_window_schedule = tuple(use_shifted_window)
            self._block_shifted = self.shifted_window_schedule
            self._block_depths = tuple(range(1, encoder_layers + 1))
        else:
            raise ValueError(
                'use_shifted_window must be boolean or a list/tuple of booleans '
                'with length encoder_layers.')
        self.use_shifted_window = use_shifted_window
        self.use_noise_text = use_noise_text
        self.use_outer_residual = use_outer_residual
        self.freeze_text_updates = bool(freeze_text_updates)
        self.input_projection = nn.Linear(channels, dims)
        self.input_norm = nn.LayerNorm(dims)
        # Same pointwise concatenation fusion as FuseNeck; no activation.
        norm = (nn.Identity() if fusion_norm_cfg is None
                else build_norm_layer(fusion_norm_cfg, channels)[1])
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(2 * channels, channels, kernel_size=1, bias=False), norm)
        self.fusion_projection = nn.Linear(channels, dims)
        self.fusion_norm = nn.LayerNorm(dims)
        # pre, post, fusion, damage-class text; optional nuisance text.
        self.type_embeddings = nn.Parameter(torch.empty(5 if use_noise_text else 4, dims))
        nn.init.normal_(self.type_embeddings, std=0.02)
        self.blocks = nn.ModuleList([
            _JointAttentionBlock(dims, heads, expansion, dropout)
            for _ in self._block_shifted
        ])
        if use_relative_position_bias:
            w = window_size
            self.relative_position_bias_table = nn.Parameter(
                torch.empty((2 * w - 1) ** 2, heads))
            nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
            coords = torch.stack(torch.meshgrid(
                torch.arange(w), torch.arange(w), indexing='ij')).flatten(1)
            relative = coords[:, :, None] - coords[:, None, :]
            index = (relative[0] + w - 1) * (2 * w - 1) + relative[1] + w - 1
            self.register_buffer('relative_position_index', index,
                                 persistent=False)
        else:
            self.register_parameter('relative_position_bias_table', None)
        self.output_norm = nn.LayerNorm(dims)
        self.output_projection = nn.Linear(dims, channels)

    def _attention_mask(self, layout, text_count, reference):
        """Compact additive bias: broadcast interior and masked boundary windows.

        Repeating the SxS block covers all pre/post/fusion pairings, including
        cross-time ones. Text has no relative 2-D position bias.
        Padding keys in ALL three spatial groups are masked; padded queries
        can read valid keys, but their outputs are discarded on cropping.
        Only fusion queries subtract log(N_visual) / log(N_text) from the
        respective key groups. Counts include all text (also nuisance prompts)
        and only valid visual keys, independently for each boundary window.
        """
        valid_windows = layout['valid']
        count, spatial_count = valid_windows.shape
        length = 3 * spatial_count + text_count
        if self.relative_position_bias_table is None:
            bias = None
        else:
            bias = self.relative_position_bias_table[
                self.relative_position_index.reshape(-1)]
            bias = bias.reshape(spatial_count, spatial_count, self.num_heads)
            bias = bias.permute(2, 0, 1).to(dtype=reference.dtype)
            bias = F.pad(bias.repeat(1, 3, 3), (0, text_count, 0, text_count))
        fusion_begin, visual_end = 2 * spatial_count, 3 * spatial_count
        if text_count:
            if bias is None:
                bias = reference.new_zeros(self.num_heads, length, length)
            bias[:, fusion_begin:visual_end, :visual_end] -= math.log(visual_end)
            bias[:, fusion_begin:visual_end, visual_end:] -= math.log(text_count)
        invalid = torch.cat((~valid_windows, ~valid_windows, ~valid_windows,
            torch.zeros(count, text_count, dtype=torch.bool, device=reference.device)), dim=1)
        # One broadcast bias for interior windows; only boundary windows
        # need per-window padding. Two SDPA calls avoid a kernel launch for
        # every corner/edge pattern, keeping FP32 training throughput.
        interior = [i for (y0, y1, x0, x1), indices in layout['groups']
                    if (y1 - y0) * (x1 - x0) == spatial_count for i in indices]
        if len(interior) == count:
            return _WindowAttentionMask(bias, ((count, bias),), None, None,
                                        invalid, self.num_heads)
        interior_set = set(interior)
        boundary = [i for i in range(count) if i not in interior_set]
        ordering = interior + boundary
        boundary_indices = torch.tensor(boundary, dtype=torch.long, device=reference.device)
        boundary_bias = (reference.new_zeros(self.num_heads, length, length)
                         if bias is None else bias)
        boundary_bias = boundary_bias.unsqueeze(0).masked_fill(
            invalid.index_select(0, boundary_indices)[:, None, None, :], float('-inf'))
        if text_count:
            # Layout metadata supplies exact counts without a GPU -> CPU sync.
            valid_counts = {i: 3 * (y1 - y0) * (x1 - x0)
                            for (y0, y1, x0, x1), indices in layout['groups']
                            for i in indices}
            correction = reference.new_tensor([
                math.log(visual_end) - math.log(valid_counts[i]) for i in boundary])
            boundary_bias[:, :, fusion_begin:visual_end, :visual_end] += (
                correction[:, None, None, None])
        if not interior:
            return _WindowAttentionMask(bias, ((count, boundary_bias),), None, None,
                                        invalid, self.num_heads)
        groups = ((len(interior), bias), (len(boundary), boundary_bias))
        order = torch.tensor(ordering, dtype=torch.long, device=reference.device)
        inverse_order = [0] * count
        for new, old in enumerate(ordering):
            inverse_order[old] = new
        inverse = torch.tensor(inverse_order, dtype=torch.long, device=reference.device)
        return _WindowAttentionMask(bias, groups, order, inverse, invalid, self.num_heads)

    def _restore_spatial(self, windows, layout, batch, height, width):
        dense = _window_reverse(windows, self.window_size, batch,
                                *layout['padded_shape'])
        top, left = layout['offset']
        return dense[:, top:top + height, left:left + width]

    def _repartition(self, tokens, source, target, neighbors, batch, height, width):
        """Move three visual grids and their updated text to the next grid."""
        spatial_count = self.window_size ** 2
        visual = []
        for group in range(3):
            dense = self._restore_spatial(
                tokens[:, group * spatial_count:(group + 1) * spatial_count],
                source, batch, height, width)
            visual.append(_window_partition(
                F.pad(dense, target['padding']), self.window_size))
        text = tokens[:, 3 * spatial_count:]
        text = text.reshape(batch, len(source['boxes']), -1)
        weights, indices = neighbors
        remapped = text.new_zeros((batch, len(target['boxes']), text.shape[-1]))
        # A window overlaps at most four windows of the other grid. Gather
        # only those states instead of multiplying by a dense Nw x Nw matrix
        # or materializing [B,Nw,4,M,D]. This also bounds remapping memory.
        for index in range(weights.shape[1]):
            remapped = remapped + text.index_select(1, indices[:, index]) * (
                weights[:, index].to(dtype=text.dtype)[None, :, None])
        remapped = remapped.reshape(batch * len(target['boxes']), -1, tokens.shape[-1])
        return torch.cat((*visual, remapped), dim=1)

    def _fusion_output(self, tokens, layout, initial_fusion):
        batch, _, height, width = initial_fusion.shape
        spatial_count = self.window_size ** 2
        fused = self._restore_spatial(
            tokens[:, 2 * spatial_count:3 * spatial_count], layout, batch, height, width)
        correction = self.output_projection(self.output_norm(fused))
        output = correction.permute(0, 3, 1, 2)
        if self.use_outer_residual:
            output = initial_fusion + output
        return output.contiguous()

    def _class_context(self, tokens, layout, batch, height, width, class_count):
        window_text = tokens[:, 3 * self.window_size ** 2:
                             3 * self.window_size ** 2 + class_count]
        return dict(tokens=window_text.reshape(
            batch, -1, class_count, window_text.shape[-1]).contiguous(),
            window_boxes=layout['boxes'], feature_shape=(height, width))

    def forward(self, pre, post, class_text, noise_text,
                return_class_attention=False, return_context=False,
                supervision_depth=0, supervision_weights=None,
                return_visual=False):
        batch, _, height, width = pre.shape
        w = self.window_size
        initial_fusion = self.fusion_conv(torch.cat((pre, post), dim=1))
        fusion = self.fusion_norm(self.fusion_projection(
            initial_fusion.permute(0, 2, 3, 1)))
        pre = self.input_norm(self.input_projection(pre.permute(0, 2, 3, 1)))
        post = self.input_norm(self.input_projection(post.permute(0, 2, 3, 1)))
        position = _position_encoding(height, width, pre.shape[-1], pre)
        regular = _window_layout(height, width, w, 0, pre, batch)
        has_requested_shift = any(self._block_shifted)
        shifted = (_window_layout(height, width, w, w // 2, pre, batch)
                   if has_requested_shift else regular)
        has_shift = shifted['offset'] != regular['offset']
        layouts = (regular, shifted)
        # Pad BHWD tensors along H/W, leaving the feature dimension intact.
        padding = regular['padding']
        pre = _window_partition(F.pad(pre, padding), w)
        post = _window_partition(F.pad(post, padding), w)
        fusion = _window_partition(F.pad(fusion, padding), w)
        position = _window_partition(
            F.pad(position, padding).expand(batch, -1, -1, -1), w)
        types = self.type_embeddings.to(dtype=pre.dtype)
        fusion = fusion + position + types[2]
        # Text is the same fixed prompt bank for every image/window. expand
        # keeps the batch/window ordering aligned without learning per-window text.
        class_tokens = class_text.to(dtype=pre.dtype).expand(pre.shape[0], -1, -1)
        groups = [pre + position + types[0], post + position + types[1],
                  fusion, class_tokens + types[3]]
        text_count = class_tokens.shape[1]
        if self.use_noise_text:
            noise_tokens = noise_text.to(dtype=pre.dtype).expand(pre.shape[0], -1, -1)
            groups.append(noise_tokens + types[4])
            text_count += noise_tokens.shape[1]
        tokens = torch.cat(groups, dim=1)
        static_prompt_tokens = [class_text[0].to(dtype=pre.dtype) + types[3]]
        if self.use_noise_text:
            static_prompt_tokens.append(noise_text[0].to(dtype=pre.dtype) + types[4])
        static_prompt_tokens = torch.cat(static_prompt_tokens, dim=0)
        mask = self._attention_mask(regular, text_count, tokens)
        masks = (mask, self._attention_mask(shifted, text_count, tokens) if has_shift else mask)
        if has_shift:
            # Constant geometry is reused by every encoder pair in this call.
            to_shifted = _window_overlap_weights(regular['boxes'], shifted['boxes']).topk(
                min(4, len(regular['boxes'])), dim=1)
            to_regular = _window_overlap_weights(shifted['boxes'], regular['boxes']).topk(
                min(4, len(shifted['boxes'])), dim=1)
        current = 0
        spatial_count = w * w
        auxiliary_states = []
        attention = None
        for i, block in enumerate(self.blocks):
            target = int(has_shift and self._block_shifted[i])
            if target != current:
                tokens = self._repartition(
                    tokens, layouts[current], layouts[target],
                    to_shifted if target else to_regular, batch, height, width)
                current = target
            depth = self._block_depths[i]
            supervised = depth <= supervision_depth and (
                i == len(self.blocks) - 1
                or self._block_depths[i + 1] != depth)
            enabled = supervised and (supervision_weights is None
                                      or supervision_weights[depth - 1] > 0)
            final_readout = return_class_attention and i == len(self.blocks) - 1
            if enabled or final_readout:
                result = block(
                    tokens, masks[current], class_count=class_tokens.shape[1],
                    spatial_count=spatial_count, log_attention=enabled,
                    return_visual_attention=enabled)
                tokens, class_attention = result[:2]
                spatial_attention = self._restore_spatial(
                    class_attention, layouts[current], batch, height, width)
                spatial_attention = spatial_attention.permute(0, 3, 1, 2).contiguous()
                if enabled:
                    visual_attention = self._restore_spatial(
                        result[2], layouts[current], batch, height, width)
                    visual_attention = visual_attention.permute(0, 3, 1, 2).contiguous()
                if final_readout:
                    # Public visualization keeps its probability-valued API.
                    attention = spatial_attention.exp() if enabled else spatial_attention
            else:
                tokens = block(tokens, masks[current])
            if self.freeze_text_updates:
                tokens[:, 3 * spatial_count:] = static_prompt_tokens.unsqueeze(0).expand(
                    tokens.shape[0], -1, -1)
            if supervised:
                state = dict(attention=spatial_attention if enabled else None,
                             attention_is_log=True,
                             visual_attention=visual_attention if enabled else None)
                auxiliary_states.append(state)
        final_layout = layouts[current]
        fused = self._fusion_output(tokens, final_layout, initial_fusion)
        if return_context:
            context = self._class_context(
                tokens, final_layout, batch, height, width, class_tokens.shape[1])
        else:
            context = None
        if return_visual:
            return fused, attention, context, tuple(auxiliary_states), initial_fusion
        if supervision_depth:
            return fused, attention, context, tuple(auxiliary_states)
        if return_class_attention or return_context:
            return fused, attention, context
        return fused


@MODELS.register_module()
class VisualFusionNeck(BaseModule):
    """Text-free concatenation/1x1-convolution baseline for ablations.

    Construct only the used convolution parameters, so DDP needs no unused
    attention/text parameters when use_text_fusion=False.
    """

    def __init__(self, channels, fusion_norm_cfg=None, init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        if not channels or any(type(c) is not int or c <= 0 for c in channels):
            raise ValueError('channels must contain positive integers.')
        self.channels = tuple(channels)
        self.levels = nn.ModuleList([nn.Sequential(
            nn.Conv2d(2 * c, c, 1, bias=False),
            nn.Identity() if fusion_norm_cfg is None else build_norm_layer(fusion_norm_cfg, c)[1])
            for c in channels])

    def forward(self, features):
        pre, post = features
        if len(pre) != len(self.channels) or len(post) != len(pre):
            raise ValueError('Visual fusion pyramid levels do not match channels.')
        outputs = []
        for c, a, b, level in zip(self.channels, pre, post, self.levels):
            if a.ndim != 4 or a.shape != b.shape or a.shape[1] != c:
                raise ValueError('Visual fusion requires matching BCHW features.')
            outputs.append(level(torch.cat((a, b), dim=1)))
        return tuple(outputs)


@MODELS.register_module()
class TextGuidedFusionNeck(BaseModule):
    """Bitemporal window joint attention with text and dense fusion queries.

    Inputs: ``(pre_pyramid, post_pyramid)``; paired BCHW shapes must match.
    Outputs: a tuple with the same per-level BCHW shapes, suitable for FPN.
    Default per-window sequence: 64 pre + 64 post + 64 fusion + 3 class +
    5 nuisance = 200 tokens. Text prompts are shared by all images; GT is
    not an input. No token has a forced class/noise interpretation.

    Args:
        channels: Positive channel counts, e.g. [64, 128, 320, 512].
        text_encoder_path: Local BGE model and tokenizer directory.
        class_prompts: Class names, e.g. ('intact', 'damaged', 'destroyed'),
            or per-class phrase groups, e.g. (('intact', 'undamaged building'),
            ('damaged', 'partially damaged building'), ('destroyed',)). Each
            phrase is independently BGE-encoded, L2-normalized and projected
            into its own token. Groups are flattened in class order without
            averaging; TextBank records each token's class index.
            More phrases increase the attention sequence length.
        noise_prompts: Unique nuisance phrases. None/empty disables nuisance
            text and its type embedding in a standalone neck.
        use_noise_text: Whether to include the fifth, nuisance type. For an
            internal text bank, None infers it from noise_prompts. With external
            text, None defaults to True; TextHTC sets it from its shared bank.
            Set False for standalone external calls with noise_text=None/empty.
        embed_dims: Attention width, divisible by 4 and num_heads.
        window_size: Positive square window width. Default 8.
        num_heads: Number of attention heads. Default 4.
        encoder_layers: Ordinary/shifted joint-attention pairs per level.
            Default 1 means two attention+FFN blocks when shifting is enabled.
        attn_level: Number of encoder-layer outputs to supervise, starting
            from layer 1. Must be in [1, encoder_layers]; None uses all layers.
            This is encoder depth, not a backbone pyramid index. Every depth
            reads fusion-query to class-text attention from its last block.
        attn_loss_weights: List/tuple of nonnegative finite weights, with
            length attn_level. None uses ones. Zero omits that depth's loss.
        attn_loss_type: 'instance' pools GT-mask attention and detached visual
            semantic reliability before budget KL. 'position' computes the
            same KL before mask pooling. 'sampled' is no longer supported.
        semantic_num_classes: Foreground class count (background is channel
            zero). TextHTC supplies its box/text class count. Standalone
            external-text callers must supply this when supervision is on.
        semantic_channels: Training-only visual semantic decoder width, 64.
        lambda_sem: Auxiliary visual semantic CE weight, default 0.2, matching
            native HTC semantic loss. No decoder is built if attention
            supervision is disabled at construction. GT semantics must use
            background=0 and foreground=text class index+1; 255 is ignored.
        lambda_attn: Finite nonnegative overall attention loss weight.
            Default 0.1. Zero disables both auxiliary supervision losses.
        stride_level: Supervise the first N pyramid scales, from fine to
            coarse. For PVT, 2 selects stride 4 and 8. Must be in
            [1, len(channels)]. None preserves the stride-8-only behavior.
        stride_loss_weights: Nonnegative finite scale weights, with length
            stride_level (one for legacy None). None uses ones. Scale and
            depth weights multiply; zero skips that scale's readout/loss.
        use_shifted_window: Enable half-window shifts. A boolean preserves the
            legacy behavior: True builds ordinary/shifted pairs and False
            builds encoder_layers ordinary blocks. A list/tuple of booleans of
            length encoder_layers selects shifted-window use independently for
            each layer and builds exactly encoder_layers blocks.
        ffn_expansion: FFN hidden width / embed_dims. Default 4.
        dropout: Attention/projection/FFN dropout. Default 0.
        use_relative_position_bias: Learn spatial offset biases. Default True.
        max_text_length: Reject overlong prompts instead of truncating.
        external_text: Receive (M, D)/(K, D) tensors from TextHTC. Default False.
        fusion_norm_cfg: Optional MMCV normalization after the initial
            1x1 convolution, as in FuseNeck. Default None (identity).
            Projected fusion tokens always have LayerNorm.
        use_outer_residual: Add the initial convolutional fusion feature to
            the final projected fusion tokens. Default False. Attention and
            FFN block residuals are always retained.

    Pre/post projection weights are shared within each level. Each level has
    independent convolutional fusion, projections, types, attention blocks
    and spatial biases;
    text projection is shared across levels and both text groups. Positions
    are computed before padding, and padded spatial keys cannot influence
    valid queries. Windows smaller than window_size are supported by padding.
    Updated text follows each new grid by area-weighting states of the same
    prompt in overlapping old windows. The final context uses the actual
    shifted window boxes for classification. Auxiliary attention maps are
    restored to their original spatial positions before mask-based GT loss.
    Classification text and optional visualization stay at the second feature
    level (stride 8 for PVT), or the only level of a single-level neck.
    Auxiliary attention supervision may cover the first stride_level scales.
    Classification and FPN always read the final encoder output; attn_level
    only selects how many encoder depths receive auxiliary attention losses.
    """

    def __init__(self, channels: Sequence[int], text_encoder_path=None,
                 class_prompts=None, noise_prompts=None,
                 embed_dims=128, window_size=8, num_heads=4, encoder_layers=1,
                 ffn_expansion=4, dropout=0., use_relative_position_bias=True,
                 max_text_length=128, external_text=False,
                 init_cfg=None, fusion_norm_cfg=None,
                 use_shifted_window=True, attn_level=None,
                 attn_loss_weights=None, stride_level=None, stride_loss_weights=None,
                 use_noise_text=None, attn_loss_type='instance', lambda_attn=0.1,
                 semantic_num_classes=None, semantic_channels=64, lambda_sem=0.2,
                 reliability_mode='gt', preference_mode='learned',
                 use_outer_residual=False, freeze_text_updates=False):
        super().__init__(init_cfg=init_cfg)
        if (not channels or any(type(c) is not int or c <= 0 for c in channels)):
            raise ValueError('channels must contain positive integers.')
        for name, value in (('embed_dims', embed_dims), ('window_size', window_size),
                            ('num_heads', num_heads), ('encoder_layers', encoder_layers),
                            ('ffn_expansion', ffn_expansion),
                            ('semantic_channels', semantic_channels),
                            ('max_text_length', max_text_length)):
            if type(value) is not int or value < 1:
                raise ValueError(f'{name} must be a positive integer.')
        if embed_dims % 4 or embed_dims % num_heads:
            raise ValueError('embed_dims must be divisible by 4 and num_heads.')
        if not isinstance(use_relative_position_bias, bool):
            raise ValueError('use_relative_position_bias must be boolean.')
        if not isinstance(external_text, bool):
            raise ValueError('external_text must be boolean.')
        if isinstance(use_shifted_window, bool):
            shifted_window_schedule = None
        elif (isinstance(use_shifted_window, (list, tuple))
              and len(use_shifted_window) == encoder_layers
              and all(isinstance(value, bool) for value in use_shifted_window)):
            shifted_window_schedule = tuple(use_shifted_window)
        else:
            raise ValueError(
                'use_shifted_window must be boolean or a list/tuple of booleans '
                'with length encoder_layers.')
        if not isinstance(use_outer_residual, bool):
            raise ValueError('use_outer_residual must be boolean.')
        if use_noise_text is not None and not isinstance(use_noise_text, bool):
            raise ValueError('use_noise_text must be boolean or None.')
        if not 0 <= dropout < 1:
            raise ValueError('dropout must be in [0, 1).')
        if (isinstance(lambda_attn, bool) or not isinstance(lambda_attn, (int, float))
                or not math.isfinite(lambda_attn) or lambda_attn < 0):
            raise ValueError('lambda_attn must be a finite nonnegative number.')
        self.lambda_attn = float(lambda_attn)
        self.channels = tuple(channels)
        self.embed_dims = embed_dims
        self.window_size = window_size
        self.external_text = external_text
        self._class_prompt_counts = None
        if attn_loss_type not in ('instance', 'position'):
            raise ValueError('attn_loss_type must be instance or position; sampled was removed.')
        self.attn_loss_type = attn_loss_type
        if reliability_mode not in ('gt', 'fixed', 'shuffle', 'maxsoft', 'entropy'):
            raise ValueError('reliability_mode must be gt, fixed, shuffle, maxsoft, or entropy.')
        if preference_mode not in ('learned', 'uniform', 'shuffle'):
            raise ValueError('preference_mode must be learned, uniform, or shuffle.')
        self.reliability_mode = reliability_mode
        self.preference_mode = preference_mode
        if (isinstance(lambda_sem, bool) or not isinstance(lambda_sem, (int, float))
                or not math.isfinite(lambda_sem) or lambda_sem <= 0):
            raise ValueError('lambda_sem must be finite and positive.')
        self.lambda_sem = float(lambda_sem)
        self.use_shifted_window = (use_shifted_window if shifted_window_schedule is None
                                   else shifted_window_schedule)
        self.use_outer_residual = use_outer_residual
        self._readout_level = min(1, len(self.channels) - 1)
        self.stride_level = stride_level
        if stride_level is not None and (
                type(stride_level) is not int or not 1 <= stride_level <= len(self.channels)):
            raise ValueError('stride_level must be an integer in [1, len(channels)], or None.')
        self._supervision_levels = ((self._readout_level,) if stride_level is None
                                    else tuple(range(stride_level)))
        if stride_loss_weights is None:
            stride_loss_weights = (1.,) * len(self._supervision_levels)
        if (not isinstance(stride_loss_weights, (list, tuple))
                or len(stride_loss_weights) != len(self._supervision_levels)):
            raise ValueError('stride_loss_weights length must match the supervised scale count.')
        if any(isinstance(weight, bool) or not isinstance(weight, (int, float))
               or not math.isfinite(weight) or weight < 0 for weight in stride_loss_weights):
            raise ValueError('stride_loss_weights must contain finite nonnegative numbers.')
        self.stride_loss_weights = tuple(float(weight) for weight in stride_loss_weights)
        self.encoder_layers = encoder_layers
        self.attn_level = encoder_layers if attn_level is None else attn_level
        if type(self.attn_level) is not int or not 1 <= self.attn_level <= encoder_layers:
            raise ValueError('attn_level must be an integer in [1, encoder_layers].')
        if attn_loss_weights is None:
            attn_loss_weights = (1.,) * self.attn_level
        if (not isinstance(attn_loss_weights, (list, tuple))
                or len(attn_loss_weights) != self.attn_level):
            raise ValueError('attn_loss_weights must be a list/tuple of length attn_level.')
        if any(isinstance(weight, bool) or not isinstance(weight, (int, float))
               or not math.isfinite(weight) or weight < 0 for weight in attn_loss_weights):
            raise ValueError('attn_loss_weights must contain finite nonnegative numbers.')
        self.attn_loss_weights = tuple(float(weight) for weight in attn_loss_weights)
        if external_text:
            if any(v is not None for v in
                   (text_encoder_path, class_prompts, noise_prompts)):
                raise ValueError('External text belongs to detector.text_bank; '
                                 'do not configure another bank in the neck.')
        else:
            if text_encoder_path is None:
                raise ValueError('Standalone fusion needs text_encoder_path.')
            self.text_bank = TextBank(
                text_encoder_path, class_prompts, noise_prompts,
                embed_dims=embed_dims, max_text_length=max_text_length)
        if not external_text:
            self._class_prompt_counts = self.text_bank.class_prompt_counts
            if semantic_num_classes is None:
                semantic_num_classes = self.text_bank.num_classes
            elif semantic_num_classes != self.text_bank.num_classes:
                raise ValueError('semantic_num_classes must match the text bank.')
            has_noise = bool(self.text_bank.noise_prompts)
            if use_noise_text is not None and use_noise_text != has_noise:
                raise ValueError('use_noise_text must match the internal text bank.')
            use_noise_text = has_noise
        self.use_noise_text = True if use_noise_text is None else use_noise_text
        self.semantic_head = None
        # Keep the visual semantic probe available when lambda_attn=0.  This
        # is the controlled "no attention-loss" variant: it removes the
        # budget supervision while retaining the similarity/reliability
        # readout for diagnostics.
        if self.lambda_sem > 0 or self.lambda_attn > 0:
            if type(semantic_num_classes) is not int or semantic_num_classes < 1:
                raise ValueError('Set positive semantic_num_classes for visual supervision.')
            self.semantic_head = VisualSemanticHead(
                self.channels, semantic_num_classes, semantic_channels)
        self.levels = nn.ModuleList([
            _WindowFusionLevel(c, embed_dims, window_size, num_heads,
                               encoder_layers, ffn_expansion, dropout,
                               use_relative_position_bias, fusion_norm_cfg,
                               self.use_shifted_window, self.use_noise_text,
                               use_outer_residual=use_outer_residual,
                               freeze_text_updates=freeze_text_updates)
            for c in self.channels
        ])

    @property
    def has_attn_supervision(self):
        """Whether the semantic/attention readout is constructed.

        The readout remains active when ``lambda_attn=0`` so controlled
        no-attention-loss runs can still export reliability diagnostics.
        """
        return (self.semantic_head is not None and any(self.attn_loss_weights)
                and any(self.stride_loss_weights))

    def forward(self, inputs, class_text=None, noise_text=None,
                return_attention=False, return_context=False, return_aux=False):
        """Return features and optionally attention/context without detaching.

        With return_context, return (features, attention_or_None, context).
        Otherwise return features or (features, BxMxHxW class attention).
        With return_aux, return (features, attention_or_None, context_or_None,
        auxiliary_states). With stride_level=None this is the legacy tuple of
        depth states. Otherwise it maps pyramid index to depth-state tuples,
        omitting zero-weight scales. GT never enters this forward path.
        Auxiliary states store log probabilities in ``attention`` and mark
        ``attention_is_log=True``; public return_attention remains probabilities.
        Prediction does not request maps.
        """
        if not isinstance(inputs, (tuple, list)) or len(inputs) != 2:
            raise ValueError('Expected (pre_pyramid, post_pyramid).')
        pre_features, post_features = inputs
        if (len(pre_features) != len(self.channels)
                or len(post_features) != len(self.channels)):
            raise ValueError('Feature pyramid levels do not match channels.')
        if self.external_text:
            if class_text is None:
                raise ValueError('External fusion requires class_text.')
            if noise_text is None:
                if self.use_noise_text:
                    raise ValueError('Provide noise_text or build with use_noise_text=False.')
                noise_text = class_text.new_empty((0, self.embed_dims))
        else:
            if class_text is not None or noise_text is not None:
                raise ValueError('Set external_text=True to supply shared text.')
            class_text, noise_text = self.text_bank()
        for name, text in (('class_text', class_text), ('noise_text', noise_text)):
            minimum = 0 if name == 'noise_text' and not self.use_noise_text else 1
            if (text.ndim != 2 or text.shape[0] < minimum
                    or text.shape[1] != self.embed_dims):
                raise ValueError(f'{name} must have shape (tokens, embed_dims).')
            if text.device != pre_features[0].device:
                raise ValueError('Text and visual features must share a device.')
        if not self.use_noise_text and len(noise_text):
            raise ValueError('use_noise_text=False requires noise_text=None or an empty tensor.')
        class_text = class_text.unsqueeze(0)
        noise_text = noise_text.unsqueeze(0)
        outputs, attention, context = [], None, None
        auxiliary_states = {} if self.stride_level is not None else None
        collect_visual = return_aux and self.has_attn_supervision
        visual_features = []
        scale_weights = dict(zip(self._supervision_levels, self.stride_loss_weights))
        for i, (pre, post, level) in enumerate(zip(
                pre_features, post_features, self.levels)):
            if (pre.ndim != 4 or post.shape != pre.shape
                    or pre.shape[1] != self.channels[i]
                    or min(pre.shape[0], pre.shape[2], pre.shape[3]) < 1
                    or pre.device != post.device or pre.dtype != post.dtype):
                raise ValueError(f'Level {i} requires matching nonempty BCHW '
                                 f'features with C={self.channels[i]}, '
                                 'on the same device and with the same dtype.')
            need_attention = return_attention and i == self._readout_level
            need_context = return_context and i == self._readout_level
            need_aux = (return_aux and self.has_attn_supervision
                        and scale_weights.get(i, 0) > 0)
            if need_attention or need_context or need_aux or collect_visual:
                result = level(
                    pre, post, class_text, noise_text,
                    return_class_attention=need_attention, return_context=need_context,
                    supervision_depth=self.attn_level if need_aux else 0,
                    supervision_weights=self.attn_loss_weights if need_aux else None,
                    return_visual=collect_visual)
                fused, level_attention, level_context = result[:3]
                if need_aux:
                    if self.stride_level is None:
                        auxiliary_states = result[3]
                    else:
                        auxiliary_states[i] = result[3]
                if need_attention:
                    attention = level_attention
                if need_context:
                    context = level_context
                if collect_visual:
                    visual_features.append(result[4])
                outputs.append(fused)
            else:
                outputs.append(level(pre, post, class_text, noise_text))
        if (not collect_visual and self.semantic_head is not None
                and self.training and torch.is_grad_enabled()):
            # Runtime ablations may disable weights after DDP construction.
            # Skip the decoder, but retain zero gradients for its parameters.
            zero = sum(p.reshape(-1)[:0].sum() for p in self.semantic_head.parameters())
            outputs[0] = outputs[0] + zero
        if return_aux:
            if collect_visual:
                logits = self.semantic_head(visual_features)
                scales = (auxiliary_states.values() if isinstance(auxiliary_states, dict)
                          else (auxiliary_states,))
                for states in scales:
                    for state in states:
                        if state['attention'] is not None:
                            state['semantic_logits'] = logits
            return tuple(outputs), attention, context, auxiliary_states
        if return_context:
            return tuple(outputs), attention, context
        return (tuple(outputs), attention) if return_attention else tuple(outputs)

    def loss_attn(self, auxiliary_states, class_token_class_ids,
                  batch_data_samples, input_shape, *, prepared=None, normalizer=None,
                  prompt_context=None):
        """Semantic CE once, and scale/depth-weighted attention budget KL.

        Reuse one geometry preparation and one detached semantic probability
        map per resolution. All counts share the existing asynchronous
        collective; loss logs retain the native rank-local convention.
        """
        if not self.has_attn_supervision:
            return {}
        scales = ({self._readout_level: auxiliary_states} if self.stride_level is None
                  else auxiliary_states)
        active_scales = tuple(i for i, weight in zip(
            self._supervision_levels, self.stride_loss_weights) if weight > 0)
        if not isinstance(scales, dict) or set(scales) != set(active_scales):
            raise ValueError('Auxiliary scale states do not match positive scale weights.')
        if any(len(states) != self.attn_level for states in scales.values()):
            raise ValueError('Auxiliary state count must equal attn_level.')
        states = [state for group in scales.values()
                  for state, weight in zip(group, self.attn_loss_weights) if weight > 0]
        spatial_shapes = {tuple(state['attention'].shape[-2:]) for state in states}
        semantic_logits = states[0]['semantic_logits']
        if prepared is None:
            prepared = self._prepare_attn_targets(
                states[0]['attention'], class_token_class_ids, batch_data_samples,
                input_shape, spatial_shapes=spatial_shapes)
        if semantic_logits.shape[1] != len(prepared['groups']) + 1:
            raise ValueError('Semantic channels must be background plus text classes.')
        semantic_loss, pixel_count = self.semantic_head.loss_terms(
            semantic_logits, batch_data_samples, input_shape)
        own_normalizer = normalizer is None
        if own_normalizer:
            normalizer = TextLossNormalizer(len(active_scales) + 1)
        normalizer.register(('visual_semantic',), pixel_count)
        normalizer.add('loss_attn_sem', self.lambda_sem * semantic_loss, ('visual_semantic',))
        for scale in active_scales:
            shape = next(tuple(state['attention'].shape[-2:]) for state in scales[scale]
                         if state['attention'] is not None)
            normalizer.register(('attn', scale), prepared['geometry'][shape]['validity'].sum())
        # Detach this path only. CE above still trains the semantic decoder and
        # initial visual fusion features. No random prompt selection is used.
        with torch.no_grad():
            semantic_probabilities = {
                shape: F.interpolate(semantic_logits.detach().float(), size=shape,
                    mode='bilinear', align_corners=False).softmax(1)[:, 1:]
                for shape in spatial_shapes}
        if prompt_context is not None:
            # Call-local readouts only; defer the conditional prompt KL until
            # the last RoI stage supplies its teacher. Register counts now to
            # retain the existing asynchronous collective and overlap.
            prompt_context.update(prepared=prepared,
                                  semantic_probabilities=semantic_probabilities, states=[])
        for scale, scale_weight in zip(self._supervision_levels, self.stride_loss_weights):
            if scale_weight == 0:
                continue
            for index, (state, depth_weight) in enumerate(zip(scales[scale], self.attn_loss_weights)):
                if depth_weight == 0:
                    continue
                shape = tuple(state['attention'].shape[-2:])
                readout = {} if prompt_context is not None else None
                numerator = self._loss_attn_state(
                    state['attention'], class_token_class_ids, batch_data_samples,
                    input_shape, prepared=prepared, avg_factor=1.,
                    log_attention=state.get('attention_is_log', False),
                    visual_attention=state['visual_attention'],
                    semantic_probabilities=semantic_probabilities[shape],
                    readout_cache=readout)
                normalizer.add(f'l{index + 1}.loss_attn',
                    self.lambda_attn * scale_weight * depth_weight * numerator, ('attn', scale))
                if prompt_context is not None:
                    prompt_context['states'].append((f'l{index + 1}.loss_attn',
                        self.lambda_attn * scale_weight * depth_weight, ('attn', scale),
                        shape, readout))
        return normalizer.finalize() if own_normalizer else {}

    def loss_attn_prompt(self, context, probabilities, *, normalizer):
        """Add rho * KL(classifier || conditional text attention).

        Together with the existing grouped budget KL this is exactly the
        full target (visual=1-rho, correct prompt k=rho*pi_k, other text=0).
        Reuse registered counts; zero-probability GT rows add no constraint.
        """
        prepared = context['prepared']
        expected = (len(prepared['targets']), prepared['shape'][1])
        if tuple(probabilities.shape) != expected:
            raise ValueError('Classifier prompt probabilities must match GT rows and prompts.')
        probabilities = probabilities.detach().float()
        if self.preference_mode == 'uniform':
            probabilities = torch.ones_like(probabilities) / probabilities.shape[-1]
        elif self.preference_mode == 'shuffle' and probabilities.shape[0] > 1:
            probabilities = probabilities.index_select(
                0, torch.randperm(probabilities.shape[0], device=probabilities.device))
        mass = probabilities.sum(-1)
        entropy = torch.xlogy(probabilities, probabilities).sum(-1)
        for name, weight, key, shape, readout in context['states']:
            numerator = self._prompt_distribution_loss(
                readout, prepared, context['semantic_probabilities'][shape],
                probabilities, mass, entropy)
            normalizer.add(name, weight * numerator, key)

    def _prompt_distribution_loss(self, readout, prepared, quality,
                                  probabilities, mass=None, entropy=None):
        """Instance-specific soft prompt KL, without dense GT x pixel maps."""
        probabilities = probabilities.detach().float()
        mass = probabilities.sum(-1) if mass is None else mass
        entropy = (torch.xlogy(probabilities, probabilities).sum(-1)
                   if entropy is None else entropy)
        log_weights, log_correct = readout['log_weights'], readout['log_correct']
        shape = tuple(log_weights.shape[-2:])
        target = prepared['geometry'][shape]
        if self.attn_loss_type == 'instance':
            # Pool probability first (arithmetic GT mean), then compare the
            # conditional prompt distribution. The budget path supplies rho.
            pixels = log_weights.permute(0, 2, 3, 1).reshape(-1, log_weights.shape[1])
            prompts = pool_log_instances(target['pool'], pixels)
            conditional = (entropy - (probabilities * prompts).sum(-1)
                           + mass * readout['correct'])
            return (readout['rho'] * conditional * target['validity']).sum()
        # Position mode weights each pixel's conditional KL by its own rho.
        # COO is storage only: pool_instances uses bounded indexed sums for
        # forward/backward and never CUDA sparse matrix multiplication.
        cache = prepared.setdefault('prompt_geometry', {})
        if shape not in cache:
            with torch.no_grad():
                pool = target['pool']
                rows, columns = pool.indices()
                labels = prepared['targets'].index_select(0, rows)
                pixels = quality.detach().float().permute(0, 2, 3, 1).reshape(-1, quality.shape[1])
                rho = 1. - pixels[columns, labels]
                values = pool.values() * rho
                weighted = torch.sparse_coo_tensor(pool.indices(), values, pool.shape,
                    device=pool.device, is_coalesced=True)
                rho_mass = values.new_zeros(pool.shape[0]).index_add_(0, rows, values)
                cache[shape] = weighted, rho_mass
        weighted, rho_mass = cache[shape]
        features = torch.cat((log_weights, log_correct), dim=1)
        pixels = features.permute(0, 2, 3, 1).reshape(-1, features.shape[1])
        pooled = pool_instances(weighted, pixels)
        prompts, classes = pooled.split((probabilities.shape[1], quality.shape[1]), dim=1)
        correct = classes.gather(1, prepared['targets'][:, None]).squeeze(1)
        return (rho_mass * entropy - (probabilities * prompts).sum(-1) + mass * correct).sum()

    @staticmethod
    def _global_instance_avg_factor(count):
        """Global valid GT count / world size for DDP gradient averaging.

        Clamp the GLOBAL count before dividing: even a single GT on just
        one rank must retain unit weight in the global mean. Only geometry
        counts are detached; visual/text gradients are unchanged. Vectors
        batch scale collectives; scalars support standalone loss calls.
        """
        count = count.detach().float().clone()
        distributed = torch.distributed
        if distributed.is_available() and distributed.is_initialized():
            distributed.all_reduce(count)
            return count.clamp_min(1.) / distributed.get_world_size()
        return count.clamp_min(1.)

    def _prepare_attn_targets(self, attention, ids, samples, input_shape,
                              *, spatial_shapes=None):
        """Area-map own masks and cache sparse instance/spatial pooling matrices.

        BitmapMasks lives on CPU. Validate/filter boxes there once, avoiding
        GPU nonzero -> CPU indices -> GPU masks at every encoder depth.
        Each row is one GT instance, each column an image/spatial position.
        Coverage sums to one per valid instance. Sparse storage keeps only
        covered cells rather than dense (instances, H, W) caches. Original GT
        row indices are preserved for positive-RoI assignments. No RoIAlign.
        Position mode additionally caches class-wise mask-pooling weights.
        """
        if ids.ndim != 1 or ids.dtype != torch.long or ids.device != attention.device:
            raise ValueError('Class token IDs and attention must share a device.')
        groups, counts = _prompt_groups(ids, attention.shape[1], counts=self._class_prompt_counts)
        height, width = input_shape
        if min(height, width, *attention.shape[-2:]) <= 0:
            raise ValueError('Attention and input shapes must be positive.')
        shapes = {tuple(attention.shape[-2:])} if spatial_shapes is None else set(spatial_shapes)
        if any(len(shape) != 2 or min(shape) <= 0 for shape in shapes):
            raise ValueError('Supervised spatial shapes must be positive (H, W).')
        offsets, labels, total = [], [], 0
        for sample in samples:
            offsets.append(total)
            labels.append(sample.gt_instances.labels.to(device=attention.device))
            total += len(labels[-1])
        targets = torch.cat(labels) if labels else ids.new_empty(0)
        geometry = {shape: dict(indices=[], values=[],
            validity=attention.new_zeros(total, dtype=torch.float32),
            count=attention.new_zeros((), dtype=torch.float32)) for shape in shapes}
        if self.attn_loss_type == 'position':
            for shape, target in geometry.items():
                target['weights'] = attention.new_zeros(
                    (len(samples), len(counts), *shape), dtype=torch.float32)
        with torch.no_grad():
            for image_index, sample in enumerate(samples):
                gt = sample.gt_instances
                labels_cpu = gt.labels.detach().to(device='cpu')
                if ((labels_cpu < 0) | (labels_cpu >= len(counts))).any():
                    raise ValueError('GT class labels do not match the text bank.')
                if len(labels_cpu) == 0:
                    continue
                img_h, img_w = sample.img_shape[:2]
                h, w = min(img_h, height, gt.masks.height), min(img_w, width, gt.masks.width)
                if h <= 0 or w <= 0:
                    continue
                boxes_cpu = get_box_tensor(gt.bboxes).detach().to(
                    device='cpu', dtype=torch.float32).clone()
                finite = torch.isfinite(boxes_cpu).all(1)
                boxes_cpu[:, 0::2].clamp_(0, w)
                boxes_cpu[:, 1::2].clamp_(0, h)
                indices = (finite & (boxes_cpu[:, 2] > boxes_cpu[:, 0])
                           & (boxes_cpu[:, 3] > boxes_cpu[:, 1])).nonzero().flatten()
                if len(indices) == 0:
                    continue
                labels = labels_cpu[indices].to(device=attention.device)
                rows = indices.to(device=attention.device) + offsets[image_index]
                mask_indices = indices.numpy()
                for start in range(0, len(indices), 32):
                    chunk_indices = mask_indices[start:start + 32]
                    if isinstance(gt.masks, BitmapMasks):
                        # BitmapMasks slicing constructs another container and
                        # np.stack copy. Read its array with the same row order.
                        masks = torch.tensor(gt.masks.masks[chunk_indices],
                            dtype=torch.float32, device=attention.device)
                    else:
                        masks = gt.masks[chunk_indices].to_tensor(
                            dtype=torch.float32, device=attention.device)
                    masks = masks[:, :h, :w]
                    # Match network input coordinates; masks outside img_shape
                    # and batch padding never become supervision.
                    masks = masks[:, None]
                    if h != height or w != width:
                        masks = F.pad(masks, (0, width - w, 0, height - h))
                    for shape, target in geometry.items():
                        coverage = F.interpolate(masks, size=shape, mode='area').squeeze(1)
                        mass = coverage.sum((1, 2))
                        coverage = coverage / mass.masked_fill(mass <= 0, 1.)[:, None, None]
                        chunk_rows = rows[start:start + 32]
                        target['validity'].index_copy_(0, chunk_rows, (mass > 0).float())
                        sparse = coverage.flatten(1).to_sparse().coalesce()
                        local = sparse.indices()
                        target['indices'].append(torch.stack((chunk_rows[local[0]],
                            local[1] + image_index * math.prod(shape))))
                        target['values'].append(sparse.values())
                        if self.attn_loss_type == 'position':
                            target['weights'][image_index].index_add_(
                                0, labels[start:start + 32], coverage)
                        target['count'].add_((mass > 0).sum())
                    del masks
            for shape, target in geometry.items():
                indices, values = target.pop('indices'), target.pop('values')
                indices = torch.cat(indices, dim=1) if indices else ids.new_empty((2, 0))
                values = (torch.cat(values) if values
                          else attention.new_empty(0, dtype=torch.float32))
                target['pool'] = torch.sparse_coo_tensor(
                    indices, values, (total, len(samples) * math.prod(shape)),
                    device=attention.device).coalesce()
        return dict(groups=groups, shape=tuple(attention.shape[:2]),
                    input_shape=tuple(input_shape), geometry=geometry,
                    targets=targets, offsets=tuple(offsets))

    @staticmethod
    def pool_instance_attention(attention, prepared):
        """Read prompt weights per original GT row; keep attention gradients."""
        target = prepared['geometry'][tuple(attention.shape[-2:])]
        pixels = attention.float().permute(0, 2, 3, 1).reshape(-1, attention.shape[1])
        return pool_instances(target['pool'], pixels)

    @staticmethod
    def pool_instance_log_attention(log_attention, prepared):
        """Log of the SAME sparse arithmetic GT mean, without exp underflow.

        Segment logsumexp touches only nonzero mask coverage, retaining the
        sparse geometry shared across depths. Empty/invalid GT rows receive
        a finite placeholder (zero log score) and are excluded by validity;
        never reduce an all--inf row or evaluate log(0) for those rows.
        """
        pool = prepared['geometry'][tuple(log_attention.shape[-2:])]['pool']
        rows, columns = pool.indices()
        channels = log_attention.shape[1]
        pixels = log_attention.float().permute(0, 2, 3, 1).reshape(-1, channels)
        terms = pixels.index_select(0, columns) + pool.values().log()[:, None]
        index = rows[:, None].expand(-1, channels)
        maximum = terms.new_full((pool.shape[0], channels), float('-inf'))
        maximum = maximum.scatter_reduce(0, index, terms, reduce='amax', include_self=True)
        # Occupancy follows sparse geometry, not attention magnitude. No
        # feature-dependent threshold or detached visual/text weight is used.
        occupied = torch.zeros(pool.shape[0], dtype=torch.bool, device=pixels.device)
        occupied[rows] = True
        maximum = maximum.masked_fill(~occupied[:, None], 0.)
        shifted = (terms - maximum.index_select(0, rows)).exp()
        total = terms.new_zeros(maximum.shape).index_add(0, rows, shifted)
        total = total.masked_fill(~occupied[:, None], 1.)
        # An empty view connects empty GT batches to attention without
        # summing possible masked -inf values and multiplying by zero.
        return maximum + total.log() + log_attention.reshape(-1)[:0].sum()

    @staticmethod
    def _budget_kl(log_correct, log_visual, rho):
        """KL to (correct text, visual, other text)=(rho, 1-rho, 0).

        Targets are constants for this objective. xlogy handles exact 0/1
        targets without evaluating 0*log(0). Production log masses remain
        finite even when their probabilities underflow to zero.
        """
        rho = rho.detach().float().clamp(0., 1.)
        visual_target = 1. - rho
        return (torch.xlogy(rho, rho) + torch.xlogy(visual_target, visual_target)
                - rho * log_correct - visual_target * log_visual)

    def _loss_attn_state(self, attention, class_token_class_ids,
                         batch_data_samples, input_shape, *, prepared=None,
                         semantic_probabilities, visual_attention,
                         avg_factor=None, log_attention=False, readout_cache=None):
        """Absolute attention budget KL, with equal valid-instance weights.

        instance: pool actual probability masses and GT-class reliability,
                  then compare the grouped distribution with its target.
        position: compare at each pixel, then mask-pool the KL per instance.
        Prompts are summed, not averaged: supervision concerns actual mass.
        Semantic foreground channels follow zero-based instance class order.
        Nuisance tokens, when present, belong to the zero-target remainder.
        """
        ids = class_token_class_ids
        if (attention is None or attention.ndim != 4
                or attention.shape[0] != len(batch_data_samples)
                or ids.ndim != 1 or attention.shape[1] != len(ids)
                or ids.dtype != torch.long or ids.device != attention.device):
            raise ValueError('Expected attention (B, prompts, H, W) and matching class IDs.')
        if prepared is None:
            prepared = self._prepare_attn_targets(attention, ids, batch_data_samples, input_shape)
        if (prepared['shape'] != tuple(attention.shape[:2])
                or prepared['input_shape'] != tuple(input_shape)):
            raise ValueError('Supervised scales must share batch, prompts and input shapes.')
        expected = (attention.shape[0], len(prepared['groups']), *attention.shape[-2:])
        if (tuple(semantic_probabilities.shape) != expected
                or tuple(visual_attention.shape) != (expected[0], 1, *expected[-2:])):
            raise ValueError('Semantic probabilities / visual mass must align with attention.')
        shape = tuple(attention.shape[-2:])
        target = prepared['geometry'][shape]
        if avg_factor is None:
            avg_factor = self._global_instance_avg_factor(target['count'])
        # Probability-valued direct callers have already lost any underflowed
        # information. Forward-produced states always use the exact log path.
        log_weights = (attention.float() if log_attention else
                       attention.float().clamp_min(torch.finfo(torch.float32).tiny).log())
        log_visual = (visual_attention.float() if log_attention else
                      visual_attention.float().clamp_min(torch.finfo(torch.float32).tiny).log())
        log_correct = torch.stack([
            torch.logsumexp(log_weights[:, group], dim=1)
            for group in prepared['groups']], dim=1)
        quality = semantic_probabilities.detach().float()
        if readout_cache is not None:
            readout_cache.update(log_weights=log_weights, log_correct=log_correct)
        if self.attn_loss_type == 'instance':
            labels = prepared['targets'][:, None]
            correct = self.pool_instance_log_attention(log_correct, prepared).gather(1, labels).squeeze(1)
            visual = self.pool_instance_log_attention(log_visual, prepared).squeeze(1)
            with torch.no_grad():
                # Shared by all supervised depths at this spatial resolution.
                cache = prepared.setdefault('visual_quality', {})
                if shape not in cache:
                    cache[shape] = self.pool_instance_attention(quality, prepared).gather(1, labels).squeeze(1)
                rho = 1. - cache[shape]
                if self.reliability_mode == 'fixed':
                    rho = rho.new_full(rho.shape, 0.5)
                elif self.reliability_mode == 'shuffle' and rho.numel() > 1:
                    rho = rho.index_select(0, torch.randperm(rho.numel(), device=rho.device))
                elif self.reliability_mode in ('maxsoft', 'entropy'):
                    # Alternative reliability estimates used only for the
                    # supplementary control runs.  They are computed from
                    # the same online semantic probabilities, without using
                    # the ground-truth class identity.
                    pooled = self.pool_instance_attention(quality, prepared)
                    probs = pooled / pooled.sum(dim=1, keepdim=True).clamp_min(1e-6)
                    if self.reliability_mode == 'maxsoft':
                        rho = probs.max(dim=1).values
                    else:
                        entropy = -(probs.clamp_min(1e-6) * probs.clamp_min(1e-6).log()).sum(dim=1)
                        rho = 1. - entropy / math.log(max(2, probs.shape[1]))
            if readout_cache is not None:
                readout_cache.update(correct=correct, rho=rho)
            penalty = self._budget_kl(correct, visual, rho)
            numerator = (penalty * target['validity']).sum()
        else:
            penalty = self._budget_kl(log_correct, log_visual, 1. - quality)
            numerator = (penalty * target['weights']).sum()
        # Keep empty / invalid GT batches connected to both Q/K readouts.
        zero = attention.reshape(-1)[:0].sum() + visual_attention.reshape(-1)[:0].sum()
        return numerator / avg_factor + zero
