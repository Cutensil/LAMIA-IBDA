"""One shared, differentiable text projection for fusion and classification."""

from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from mm.engine.model import BaseModule
from mm.det.registry import MODELS


def _encode_texts(encoder_path, prompts, max_length):
    """Encode local BGE prompts once on CPU; do not retain the encoder."""
    from transformers import AutoModel, AutoTokenizer

    encoder_path = str(Path(encoder_path).expanduser().resolve())
    tokenizer = AutoTokenizer.from_pretrained(
        encoder_path, local_files_only=True)
    encoded = tokenizer(
        list(prompts), padding=True, truncation=False, return_tensors='pt')
    encoder = AutoModel.from_pretrained(
        encoder_path, local_files_only=True).cpu().eval().requires_grad_(False)
    limit = min(max_length, encoder.config.max_position_embeddings)
    if encoded['input_ids'].shape[1] > limit:
        raise ValueError(f'Text exceeds {limit} tokens; shorten the prompts '
                         'or increase max_text_length within the encoder limit.')
    with torch.no_grad():
        vectors = encoder(**encoded).last_hidden_state[:, 0].float()
        if (not torch.isfinite(vectors).all()
                or (vectors.norm(dim=-1) == 0).any()):
            raise ValueError('The text encoder returned invalid text vectors.')
        vectors = F.normalize(vectors, dim=-1)
    return vectors.detach()


def _class_prompt_groups(prompts):
    """Accept flat class names or unequal-sized per-class phrase groups."""
    if not isinstance(prompts, (tuple, list)) or not prompts:
        raise ValueError('class_prompts must be a nonempty list or tuple.')
    groups = []
    for entry in prompts:
        group = (entry,) if isinstance(entry, str) else entry
        if (not isinstance(group, (tuple, list)) or not group
                or any(not isinstance(p, str) or not p.strip() for p in group)):
            raise ValueError('Each class must contain one or more nonempty phrases.')
        groups.append(tuple(group))
    flattened = [p for group in groups for p in group]
    if len(set(flattened)) != len(flattened):
        raise ValueError('Class prompt phrases must be unique.')
    return tuple(groups)


def _prompt_groups(ids, count, num_classes=None, counts=None):
    """Static class slices for TextHTC; validated generic mapping otherwise.

    TextHTC supplies immutable counts from the same TextBank that owns ids.
    This avoids GPU reductions and boolean indexing of a constant mapping in
    every decoder stage/loss. Standalone callers may still use arbitrary
    prompt order, including unequal class sizes.
    """
    if ids is not None and (ids.ndim != 1 or len(ids) != count or ids.dtype != torch.long):
        raise ValueError('Invalid prompt-to-class mapping.')
    if counts is not None:
        if sum(counts) != count or (num_classes is not None and len(counts) != num_classes):
            raise ValueError('Prompt counts do not match the shared text bank.')
        offset, groups = 0, []
        for size in counts:
            groups.append(slice(offset, offset + size))
            offset += size
        return tuple(groups), tuple(counts)
    if ids is None:
        if count != num_classes:
            raise ValueError('Multiple prompts require class_token_class_ids.')
        values = list(range(count))
    else:
        values = ids.detach().cpu().tolist()
    if not values or min(values) < 0:
        raise ValueError('Class token IDs must be nonnegative.')
    classes = max(values) + 1 if num_classes is None else num_classes
    if max(values) >= classes:
        raise ValueError('Invalid prompt-to-class mapping.')
    groups = tuple([i for i, value in enumerate(values) if value == c] for c in range(classes))
    if any(not group for group in groups):
        raise ValueError('Every foreground class needs at least one prompt.')
    return groups, tuple(len(group) for group in groups)


@MODELS.register_module()
class TextBank(BaseModule):
    """Frozen BGE buffers plus a single trainable Linear/LayerNorm pair.

    ``forward`` returns ALL class/noise tokens, shaped (M, D) and (K, D).
    Every phrase remains an independent token. ``sample_class_indices``
    chooses one index per foreground class, in dataset class order, so its
    result can index the already projected class tensor without re-encoding.
    Class order must match ``metainfo.classes``; nuisance text is not background.
    None or an empty noise prompt sequence disables nuisance text; its returned
    tensor then has shape (0, D), with no placeholder token.
    """

    def __init__(self, text_encoder_path, class_prompts, noise_prompts=None,
                 embed_dims=128, max_text_length=128,
                 ignore_checkpoint_text_cache=False,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        if type(embed_dims) is not int or embed_dims < 1:
            raise ValueError('embed_dims must be a positive integer.')
        if type(max_text_length) is not int or max_text_length < 2:
            raise ValueError('max_text_length must be an integer of at least 2.')
        groups = _class_prompt_groups(class_prompts)
        flattened = tuple(p for group in groups for p in group)
        if noise_prompts is None:
            noise_prompts = ()
        if (not isinstance(noise_prompts, (tuple, list))
                or any(not isinstance(p, str) or not p.strip() for p in noise_prompts)
                or len(set(noise_prompts)) != len(noise_prompts)):
            raise ValueError('noise_prompts must be None or a sequence of unique nonempty strings.')
        if set(flattened) & set(noise_prompts):
            raise ValueError('Class and nuisance groups must use distinct prompts.')
        self.class_prompt_groups = groups
        self.class_prompt_counts = tuple(len(group) for group in groups)
        self.class_prompts = flattened
        self.noise_prompts = tuple(noise_prompts)
        self.num_classes = len(groups)
        self.embed_dims = embed_dims
        self.ignore_checkpoint_text_cache = bool(ignore_checkpoint_text_cache)
        self.text_encoder_path = str(Path(text_encoder_path).expanduser().resolve())
        vectors = _encode_texts(
            self.text_encoder_path, flattened + self.noise_prompts, max_text_length)
        count = len(flattened)
        self.register_buffer('class_vectors', vectors[:count].clone())
        self.register_buffer('noise_vectors', vectors[count:].clone())
        self.register_buffer('class_token_class_ids', torch.tensor([
            c for c, group in enumerate(groups) for _ in group], dtype=torch.long))
        counts = torch.tensor([len(group) for group in groups], dtype=torch.long)
        self.register_buffer('prompt_counts', counts, persistent=False)
        self.register_buffer('prompt_offsets', counts.cumsum(0) - counts,
                             persistent=False)
        self.text_projection = nn.Linear(vectors.shape[1], embed_dims)
        self.text_norm = nn.LayerNorm(embed_dims)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata,
                               strict, missing_keys, unexpected_keys,
                               error_msgs):
        """Keep config-encoded text when an inference ablation changes prompts.

        The normal checkpoint contains frozen text buffers.  For prompt
        permutation/subset tests those buffers must not replace the vectors
        encoded from the inference config, even when their shapes happen to
        match the checkpoint.
        """
        ignored = []
        if self.ignore_checkpoint_text_cache:
            for name in ('class_vectors', 'noise_vectors',
                         'class_token_class_ids'):
                key = prefix + name
                if key in state_dict:
                    ignored.append(key)
                    state_dict.pop(key)
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys,
            unexpected_keys, error_msgs)
        for key in ignored:
            if key in missing_keys:
                missing_keys.remove(key)

    def forward(self):
        text = self.text_norm(self.text_projection(torch.cat(
            (self.class_vectors, self.noise_vectors), dim=0)))
        return text[:len(self.class_prompts)], text[len(self.class_prompts):]

    def sample_class_indices(self):
        """One uniform draw per class; singleton groups always return index 0 locally."""
        # Avoid consuming RNG when every class has exactly one description.
        if all(len(group) == 1 for group in self.class_prompt_groups):
            return self.prompt_offsets
        return torch.stack([
            torch.randint(len(group), (), device=self.class_vectors.device)
            if len(group) > 1 else self.prompt_offsets.new_zeros(())
            for group in self.class_prompt_groups
        ]) + self.prompt_offsets
