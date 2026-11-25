"""Shared spatial readout of updated window text for fusion and HTC losses."""

import torch


@torch.no_grad()
def prepare_class_pooling(class_context, rois, img_shapes):
    """Prepare detached RoI/window geometry once, independently of text."""

    tokens = class_context['tokens']
    windows = class_context['window_boxes']
    if (tokens.ndim != 4 or windows.shape != (tokens.shape[1], 4)
            or rois.ndim != 2 or rois.shape[1] != 5
            or tokens.device != rois.device or len(img_shapes) != tokens.shape[0]):
        raise ValueError('Invalid window text context or RoI geometry.')
    feature_h, feature_w = class_context['feature_shape']
    input_h, input_w = class_context['input_shape']
    if min(feature_h, feature_w, input_h, input_w) <= 0:
        raise ValueError('Context image/feature sizes must be positive.')
    if len(rois) and ((rois[:, 0] < 0) | (rois[:, 0] >= len(img_shapes))).any():
        raise ValueError('RoI batch index is outside text context.')
    prepared = []
    for image_index, (img_h, img_w) in enumerate(img_shapes):
        indices = (rois[:, 0] == image_index).nonzero(as_tuple=False).flatten()
        if len(indices) == 0:
            continue
        h, w = min(img_h, input_h), min(img_w, input_w)
        if min(h, w) <= 0:
            raise ValueError('img_shape must have positive height/width.')
        boxes = torch.nan_to_num(rois[indices, 1:].detach().float(),
                                 nan=0., posinf=0., neginf=0.)
        boxes[:, 0::2].clamp_(0, w)
        boxes[:, 1::2].clamp_(0, h)
        scale = boxes.new_tensor([feature_w / input_w, feature_h / input_h] * 2)
        boxes = boxes * scale
        window_boxes = windows.detach().to(device=rois.device, dtype=torch.float32)
        extent = (torch.minimum(boxes[:, None, 2:], window_boxes[None, :, 2:])
                  - torch.maximum(boxes[:, None, :2], window_boxes[None, :, :2]))
        area = extent.clamp_min(0).prod(-1)
        mass = area.sum(-1, keepdim=True)
        # Only valid image area contributes to fallback, even when the
        # batch input contains extra padded rows/columns.
        valid_end = boxes.new_tensor([w * feature_w / input_w,
                                       h * feature_h / input_h])
        valid_extent = (torch.minimum(window_boxes[:, 2:], valid_end)
                        - window_boxes[:, :2]).clamp_min(0)
        fallback = valid_extent.prod(-1)
        fallback = fallback / fallback.sum().clamp_min(1e-8)
        weights = torch.where(mass > 0, area / mass.clamp_min(1e-8), fallback[None])
        prepared.append((image_index, indices, weights))
    return tuple(prepared)


def pool_class_prototypes(class_context, rois, img_shapes, *, prepared=None):
    """Pool each prompt across windows; optionally reuse detached geometry.

    No averaging across different prompts. A degenerate/outside RoI uses
    that image's valid-area window average. Only geometry is detached;
    gradients reach every selected text state at its own encoder depth.
    Prepared geometry is strictly call-local, never stored on a module.
    """
    tokens = class_context['tokens']
    if prepared is None:
        prepared = prepare_class_pooling(class_context, rois, img_shapes)
    output = tokens.new_zeros((len(rois), tokens.shape[2], tokens.shape[3]))
    output = output + tokens.sum() * 0.
    for image_index, indices, weights in prepared:
        pooled = weights.to(tokens.dtype) @ tokens[image_index].flatten(1)
        # Autocast can lower the matmul dtype even when tokens are float32.
        pooled = pooled.reshape(-1, tokens.shape[2], tokens.shape[3])
        output.index_copy_(0, indices, pooled.to(output.dtype))
    return output

