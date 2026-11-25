"""Training-only semantic probe of the visual features BEFORE text attention."""

import torch
from torch import nn
from torch.nn import functional as F


class VisualSemanticHead(nn.Module):
    """Fuse the visual pyramid at its finest resolution, without text input.

    Channel zero is background; channels 1..C follow the text-bank class order.
    No normalization with running statistics or cross-rank synchronization is
    needed. The decoder is only called when auxiliary supervision is requested.
    """

    def __init__(self, channels, num_classes, hidden_channels=64):
        super().__init__()
        self.num_classes = num_classes
        self.projections = nn.ModuleList([
            nn.Conv2d(c, hidden_channels, 1) for c in channels])
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, num_classes + 1, 1))

    def forward(self, features):
        shape = features[0].shape[-2:]
        merged = None
        for feature, projection in zip(features, self.projections):
            projected = projection(feature)
            if projected.shape[-2:] != shape:
                projected = F.interpolate(
                    projected, size=shape, mode='bilinear', align_corners=False)
            merged = projected if merged is None else merged + projected
        return self.decoder(merged / len(features))

    @staticmethod
    def loss_terms(logits, samples, input_shape, ignore_index=255):
        """Return CE numerator / valid-pixel count, including empty supervision.

        Only real image pixels are supervised. Batch padding and explicitly
        ignored pixels never contribute, even if preprocessing pads with zero.
        """
        height, width = input_shape
        target = torch.full((len(samples), height, width), ignore_index,
                            dtype=torch.long, device=logits.device)
        for index, sample in enumerate(samples):
            if 'gt_sem_seg' not in sample:
                raise ValueError('Visual semantic supervision requires gt_sem_seg.')
            label = sample.gt_sem_seg.sem_seg
            if label.ndim != 3 or label.shape[0] != 1:
                raise ValueError('gt_sem_seg must have shape (1, H, W).')
            h = min(height, sample.img_shape[0], label.shape[-2])
            w = min(width, sample.img_shape[1], label.shape[-1])
            target[index, :h, :w] = label[0, :h, :w].to(logits.device, torch.long)
        prediction = F.interpolate(logits.float(), size=input_shape,
                                   mode='bilinear', align_corners=False)
        numerator = F.cross_entropy(
            prediction, target, ignore_index=ignore_index, reduction='sum')
        return numerator, (target != ignore_index).sum()
