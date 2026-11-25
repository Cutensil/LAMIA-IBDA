# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import nn
from torch.nn.modules.utils import _pair
from torchvision.ops import roi_align as torchvision_roi_align


def roi_align(input: torch.Tensor,
              rois: torch.Tensor,
              output_size,
              spatial_scale: float = 1.0,
              sampling_ratio: int = 0,
              pool_mode: str = 'avg',
              aligned: bool = True) -> torch.Tensor:
    """Pool RoI features using the original sampling and alignment settings."""
    if pool_mode != 'avg':
        raise ValueError('RoIAlign only supports average pooling in Lamia.')
    output = torchvision_roi_align(
        input, rois, output_size, spatial_scale, sampling_ratio, aligned)
    # The extractor writes the pooled features into a tensor of input dtype.
    return output.to(dtype=input.dtype)


class RoIAlign(nn.Module):
    """RoIAlign layer used by the HTC box, mask and semantic branches."""

    def __init__(self,
                 output_size,
                 spatial_scale: float = 1.0,
                 sampling_ratio: int = 0,
                 pool_mode: str = 'avg',
                 aligned: bool = True):
        super().__init__()
        self.output_size = _pair(output_size)
        self.spatial_scale = float(spatial_scale)
        self.sampling_ratio = int(sampling_ratio)
        self.pool_mode = pool_mode
        self.aligned = aligned

    def forward(self, input: torch.Tensor, rois: torch.Tensor) -> torch.Tensor:
        return roi_align(input, rois, self.output_size, self.spatial_scale,
                         self.sampling_ratio, self.pool_mode, self.aligned)
