import torch
from torch import nn
from mm.engine.model import BaseModule
from mm.det.registry import MODELS
from mm.cv.cnn import build_norm_layer

@MODELS.register_module()
class SiamBackboneSharedWeights(BaseModule):
    def __init__(self,
                 btype: str,
                 init_cfg: str = None,
                 **kwargs):
        super().__init__(init_cfg=None)
        
        self.siam_backbone = MODELS.build(dict(
            type=btype,
            init_cfg=init_cfg,
            **kwargs
        ))
        
    def forward(self, x):
        assert x.shape[1] in (4, 6), f'SiamBackboneSharedWeights expects 4 or 6 channel input, but got {x.shape[1]} channels.'
        assert x.ndim == 4, f'SiamBackbone expects batch input.'
        x_pre = x[:, :3, :, :]
        x_post = x[:, 3:, :, :]
        if x_post.shape[1] == 1:
            x_post = x_post.repeat(1, 3, 1, 1)
        
        feats_pre = self.siam_backbone(x_pre)
        feats_post = self.siam_backbone(x_post)

        return (feats_pre, feats_post)

@MODELS.register_module()
class SiamBackbone(BaseModule):
    def __init__(self,
                 btype: str,
                 init_cfg: str = None,
                 first_conv_attr: str = 'conv1',
                 **kwargs):
        super().__init__(init_cfg=None)
        
        self.opt_backbone = MODELS.build(dict(
            type=btype,
            init_cfg=init_cfg,
            **kwargs
        ))
        self.sar_backbone = MODELS.build(dict(
            type=btype,
            init_cfg=init_cfg,
            **kwargs
        ))
        self._init_sar_from_opt(first_conv_attr)
    
    def get_nested_attr(self, obj, path):
        attrs = path.split('.')
        for attr in attrs:
            if attr.isdigit():
                obj = obj[int(attr)]
            else:
                obj = getattr(obj, attr)
        return obj
    
    def _init_sar_from_opt(self, first_conv_attr):
        # opt_conv1 = self.opt_backbone.conv1
        # sar_conv1 = self.sar_backbone.conv1
        
        opt_conv1 = self.get_nested_attr(self.opt_backbone, first_conv_attr)
        sar_conv1 = self.get_nested_attr(self.sar_backbone, first_conv_attr)
        
        opt_weight = opt_conv1.weight.data
        opt_weight_mean = opt_weight.mean(dim=1, keepdim=True)
        sar_conv1.weight.data = opt_weight_mean.repeat(1, 3, 1, 1)

    def forward(self, x):
        assert x.shape[1] == 4, f'SiamBackbone expects 4 channel input, but got {x.shape[1]} channels.'
        assert x.ndim == 4, f'SiamBackbone expects batch input.'
        x_pre = x[:, :3, :, :]
        x_post = x[:, 3:, :, :]
        
        assert x_post.shape[1] == 3 or x_post.shape[1] == 1
        if x_post.shape[1] == 1:
            x_post_3ch = torch.cat([x_post, x_post, x_post], dim=1)
        else:
            x_post_3ch = x_post

        feats_pre = self.opt_backbone(x_pre)
        feats_post = self.sar_backbone(x_post_3ch)

        return (feats_pre, feats_post)

@MODELS.register_module()
class FuseNeck(BaseModule):
    def __init__(self, channels, norm_cfg=None, input_img=False):
        super().__init__()
        self.input_img = input_img
        self.fusion_convs = nn.ModuleList()
        for c in channels:
            conv = nn.Conv2d(c*2, c, kernel_size=1, bias=False)
            if norm_cfg is not None:
                _, norm = build_norm_layer(norm_cfg, c)
            else:
                norm = nn.Identity()
            self.fusion_convs.append(nn.Sequential(conv, norm))
    
    def forward(self, inputs):
        fused_feats = []
        for i, (f_pre, f_post) in enumerate(zip(*inputs), start=-1 if self.input_img else 0):
            f_cat = torch.cat([f_pre, f_post], dim=1)
            if i < 0:
                f_fused = f_cat
            else:
                f_fused = self.fusion_convs[i](f_cat)
            fused_feats.append(f_fused)
        return tuple(fused_feats)

@MODELS.register_module()
class SequentialNeck(nn.Module):
    def __init__(self, neck_list: list):
        super().__init__()
        self.model_list = nn.ModuleList([MODELS.build(neck) for neck in neck_list])
    
    def forward(self, inputs):
        x = inputs
        difference = None
        pre_features = None
        temporal_pair = None
        for model in self.model_list:
            joint_difference_processed = False
            # NativeContentChangeNeck exposes a second pyramid with the same
            # per-level shapes as ``x``.  FPN is batch-independent, so run
            # the content and change pyramids together and split the result.
            # The old code called FPN twice serially, duplicating all lateral
            # and top-down convolutions without changing either output.
            if (difference is not None and
                    model.__class__.__name__ == 'FPN' and
                    isinstance(x, (tuple, list)) and
                    isinstance(difference, (tuple, list)) and
                    len(x) == len(difference) and
                    all(a.ndim == b.ndim == 4 and a.shape[1:] == b.shape[1:]
                        for a, b in zip(x, difference))):
                split_at = x[0].shape[0]
                joint = model(tuple(torch.cat((a, b), dim=0)
                                    for a, b in zip(x, difference)))
                x = tuple(level[:split_at] for level in joint)
                difference = tuple(level[split_at:] for level in joint)
                joint_difference_processed = True
            else:
                x = model(x)
            if pre_features is None:
                pre_features = getattr(model, '_cached_pre_features', None)
            elif pre_features is not None:
                pre_features = model(pre_features)
            current_pair = getattr(model, '_cached_temporal_pair_features', None)
            if temporal_pair is None:
                temporal_pair = current_pair
            elif temporal_pair is not None:
                temporal_pair = (model(temporal_pair[0]),
                                 model(temporal_pair[1]))
            if difference is None:
                difference = getattr(model, '_cached_difference_features', None)
            elif not joint_difference_processed:
                difference = model(difference)
        self._cached_difference_features = difference
        self._cached_pre_features = pre_features
        self._cached_temporal_pair_features = temporal_pair
        return x
