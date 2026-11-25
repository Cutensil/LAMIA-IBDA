"""PVTv1/PVTv2 backbones registered for MMDetection 3.x."""

import math
from functools import partial

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.init import trunc_normal_

from mm.cv.cnn.bricks import DropPath
from mm.det.registry import MODELS
from mm.engine.model import BaseModule
from mm.engine.utils import to_2tuple


# PVTv1/PVTv2 adapted from tools/models/PVT-2_github/detection/{pvt,pvt_v2}.py.
# Original implementation: Wenhai Wang et al., Apache-2.0; see that repo's LICENSE.
# Keep upstream parameter names so official ImageNet checkpoints load directly.

class _PVTv2Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., linear=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = _PVTDWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.linear = linear
        if self.linear:
            self.relu = nn.ReLU(inplace=True)


    def forward(self, x, H, W):
        x = self.fc1(x)
        if self.linear:
            x = self.relu(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class _PVTAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1, linear=False):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.linear = linear
        self.sr_ratio = sr_ratio
        if not linear:
            if sr_ratio > 1:
                self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
                self.norm = nn.LayerNorm(dim)
        else:
            self.pool = nn.AdaptiveAvgPool2d(7)
            self.sr = nn.Conv2d(dim, dim, kernel_size=1, stride=1)
            self.norm = nn.LayerNorm(dim)
            self.act = nn.GELU()


    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if not self.linear:
            if self.sr_ratio > 1:
                x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
                x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
                x_ = self.norm(x_)
                kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            else:
                kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            # Pooling preserves channels-last storage from the token view.
            # Keep the 1x1 SRA convolution's gradients compatible with DDP too.
            x_ = self.sr(self.pool(x_).contiguous()).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            x_ = self.act(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class _PVTv2Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1, linear=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = _PVTAttention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio, linear=linear)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = _PVTv2Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, linear=linear)


    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        return x


class _PVTOverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        assert max(patch_size) > stride, "Set larger patch_size than stride"

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // stride, img_size[1] // stride
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)


    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W


class _PVTDWConv(nn.Module):
    def __init__(self, dim=768):
        super(_PVTDWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        # Token-major storage becomes channels-last after this view. On CUDA,
        # depthwise convolution can then return weight gradients with strides
        # (9, 1, 3, 1), while DDP expects the parameter's (9, 9, 3, 1).
        # Normalize the activation layout before convolution, including AMP.
        x = x.transpose(1, 2).reshape(B, C, H, W).contiguous()
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x


class _PVTv1Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class _PVTv1Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = _PVTAttention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = _PVTv1Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class _PVTPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        assert img_size[0] % patch_size[0] == 0 and img_size[1] % patch_size[1] == 0, \
            f"img_size {img_size} should be divided by patch_size {patch_size}."
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape

        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        H, W = H // self.patch_size[0], W // self.patch_size[1]

        return x, (H, W)


class _PVTBase(BaseModule):
    """MMEngine initialization and stage freezing shared by both PVT versions."""

    def __init__(self, pretrained, init_cfg, out_indices, num_stages,
                 frozen_stages, style):
        if pretrained is not None:
            if init_cfg is not None:
                raise ValueError('Specify only one of pretrained and init_cfg.')
            if not isinstance(pretrained, str):
                raise TypeError('pretrained must be a checkpoint path/URL or None.')
            init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        super().__init__(init_cfg=init_cfg)
        if not 1 <= num_stages <= 4:
            raise ValueError('num_stages must be between 1 and 4.')
        self.num_stages = num_stages
        self.out_indices = tuple(out_indices)
        if (not self.out_indices or
                tuple(sorted(set(self.out_indices))) != self.out_indices or
                any(i < 0 or i >= num_stages for i in self.out_indices)):
            raise ValueError('out_indices must be increasing unique stage indices.')
        # 0 freezes the first complete stage; -1 leaves all stages trainable.
        if not -1 <= frozen_stages < num_stages:
            raise ValueError('frozen_stages must be in [-1, num_stages - 1].')
        self.frozen_stages = frozen_stages
        if style != 'pytorch':
            raise ValueError("PVT only supports style='pytorch'.")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)
        elif isinstance(module, nn.Conv2d) and self._version == 2:
            fan_out = math.prod(module.kernel_size) * module.out_channels // module.groups
            nn.init.normal_(module.weight, std=math.sqrt(2.0 / fan_out))
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def init_weights(self):
        if not self.is_init:
            self.apply(self._init_weights)
            if self._version == 1:
                for i in range(self.num_stages):
                    trunc_normal_(getattr(self, f'pos_embed{i + 1}'), std=.02)
        # Pretrained initialization runs last and supports checkpoint prefixes.
        super().init_weights()

    def _freeze_stages(self):
        for i in range(self.frozen_stages + 1):
            for prefix in ('patch_embed', 'block', 'norm', 'pos_drop'):
                module = getattr(self, f'{prefix}{i + 1}', None)
                if module is not None:
                    module.eval()
                    module.requires_grad_(False)
            pos_embed = getattr(self, f'pos_embed{i + 1}', None)
            if pos_embed is not None:
                pos_embed.requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        self._freeze_stages()
        return self

    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad_(False)

    def no_weight_decay(self):
        return {name for name, _ in self.named_parameters()
                if name.startswith('pos_embed')}

    def forward(self, x):
        return self.forward_features(x)

    def _validate_stages(self, embed_dims, num_heads, mlp_ratios, depths, sr_ratios):
        for values in (embed_dims, num_heads, mlp_ratios, depths, sr_ratios):
            if len(values) < self.num_stages:
                raise ValueError('Each stage setting must cover num_stages.')
            if any(value <= 0 for value in values[:self.num_stages]):
                raise ValueError('Stage settings must be positive.')
        self.embed_dims = tuple(embed_dims[:self.num_stages])
        self.depths = tuple(depths[:self.num_stages])
        self.out_channels = tuple(self.embed_dims[i] for i in self.out_indices)


@MODELS.register_module()
class PVTv2(_PVTBase):
    """Official PVTv2 for MMDetection 3.x, without a classification head.

    Outputs are contiguous NCHW features at strides 4, 8, 16 and 32.
    Select a preset with type='pvt_v2_b0' ... 'pvt_v2_b5' or
    'pvt_v2_b2_li'. Official parameter names are preserved, including
    patch_embed1.proj (also usable as SiamBackbone.first_conv_attr).
    Use init_cfg=dict(type='Pretrained', checkpoint='...') to load weights.
    Inputs must have at least 32 pixels along both spatial dimensions when
    using the default spatial reduction ratios.
    """

    _version = 2

    def __init__(self, img_size=224, patch_size=4, in_chans=3,
                 num_classes=1000, embed_dims=(64, 128, 256, 512),
                 num_heads=(1, 2, 4, 8), mlp_ratios=(4, 4, 4, 4),
                 qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=(3, 4, 6, 3), sr_ratios=(8, 4, 2, 1), num_stages=4,
                 linear=False, pretrained=None, init_cfg=None,
                 out_indices=(0, 1, 2, 3), frozen_stages=-1, style='pytorch'):
        # patch_size and num_classes are retained for upstream config compatibility;
        # PVTv2 always uses overlapping 7x7/3x3 embeddings and has no classifier.
        super().__init__(pretrained, init_cfg, out_indices, num_stages,
                         frozen_stages, style)
        self._validate_stages(embed_dims, num_heads, mlp_ratios, depths, sr_ratios)
        self.linear = linear
        img_size = to_2tuple(img_size)
        dpr = torch.linspace(0, drop_path_rate, sum(self.depths)).tolist()
        cur = 0
        for i in range(num_stages):
            patch_embed = _PVTOverlapPatchEmbed(
                img_size=img_size if i == 0 else tuple(s // (2 ** (i + 1)) for s in img_size),
                patch_size=7 if i == 0 else 3, stride=4 if i == 0 else 2,
                in_chans=in_chans if i == 0 else embed_dims[i - 1],
                embed_dim=embed_dims[i])
            block = nn.ModuleList([
                _PVTv2Block(
                    dim=embed_dims[i], num_heads=num_heads[i],
                    mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate,
                    drop_path=dpr[cur + j], norm_layer=norm_layer,
                    sr_ratio=sr_ratios[i], linear=linear)
                for j in range(depths[i])])
            setattr(self, f'patch_embed{i + 1}', patch_embed)
            setattr(self, f'block{i + 1}', block)
            setattr(self, f'norm{i + 1}', norm_layer(embed_dims[i]))
            cur += depths[i]
        self._freeze_stages()

    def forward_features(self, x):
        outs = []
        for i in range(self.num_stages):
            x, h, w = getattr(self, f'patch_embed{i + 1}')(x)
            for block in getattr(self, f'block{i + 1}'):
                x = block(x, h, w)
            x = getattr(self, f'norm{i + 1}')(x)
            x = x.reshape(x.shape[0], h, w, -1).permute(0, 3, 1, 2).contiguous()
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)


@MODELS.register_module()
class PVT(_PVTBase):
    """Official PVTv1 detection backbone with interpolated position embeddings.

    Presets: pvt_tiny, pvt_small, pvt_small_f4, pvt_medium, pvt_large.
    F4=True returns only the final stage, as in the upstream detection code.
    """

    _version = 1

    def __init__(self, img_size=224, patch_size=4, in_chans=3,
                 num_classes=1000, embed_dims=(64, 128, 256, 512),
                 num_heads=(1, 2, 4, 8), mlp_ratios=(4, 4, 4, 4),
                 qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=(3, 4, 6, 3), sr_ratios=(8, 4, 2, 1), F4=False,
                 num_stages=4, pretrained=None, init_cfg=None,
                 out_indices=(0, 1, 2, 3), frozen_stages=-1, style='pytorch'):
        if F4:
            out_indices = (num_stages - 1,)
        super().__init__(pretrained, init_cfg, out_indices, num_stages,
                         frozen_stages, style)
        self._validate_stages(embed_dims, num_heads, mlp_ratios, depths, sr_ratios)
        self.F4 = F4
        stage_size = to_2tuple(img_size)
        dpr = torch.linspace(0, drop_path_rate, sum(self.depths)).tolist()
        cur = 0
        for i in range(num_stages):
            patch_embed = _PVTPatchEmbed(
                img_size=stage_size, patch_size=patch_size if i == 0 else 2,
                in_chans=in_chans if i == 0 else embed_dims[i - 1],
                embed_dim=embed_dims[i])
            stage_size = (patch_embed.H, patch_embed.W)
            # Upstream classification weights include one unused CLS position.
            pos_embed = nn.Parameter(torch.zeros(
                1, patch_embed.num_patches + (i == num_stages - 1), embed_dims[i]))
            block = nn.ModuleList([
                _PVTv1Block(
                    dim=embed_dims[i], num_heads=num_heads[i],
                    mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate,
                    drop_path=dpr[cur + j], norm_layer=norm_layer,
                    sr_ratio=sr_ratios[i])
                for j in range(depths[i])])
            setattr(self, f'patch_embed{i + 1}', patch_embed)
            setattr(self, f'pos_embed{i + 1}', pos_embed)
            setattr(self, f'pos_drop{i + 1}', nn.Dropout(drop_rate))
            setattr(self, f'block{i + 1}', block)
            cur += depths[i]
        self._freeze_stages()

    def _get_pos_embed(self, pos_embed, patch_embed, h, w):
        # Compare both dimensions of this stage, not stage 1's token count.
        if (h, w) == (patch_embed.H, patch_embed.W):
            return pos_embed
        pos_embed = pos_embed.reshape(1, patch_embed.H, patch_embed.W, -1)
        return F.interpolate(
            pos_embed.permute(0, 3, 1, 2), size=(h, w),
            mode='bilinear', align_corners=False).flatten(2).transpose(1, 2)

    def forward_features(self, x):
        outs = []
        for i in range(self.num_stages):
            patch_embed = getattr(self, f'patch_embed{i + 1}')
            x, (h, w) = patch_embed(x)
            pos_embed = getattr(self, f'pos_embed{i + 1}')
            if i == self.num_stages - 1:
                pos_embed = pos_embed[:, 1:]
            x = getattr(self, f'pos_drop{i + 1}')(
                x + self._get_pos_embed(pos_embed, patch_embed, h, w))
            for block in getattr(self, f'block{i + 1}'):
                x = block(x, h, w)
            x = x.reshape(x.shape[0], h, w, -1).permute(0, 3, 1, 2).contiguous()
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)


def _pvt_preset(depths, **overrides):
    cfg = dict(
        patch_size=4, embed_dims=(64, 128, 320, 512), num_heads=(1, 2, 5, 8),
        mlp_ratios=(8, 8, 4, 4), qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=depths,
        sr_ratios=(8, 4, 2, 1), drop_rate=0., drop_path_rate=0.1)
    cfg.update(overrides)
    return cfg


@MODELS.register_module()
class pvt_v2_b0(PVTv2):
    def __init__(self, **kwargs):
        cfg = _pvt_preset((2, 2, 2, 2), embed_dims=(32, 64, 160, 256))
        cfg.update(kwargs)
        super().__init__(**cfg)


@MODELS.register_module()
class pvt_v2_b1(PVTv2):
    def __init__(self, **kwargs):
        cfg = _pvt_preset((2, 2, 2, 2))
        cfg.update(kwargs)
        super().__init__(**cfg)


@MODELS.register_module()
class pvt_v2_b2(PVTv2):
    def __init__(self, **kwargs):
        cfg = _pvt_preset((3, 4, 6, 3))
        cfg.update(kwargs)
        super().__init__(**cfg)


@MODELS.register_module()
class pvt_v2_b2_li(PVTv2):
    def __init__(self, **kwargs):
        cfg = _pvt_preset((3, 4, 6, 3), linear=True)
        cfg.update(kwargs)
        super().__init__(**cfg)


@MODELS.register_module()
class pvt_v2_b3(PVTv2):
    def __init__(self, **kwargs):
        cfg = _pvt_preset((3, 4, 18, 3))
        cfg.update(kwargs)
        super().__init__(**cfg)


@MODELS.register_module()
class pvt_v2_b4(PVTv2):
    def __init__(self, **kwargs):
        cfg = _pvt_preset((3, 8, 27, 3))
        cfg.update(kwargs)
        super().__init__(**cfg)


@MODELS.register_module()
class pvt_v2_b5(PVTv2):
    def __init__(self, **kwargs):
        cfg = _pvt_preset((3, 6, 40, 3), mlp_ratios=(4, 4, 4, 4))
        cfg.update(kwargs)
        super().__init__(**cfg)


@MODELS.register_module()
class pvt_tiny(PVT):
    def __init__(self, **kwargs):
        cfg = _pvt_preset((2, 2, 2, 2))
        cfg.update(kwargs)
        super().__init__(**cfg)


@MODELS.register_module()
class pvt_small(PVT):
    def __init__(self, **kwargs):
        cfg = _pvt_preset((3, 4, 6, 3))
        cfg.update(kwargs)
        super().__init__(**cfg)


@MODELS.register_module()
class pvt_small_f4(PVT):
    def __init__(self, **kwargs):
        cfg = _pvt_preset((3, 4, 6, 3), F4=True)
        cfg.update(kwargs)
        super().__init__(**cfg)


@MODELS.register_module()
class pvt_medium(PVT):
    def __init__(self, **kwargs):
        cfg = _pvt_preset((3, 4, 18, 3), drop_path_rate=0.0)
        cfg.update(kwargs)
        super().__init__(**cfg)


@MODELS.register_module()
class pvt_large(PVT):
    def __init__(self, **kwargs):
        cfg = _pvt_preset((3, 8, 27, 3), drop_path_rate=0.0)
        cfg.update(kwargs)
        super().__init__(**cfg)
