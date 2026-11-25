import warnings
import os.path as osp
import numpy as np
import mm.cv as mmcv
from mm.cv.transforms import BaseTransform
import torch
import torch.nn.functional as F
from typing import Sequence, Union, Optional, Tuple
from numbers import Number
from mm.det.datasets.transforms import RandomCrop
from mm.det.models.data_preprocessors import DetDataPreprocessor
from mm.det.registry import MODELS, TRANSFORMS
from mm.engine import fileio
from mm.engine.utils import is_seq_of
from mm.engine.structures import InstanceData

@MODELS.register_module()
class BiDetDataPreprocessor(DetDataPreprocessor):
    def __init__(self,
                 mean: Sequence[Number] = None,
                 std: Sequence[Number] = None,
                 bgr_to_rgb: bool = False,
                 rgb_to_bgr: bool = False,
                 **kwargs):
        
        super().__init__(
            mean=None,
            std=None,
            bgr_to_rgb=False,
            rgb_to_bgr=False,
            **kwargs
        )

        assert not (bgr_to_rgb and rgb_to_bgr), (
            '`bgr2rgb` and `rgb2bgr` cannot be set to True at the same time')
        self.bi_channel_conversion = bgr_to_rgb or rgb_to_bgr
        
        assert (mean is None) == (std is None), (
            'mean and std should be both None or tuple')
        if mean is not None or std is not None:
            # assert len(mean) == 4, 'length of mean must be 4.'
            # assert len(std) == 4, 'length of std must be 4.'
                        
            self.register_buffer('mean', torch.tensor(mean).view(-1, 1, 1), False)
            self.register_buffer('std', torch.tensor(std).view(-1, 1, 1), False)

            self._enable_normalize = True
        else:
            self._enable_normalize = False

    def forward(self, data: dict, training: bool = False) -> Union[dict, list]:
        if self.bi_channel_conversion and 'inputs' in data:
            inputs = data['inputs']
            if is_seq_of(inputs, torch.Tensor):
                new_inputs = []
                for x in inputs:
                    if x.shape[0] == 4:
                        new_inputs.append(x[[2, 1, 0, 3], ...])
                    else:
                        new_inputs.append(x)
                data['inputs'] = new_inputs
            elif isinstance(inputs, torch.Tensor):
                if inputs.ndim == 4 and inputs.shape[1] == 4:
                    data['inputs'] = inputs[:, [2, 1, 0, 3], ...]
                    
        return super().forward(data, training=training)

@TRANSFORMS.register_module()
class LoadBiImageFromFile(BaseTransform):
    def __init__(self,
                 to_float32: bool = False,
                 pre_color_type: str = 'color',
                 post_color_type: str = 'grayscale',
                 imdecode_backend: str = 'cv2',
                 file_client_args: Optional[dict] = None,
                 ignore_empty: bool = False,
                 *,
                 backend_args: Optional[dict] = None) -> None:
        self.ignore_empty = ignore_empty
        self.to_float32 = to_float32
        self.pre_color_type = pre_color_type
        self.post_color_type = post_color_type
        self.imdecode_backend = imdecode_backend

        self.file_client_args: Optional[dict] = None
        self.backend_args: Optional[dict] = None
        if file_client_args is not None:
            warnings.warn(
                '"file_client_args" will be deprecated in future. '
                'Please use "backend_args" instead', DeprecationWarning)
            if backend_args is not None:
                raise ValueError(
                    '"file_client_args" and "backend_args" cannot be set '
                    'at the same time.')

            self.file_client_args = file_client_args.copy()
        if backend_args is not None:
            self.backend_args = backend_args.copy()
    
    def transform(self, results: dict) -> Optional[dict]:
        # mmdet.inference_detector wraps a dict input as ``img_path``. Unpack
        # it here so direct inference can provide pre/post image paths.
        if isinstance(results.get('img_path'), dict):
            packed = results['img_path']
            results = dict(packed, **{k: v for k, v in results.items()
                                      if k != 'img_path'})
        root_path = osp.dirname(results['img_path'])
        pre_filename = osp.join(root_path, results['pre_img_path'])
        post_filename = osp.join(root_path, results['post_img_path'])
        try:
            if self.file_client_args is not None:
                pre_file_client = fileio.FileClient.infer_client(
                    self.file_client_args, pre_filename)
                pre_img_bytes = pre_file_client.get(pre_filename)
                post_file_client = fileio.FileClient.infer_client(
                    self.file_client_args, post_filename)
                post_img_bytes = post_file_client.get(post_filename)
            else:
                pre_img_bytes = fileio.get(
                    pre_filename, backend_args=self.backend_args)
                post_img_bytes = fileio.get(
                    post_filename, backend_args=self.backend_args)
            pre_img = mmcv.imfrombytes(
                pre_img_bytes, flag=self.pre_color_type, backend=self.imdecode_backend)
            post_img = mmcv.imfrombytes(
                post_img_bytes, flag=self.post_color_type, backend=self.imdecode_backend)
        except Exception as e:
            if self.ignore_empty:
                return None
            else:
                raise e
        assert pre_img is not None, f'failed to load image: {pre_filename}'
        assert post_img is not None, f'failed to load image: {pre_filename}'

        if pre_img.ndim == 2:
            pre_img = np.expand_dims(pre_img, axis=-1)
        if post_img.ndim == 2:
            post_img = np.expand_dims(post_img, axis=-1)
        assert pre_img.ndim == post_img.ndim == 3, 'Bitemporal images should have either 2 or 3 dims.'

        img = np.concatenate([pre_img, post_img], axis=-1)

        if self.to_float32:
            img = img.astype(np.float32)

        results['img'] = img
        results['img_shape'] = img.shape[:2]
        results['ori_shape'] = img.shape[:2]
        return results

@TRANSFORMS.register_module()
class FixedQuadCrop(RandomCrop):
    """Crop one quadrant and assign it an independent COCO image id."""

    def __init__(self, tile_index, image_size, **kwargs):
        if tile_index not in range(4):
            raise ValueError('tile_index must be 0, 1, 2 or 3.')

        img_h, img_w = image_size
        crop_h = img_h // 2
        crop_w = img_w // 2

        OFFSETS = (
            (0, 0),      # left-top
            (0, crop_w),    # right-top
            (crop_h, 0),    # left-bottom
            (crop_h, crop_w),  # right-bottom
        )

        self.tile_index = tile_index
        self.expected_image_size = image_size
        self.fixed_offset = OFFSETS[tile_index]
        self.crop_size = (crop_h, crop_w)

        super().__init__(crop_size=self.crop_size, **kwargs)

    def _rand_offset(self, margin):
        top, left = self.fixed_offset
        margin_h, margin_w = margin
        if top > margin_h or left > margin_w:
            raise ValueError(
                f'Cannot crop tile {self.tile_index} from margin {margin}.')
        return top, left

    def transform(self, results):
        height, width = results['img'].shape[:2]
        expected = self.expected_image_size
        if (height, width) != expected:
            raise ValueError(
                f'FixedQuadCrop expects {expected}, '
                f'but received {(height, width)}.')

        parent_img_id = results['img_id']
        if not isinstance(parent_img_id, (int, np.integer)):
            raise TypeError('FixedQuadCrop requires an integer img_id.')

        results = super().transform(results)
        if results is None:
            raise RuntimeError(
                'allow_negative_crop=True should always retain the tile.')

        results['parent_img_id'] = int(parent_img_id)
        results['tile_index'] = self.tile_index
        results['img_id'] = int(parent_img_id) * 4 + self.tile_index
        results['img_shape'] = self.crop_size
        results['ori_shape'] = self.crop_size
        results['scale_factor'] = (1.0, 1.0)
        # ScaledCocoMetric reconstructs these records from cropped GT.
        ignore_flags = results.get('gt_ignore_flags')
        if ignore_flags is None:
            instance_count = len(results.get('gt_bboxes', []))
        else:
            instance_count = int((ignore_flags == 0).sum())
        results['instances'] = [dict() for _ in range(instance_count)]

        return results

# @MODELS.register_module()
# class BuildingDamagePostProcessor:
#     """基于包含度感知 (IoU + IoS) 的极简灾损后处理。
    
#     仅保留两项核心职能:
#     1. 零超参的无偏序数判决 (round 或 argmax)。
#     2. 类别无关的 IoMIN-NMS: 一行公式同时解决常规重叠与大楼包裹碎块问题。
#     """

#     def __init__(self, 
#                  iou_threshold: float = 0.5,
#                  score_threshold: float = 0.05,
#                  decision_mode: str = 'round'):
#         self.iou_threshold = iou_threshold
#         self.score_threshold = score_threshold
#         self.decision_mode = decision_mode

#     def __call__(self, instance_data: InstanceData, img_shape: Tuple[int, int]) -> InstanceData:
#         if len(instance_data) == 0:
#             return instance_data

#         # 1. 基础置信度过滤
#         keep_mask = instance_data.scores >= self.score_threshold
#         instance_data = instance_data[keep_mask]
#         if len(instance_data) == 0:
#             return instance_data

#         scores = instance_data.scores
#         class_probs = getattr(instance_data, 'class_probs', None)
#         masks = instance_data.masks

#         # 2. 零超参类别判决 (直接支持 BRIGHT 3 分类)
#         if class_probs is not None and class_probs.shape[1] == 3:
#             if self.decision_mode == 'round':
#                 weights = torch.tensor([0.0, 1.0, 2.0], device=class_probs.device)
#                 expected_damage = (class_probs * weights[None, :]).sum(dim=-1)
#                 labels = torch.round(expected_damage).long().clamp(0, 2)
#             else:
#                 labels = class_probs.argmax(dim=-1)
#         else:
#             labels = instance_data.labels

#         # 3. 极简的包含感知 NMS (IoU 与 IoS 双重判定)
#         N = len(instance_data)
#         if N <= 1:
#             instance_data.labels = labels
#             return instance_data

#         # 转换为扁平二值张量计算重合
#         flat_masks = (masks > 0.5).flatten(1).float()
#         areas = flat_masks.sum(dim=1)  # [N]
#         inter = torch.mm(flat_masks, flat_masks.t())  # [N, N]

#         # 计算标准 IoU 与 包含率 IoS
#         unions = areas[:, None] + areas[None, :] - inter
#         ious = inter / unions.clamp_min(1e-6)
        
#         min_areas = torch.minimum(areas[:, None], areas[None, :])
#         ios = inter / min_areas.clamp_min(1e-6)  # 相交面积占较小目标的比例

#         # 核心：只要满足“重叠过大”或“被大幅度包含(>0.6)”，即视为同一物理实体
#         overlap_matrix = torch.maximum(ious, ios)

#         # 标准贪心抑制
#         order = scores.argsort(descending=True)
#         keep = []
#         suppressed = torch.zeros(N, dtype=torch.bool, device=scores.device)

#         for idx in range(N):
#             i = order[idx].item()
#             if suppressed[i]:
#                 continue
#             keep.append(i)
#             # 抑制重叠或包含于当前实体的低分实体
#             ovr = overlap_matrix[i, order[idx + 1:]]
#             suppressed_idx = order[idx + 1:][ovr > self.iou_threshold]
#             suppressed[suppressed_idx] = True

#         # 原位更新并安全返回
#         instance_data = instance_data[keep]
#         instance_data.labels = labels[keep]
#         return instance_data