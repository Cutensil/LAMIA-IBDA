from mm.engine.hooks import CheckpointHook
from mm.det.engine.hooks import DetVisualizationHook
from mm.engine.registry import HOOKS
from mm.det.registry import HOOKS as DETHOOKS, VISUALIZERS
from mm.det.visualization import DetLocalVisualizer
import os.path as osp
import numpy as np
from typing import Sequence, Optional
import mm.cv as mmcv
from mm.engine.fileio import get
from mm.engine.runner import Runner
from mm.engine.utils import mkdir_or_exist
from mm.det.structures import DetDataSample
from mm.engine.dist import master_only


@HOOKS.register_module()
class SafeCheckpointHook(CheckpointHook):
    def __init__(self,
                 *args,
                 refresh_last_checkpoint: bool = False,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.refresh_last_checkpoint = refresh_last_checkpoint

    def before_train(self, runner):
        # Older checkpoints stored one pair of keys per evaluation metric.
        if self.save_best is not None and self.key_indicators == ['coco/segm_mAP']:
            hub = runner.message_hub
            for key in ('best_score', 'best_ckpt'):
                previous_key = f'{key}_coco/segm_mAP'
                if key not in hub.runtime_info and previous_key in hub.runtime_info:
                    hub.update_info(key, hub.get_info(previous_key))
        super().before_train(runner)

    def _save_best_checkpoint(self, runner, *args, **kwargs):
        try:
            if self.refresh_last_checkpoint:
                super()._save_best_checkpoint(runner, *args, **kwargs)
                return

            last_ckpt = self.last_ckpt
            self.last_ckpt = None
            try:
                super()._save_best_checkpoint(runner, *args, **kwargs)
            finally:
                self.last_ckpt = last_ckpt
        except:
            runner.logger.warning('Skipped saving best model since it was failed.')

@DETHOOKS.register_module()
class BiDetVisualizationHook(DetVisualizationHook):
    def __init__(self,
                 pre_dir: str,
                 post_dir: str,
                 pre_colortype: str = 'color',
                 post_colortype: str = 'grayscale',
                 **kwargs):
        super().__init__(**kwargs)
        self.pre_dir = pre_dir
        self.post_dir = post_dir
        self.pre_colortype = pre_colortype
        self.post_colortype = post_colortype
        self._resolved_test_out_dir = None

    @staticmethod
    def _metainfo(data_sample: DetDataSample, key, default=None):
        return data_sample.metainfo.get(key, default)

    def _image_paths(self, data_sample: DetDataSample):
        """Prefer the exact paths produced by LoadBiImageFromFile."""
        img_path = data_sample.img_path
        img_dir = osp.dirname(img_path)
        img_basename = osp.basename(img_path)
        pre_path = self._metainfo(data_sample, 'pre_img_path')
        post_path = self._metainfo(data_sample, 'post_img_path')

        def resolve(path, directory):
            if path is None:
                return osp.join(img_dir, directory, img_basename)
            return path if osp.isabs(path) else osp.join(img_dir, path)

        return (
            resolve(pre_path, self.pre_dir),
            resolve(post_path, self.post_dir),
        )

    def _read_image(self, img_path: str, color_type: str) -> np.ndarray:
        img_bytes = get(img_path, backend_args=self.backend_args)
        img = mmcv.imfrombytes(
            img_bytes, channel_order='rgb', flag=color_type)
        if img.ndim == 2:
            img = np.repeat(img[..., None], 3, axis=2)
        elif img.shape[2] == 1:
            img = np.repeat(img, 3, axis=2)
        return img

    def _tile_index(self, data_sample: DetDataSample,
                    image_shape) -> Optional[int]:
        tile_index = self._metainfo(data_sample, 'tile_index')
        target_shape = self._metainfo(data_sample, 'ori_shape')
        if target_shape is None:
            return tile_index
        target_h, target_w = target_shape[:2]
        image_h, image_w = image_shape[:2]

        # FixedQuadCrop metadata was historically omitted by PackDetInputs.
        # Its image-id mapping is lossless, so recover the quadrant when the
        # source image is twice the predicted tile size.
        if (tile_index is None and image_h == 2 * target_h
                and image_w == 2 * target_w):
            img_id = self._metainfo(data_sample, 'img_id')
            if isinstance(img_id, (int, np.integer)):
                tile_index = int(img_id) % 4
        if tile_index is not None and int(tile_index) not in range(4):
            raise ValueError(f'Invalid tile_index: {tile_index}.')
        return None if tile_index is None else int(tile_index)

    def _prepare_image(self, image: np.ndarray,
                       data_sample: DetDataSample,
                       tile_index: Optional[int]) -> np.ndarray:
        target_shape = self._metainfo(data_sample, 'ori_shape')
        if target_shape is None:
            return image
        target_h, target_w = target_shape[:2]
        if image.shape[:2] == (target_h, target_w):
            return image

        if tile_index is not None:
            full_h, full_w = 2 * target_h, 2 * target_w
            if image.shape[:2] != (full_h, full_w):
                image = mmcv.imresize(image, (full_w, full_h))
            top = (tile_index // 2) * target_h
            left = (tile_index % 2) * target_w
            return image[top:top + target_h, left:left + target_w]

        # Preserve ordinary, non-tiled evaluation support.
        return mmcv.imresize(image, (target_w, target_h))

    def _load_views(self, data_sample: DetDataSample):
        pre_path, post_path = self._image_paths(data_sample)
        pre_image = self._read_image(pre_path, self.pre_colortype)
        post_image = self._read_image(post_path, self.post_colortype)
        tile_index = self._tile_index(data_sample, pre_image.shape)
        return (
            ('pre', pre_path,
             self._prepare_image(pre_image, data_sample, tile_index)),
            ('post', post_path,
             self._prepare_image(post_image, data_sample, tile_index)),
        ), tile_index

    def _parent_img_id(self, data_sample: DetDataSample,
                       tile_index: Optional[int]):
        parent_img_id = self._metainfo(data_sample, 'parent_img_id')
        if parent_img_id is not None:
            return int(parent_img_id)
        img_id = self._metainfo(data_sample, 'img_id')
        if tile_index is not None and isinstance(img_id, (int, np.integer)):
            return int(img_id) // 4
        return None

    def _tile_hint(self, data_sample: DetDataSample) -> Optional[int]:
        """Recover a tile index without image I/O for test-time sampling."""
        tile_index = self._metainfo(data_sample, 'tile_index')
        if tile_index is not None:
            return int(tile_index)
        # All fixed-quadrant evaluation pipelines retain ``instances`` for
        # ScaledCocoMetric and encode the tile in img_id.
        img_id = self._metainfo(data_sample, 'img_id')
        instances = self._metainfo(data_sample, 'instances')
        if instances is not None and isinstance(img_id, (int, np.integer)):
            return int(img_id) % 4
        return None

    @staticmethod
    def _view_name(img_path: str, view: str,
                   tile_index: Optional[int]) -> str:
        tile_suffix = '' if tile_index is None else f'_tile{tile_index}'
        return f'{osp.basename(img_path)}_{view}{tile_suffix}'

    def _out_file(self, img_path: str, view: str,
                  tile_index: Optional[int]) -> Optional[str]:
        if self._resolved_test_out_dir is None:
            return None
        stem = osp.splitext(osp.basename(img_path))[0]
        tile_suffix = '' if tile_index is None else f'_tile{tile_index}'
        return osp.join(
            self._resolved_test_out_dir,
            f'{stem}_{view}{tile_suffix}.png')

    def _visualize_sample(self, data_sample: DetDataSample,
                          step: int, test: bool = False):
        views, tile_index = self._load_views(data_sample)
        for view, img_path, image in views:
            self._visualizer.add_datasample(
                self._view_name(img_path, view, tile_index),
                image,
                data_sample=data_sample,
                show=self.show,
                wait_time=self.wait_time,
                pred_score_thr=self.score_thr,
                out_file=self._out_file(img_path, view, tile_index)
                if test else None,
                step=step)
        return tile_index

    def after_val_iter(self, runner: Runner, batch_idx: int, data_batch: dict,
                       outputs: Sequence[DetDataSample]) -> None:
        if not self.draw:
            return
        step = runner.iter + batch_idx
        if step % self.interval != 0:
            return
        for output in outputs:
            self._visualize_sample(output, step)

    def after_test_iter(self, runner: Runner, batch_idx: int, data_batch: dict,
                       outputs: Sequence[DetDataSample]) -> None:
        if not self.draw:
            return
        if (self.test_out_dir is not None
                and self._resolved_test_out_dir is None):
            self._resolved_test_out_dir = osp.join(
                runner.work_dir, runner.timestamp, self.test_out_dir)
            mkdir_or_exist(self._resolved_test_out_dir)

        for output in outputs:
            self._test_index += 1
            tile_hint = self._tile_hint(output)
            parent_img_id = self._parent_img_id(output, tile_hint)
            sample_index = (
                parent_img_id if parent_img_id is not None
                else self._test_index - 1)
            if sample_index % self.interval != 0:
                continue
            views, tile_index = self._load_views(output)
            for view, img_path, image in views:
                self._visualizer.add_datasample(
                    self._view_name(img_path, view, tile_index),
                    image,
                    data_sample=output,
                    show=self.show,
                    wait_time=self.wait_time,
                    pred_score_thr=self.score_thr,
                    out_file=self._out_file(img_path, view, tile_index),
                    step=self._test_index)

@VISUALIZERS.register_module()
class BiDetLocalVisualizer(DetLocalVisualizer):
    @master_only
    def add_datasample(
            self,
            name: str,
            image: np.ndarray,
            data_sample: Optional['DetDataSample'] = None,
            draw_gt: bool = True,
            draw_pred: bool = True,
            show: bool = False,
            wait_time: float = 0,
            out_file: Optional[str] = None,
            pred_score_thr: float = 0.3,
            step: int = 0) -> None:
        image = image.clip(0, 255).astype(np.uint8)
        classes = self.dataset_meta.get('classes', None)
        palette = self.dataset_meta.get('palette', None)

        gt_img_data = None
        pred_img_data = None

        if data_sample is not None:
            data_sample = data_sample.cpu()

        if draw_gt and data_sample is not None:
            gt_img_data = image
            if 'gt_instances' in data_sample:
                gt_img_data = self._draw_instances(image,
                                                   data_sample.gt_instances,
                                                   classes, palette)

        if draw_pred and data_sample is not None:
            pred_img_data = image
            if 'pred_instances' in data_sample:
                pred_instances = data_sample.pred_instances
                pred_instances = pred_instances[
                    pred_instances.scores > pred_score_thr]
                pred_img_data = self._draw_instances(image, pred_instances,
                                                     classes, palette)

        if gt_img_data is not None and pred_img_data is not None:
            drawn_img = np.concatenate((gt_img_data, pred_img_data), axis=1)
        elif gt_img_data is not None:
            drawn_img = gt_img_data
        elif pred_img_data is not None:
            drawn_img = pred_img_data
        else:
            # Display the original image directly if nothing is drawn.
            drawn_img = image

        # It is convenient for users to obtain the drawn image.
        # For example, the user wants to obtain the drawn image and
        # save it as a video during video inference.
        self.set_image(drawn_img)

        if show:
            self.show(drawn_img, win_name=name, wait_time=wait_time)

        if out_file is not None:
            mmcv.imwrite(drawn_img[..., ::-1], out_file)
        else:
            self.add_image(name, drawn_img, step)
