import numpy as np
import pycocotools.mask as mask_util

from mm.det.evaluation.metrics import CocoMetric
from mm.det.registry import METRICS


@METRICS.register_module()
class ScaledCocoMetric(CocoMetric):
    """Evaluate cropped GT with bounded RLE conversion buffers."""

    _RLE_CHUNK_BYTES = 1 << 20

    def bitmap_to_rle(self, bitmap_masks, transpose=True):
        mask_array = bitmap_masks.to_ndarray()
        if transpose:
            mask_array = mask_array.transpose(1, 2, 0)
        # Convert layout and dtype together, avoiding a full-size astype copy.
        # Each mask is independent, so chunking preserves its exact RLE bytes.
        pixels = mask_array.shape[0] * mask_array.shape[1]
        chunk_size = max(1, self._RLE_CHUNK_BYTES // max(1, pixels))
        rles = []
        for start in range(0, mask_array.shape[2], chunk_size):
            masks = np.asfortranarray(
                mask_array[:, :, start:start + chunk_size], dtype=np.uint8)
            rles.extend(mask_util.encode(masks))
        return rles

    def process(self, data_batch, data_samples):
        # Once COCO GT is initialized, the parent ignores per-sample GT records.
        if self._coco_api is not None:
            super().process(data_batch, data_samples)
            return
        for data_sample in data_samples:
            assert "instances" in data_sample, \
                    'ground truth is required for evaluation when ' \
                    '`ann_file` is not provided'
            assert len(data_sample['instances']) == len(data_sample['gt_instances']['masks']), \
                    'length of instance and gt_instance should be the same'
            n = len(data_sample['instances'])
            gt_instance = data_sample['gt_instances']
            gt_bbox = gt_instance['bboxes'].cpu().numpy()
            gt_mask_rles = self.bitmap_to_rle(gt_instance['masks'])
            gt_label = gt_instance['labels'].cpu().numpy()
            for i in range(n):
                sample = data_sample['instances'][i]
                sample['bbox'] = gt_bbox[i].tolist()
                sample['bbox_label'] = gt_label[i].item()
                sample['mask'] = gt_mask_rles[i]
        super().process(data_batch, data_samples)
