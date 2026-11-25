"""HTC RoI classification with spatially read contextual text prototypes."""

import math

import torch
from torch.nn import functional as F

from mm.det.models.losses import accuracy
from mm.det.models.roi_heads.htc_roi_head import HybridTaskCascadeRoIHead
from mm.det.registry import MODELS
from mm.det.structures.bbox import bbox2roi

from .text_bbox_head import TextBBoxHead
from .text_pooling import pool_class_prototypes
from .text_loss_normalization import TextLossNormalizer


class _TextRoIFeatures(tuple):
    """Call-local context carrier through native HTC refinement/loss loops."""

    def __new__(cls, features, class_context, class_token_class_ids,
                img_shapes, score_collector=None, normalizer=None, prompt_targets=None):
        result = super().__new__(cls, features)
        result.class_context = class_context
        result.class_token_class_ids = class_token_class_ids
        result.img_shapes = img_shapes
        result.score_collector = score_collector
        result.inactive_cls_zero = None
        result.normalizer = normalizer
        result.prompt_targets = prompt_targets
        return result


@MODELS.register_module()
class TextRoIHead(HybridTaskCascadeRoIHead):
    """Read window text using each stage's current boxes; keep native masks."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not all(isinstance(head, TextBBoxHead) for head in self.bbox_head):
            raise TypeError('Every cascade stage must use TextBBoxHead.')
        if len({(h.num_classes, h.text_dims) for h in self.bbox_head}) != 1:
            raise ValueError('All stages must agree on class count and text width.')

    @staticmethod
    def _with_context(x, class_context, class_token_class_ids, batch_data_samples,
                      normalizer=None, prompt_targets=None):
        if class_context is None:
            raise ValueError('TextRoIHead requires updated window text context.')
        return _TextRoIFeatures(
            x, class_context, class_token_class_ids,
            tuple(sample.img_shape[:2] for sample in batch_data_samples),
            normalizer=normalizer, prompt_targets=prompt_targets)

    pool_class_prototypes = staticmethod(pool_class_prototypes)

    def loss(self, x, rpn_results_list, batch_data_samples, *, class_context,
             class_token_class_ids=None, normalizer=None, prompt_targets=None):
        own_normalizer = normalizer is None
        if own_normalizer:
            normalizer = TextLossNormalizer(sum(
                head.lambda_align > 0 for head in self.bbox_head))
        if prompt_targets is not None:
            offsets, total = [], 0
            for sample in batch_data_samples:
                offsets.append(total)
                total += len(sample.gt_instances.labels)
            tokens = class_context['tokens']
            prompt_targets.update(
                offsets=tuple(offsets),
                probabilities=tokens.new_zeros((total, tokens.shape[2]), dtype=torch.float32),
                counts=tokens.new_zeros(total, dtype=torch.float32))
        features = self._with_context(
            x, class_context, class_token_class_ids, batch_data_samples, normalizer,
            prompt_targets)
        losses = super().loss(features, rpn_results_list, batch_data_samples)
        if prompt_targets is not None:
            # Zero rows stay zero: no final-stage positive RoI means budget-only
            # supervision for that GT, not an artificial uniform prompt target.
            prompt_targets['probabilities'].div_(prompt_targets['counts'].clamp_min(1.)[:, None])
        if features.inactive_cls_zero is not None:
            for name in losses:
                if 'loss' in name:
                    losses[name] = losses[name] + features.inactive_cls_zero
                    break
        return normalizer.finalize(losses) if own_normalizer else losses

    def predict(self, x, rpn_results_list, batch_data_samples, rescale=False,
                *, class_context, class_token_class_ids=None):
        return super().predict(
            self._with_context(x, class_context, class_token_class_ids, batch_data_samples),
            rpn_results_list, batch_data_samples, rescale=rescale)

    def forward(self, x, rpn_results_list, batch_data_samples, *, class_context,
                class_token_class_ids=None):
        return super().forward(
            self._with_context(x, class_context, class_token_class_ids, batch_data_samples),
            rpn_results_list, batch_data_samples)

    def bbox_loss(self, stage, x, sampling_results, semantic_feat=None):
        head = self.bbox_head[stage]
        collect_prompts = x.prompt_targets is not None and stage == self.num_stages - 1
        rois = bbox2roi([result.priors for result in sampling_results])
        results = self._bbox_forward(stage, x, rois, semantic_feat=semantic_feat,
                                     return_similarities=head.lambda_align > 0 or collect_prompts)
        results.update(rois=rois)
        losses_and_targets = head.loss_and_target(
            cls_score=results['cls_score'] if head.lambda_cls > 0 else None,
            bbox_pred=results['bbox_pred'], rois=rois,
            sampling_results=sampling_results, rcnn_train_cfg=self.train_cfg[stage])
        bbox_losses = losses_and_targets['loss_bbox']
        labels = losses_and_targets['bbox_targets'][0]
        similarities = results.pop('prompt_similarities', None)
        if collect_prompts:
            self._collect_prompt_targets(
                head, similarities, sampling_results, x.class_token_class_ids, x.prompt_targets)
        if head.lambda_align > 0:
            key = ('align', stage)
            count = ((labels >= 0) & (labels < head.num_classes)).sum()
            x.normalizer.register(key, count)
            # Native HTC weights bbox losses in its stage loop. This deferred
            # numerator is outside that dict, so apply that weight here once.
            alignment = head.loss_align(
                similarities, labels, x.class_token_class_ids, avg_factor=1.)
            x.normalizer.add(f's{stage}.loss_align',
                self.stage_loss_weights[stage] * alignment, key)
        if head.lambda_cls > 0:
            classification = bbox_losses.get('loss_cls', results['cls_score'].sum() * 0.)
            bbox_losses['loss_cls'] = head.lambda_cls * classification
        else:
            if results['cls_score'].numel() > 0:
                bbox_losses['acc'] = accuracy(results['cls_score'], labels)
            # Keep classifier parameters in DDP without a zero loss log entry.
            zero = results['cls_score'].sum() * 0.
            x.inactive_cls_zero = (zero if x.inactive_cls_zero is None
                                   else x.inactive_cls_zero + zero)
        results.update(losses_and_targets)
        return results

    @staticmethod
    @torch.no_grad()
    def _collect_prompt_targets(head, similarities, sampling_results, ids, targets):
        """Mean final-stage GT-class probabilities per original GT instance.

        Reuse actual classifier cosines and its current (clamped) logit scale.
        Positives precede negatives in each SamplingResult.priors tensor.
        This deterministic teacher never participates in autograd or sampling.
        """
        if ids is None:
            ids = torch.arange(head.num_classes, device=similarities.device)
        scale = head.logit_scale.detach().clamp(max=math.log(100.)).exp()
        begin = 0
        for result, gt_offset in zip(sampling_results, targets['offsets']):
            count = len(result.pos_assigned_gt_inds)
            scores = similarities[begin:begin + count].detach().float() * scale
            same_class = ids[None] == result.pos_gt_labels[:, None]
            probabilities = scores.masked_fill(~same_class, float('-inf')).softmax(-1)
            rows = result.pos_assigned_gt_inds + gt_offset
            targets['probabilities'].index_add_(0, rows, probabilities)
            targets['counts'].index_add_(0, rows, scores.new_ones(count))
            begin += len(result.priors)

    def _refine_roi(self, x, rois, batch_img_metas, num_proposals_per_img, **kwargs):
        """Average stage objectness/class logits before composing probabilities."""
        collector = []
        features = _TextRoIFeatures(
            x, x.class_context, x.class_token_class_ids, x.img_shapes, collector)
        refined, _, bbox_preds = super()._refine_roi(
            features, rois, batch_img_metas, num_proposals_per_img, **kwargs)
        objectness = sum(item[0] for item in collector) / len(collector)
        text_logits = sum(item[1] for item in collector) / len(collector)
        scores = self.bbox_head[-1].compose_scores(objectness, text_logits)
        return refined, scores.split(num_proposals_per_img, 0), bbox_preds

    def _bbox_forward(self, stage, x, rois, semantic_feat=None,
                      return_similarities=False):
        if not isinstance(x, _TextRoIFeatures):
            raise ValueError('Call loss/predict/forward with explicit class_context.')
        extractor = self.bbox_roi_extractor[stage]
        bbox_feats = extractor(x[:extractor.num_inputs], rois)
        if self.with_semantic and 'bbox' in self.semantic_fusion:
            semantic = self.semantic_roi_extractor([semantic_feat], rois)
            if semantic.shape[-2:] != bbox_feats.shape[-2:]:
                semantic = F.adaptive_avg_pool2d(semantic, bbox_feats.shape[-2:])
            bbox_feats = bbox_feats + semantic
        prototypes = self.pool_class_prototypes(x.class_context, rois, x.img_shapes)
        output = self.bbox_head[stage](
            bbox_feats, prototypes, x.class_token_class_ids, return_components=True,
            return_similarities=return_similarities)
        score, regression, objectness, text_logits = output[:4]
        if x.score_collector is not None:
            x.score_collector.append((objectness, text_logits))
        results = dict(cls_score=score, bbox_pred=regression)
        if return_similarities:
            results['prompt_similarities'] = output[4]
        return results
