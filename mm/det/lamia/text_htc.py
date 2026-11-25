"""HTC with text-guided fusion and optional contextual-text classification."""

import copy

import torch
import torch.nn.functional as F
from pycocotools import mask as mask_utils

from mm.det.models.detectors.htc import HybridTaskCascade
from mm.det.models.roi_heads.htc_roi_head import HybridTaskCascadeRoIHead
from mm.det.registry import MODELS

from .text_bank import TextBank
from .text_fusion_neck import TextGuidedFusionNeck
from .text_roi_head import TextRoIHead
from .text_loss_normalization import TextLossNormalizer


@MODELS.register_module()
class TextHTC(HybridTaskCascade):
    """Explicit text routing; no forward caches or duplicated text modules.

    ``neck`` is the externally conditioned fusion neck; ``fpn`` performs
    multi-scale fusion afterwards. One full prompt bank feeds fusion. Updated
    window class tokens are passed explicitly to every RoI stage and spatially
    pooled again for each stage's current boxes. All prompts participate in
    both training and inference. Only the optional alignment loss samples a
    GT-class prompt; classification itself always uses all prompts.
    A native HybridTaskCascadeRoIHead disables text classification entirely:
    the bank still guides fusion and its auxiliary losses, while native HTC
    computes loss_cls and predicts without receiving any text arguments.
    The neck owns a training-only pure-visual semantic probe and the weighted
    semantic-guided visual/text attention budget loss. Fusion-only uses native
    HTC heads; classification-only uses projected TextBank text without updates.
    Existing image-only ``extract_feat`` callers still receive an FPN tuple.
    ``attn_use_cls_prompt`` optionally uses detached final-stage, GT-class
    prompt probabilities to distribute the semantic text attention budget.
    """

    def __init__(self, text_bank, fpn,
                 use_text_fusion=True, use_text_classification=None,
                 attn_use_cls_prompt=False, text_inference_mode='contextual',
                 diagnostic_attention=False, **kwargs):
        if not isinstance(use_text_fusion, bool):
            raise ValueError('use_text_fusion must be boolean.')
        if use_text_classification is not None and not isinstance(use_text_classification, bool):
            raise ValueError('use_text_classification must be boolean or None.')
        if not isinstance(attn_use_cls_prompt, bool):
            raise ValueError('attn_use_cls_prompt must be boolean.')
        if text_inference_mode not in ('contextual', 'no_text_update',
                                       'static_classifier', 'decoupled', 'random_text'):
            raise ValueError('Unsupported text_inference_mode.')
        roi_cfg = copy.deepcopy(kwargs['roi_head'])
        if use_text_classification is None:
            use_text_classification = roi_cfg['type'] in ('TextRoIHead', TextRoIHead)
        if attn_use_cls_prompt and not (use_text_fusion and use_text_classification):
            raise ValueError('attn_use_cls_prompt requires text fusion and text classification.')
        # Flags select real modules at construction time, not dormant branches
        # with unused parameters. Existing native fusion-only configs still work.
        roi_cfg['type'] = 'TextRoIHead' if use_text_classification else 'HybridTaskCascadeRoIHead'
        for head in roi_cfg['bbox_head']:
            head['type'] = 'TextBBoxHead' if use_text_classification else 'Shared2FCBBoxHead'
            if use_text_classification:
                head.setdefault('text_dims', text_bank.get('embed_dims', 128))
            else:
                for key in ('text_dims', 'temperature', 'learnable_temperature',
                            'lambda_cls', 'lambda_align', 'sampling_temperature',
                            'random_align_sampling'):
                    head.pop(key, None)
        kwargs['roi_head'] = roi_cfg
        # Decide the parameter shape before constructing the neck/optimizer/DDP.
        # One shared bank controls both token presence and type-embedding count.
        has_noise = bool(text_bank.get('noise_prompts'))
        neck = dict(kwargs['neck'])
        if use_text_fusion:
            # TextHTC owns the shared bank and supplies text to the neck.
            neck['external_text'] = True
            neck.setdefault('semantic_num_classes', roi_cfg['bbox_head'][0]['num_classes'])
            configured = neck.get('use_noise_text')
            if configured is not None and (
                    not isinstance(configured, bool) or configured != has_noise):
                raise ValueError('neck.use_noise_text must match text_bank.noise_prompts.')
            neck['use_noise_text'] = has_noise
            if text_inference_mode in ('no_text_update', 'decoupled'):
                neck['freeze_text_updates'] = True
        else:
            neck = dict(type='VisualFusionNeck', channels=neck['channels'],
                        fusion_norm_cfg=neck.get('fusion_norm_cfg'))
        kwargs['neck'] = neck
        super().__init__(**kwargs)
        self.use_text_fusion = use_text_fusion
        self.attn_use_cls_prompt = attn_use_cls_prompt
        self.text_inference_mode = text_inference_mode
        self.diagnostic_attention = bool(diagnostic_attention)
        self._random_class_text = None
        self.text_bank = (MODELS.build(text_bank)
                          if use_text_fusion or use_text_classification else None)
        self.fpn = MODELS.build(fpn)
        if self.text_bank is not None and not isinstance(self.text_bank, TextBank):
            raise TypeError('text_bank must be TextBank.')
        if use_text_fusion and (not isinstance(self.neck, TextGuidedFusionNeck)
                or not self.neck.external_text):
            raise TypeError('TextHTC needs TextGuidedFusionNeck(external_text=True).')
        if not isinstance(self.roi_head, HybridTaskCascadeRoIHead):
            raise TypeError('TextHTC needs an HTC RoI head.')
        self.with_text_classification = isinstance(self.roi_head, TextRoIHead)
        if use_text_fusion and self.neck.embed_dims != self.text_bank.embed_dims:
            raise ValueError('Fusion and text-bank dimensions must agree.')
        if (use_text_fusion and self.neck.semantic_head is not None
                and self.neck.semantic_head.num_classes != self.text_bank.num_classes):
            raise ValueError('Visual semantic and text-bank class counts must agree.')
        for head in self.roi_head.bbox_head:
            if self.text_bank is not None and (head.num_classes != self.text_bank.num_classes
                    or (self.with_text_classification
                        and head.text_dims != self.text_bank.embed_dims)):
                raise ValueError('Text bank and box heads must agree on classes/dimensions.')
            if not self.with_text_classification and hasattr(head, 'text_dims'):
                raise ValueError('Native HTC RoI heads require native box heads; '
                                 'use Shared2FCBBoxHead for fusion-only models.')
        if tuple(self.fpn.in_channels) != self.neck.channels:
            raise ValueError('FPN input channels must match fusion output channels.')
        # The bank owns this fixed, contiguous prompt order. Bind CPU shape
        # metadata once instead of reading constant GPU IDs in every stage.
        if use_text_fusion:
            self.neck._class_prompt_counts = self.text_bank.class_prompt_counts
        if self.with_text_classification:
            for head in self.roi_head.bbox_head:
                head._class_prompt_counts = self.text_bank.class_prompt_counts

    def _extract_feat_with_text(self, batch_inputs, return_aux=False):
        class_text, noise_text = self.text_bank() if self.text_bank is not None else (None, None)
        if self.text_inference_mode == 'random_text' and class_text is not None:
            if (self._random_class_text is None
                    or self._random_class_text.shape != class_text.shape
                    or self._random_class_text.device != class_text.device):
                self._random_class_text = torch.randn_like(class_text)
                self._random_class_text = torch.nn.functional.normalize(
                    self._random_class_text, dim=-1)
            class_text = self._random_class_text
        x = self.backbone(batch_inputs)
        auxiliary_states, context = None, None
        if self.use_text_fusion:
            need_fusion_loss = return_aux and self.neck.has_attn_supervision
            result = self.neck(x, class_text=class_text, noise_text=noise_text,
                               return_context=self.with_text_classification,
                               return_aux=need_fusion_loss)
            if need_fusion_loss:
                x, attention, context, auxiliary_states = result
            elif self.with_text_classification:
                x, attention, context = result
            else:
                x, attention = result, None
        else:
            x = self.neck(x)
            if self.with_text_classification:
                # A single global window makes every RoI read the same original
                # projected prompt, with no text-guided visual fusion involved.
                context = dict(tokens=class_text[None, None].expand(
                    batch_inputs.shape[0], 1, -1, -1),
                    window_boxes=class_text.new_tensor([[0., 0., 1., 1.]]),
                    feature_shape=(1, 1))
        if context is not None:
            if (self.text_inference_mode in ('static_classifier', 'decoupled')
                    and self.with_text_classification and class_text is not None):
                context = dict(
                    context,
                    tokens=class_text[None, None].expand(
                        batch_inputs.shape[0], context['tokens'].shape[1], -1, -1))
            context = dict(context, input_shape=tuple(batch_inputs.shape[-2:]))
        features = self.fpn(x)
        if return_aux:
            return features, class_text, noise_text, auxiliary_states, context
        return features, context

    def extract_feat(self, batch_inputs):
        return self._extract_feat_with_text(batch_inputs)[0]

    def _classification_kwargs(self, context):
        if not self.with_text_classification:
            return {}
        return dict(class_context=context,
                    class_token_class_ids=self.text_bank.class_token_class_ids)

    @staticmethod
    def _provided_proposals(batch_data_samples):
        if not all(sample.get('proposals', None) is not None
                   for sample in batch_data_samples):
            raise ValueError('Every sample must provide proposals when RPN is absent.')
        return [sample.proposals for sample in batch_data_samples]

    def loss(self, batch_inputs, batch_data_samples):
        x, _, _, auxiliary_states, context = self._extract_feat_with_text(
            batch_inputs, return_aux=True)
        classification = self._classification_kwargs(context)
        losses = {}
        need_fusion_loss = self.use_text_fusion and self.neck.has_attn_supervision
        count_groups = (sum(weight > 0 for weight in self.neck.stride_loss_weights)
                        + 1 if need_fusion_loss else 0)
        if self.with_text_classification:
            count_groups += sum(head.lambda_align > 0 for head in self.roi_head.bbox_head)
        normalizer = TextLossNormalizer(count_groups)
        prompt_context = {} if need_fusion_loss and self.attn_use_cls_prompt else None
        if need_fusion_loss:
            losses.update(self.neck.loss_attn(
                auxiliary_states, self.text_bank.class_token_class_ids,
                batch_data_samples, batch_inputs.shape[-2:], normalizer=normalizer,
                prompt_context=prompt_context))
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal', self.test_cfg.rpn)
            # RPN only changes labels. Keep its containers independent while
            # sharing read-only boxes, masks and semantic GT with the RoI path.
            rpn_samples = []
            for sample in batch_data_samples:
                rpn_sample = sample.new()
                rpn_sample.gt_instances = sample.gt_instances.new()
                rpn_sample.gt_instances.labels = torch.zeros_like(sample.gt_instances.labels)
                rpn_samples.append(rpn_sample)
            rpn_losses, proposals = self.rpn_head.loss_and_predict(
                x, rpn_samples, proposal_cfg=proposal_cfg)
            for key in list(rpn_losses):
                if 'loss' in key and 'rpn' not in key:
                    rpn_losses[f'rpn_{key}'] = rpn_losses.pop(key)
            losses.update(rpn_losses)
        else:
            proposals = self._provided_proposals(batch_data_samples)
        prompt_targets = {} if prompt_context is not None else None
        roi_kwargs = dict(normalizer=normalizer) if self.with_text_classification else {}
        if prompt_targets is not None:
            roi_kwargs['prompt_targets'] = prompt_targets
        roi_losses = self.roi_head.loss(
            x, proposals, batch_data_samples, **classification, **roi_kwargs)
        losses.update(roi_losses)
        if prompt_context is not None:
            self.neck.loss_attn_prompt(
                prompt_context, prompt_targets['probabilities'], normalizer=normalizer)
        return normalizer.finalize(losses)

    def predict(self, batch_inputs, batch_data_samples, rescale=True):
        try:
            with open('/tmp/text_htc_predict_debug.log', 'a') as handle:
                handle.write(f'predict diagnostic={self.diagnostic_attention}\n')
        except Exception:
            pass
        if self.diagnostic_attention:
            x, _, _, auxiliary_states, context = self._extract_feat_with_text(
                batch_inputs, return_aux=True)
            diagnostics = self._collect_attention_diagnostics(
                auxiliary_states, batch_data_samples, batch_inputs.shape[-2:])
        else:
            x, context = self._extract_feat_with_text(batch_inputs)
        if batch_data_samples[0].get('proposals', None) is None:
            if not self.with_rpn:
                raise ValueError('Prediction needs RPN or provided proposals.')
            proposals = self.rpn_head.predict(x, batch_data_samples, rescale=False)
        else:
            proposals = self._provided_proposals(batch_data_samples)
        results = self.roi_head.predict(
            x, proposals, batch_data_samples, rescale=rescale,
            **self._classification_kwargs(context))
        output = self.add_pred_to_datasample(batch_data_samples, results)
        if self.diagnostic_attention:
            for sample, diag in zip(output, diagnostics):
                if diag is not None:
                    sample.set_field(diag, 'attention_diagnostics')
        return output

    @torch.no_grad()
    def _collect_attention_diagnostics(self, auxiliary_states, batch_data_samples, input_shape):
        """Attach compact per-image reliability/allocation summaries."""
        if auxiliary_states is None:
            return [None] * len(batch_data_samples)
        groups = auxiliary_states.values() if isinstance(auxiliary_states, dict) else (auxiliary_states,)
        try:
            with open('/tmp/attention_state_debug.log', 'a') as handle:
                handle.write(str([(list(state.keys()), state.get('attention') is not None,
                                   state.get('visual_attention') is not None,
                                   state.get('semantic_logits') is not None)
                                  for group in groups for state in group]) + '\n')
        except Exception:
            pass
        groups = auxiliary_states.values() if isinstance(auxiliary_states, dict) else (auxiliary_states,)
        states = [state for group in groups for state in group
                  if state.get('attention') is not None
                  and state.get('visual_attention') is not None
                  and state.get('semantic_logits') is not None]
        if not states:
            return [None] * len(batch_data_samples)
        state = states[-1]
        text = state['attention'].float().exp()
        visual = state['visual_attention'].float().exp()
        logits = state['semantic_logits'].float()
        probs = F.interpolate(logits, size=text.shape[-2:], mode='bilinear',
                              align_corners=False).softmax(1)[:, 1:]
        prompt_counts = tuple(self.text_bank.class_prompt_counts)
        offsets = []
        total = 0
        for count in prompt_counts:
            offsets.append(total)
            total += int(count)
        diagnostics = []
        for image_index, sample in enumerate(batch_data_samples):
            gt = sample.gt_instances
            raw_instances = getattr(sample, 'instances', None)
            if (not hasattr(gt, 'masks') or len(gt) == 0) and raw_instances:
                decoded = []
                labels_raw = []
                for item in raw_instances:
                    if item.get('mask') is None:
                        continue
                    decoded.append(torch.as_tensor(
                        mask_utils.decode(item['mask']), dtype=torch.float32,
                        device=text.device))
                    labels_raw.append(int(item.get('bbox_label', 0)))
                if decoded:
                    gt_masks = torch.stack(decoded, dim=0)
                    masks = F.interpolate(gt_masks[:, None], size=text.shape[-2:],
                                          mode='area').squeeze(1)
                    mass = masks.flatten(1).sum(1).clamp_min(1e-6)
                    labels = torch.tensor(labels_raw, device=text.device, dtype=torch.long)
                    q = probs[image_index, labels]
                    rho = (q * masks).flatten(1).sum(1) / mass
                    v = (visual[image_index, 0][None] * masks).flatten(1).sum(1) / mass
                    group_values = []
                    for start, count in zip(offsets, prompt_counts):
                        group = text[image_index, start:start + int(count)].sum(0)
                        group_values.append((group[None] * masks).flatten(1).sum(1) / mass)
                    group_values = torch.stack(group_values, dim=0)
                    t = group_values[labels, torch.arange(len(labels), device=text.device)]
                    diagnostics.append(dict(rho=float(rho.mean().cpu()),
                                            visual_attention=float(v.mean().cpu()),
                                            text_attention=float(t.mean().cpu())))
                    continue
            if not hasattr(gt, 'masks') or len(gt) == 0:
                # Empty tile GT is common for the full validation loader. Use
                # a compact global semantic-confidence proxy so every image
                # still contributes to corruption diagnostics; mask-pooled
                # records above remain the preferred reliability definition.
                rho_global = probs[image_index].max(0).values.mean()
                visual_global = visual[image_index, 0].mean()
                text_global = text[image_index].sum(0).mean()
                diagnostics.append(dict(rho=float(rho_global.cpu()),
                                        visual_attention=float(visual_global.cpu()),
                                        text_attention=float(text_global.cpu())))
                continue
            masks = gt.masks.to_tensor(dtype=torch.float32,
                                       device=text.device)
            masks = F.interpolate(masks[:, None], size=text.shape[-2:], mode='area').squeeze(1)
            mass = masks.flatten(1).sum(1).clamp_min(1e-6)
            labels = gt.labels.to(text.device)
            q = probs[image_index, labels]
            rho = (q * masks).flatten(1).sum(1) / mass
            v = (visual[image_index, 0][None] * masks).flatten(1).sum(1) / mass
            group_values = []
            for start, count in zip(offsets, prompt_counts):
                group = text[image_index, start:start + int(count)].sum(0)
                group_values.append((group[None] * masks).flatten(1).sum(1) / mass)
            group_values = torch.stack(group_values, dim=0)
            t = group_values[labels, torch.arange(len(labels), device=text.device)]
            diagnostics.append(dict(rho=float(rho.mean().cpu()),
                                    visual_attention=float(v.mean().cpu()),
                                    text_attention=float(t.mean().cpu())))
        return diagnostics

    def _forward(self, batch_inputs, batch_data_samples):
        x, context = self._extract_feat_with_text(batch_inputs)
        classification = self._classification_kwargs(context)
        if self.with_rpn:
            proposals = self.rpn_head.predict(x, batch_data_samples, rescale=False)
        else:
            proposals = self._provided_proposals(batch_data_samples)
        return (self.roi_head.forward(
            x, proposals, batch_data_samples, **classification),)
