"""HTC classification by contextual text prototypes and visual objectness."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from mm.det.models.roi_heads.bbox_heads import Shared2FCBBoxHead
from mm.det.registry import MODELS

from .text_bank import _prompt_groups
from .text_loss_normalization import TextLossNormalizer


@MODELS.register_module()
class TextBBoxHead(Shared2FCBBoxHead):
    """Use updated RoI-specific text as the foreground classifier.

    Shared RoI FC features feed the original box regressor, a scalar foreground
    logit and a normalized instance embedding. Each prompt remains a separate
    prototype; class logits use log-mean-exp of prompt cosine logits. The final
    K+1 log probabilities are supervised once by the native classification
    loss. An optional foreground-only cosine alignment loss samples one GT-
    class prompt according to a softmax of same-class cosine similarities,
    with a fixed temperature used only for alignment sampling. Disabling
    ``random_align_sampling`` instead aligns to the highest-similarity prompt.
    GT never enters forward or prototype construction.
    """

    def __init__(self, text_dims=128, temperature=0.07,
                 learnable_temperature=True, lambda_cls=1.0,
                 lambda_align=1.0, sampling_temperature=0.07,
                 random_align_sampling=True, **kwargs):
        super().__init__(**kwargs)
        if type(text_dims) is not int or text_dims < 1:
            raise ValueError('text_dims must be a positive integer.')
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError('temperature must be finite and positive.')
        if not math.isfinite(lambda_cls) or lambda_cls < 0:
            raise ValueError('lambda_cls must be finite and nonnegative.')
        if not math.isfinite(lambda_align) or lambda_align < 0:
            raise ValueError('lambda_align must be finite and nonnegative.')
        if not math.isfinite(sampling_temperature) or sampling_temperature <= 0:
            raise ValueError('sampling_temperature must be finite and positive.')
        if not isinstance(random_align_sampling, bool):
            raise ValueError('random_align_sampling must be boolean.')
        if (not self.with_cls or self.custom_cls_channels
                or self.custom_activation
                or getattr(self.loss_cls, 'use_sigmoid', False)):
            raise ValueError('TextBBoxHead requires ordinary softmax classification.')
        self.text_dims = text_dims
        self.lambda_cls = float(lambda_cls)
        self.lambda_align = float(lambda_align)
        self._class_prompt_counts = None  # Bound once by TextHTC, no tensor cache.
        # A fixed sampling scale, independent of the classifier's logit_scale.
        self.sampling_temperature = float(sampling_temperature)
        self.random_align_sampling = random_align_sampling
        # Keep the original FC/regression paths, but fc_cls now predicts only
        # foreground objectness. All foreground grades come from text matching.
        self.fc_cls = nn.Linear(self.cls_last_dim, 1)
        self.visual_projection = nn.Linear(self.cls_last_dim, text_dims, bias=False)
        scale = torch.tensor(math.log(1. / temperature))
        if learnable_temperature:
            self.logit_scale = nn.Parameter(scale)
        else:
            self.register_buffer('logit_scale', scale)
        if kwargs.get('init_cfg') is None:
            self.init_cfg.append(dict(
                type='Xavier', distribution='uniform',
                override=dict(name='visual_projection')))

    def _text_logits(self, shared_features, class_prototypes, class_token_class_ids,
                     return_similarities=False):
        if (class_prototypes is None or class_prototypes.ndim != 3
                or class_prototypes.shape[0] != shared_features.shape[0]
                or class_prototypes.shape[1] < 1
                or class_prototypes.shape[2] != self.text_dims
                or class_prototypes.device != shared_features.device):
            raise ValueError('class_prototypes must be (RoIs, prompts, text_dims) '
                             'on the RoI device.')
        count = class_prototypes.shape[1]
        if class_token_class_ids is not None and class_token_class_ids.device != shared_features.device:
            raise ValueError('Prompt IDs and RoI features must share a device.')
        groups, counts = _prompt_groups(class_token_class_ids, count, self.num_classes,
                                        self._class_prompt_counts)
        projected = self.visual_projection(shared_features)
        with torch.autocast(device_type=shared_features.device.type, enabled=False):
            # Bound small-norm amplification in visual/text cosine gradients.
            visual = F.normalize(projected.float(), dim=-1, eps=1e-2)
            text = F.normalize(class_prototypes.float(), dim=-1, eps=1e-2)
            similarities = torch.einsum('nd,nmd->nm', visual, text)
            per_prompt = similarities * self.logit_scale.clamp(max=math.log(100.)).exp()
            # Subtract log(prompt count), so repeated identical descriptions or
            # unequal group sizes do not give a class an automatic advantage.
            logits = torch.stack([
                torch.logsumexp(per_prompt[:, groups[c]], dim=-1)
                - math.log(counts[c])
                for c in range(self.num_classes)
            ], dim=-1)
        return (logits, similarities) if return_similarities else logits

    @torch.no_grad()
    def _alignment_probabilities(self, similarities):
        """Sample GT-class prompts by their temperature-scaled cosine affinity.

        Input contains only one GT class's raw cosine similarities (N, K).
        Selection is detached: gradients come from the selected cosine loss,
        not from changing its sampling distribution. Singleton classes do not
        require sampling. Equal similarities receive equal probabilities.
        """
        if similarities.shape[1] == 1:
            return torch.ones_like(similarities)
        return (similarities.float() / self.sampling_temperature).softmax(-1)

    def loss_align(self, similarities, labels, class_token_class_ids=None,
                   *, avg_factor=None):
        """Global mean selected cosine distance over foreground RoIs.

        Background/ignored labels never choose prompts. All class prompts stay
        in the normal classifier; this sampling affects only the auxiliary
        loss. With sampling disabled, choose the same-class argmax (first
        index on ties) without using RNG or sampling_temperature. Empty
        foreground returns a graph-connected zero for DDP.
        """
        if (similarities.ndim != 2 or similarities.shape[1] < 1
                or labels.shape != similarities.shape[:1]
                or labels.dtype != torch.long or labels.device != similarities.device):
            raise ValueError('Alignment requires cosine scores (N, M) and labels (N,).')
        if class_token_class_ids is not None and class_token_class_ids.device != similarities.device:
            raise ValueError('Prompt IDs and RoI features must share a device.')
        groups, _ = _prompt_groups(class_token_class_ids, similarities.shape[1],
                                   self.num_classes, self._class_prompt_counts)
        total = similarities.sum() * 0.
        if self.lambda_align == 0:
            return total
        # Register before prompt sampling so standalone use can overlap the
        # collective too. HTC supplies avg_factor=1 and normalizes at return.
        normalizer = None
        if avg_factor is None:
            normalizer = TextLossNormalizer(1)
            count = ((labels >= 0) & (labels < self.num_classes)).sum()
            normalizer.register('align', count)
        for class_index in range(self.num_classes):
            scores = similarities[labels == class_index][:, groups[class_index]]
            if scores.shape[0] == 0:
                continue
            if scores.shape[1] == 1:
                selected = scores[:, 0]
            elif not self.random_align_sampling:
                indices = scores.detach().argmax(dim=-1, keepdim=True)
                selected = scores.gather(1, indices).squeeze(1)
            else:
                probabilities = self._alignment_probabilities(scores)
                indices = torch.multinomial(probabilities, 1)
                selected = scores.gather(1, indices).squeeze(1)
            total = total + (1. - selected).sum()
        total = self.lambda_align * total
        if normalizer is not None:
            normalizer.add('loss_align', total, 'align')
            return normalizer.finalize()['loss_align']
        return total / avg_factor

    @staticmethod
    def compose_scores(objectness_logits, text_logits):
        """Return normalized log P(c)=log(sigmoid(o))+log Q_text(c).

        The background is log(sigmoid(-o)); it never depends on nuisance
        prompts. Log-space arithmetic remains finite for extreme logits.
        Native MMDetection softmax/NMS and cross entropy accept this output.
        """
        if (objectness_logits.ndim != 2 or objectness_logits.shape[1] != 1
                or text_logits.ndim != 2
                or text_logits.shape[0] != objectness_logits.shape[0]):
            raise ValueError('Expected objectness (N, 1) and class logits (N, K).')
        objectness = objectness_logits.float()
        foreground = F.logsigmoid(objectness) + F.log_softmax(text_logits.float(), dim=-1)
        return torch.cat((foreground, F.logsigmoid(-objectness)), dim=-1)

    def forward(self, x, class_prototypes=None, class_token_class_ids=None,
                return_components=False, return_similarities=False):
        """Optionally append raw prompt cosines for the training-only loss."""
        if self.with_avg_pool:
            x = self.avg_pool(x)
        x = x.flatten(1)
        for fc in self.shared_fcs:
            x = self.relu(fc(x))
        objectness = self.fc_cls(x)
        bbox_pred = self.fc_reg(x) if self.with_reg else None
        text_logits = self._text_logits(
            x, class_prototypes, class_token_class_ids, return_similarities)
        if return_similarities:
            text_logits, similarities = text_logits
        score = self.compose_scores(objectness, text_logits)
        result = ((score, bbox_pred, objectness, text_logits) if return_components
                  else (score, bbox_pred))
        return result + (similarities,) if return_similarities else result
