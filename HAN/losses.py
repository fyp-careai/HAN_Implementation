"""
InfoNCE Contrastive Link Prediction Loss for HANPP
====================================================

Multi-positive InfoNCE loss for disease link prediction.
For each patient, their true diseases should score higher than all non-diseases
in the shared embedding space.

Mathematical formulation:
    score(i, d) = cos(z_patient_i, z_disease_d) / τ

    loss_i = -log( Σ_{d ∈ pos(i)} exp(score(i, d))  /  Σ_{d ∈ all} exp(score(i, d)) )

    This pushes positive disease embeddings closer to the patient embedding
    and negative disease embeddings further away, with all diseases competing
    against each other in a softmax.

Reference:
    Oord et al. "Representation Learning with Contrastive Predictive Coding." 2018.
    Radford et al. "Learning Transferable Visual Models (CLIP)." ICML 2021.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELinkLoss(nn.Module):
    """
    Multi-positive InfoNCE contrastive loss for link prediction.

    For each patient i with positive disease set D+(i) and all diseases D:
        loss_i = -log( Σ exp(s_{i,d+}) / Σ exp(s_{i,d}) )

    This is equivalent to a multi-label softmax where all positive diseases
    collectively compete against all diseases.

    Args:
        hard_neg_weight: upweight factor for hard negatives (diseases that score
                         high but should be negative). Default 1.0 = no upweighting.
        label_smoothing: smooth labels by ε to prevent overconfident embeddings.
                         Default 0.0 = no smoothing.
    """

    def __init__(self, hard_neg_weight=1.0, label_smoothing=0.0):
        super().__init__()
        self.hard_neg_weight = hard_neg_weight
        self.label_smoothing = label_smoothing

    def forward(self, scores, labels):
        """
        Args:
            scores: [N, D] raw cosine similarity / τ  (from model forward)
            labels: [N, D] binary multi-label targets (1 = positive link)

        Returns:
            loss: scalar contrastive loss
        """
        N, D = scores.shape

        # Apply label smoothing if requested
        if self.label_smoothing > 0:
            labels = labels * (1.0 - self.label_smoothing) + self.label_smoothing / D

        # ── Denominator: log-sum-exp over ALL diseases ────────────────────────
        # If hard negative weighting is active, upweight negative disease scores
        if self.hard_neg_weight > 1.0:
            neg_mask = (labels < 0.5).float()
            # Scale up negative scores to make them harder to distinguish
            weighted_scores = scores + neg_mask * torch.log(
                torch.tensor(self.hard_neg_weight, device=scores.device)
            )
            log_sum_all = torch.logsumexp(weighted_scores, dim=1)  # [N]
        else:
            log_sum_all = torch.logsumexp(scores, dim=1)  # [N]

        # ── Numerator: log-sum-exp over POSITIVE diseases only ────────────────
        pos_mask = (labels > 0.5)
        neg_inf = torch.full_like(scores, -1e9)
        pos_scores = torch.where(pos_mask, scores, neg_inf)
        log_sum_pos = torch.logsumexp(pos_scores, dim=1)  # [N]

        # ── Loss: -log(sum_pos / sum_all) ─────────────────────────────────────
        loss_per_patient = -(log_sum_pos - log_sum_all)

        # Only compute loss for patients with at least one positive disease
        has_pos = labels.sum(dim=1) > 0.5
        if has_pos.sum() > 0:
            return loss_per_patient[has_pos].mean()
        else:
            return torch.tensor(0.0, device=scores.device, requires_grad=True)


class MarginLinkLoss(nn.Module):
    """
    Margin-based link prediction loss (TransE-style).

    For each patient, each (positive, negative) disease pair:
        loss = max(0, margin + score(neg) - score(pos))

    This explicitly enforces that positive disease scores exceed negative
    disease scores by at least `margin`.

    Args:
        margin: minimum required score gap between positive and negative diseases.
    """

    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, scores, labels):
        """
        Args:
            scores: [N, D] raw scores
            labels: [N, D] binary targets

        Returns:
            loss: scalar margin loss
        """
        pos_mask = labels.bool()
        neg_mask = ~pos_mask

        losses = []
        for i in range(scores.shape[0]):
            pos_scores = scores[i][pos_mask[i]]
            neg_scores = scores[i][neg_mask[i]]

            if pos_scores.numel() == 0 or neg_scores.numel() == 0:
                continue

            # All pairs: [num_pos, 1] - [1, num_neg] → [num_pos, num_neg]
            margin_loss = self.margin + neg_scores.unsqueeze(0) - pos_scores.unsqueeze(1)
            margin_loss = F.relu(margin_loss)
            losses.append(margin_loss.mean())

        if losses:
            return torch.stack(losses).mean()
        else:
            return torch.tensor(0.0, device=scores.device, requires_grad=True)
