"""
MC Dropout Uncertainty Quantification for HAN++ (CareAI)
=========================================================

Monte Carlo Dropout approximates Bayesian inference by running the model
multiple times with dropout ENABLED at inference time. Each forward pass
samples a different dropout mask, producing a distribution over predictions.

Clinical value:
- Mean prediction: the best estimate of organ severity
- Std (uncertainty): how confident the model is — a std of 0.05 means
  the model is very confident; std of 0.25 means "ask the doctor"
- Clinically essential: regulators (FDA, MHRA) require uncertainty
  estimates for AI-assisted diagnosis tools.

Reference:
    Gal & Ghahramani. "Dropout as a Bayesian Approximation." ICML 2016.
"""

import torch
import numpy as np


def mc_dropout_predict(model, features, neighbor_dicts, n_samples=50, device=None):
    """
    Monte Carlo Dropout inference for uncertainty quantification.

    Runs n_samples stochastic forward passes with dropout ENABLED,
    then computes mean and standard deviation of the predicted
    class probability distribution.

    Args:
        model:          trained HANPP model (must have dropout layers)
        features:       patient feature tensor [N, in_dim]
        neighbor_dicts: dict of {metapath_name: neighbor_dict}
        n_samples:      number of MC samples (default 50; 30 min, 100 max)
        device:         torch device (inferred from model if None)

    Returns:
        predictions:      np.ndarray [N, num_organs]  argmax of mean probs
        confidence:       np.ndarray [N, num_organs]  max mean probability (0-1)
        uncertainty:      np.ndarray [N, num_organs]  std of winning class prob
        mean_probs:       np.ndarray [N, num_organs, num_severity]  full distribution
        damage_score_mean:np.ndarray [N, num_organs]  mean regression score
        damage_score_std: np.ndarray [N, num_organs]  std of regression score
    """
    if device is None:
        device = next(model.parameters()).device

    features = features.to(device)

    # --- Enable dropout (set to train mode so dropout is active) ---
    model.train()

    all_probs = []
    all_scores = []

    with torch.no_grad():
        for _ in range(n_samples):
            logits, organ_scores, _, _ = model(features, neighbor_dicts)
            probs = torch.softmax(logits, dim=2)   # [N, O, num_severity]
            all_probs.append(probs.cpu())
            all_scores.append(organ_scores.cpu())  # [N, O]

    # --- Restore eval mode (no dropout) ---
    model.eval()

    # Stack samples: [S, N, O, num_severity]
    all_probs = torch.stack(all_probs, dim=0)
    all_scores = torch.stack(all_scores, dim=0)   # [S, N, O]

    # --- Aggregate statistics ---
    mean_probs = all_probs.mean(dim=0).numpy()    # [N, O, num_severity]
    std_probs  = all_probs.std(dim=0).numpy()     # [N, O, num_severity]

    # Predicted class = argmax of mean probabilities
    predictions = mean_probs.argmax(axis=2)        # [N, O]

    # Confidence = probability of the predicted class under mean distribution
    N, O = predictions.shape
    confidence = mean_probs[np.arange(N)[:, None],
                            np.arange(O)[None, :],
                            predictions]            # [N, O]

    # Uncertainty = std of the predicted class probability across samples
    uncertainty = std_probs[np.arange(N)[:, None],
                            np.arange(O)[None, :],
                            predictions]            # [N, O]

    # Damage score statistics
    damage_score_mean = all_scores.mean(dim=0).numpy()   # [N, O]
    damage_score_std  = all_scores.std(dim=0).numpy()    # [N, O]

    return predictions, confidence, uncertainty, mean_probs, damage_score_mean, damage_score_std


def interpret_uncertainty(uncertainty_val):
    """
    Map a scalar uncertainty (std of winning class prob) to a clinical label.

    Thresholds calibrated for 4-class severity prediction:
      < 0.05 → Very High confidence
      0.05-0.10 → High confidence
      0.10-0.20 → Moderate confidence — flag for review
      > 0.20 → Low confidence — physician must verify

    Returns:
        (label, flag)  where flag=True means physician attention needed
    """
    if uncertainty_val < 0.05:
        return "Very High", False
    elif uncertainty_val < 0.10:
        return "High", False
    elif uncertainty_val < 0.20:
        return "Moderate", True   # flag for review
    else:
        return "Low — Verify", True  # must review


def uncertainty_report_lines(organ_name, severity_name, confidence, uncertainty,
                              damage_score_mean, damage_score_std):
    """
    Generate human-readable uncertainty report lines for one organ.

    Returns list of strings to append to a report.
    """
    conf_label, flag = interpret_uncertainty(uncertainty)
    lines = [
        f"    Predicted severity : {severity_name}",
        f"    Confidence (mean p): {confidence*100:.1f}%",
        f"    Uncertainty (std)  : {uncertainty:.4f}  [{conf_label} confidence]",
        f"    Damage score       : {damage_score_mean:.3f} ± {damage_score_std:.3f}",
    ]
    if flag:
        lines.append(f"    *** PHYSICIAN REVIEW RECOMMENDED — uncertain prediction ***")
    return lines
