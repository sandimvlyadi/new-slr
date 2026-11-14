"""RQ4: Calibration, Ordinal Conformal Prediction (APS, RAPS, Mondrian, JointxGrade Mondrian), & Selective Referral."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import torch
import torch.nn.functional as F

from .data import BE_JOINTS, NATIVE_SCORES
from .metrics import LEVEL4, compute_metrics, to_level4
from .models import logits_to_probs


# --------------------------------------------------------- Calibration Metrics (ECE, MCE, Brier, NLL)

def calibration_metrics(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> dict:
    """Compute NLL, multi-class Brier score, top-label ECE, and classwise ECE."""
    probs = np.clip(np.asarray(probs, dtype=np.float64), 1e-12, 1.0)
    probs = probs / probs.sum(axis=1, keepdims=True)
    labels = np.asarray(labels, dtype=np.int64)
    n, k = probs.shape

    nll = float(-np.mean(np.log(probs[np.arange(n), labels])))
    onehot = np.eye(k)[labels]
    brier = float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))

    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    acc = (pred == labels).astype(np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece, mce = 0.0, 0.0
    for b0, b1 in zip(bins[:-1], bins[1:]):
        mask = (conf > b0) & (conf <= b1) if b0 > 0 else (conf >= b0) & (conf <= b1)
        if mask.any():
            gap = abs(float(acc[mask].mean() - conf[mask].mean()))
            ece += (mask.sum() / n) * gap
            mce = max(mce, gap)

    cw_ece = 0.0
    for c in range(k):
        pc = probs[:, c]
        yc = onehot[:, c]
        c_ece = 0.0
        for b0, b1 in zip(bins[:-1], bins[1:]):
            mask = (pc > b0) & (pc <= b1) if b0 > 0 else (pc >= b0) & (pc <= b1)
            if mask.any():
                c_ece += (mask.sum() / n) * abs(float(yc[mask].mean() - pc[mask].mean()))
        cw_ece += c_ece / k

    return {'nll': nll, 'brier': brier, 'ece': float(ece), 'mce': float(mce), 'cw_ece': float(cw_ece)}


# --------------------------------------------------------- RQ4-P1 & RQ4-P2: Temperature & Vector/Dirichlet Scaling

def fit_temperature(val_logits: np.ndarray, val_labels: np.ndarray) -> float:
    """Optimize scalar temperature T > 0 minimizing NLL on validation logits (Guo et al., 2017; RQ4-P1)."""
    logits = np.asarray(val_logits, dtype=np.float64)
    labels = np.asarray(val_labels, dtype=np.int64)

    def nll_fn(log_t):
        t = float(np.exp(log_t[0]))
        scaled = logits / t
        scaled -= scaled.max(axis=1, keepdims=True)
        log_p = scaled - np.log(np.exp(scaled).sum(axis=1, keepdims=True))
        return -float(np.mean(log_p[np.arange(len(labels)), labels]))

    res = minimize(nll_fn, x0=[0.0], bounds=[(-2.5, 2.5)], method='L-BFGS-B')
    return float(np.exp(res.x[0]))


def apply_temperature(logits: np.ndarray, temperature: float, prior_shift: np.ndarray | None = None) -> np.ndarray:
    scaled = np.asarray(logits, dtype=np.float64) / temperature
    if prior_shift is not None:
        scaled = scaled + prior_shift[None, :]
    scaled -= scaled.max(axis=1, keepdims=True)
    exp_s = np.exp(scaled)
    return exp_s / exp_s.sum(axis=1, keepdims=True)


def fit_dirichlet_scaling(val_probs: np.ndarray, val_labels: np.ndarray, l2_offdiag: float = 1e-2, l2_bias: float = 1e-3) -> tuple[np.ndarray, np.ndarray]:
    """Fit Dirichlet calibration mapping log(p) -> softmax(W log(p) + b) with off-diagonal L2 penalty (Kull et al., 2019; M095; RQ4-P2)."""
    log_p = np.log(np.clip(np.asarray(val_probs, dtype=np.float64), 1e-12, 1.0))
    labels = np.asarray(val_labels, dtype=np.int64)
    n, k = log_p.shape
    off_mask = 1.0 - np.eye(k)

    x0 = np.concatenate([np.eye(k).ravel(), np.zeros(k)])

    def obj(params):
        W = params[:k * k].reshape(k, k)
        b = params[k * k:]
        z = log_p @ W.T + b[None, :]
        z -= z.max(axis=1, keepdims=True)
        lp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
        nll = -np.mean(lp[np.arange(n), labels])
        reg = l2_offdiag * np.sum((W * off_mask) ** 2) + l2_bias * np.sum(b ** 2)
        return float(nll + reg)

    res = minimize(obj, x0=x0, method='L-BFGS-B', options={'maxiter': 300})
    W = res.x[:k * k].reshape(k, k)
    b = res.x[k * k:]
    return W, b


def apply_dirichlet_scaling(probs: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    log_p = np.log(np.clip(np.asarray(probs, dtype=np.float64), 1e-12, 1.0))
    z = log_p @ W.T + b[None, :]
    z -= z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


# --------------------------------------------------------- RQ4-P3 & RQ4-P4: Ordinal Conformal Prediction

def nonconformity_scores(probs: np.ndarray, method: str = 'aps', lam_raps: float = 0.02, k_reg: int = 1) -> np.ndarray:
    """Compute non-conformity score matrix S[i, c] for sample i and candidate class c (APS or RAPS)."""
    n, k = probs.shape
    scores = np.zeros((n, k), dtype=np.float64)
    for i in range(n):
        p = probs[i]
        order = np.argsort(-p)
        cum = 0.0
        for rank, c in enumerate(order):
            reg = lam_raps * max(0, rank + 1 - k_reg) if method == 'raps' else 0.0
            scores[i, c] = cum + p[c] + reg
            cum += p[c]
    return scores


def _conformal_quantile(s: np.ndarray, alpha: float = 0.10) -> float:
    n = len(s)
    if n == 0:
        return 1.0
    q_level = min(1.0, np.ceil((n + 1) * (1.0 - alpha)) / n)
    return float(np.quantile(s, q_level, method='higher'))


def conformal_predict_sets(
    val_probs: np.ndarray,
    val_labels: np.ndarray,
    test_probs: np.ndarray,
    test_labels: np.ndarray,
    val_joints: np.ndarray | None = None,
    test_joints: np.ndarray | None = None,
    mode: str = 'marginal_aps',
    alpha: float = 0.10,
    min_group_n: int = 10,
) -> dict:
    """Construct contiguous ordinal prediction intervals [c_min, c_max] under 4 conformal schemes:
      - 'marginal_aps'        (RQ4-P3: standard split conformal APS)
      - 'marginal_raps'       (RQ4-P3: regularized APS, M110)
      - 'mondrian_grade'      (RQ4-P3: class-conditional Mondrian per grade)
      - 'mondrian_joint_grade' (RQ4-P4: novel joint-and-grade Mondrian with shrinkage for rare (j,c) cells)
    """
    base_method = 'raps' if 'raps' in mode else 'aps'
    s_val = nonconformity_scores(val_probs, method=base_method)
    s_test = nonconformity_scores(test_probs, method=base_method)
    val_true_s = s_val[np.arange(len(val_labels)), val_labels]
    n_test, k = test_probs.shape

    q_global = _conformal_quantile(val_true_s, alpha)
    q_by_grade = np.array([
        _conformal_quantile(val_true_s[val_labels == c], alpha) if (val_labels == c).any() else q_global
        for c in range(k)
    ])

    if mode in ('marginal_aps', 'marginal_raps'):
        thresh = np.full((n_test, k), q_global)
    elif mode == 'mondrian_grade':
        thresh = np.broadcast_to(q_by_grade[None, :], (n_test, k))
    elif mode == 'mondrian_joint_grade':
        if val_joints is None or test_joints is None:
            raise ValueError('mondrian_joint_grade requires val_joints and test_joints')
        thresh = np.empty((n_test, k), dtype=np.float64)
        q_jg = {}
        for j in np.unique(np.concatenate([val_joints, test_joints])):
            for c in range(k):
                mask = (val_joints == j) & (val_labels == c)
                n_jc = int(mask.sum())
                if n_jc == 0:
                    q_jg[(j, c)] = q_by_grade[c]
                elif n_jc < min_group_n:
                    w = n_jc / float(min_group_n)
                    q_jg[(j, c)] = w * _conformal_quantile(val_true_s[mask], alpha) + (1.0 - w) * q_by_grade[c]
                else:
                    q_jg[(j, c)] = _conformal_quantile(val_true_s[mask], alpha)
        for i, j in enumerate(test_joints):
            for c in range(k):
                thresh[i, c] = q_jg[(j, c)]
    else:
        raise ValueError(f'Unknown conformal mode: {mode}')

    raw_in = s_test <= thresh
    top1 = test_probs.argmax(axis=1)
    raw_in[np.arange(n_test), top1] = True  # never allow empty set

    # Enforce ordinal contiguity: convex hull [c_min, c_max]
    intervals = np.zeros((n_test, k), dtype=bool)
    sizes = np.zeros(n_test, dtype=np.int64)
    for i in range(n_test):
        idx = np.where(raw_in[i])[0]
        c_min, c_max = int(idx.min()), int(idx.max())
        intervals[i, c_min:c_max + 1] = True
        sizes[i] = c_max - c_min + 1

    covered = intervals[np.arange(n_test), test_labels]
    cov_by_grade = {
        int(NATIVE_SCORES[c]): (float(covered[test_labels == c].mean()) if (test_labels == c).any() else float('nan'))
        for c in range(k)
    }
    cov_ge2 = float(covered[test_labels >= 2].mean()) if (test_labels >= 2).any() else float('nan')

    return {
        'mode': mode,
        'alpha': alpha,
        'marginal_coverage': float(covered.mean()),
        'coverage_ge2': cov_ge2,
        'coverage_per_grade': cov_by_grade,
        'mean_set_size': float(sizes.mean()),
        'singleton_share': float((sizes == 1).mean()),
        'sizes': sizes,
        'covered': covered,
    }


# --------------------------------------------------------- RQ4-P5 & RQ4-P6: Selective Referral (Risk-Coverage)

def risk_coverage_metrics(gt_scores: np.ndarray, pred_scores: np.ndarray, uncertainty: np.ndarray) -> dict:
    """Compute AURC (Area Under Risk-Coverage curve on 1 - BACC4) and BACC4 at 80% & 90% coverage."""
    order = np.argsort(uncertainty)  # lowest uncertainty retained first
    gt = np.asarray(gt_scores)[order]
    pr = np.asarray(pred_scores)[order]
    n = len(gt)

    coverages = np.linspace(0.50, 1.00, 21)
    risks, baccs = [], []
    for cov in coverages:
        k = max(10, int(round(cov * n)))
        bacc = compute_metrics(gt[:k], pr[:k])['bacc4']
        baccs.append(bacc)
        risks.append(1.0 - bacc)

    aurc = float(np.trapezoid(risks, coverages))
    k80 = int(round(0.80 * n))
    k90 = int(round(0.90 * n))
    return {
        'aurc_bacc4': aurc,
        'bacc4_at_80pct': compute_metrics(gt[:k80], pr[:k80])['bacc4'],
        'bacc4_at_90pct': compute_metrics(gt[:k90], pr[:k90])['bacc4'],
        'bacc4_at_100pct': compute_metrics(gt, pr)['bacc4'],
        'qwk_at_80pct': compute_metrics(gt[:k80], pr[:k80])['qwk'],
    }
