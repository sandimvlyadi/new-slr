"""Evaluation metrics for ordinal imbalanced erosion grading (proposal Section 6).

Primary:
  - bacc4:     Balanced Accuracy on 4 levels {0, 1, 2, >=3} (merges rare scores 3 & 5)
  - bacc5:     Balanced Accuracy on the 5 native scores (0, 1, 2, 3, 5)
  - qwk:       Quadratic Weighted Cohen's Kappa on native scores

Secondary:
  - macro_f1:  unweighted macro F1 on native scores
  - mae:       mean absolute error on true SvdH scale (0, 1, 2, 3, 5)
  - recall_per_grade: dict {grade: recall}
  - recall_ge2: sensitivity for clinically significant erosion (score >= 2)
"""
import numpy as np
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, confusion_matrix, f1_score

from .data import NATIVE_SCORES

LEVEL4 = {0: 0, 1: 1, 2: 2, 3: 3, 4: 3, 5: 3}


def to_level4(scores: np.ndarray) -> np.ndarray:
    return np.vectorize(LEVEL4.get)(scores)


def compute_metrics(
    y_true_scores: np.ndarray,
    y_pred_scores: np.ndarray,
    classes: tuple[int, ...] = NATIVE_SCORES,
) -> dict:
    """Compute all primary & secondary metrics given native SvdH scores (e.g. in {0,1,2,3,5})."""
    y_true = np.asarray(y_true_scores, dtype=np.int64)
    y_pred = np.asarray(y_pred_scores, dtype=np.int64)

    score_to_pos = {s: i for i, s in enumerate(classes)}
    t_pos = np.vectorize(score_to_pos.get)(y_true)
    p_pos = np.vectorize(score_to_pos.get)(y_pred)

    t4 = to_level4(y_true)
    p4 = to_level4(y_pred)

    bacc5 = float(balanced_accuracy_score(t_pos, p_pos))
    bacc4 = float(balanced_accuracy_score(t4, p4))
    qwk = float(cohen_kappa_score(t_pos, p_pos, weights='quadratic', labels=list(range(len(classes)))))
    qwk4 = float(cohen_kappa_score(t4, p4, weights='quadratic', labels=[0, 1, 2, 3]))
    macro_f1 = float(f1_score(t_pos, p_pos, average='macro', labels=list(range(len(classes))), zero_division=0))
    mae = float(np.mean(np.abs(y_true - y_pred)))

    cm = confusion_matrix(t_pos, p_pos, labels=list(range(len(classes))))
    row_sums = cm.sum(axis=1)
    recall_per_grade = {
        int(s): (float(cm[i, i] / row_sums[i]) if row_sums[i] > 0 else float('nan'))
        for i, s in enumerate(classes)
    }

    ge2_mask = y_true >= 2
    recall_ge2 = float(np.mean(y_pred[ge2_mask] >= 2)) if ge2_mask.any() else float('nan')

    return {
        'n_joints': int(len(y_true)),
        'bacc4': bacc4,
        'bacc5': bacc5,
        'qwk': qwk,
        'qwk4': qwk4,
        'macro_f1': macro_f1,
        'mae': mae,
        'recall_ge2': recall_ge2,
        'recall_per_grade': recall_per_grade,
        'confusion_matrix': cm.tolist(),
    }


def patient_bootstrap_ci(
    patients: np.ndarray,
    y_true_scores: np.ndarray,
    y_pred_scores: np.ndarray,
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, tuple[float, float]]:
    """Cluster bootstrap at the patient level (proposal Section 6)."""
    patients = np.asarray(patients)
    y_true = np.asarray(y_true_scores, dtype=np.int64)
    y_pred = np.asarray(y_pred_scores, dtype=np.int64)
    uniq = np.unique(patients)
    p_to_idx = {p: np.where(patients == p)[0] for p in uniq}

    rng = np.random.default_rng(seed)
    keys = ('bacc4', 'bacc5', 'qwk', 'macro_f1', 'mae', 'recall_ge2')
    samples = {k: [] for k in keys}
    for _ in range(n_boot):
        drawn = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([p_to_idx[p] for p in drawn])
        m = compute_metrics(y_true[idx], y_pred[idx])
        for k in keys:
            samples[k].append(m[k])

    lo, hi = 100 * (alpha / 2), 100 * (1 - alpha / 2)
    return {k: (float(np.nanpercentile(v, lo)), float(np.nanpercentile(v, hi))) for k, v in samples.items()}
