"""Hand total erosion score agreement: ICC(A,1) and ICC(C,1) (secondary endpoint)."""
import glob
from pathlib import Path

import numpy as np
import pandas as pd

from .compare import RunPicks, flat_levels, holm, run_levels, two_sided_p


def hand_totals(df, joints=None):
    """Per case: reference and predicted totals of the native grades, and the patient."""
    if joints is not None:
        df = df[df['joint'].isin(joints)]
    return df.groupby('case', sort=True).agg(patient=('patient', 'first'), gt=('gt', 'sum'), pred=('pred', 'sum'))


def icc_a1(x1, x2, w=None):
    """ICC(A,1) of two raters over n subjects; w = frequency weights of shape (n,) or (B, n) -> (B,)."""
    x1, x2 = np.asarray(x1, float), np.asarray(x2, float)
    w = np.ones((1, len(x1))) if w is None else np.atleast_2d(w).astype(float)
    n = w.sum(1)
    r = (x1 + x2) / 2
    m = w @ r / n
    msr = 2 * (w @ r ** 2 - 2 * m * (w @ r) + m ** 2 * n) / (n - 1)
    e = (x1 - x2) / 2
    d = 2 * (w @ e) / n
    mse = 2 * (w @ e ** 2 - d * (w @ e) + d ** 2 * n / 4) / (n - 1)
    msc = n * d ** 2 / 2
    denom = msr + mse + 2 * (msc - mse) / n
    return np.where(denom > 1e-12, (msr - mse) / denom, 0.0)


def icc_c1(x1, x2):
    """ICC(C,1) = ICC(3,1): two-way, consistency, single measure (ignores a constant offset)."""
    x1, x2 = np.asarray(x1, float), np.asarray(x2, float)
    r, e = (x1 + x2) / 2, (x1 - x2) / 2
    n = len(x1)
    msr = 2 * ((r - r.mean()) ** 2).sum() / (n - 1)
    mse = 2 * ((e - e.mean()) ** 2).sum() / (n - 1)
    return float((msr - mse) / max(msr + mse, 1e-12))


def case_weights(patients_of_cases, n_boot, rng):
    patients, index = np.unique(patients_of_cases, return_inverse=True)
    w = rng.multinomial(len(patients), np.full(len(patients), 1 / len(patients)), size=n_boot)
    return w[:, index]


def bootstrap_cells(cells, n_boot=5000, seed=0, levels=None, pick_seed=0):
    rng = np.random.default_rng(seed)
    ref = next(iter(cells.values()))[0]
    w = case_weights(ref['patient'].to_numpy(), n_boot, rng)
    picks = RunPicks(n_boot, pick_seed)
    point, draws, consistency = {}, {}, {}
    for key, runs in cells.items():
        for t in runs:
            if not t.index.equals(ref.index) or not t['gt'].equals(ref['gt']):
                raise ValueError(f'{key}: runs do not score the same hands')
        boot = np.stack([icc_a1(t['gt'], t['pred'], w) for t in runs], 1)
        full = np.array([icc_a1(t['gt'], t['pred'])[0] for t in runs])
        design = (levels or {}).get(key, flat_levels(len(runs)))
        point[key] = float(full.mean())
        consistency[key] = float(np.mean([icc_c1(t['gt'], t['pred']) for t in runs]))
        draws[key] = np.take_along_axis(boot, picks(design), 1).mean(1)
    return point, draws, consistency


def ci(d, alpha=0.05):
    return np.nanpercentile(d, [100 * alpha / 2, 100 * (1 - alpha / 2)]).tolist()
