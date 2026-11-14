"""Shared helpers for paired contrast tables, Holm correction, and prediction statistics."""
import glob
from pathlib import Path
import re

import numpy as np
import pandas as pd

from .compare import bootstrap, holm, run_levels, two_sided_p
from .data import NATIVE_SCORES
from .icc import bootstrap_cells, ci, hand_totals

TAG = re.compile(r'_(d\d_s\d+)$')
KEYS = ('bacc4', 'qwk')


def load(pattern):
    return {TAG.search(Path(p).parent.name).group(1): p for p in sorted(glob.glob(f'{pattern}/test_predictions.csv'))}


def collapse(df):
    return df.assign(gt=df['gt'].clip(upper=2), pred=df['pred'].clip(upper=2))


def contrast(variant, base, levels, n_boot=5000, seed=0, icc=False, alpha=0.05):
    runs = {'variant': variant, 'base': base}
    point, _, hier, _, _ = bootstrap(runs, n_boot, seed, levels)
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    row = {'runs': len(variant)}
    for k in KEYS:
        d = hier['variant'][k] - hier['base'][k]
        row[k] = {'variant': point['variant'][k], 'base': point['base'][k], 'diff': point['variant'][k] - point['base'][k],
                  'ci': np.nanpercentile(d, [lo, hi]).tolist(), 'p': two_sided_p(d[~np.isnan(d)])}
    if icc:
        totals = {s: [hand_totals(df) for df in dfs] for s, dfs in runs.items()}
        ipoint, idraws, _ = bootstrap_cells(totals, n_boot, seed, levels={s: levels for s in runs})
        d = idraws['variant'] - idraws['base']
        row['icc'] = {'variant': ipoint['variant'], 'base': ipoint['base'], 'diff': ipoint['variant'] - ipoint['base'],
                      'ci': ci(d), 'p': two_sided_p(d[~np.isnan(d)])}
    return row


def paired(variant_glob, base_glob, transform=None):
    v, b = load(variant_glob), load(base_glob)
    tags = sorted(set(v) & set(b))
    if not tags:
        return None
    f = transform or (lambda df: df)
    return [pd.read_csv(v[t]) for t in tags], [f(pd.read_csv(b[t])) for t in tags], run_levels(tags)


def add_holm(rows):
    if rows:
        for r, p in zip(rows, holm(np.array([r['bacc4']['p'] for r in rows]))):
            r['bacc4']['p_holm'] = float(p)
    return rows


def prediction_stats(frames):
    recall = {str(g): [] for g in NATIVE_SCORES}
    pred, true = [], []
    for df in frames:
        pred.append(float((df['pred'] == 0).mean()))
        true.append(float((df['gt'] == 0).mean()))
        for g in NATIVE_SCORES:
            hit = df.loc[df['gt'] == g, 'pred'] == g
            recall[str(g)].append(float(hit.mean()) if len(hit) else float('nan'))
    return {'runs': len(frames), 'pred_share_grade0': float(np.mean(pred)), 'true_share_grade0': float(np.mean(true)),
            'recall': {g: float(np.nanmean(v)) for g, v in recall.items()}}
