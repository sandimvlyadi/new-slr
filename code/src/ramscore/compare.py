"""Paired patient-level cluster bootstrap across multi-seed run groups (proposal Section 6)."""
import argparse
import glob
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from .metrics import compute_metrics

KEYS = ('bacc4', 'bacc5', 'qwk', 'qwk4', 'macro_f1', 'mae', 'recall_ge2')
LOWER_IS_BETTER = {'mae'}
DRAW_SEED_RE = re.compile(r'_d(?P<draw>\d+)_s(?P<seed>\d+)$')


def flat_levels(n_runs: int) -> list[tuple[int]]:
    return [(i,) for i in range(n_runs)]


def run_levels(run_names: list[str]) -> list[tuple]:
    parsed = [DRAW_SEED_RE.search(n) for n in run_names]
    if all(parsed):
        draws = [int(m['draw']) for m in parsed]
        if len(set(draws)) > 1:
            return [(int(m['draw']), int(m['seed'])) for m in parsed]
    return flat_levels(len(run_names))


def aligned_levels(names_by_group: dict[str, list[str]]) -> dict[str, list[tuple]]:
    levels = {g: run_levels(ns) for g, ns in names_by_group.items()}
    first = next(iter(levels.values()))
    for g, lv in levels.items():
        if lv != first:
            raise ValueError(f'Run design of {g} ({lv}) does not match {first}')
    return levels


class RunPicks:
    def __init__(self, n_boot: int, seed: int):
        self.n_boot = n_boot
        self.seed = seed
        self._cache: dict[tuple, np.ndarray] = {}

    def __call__(self, levels: list[tuple]) -> np.ndarray:
        key = tuple(levels)
        if key not in self._cache:
            rng = np.random.default_rng(self.seed)
            depth = len(key[0])
            if depth == 1:
                m = len(key)
                self._cache[key] = rng.integers(0, m, size=(self.n_boot, m))
            elif depth == 2:
                outer = sorted({lv[0] for lv in key})
                by_outer = {d: [i for i, lv in enumerate(key) if lv[0] == d] for d in outer}
                k_out = len(outer)
                if any(len(v) != len(by_outer[outer[0]]) for v in by_outer.values()):
                    raise ValueError(f'Unbalanced nesting in {key}')
                k_in = len(by_outer[outer[0]])
                table = np.array([by_outer[d] for d in outer])
                o_pick = rng.integers(0, k_out, size=(self.n_boot, k_out, 1))
                i_pick = rng.integers(0, k_in, size=(self.n_boot, k_out, k_in))
                self._cache[key] = table[o_pick, i_pick].reshape(self.n_boot, k_out * k_in)
            else:
                raise ValueError(f'Unsupported nesting depth {depth}')
        return self._cache[key]


def load_runs(prefixes: list[str], pattern: str = 'runs/{}_s*') -> tuple[dict[str, list[pd.DataFrame]], dict[str, list[str]]]:
    groups: dict[str, list[pd.DataFrame]] = {}
    names: dict[str, list[str]] = {}
    ref_index = None
    for pref in prefixes:
        paths = sorted(glob.glob(f"{pattern.format(pref)}/test_predictions.csv"))
        if not paths:
            raise FileNotFoundError(f'No test_predictions.csv found for prefix {pref!r}')
        dfs = []
        for p in paths:
            df = pd.read_csv(p).sort_values(['case', 'joint']).reset_index(drop=True)
            idx = list(zip(df['case'], df['joint'], df['gt']))
            if ref_index is None:
                ref_index = idx
            elif idx != ref_index:
                raise ValueError(f'Mismatched test joint ordering/labels in {p}')
            dfs.append(df)
        groups[pref] = dfs
        names[pref] = [Path(p).parent.name for p in paths]
    return groups, names


def bootstrap(
    runs: dict[str, list[pd.DataFrame]],
    n_boot: int = 5000,
    seed: int = 0,
    levels: dict[str, list[tuple]] | None = None,
):
    first_df = next(iter(runs.values()))[0]
    patients = first_df['patient'].to_numpy()
    uniq_p = np.unique(patients)
    p_to_idx = {p: np.where(patients == p)[0] for p in uniq_p}
    gt_full = first_df['gt'].to_numpy(dtype=np.int64)
    preds = {g: [df['pred'].to_numpy(dtype=np.int64) for df in dfs] for g, dfs in runs.items()}

    point: dict[str, dict[str, float]] = {}
    per_seed: dict[str, dict[str, list[float]]] = {}
    for g, p_list in preds.items():
        ms = [compute_metrics(gt_full, pr) for pr in p_list]
        per_seed[g] = {k: [m[k] for m in ms] for k in KEYS}
        point[g] = {k: float(np.mean(per_seed[g][k])) for k in KEYS}
        point[g]['sd'] = {k: float(np.std(per_seed[g][k], ddof=1)) if len(ms) > 1 else 0.0 for k in KEYS}

    rng = np.random.default_rng(seed)
    boot = {g: {k: np.empty((n_boot, len(p_list)), dtype=np.float64) for k in KEYS} for g, p_list in preds.items()}
    for b in range(n_boot):
        drawn = rng.choice(uniq_p, size=len(uniq_p), replace=True)
        idx = np.concatenate([p_to_idx[p] for p in drawn])
        gt_b = gt_full[idx]
        for g, p_list in preds.items():
            for s, pr in enumerate(p_list):
                m = compute_metrics(gt_b, pr[idx])
                for k in KEYS:
                    boot[g][k][b, s] = m[k]

    fixed = {g: {k: boot[g][k].mean(axis=1) for k in KEYS} for g in preds}
    picks = RunPicks(n_boot, seed + 1)
    hier: dict[str, dict[str, np.ndarray]] = {}
    for g, p_list in preds.items():
        lv = levels[g] if levels and g in levels else flat_levels(len(p_list))
        s_idx = picks(lv)
        hier[g] = {k: np.take_along_axis(boot[g][k], s_idx, axis=1).mean(axis=1) for k in KEYS}
    return point, fixed, hier, per_seed, boot


def two_sided_p(diffs: np.ndarray) -> float:
    n = len(diffs)
    p_lo = (np.sum(diffs <= 0) + 1) / (n + 1)
    p_hi = (np.sum(diffs >= 0) + 1) / (n + 1)
    return float(min(1.0, 2.0 * min(p_lo, p_hi)))


def holm(p_values: np.ndarray) -> np.ndarray:
    m = len(p_values)
    order = np.argsort(p_values)
    adj = np.empty(m, dtype=np.float64)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p_values[idx])
        adj[idx] = min(1.0, running)
    return adj


def compare(
    runs: dict[str, list[pd.DataFrame]],
    baseline: str,
    extra_pairs: list[tuple[str, str]] | None = None,
    n_boot: int = 5000,
    seed: int = 0,
    alpha: float = 0.05,
    levels: dict[str, list[tuple]] | None = None,
) -> dict:
    point, fixed, hier, per_seed, _ = bootstrap(runs, n_boot=n_boot, seed=seed, levels=levels)
    lo, hi = 100 * (alpha / 2), 100 * (1 - alpha / 2)

    groups_out = {}
    for g in runs:
        groups_out[g] = {
            'n_seeds': len(runs[g]),
            'mean': {k: point[g][k] for k in KEYS},
            'sd': point[g]['sd'],
            'per_seed': per_seed[g],
            'ci95_fixed': {k: [float(np.nanpercentile(fixed[g][k], lo)), float(np.nanpercentile(fixed[g][k], hi))] for k in KEYS},
            'ci95_hier': {k: [float(np.nanpercentile(hier[g][k], lo)), float(np.nanpercentile(hier[g][k], hi))] for k in KEYS},
        }

    pairs = [(g, baseline) for g in runs if g != baseline]
    if extra_pairs:
        for pair in extra_pairs:
            if pair not in pairs:
                pairs.append(pair)

    pair_out = {}
    for a, b in pairs:
        key = f'{a}-{b}'
        pair_out[key] = {'a': a, 'b': b, 'vs_baseline': (b == baseline)}
        for k in KEYS:
            d_fixed = fixed[a][k] - fixed[b][k]
            d_hier = hier[a][k] - hier[b][k]
            valid_f = d_fixed[~np.isnan(d_fixed)]
            valid_h = d_hier[~np.isnan(d_hier)]
            prob_better = float(np.mean(valid_h < 0)) if k in LOWER_IS_BETTER else float(np.mean(valid_h > 0))
            pair_out[key][k] = {
                'diff': float(point[a][k] - point[b][k]),
                'ci_fixed': [float(np.percentile(valid_f, lo)), float(np.percentile(valid_f, hi))],
                'ci_hier': [float(np.percentile(valid_h, lo)), float(np.percentile(valid_h, hi))],
                'p_fixed': two_sided_p(valid_f),
                'p_hier': two_sided_p(valid_h),
                'prob_better': prob_better,
            }

    base_keys = [key for key, v in pair_out.items() if v['vs_baseline']]
    if base_keys:
        for k in KEYS:
            for p_col, out_col in (('p_fixed', 'p_holm_fixed'), ('p_hier', 'p_holm_hier')):
                raw = np.array([pair_out[key][k][p_col] for key in base_keys])
                for key, padj in zip(base_keys, holm(raw)):
                    pair_out[key][k][out_col] = float(padj)

    return {'n_boot': n_boot, 'baseline': baseline, 'groups': groups_out, 'pairs': pair_out}
