"""RQ1: pre-training and SSL against the ImageNet ResNet34 reference (RQ0-01), at 100% and 10% labels."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import NATIVE_SCORES
from .e6abl import add_holm, contrast, load, paired, prediction_stats

FRACTIONS = ('1.0', '0.1')
MIN_DIFF = 0.05
ALPHA = 0.05
BEST = 'best_existing'
BEST_NO_SKELEX = 'best_existing_no_skelex'
BEST_SSL = 'best_ssl'
BEST_QWK = 'best_existing_qwk'
BEST_NO_SKELEX_QWK = 'best_existing_no_skelex_qwk'
SELECTIONS = (BEST, BEST_NO_SKELEX, BEST_SSL, BEST_QWK, BEST_NO_SKELEX_QWK)
SENSITIVITY = {BEST: BEST_QWK, BEST_NO_SKELEX: BEST_NO_SKELEX_QWK}
QWK_FLOOR_ARM = 'rq0-01'

LABELS = {
    'rq0-01': 'ImageNet ResNet34 (rujukan)',
    'rq1-01': 'Scratch ResNet34',
    'rq1-02': 'DINO ResNet34',
    'rq1-03': 'DINO + ChronoCon ResNet34',
    'rq1-04': 'ImageNet ViT-S (LLRD)',
    'rq1-05': 'MAE ViT-S (LLRD)',
    'rq1-06': 'DINO ViT-S (LLRD)',
    'rq1-07': 'ImageNet → bone age RSNA ResNet34',
    'rq1-08': 'SimCLR ResNet34',
    'rq1-10': 'BYOL ResNet34',
    'rq1-12': 'SupCon + CE ResNet34',
    'rq1-13': 'SupCon berbobot jarak grade ResNet34 (adaptasi M219)',
    'rq1-14': 'SSL normalitas grade 0 ResNet34',
    'rq1-15': 'Gate normalitas 0 vs ≥1 ResNet34 (modifikasi)',
    'rq1-17a': 'Linear probe ImageNet ResNet34',
    'rq1-17b': 'LP-FT ImageNet ResNet34',
    'rq1-17c': 'Linear probe SSL terbaik',
    'rq1-17d': 'LP-FT SSL terbaik',
    'rq1-18a': 'SKELEX ViT-L linear probe',
    'rq1-18b': 'SKELEX ViT-L fine-tune (LLRD)',
}
SELECTION_LABELS = {
    BEST: 'metode terbaik yang sudah ada',
    BEST_NO_SKELEX: 'metode terbaik selain SKELEX',
    BEST_SSL: 'SSL terbaik',
    BEST_QWK: 'metode terbaik dengan QWK val ≥ rujukan',
    BEST_NO_SKELEX_QWK: 'metode terbaik selain SKELEX dengan QWK val ≥ rujukan',
}
BEST_CANDIDATES = ('rq0-01', 'rq1-01', 'rq1-02', 'rq1-03', 'rq1-04', 'rq1-05', 'rq1-06', 'rq1-07', 'rq1-08', 'rq1-10',
                   'rq1-12', 'rq1-14', 'rq1-18b')
SSL_CANDIDATES = ('rq1-02', 'rq1-03', 'rq1-05', 'rq1-06', 'rq1-08', 'rq1-10')
FAMILIES = {
    'F1': [(a, 'rq0-01') for a in ('rq1-01', 'rq1-02', 'rq1-07', 'rq1-08', 'rq1-10', 'rq1-12', 'rq1-14', 'rq1-18b')],
    'F2': [('rq1-03', 'rq1-02'), ('rq1-04', 'rq0-01'), ('rq1-05', 'rq1-04'), ('rq1-06', 'rq1-04'),
           ('rq1-08', 'rq1-02'), ('rq1-10', 'rq1-02'), ('rq1-18b', BEST_NO_SKELEX)],
    'F3': [('rq1-13', 'rq1-12'), ('rq1-13', BEST), ('rq1-15', 'rq1-14'), ('rq1-15', BEST)],
    'F4': [('rq1-17a', 'rq0-01'), ('rq1-17b', 'rq0-01'), ('rq1-17c', BEST_SSL), ('rq1-17d', BEST_SSL),
           ('rq1-18a', 'rq1-18b')],
}
FAMILY_TITLES = {
    'F1': 'F1: pre-training / SSL vs rujukan ImageNet ResNet34',
    'F2': 'F2: antar-metode dan arsitektur',
    'F3': 'F3: modifikasi (vs versi asli dan vs metode terbaik yang sudah ada)',
    'F4': 'F4: strategi fine-tuning',
}


def arm_glob(root, arm, frac):
    sub = 'rq0' if arm.startswith('rq0') else 'rq1'
    return str(Path(root) / sub / f'{arm}_*_b2_f{frac}_d*_s*')


def verdict(stats):
    if stats['p_holm'] >= ALPHA:
        return 'tidak terbukti'
    return 'bermakna' if abs(stats['diff']) >= MIN_DIFF else 'kecil'


def run_metrics(pattern):
    paths = sorted(Path(p).parent / 'metrics.json' for p in load(pattern).values())
    return [json.loads(p.read_text()) for p in paths if p.exists()]


def run_summary(pattern):
    ms = run_metrics(pattern)
    if not ms:
        return None
    val = np.array([m['val']['bacc4'] for m in ms])
    test = np.array([m['test']['bacc4'] for m in ms])
    return {'runs': len(ms), 'val_bacc4': float(val.mean()), 'val_qwk': float(np.mean([m['val']['qwk'] for m in ms])),
            'test_bacc4': float(test.mean()), 'test_minus_val': float((test - val).mean()),
            'best_epoch': float(np.mean([m['best_epoch'] for m in ms]))}


def select(root, candidates, fracs, expected=3, qwk_floor_arm=None):
    scores, qwks, missing = {}, {}, []
    for arm in candidates:
        ms = [m for f in fracs for m in run_metrics(arm_glob(root, arm, f))]
        if len(ms) < expected * len(fracs):
            missing.append(arm)
        if ms:
            scores[arm] = float(np.mean([m['val']['bacc4'] for m in ms]))
            qwks[arm] = float(np.mean([m['val']['qwk'] for m in ms]))
    floor = qwks.get(qwk_floor_arm) if qwk_floor_arm else None
    eligible = [a for a in scores if qwk_floor_arm is None or (floor is not None and round(qwks[a], 3) >= round(floor, 3))]
    best = max(eligible, key=lambda a: (round(scores[a], 3), -candidates.index(a))) if eligible else None
    return {'arm': best, 'scores': scores, 'qwk': qwks, 'qwk_floor': floor,
            'excluded': [a for a in scores if a not in eligible], 'missing': missing,
            'provisional': bool(missing) or best is None}


def selections(root):
    out = {}
    no_skelex = tuple(a for a in BEST_CANDIDATES if a != 'rq1-18b')
    for frac in FRACTIONS:
        out[f'{BEST}_{frac}'] = select(root, BEST_CANDIDATES, (frac,))
        out[f'{BEST_NO_SKELEX}_{frac}'] = select(root, no_skelex, (frac,))
        out[f'{BEST_QWK}_{frac}'] = select(root, BEST_CANDIDATES, (frac,), qwk_floor_arm=QWK_FLOOR_ARM)
        out[f'{BEST_NO_SKELEX_QWK}_{frac}'] = select(root, no_skelex, (frac,), qwk_floor_arm=QWK_FLOOR_ARM)
    ssl = select(root, SSL_CANDIDATES, FRACTIONS)
    for frac in FRACTIONS:
        out[f'{BEST_SSL}_{frac}'] = ssl
    return out


def resolve(base, frac, sel):
    if base not in SELECTIONS:
        return base, None
    s = sel[f'{base}_{frac}']
    if s['provisional']:
        return None, 'pilihan validasi belum final; kandidat belum selesai: ' + ', '.join(s['missing'])
    return s['arm'], None


def family(root, pairs, sel, n_boot):
    rows = []
    for variant, base in pairs:
        for frac in FRACTIONS:
            arm, why = resolve(base, frac, sel)
            row = {'arm': variant, 'base_key': base, 'base': arm, 'frac': frac, 'note': ''}
            pair = paired(arm_glob(root, variant, frac), arm_glob(root, arm, frac)) if arm else None
            if arm is not None and arm == variant:
                row.update(status='tidak dijalankan', reason='arm terpilih sama dengan arm yang diuji', bacc4={'p': 1.0})
            elif pair is None:
                row.update(status='tidak dijalankan', reason=why or 'arm belum dijalankan', bacc4={'p': 1.0})
            else:
                row.update(status='dihitung', **contrast(*pair, n_boot, icc=True))
            rows.append(row)
    add_holm(rows)
    for r in rows:
        r['verdict'] = verdict(r['bacc4']) if r['status'] == 'dihitung' else r['status']
    return rows


def analyse(root, n_boot=5000):
    root = Path(root)
    sel = selections(root)
    families = {k: family(root, pairs, sel, n_boot) for k, pairs in FAMILIES.items()}
    predictions, summaries = [], []
    for frac in FRACTIONS:
        for arm in LABELS:
            frames = [pd.read_csv(p) for p in load(arm_glob(root, arm, frac)).values()]
            if frames:
                predictions.append({'frac': frac, 'arm': arm, **prediction_stats(frames)})
            s = run_summary(arm_glob(root, arm, frac))
            if s:
                summaries.append({'frac': frac, 'arm': arm, **s})
    return {'selections': sel, 'families': families, 'predictions': predictions, 'summaries': summaries}
