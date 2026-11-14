"""RQ0: shared reference (RQ0-01, ResNet34 + ImageNet + balanced softmax) and architecture check (RQ0-02, ConvNeXt-Tiny)."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import NATIVE_SCORES
from .e6abl import add_holm, contrast, load, paired, prediction_stats

ARMS = {'rq0-01': 'resnet34', 'rq0-02': 'convnext_tiny'}
LABELS = {'rq0-01': 'ResNet34 (rujukan)', 'rq0-02': 'ConvNeXt-Tiny'}
FRACTIONS = ('1.0', '0.1')
MIN_DIFF = 0.05
ALPHA = 0.05


def arm_glob(root, arm, frac):
    return str(Path(root) / 'rq0' / f'{arm}_{ARMS[arm]}_b2_f{frac}_d*_s*')


def verdict(stats):
    if stats['p_holm'] >= ALPHA:
        return 'tidak terbukti'
    return 'bermakna' if abs(stats['diff']) >= MIN_DIFF else 'kecil'


def run_summary(pattern):
    paths = sorted(Path(p).parent / 'metrics.json' for p in load(pattern).values())
    ms = [json.loads(p.read_text()) for p in paths if p.exists()]
    if not ms:
        return None
    return {'runs': len(ms), 'val_bacc4': float(np.mean([m['val']['bacc4'] for m in ms])),
            'val_qwk': float(np.mean([m['val']['qwk'] for m in ms])),
            'best_epoch': float(np.mean([m['best_epoch'] for m in ms]))}


def analyse(root, n_boot=5000):
    root = Path(root)
    primary, predictions, summaries = [], [], []
    for frac in FRACTIONS:
        pair = paired(arm_glob(root, 'rq0-02', frac), arm_glob(root, 'rq0-01', frac))
        if pair:
            primary.append({'frac': frac, **contrast(*pair, n_boot, icc=True)})
        for arm in ARMS:
            frames = [pd.read_csv(p) for p in load(arm_glob(root, arm, frac)).values()]
            if frames:
                predictions.append({'frac': frac, 'arm': arm, **prediction_stats(frames)})
            s = run_summary(arm_glob(root, arm, frac))
            if s:
                summaries.append({'frac': frac, 'arm': arm, **s})
    add_holm(primary)
    for r in primary:
        r['verdict'] = verdict(r['bacc4'])
    return {'primary': primary, 'predictions': predictions, 'summaries': summaries}


def report(results):
    lines = ['# RQ0: Rujukan Bersama dan Cek Arsitektur (ResNet34 vs ConvNeXt-Tiny)', '',
             '## Kontras Primer: ConvNeXt-Tiny − ResNet34 (Rujukan)', '',
             '| Label | Run | BACC4 ResNet34 | BACC4 ConvNeXt-Tiny | Selisih [95% CI] | Holm p | Putusan | Selisih QWK | p | Selisih ICC | p |',
             '|---|---|---|---|---|---|---|---|---|---|---|']
    for r in results['primary']:
        a, q, i = r['bacc4'], r['qwk'], r['icc']
        lines.append(f"| {float(r['frac']):.0%} | {r['runs']} | {a['base']:.3f} | {a['variant']:.3f} | "
                     f"{a['diff']:+.3f} [{a['ci'][0]:+.3f}, {a['ci'][1]:+.3f}] | {a['p_holm']:.3f} | {r['verdict']} | "
                     f"{q['diff']:+.3f} | {q['p']:.3f} | {i['diff']:+.3f} | {i['p']:.3f} |")
    lines += ['', '## Validasi dan Epoch Terpilih (Rata-Rata Run)', '',
              '| Label | Arm | Run | Val BACC4 | Val QWK | Epoch terpilih |', '|---|---|---|---|---|---|']
    for r in results['summaries']:
        lines.append(f"| {float(r['frac']):.0%} | {LABELS[r['arm']]} | {r['runs']} | {r['val_bacc4']:.3f} | "
                     f"{r['val_qwk']:.3f} | {r['best_epoch']:.1f} |")
    grades = [str(g) for g in NATIVE_SCORES]
    lines += ['', '## Recall per Grade (Test)', '', '| Label | Arm | Run | Proporsi prediksi grade 0 | ' +
              ' | '.join(f'Recall {g}' for g in grades) + ' |', '|---|---|---|---|' + '---|' * len(grades)]
    for r in results['predictions']:
        lines.append(f"| {float(r['frac']):.0%} | {LABELS[r['arm']]} | {r['runs']} | {r['pred_share_grade0']:.3f} | " +
                     ' | '.join(f"{r['recall'][g]:.3f}" for g in grades) + ' |')
    return '\n'.join(lines) + '\n'


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs', default='runs')
    p.add_argument('--out', default='runs/rq0_analysis')
    p.add_argument('--n-boot', type=int, default=5000)
    args = p.parse_args(argv)
    dest = Path(args.out)
    dest.mkdir(parents=True, exist_ok=True)
    results = analyse(args.runs, args.n_boot)
    text = report(results)
    (dest / 'report.md').write_text(text)
    (dest / 'rq0.json').write_text(json.dumps(results, indent=2))
    print(text)


if __name__ == '__main__':
    main()
