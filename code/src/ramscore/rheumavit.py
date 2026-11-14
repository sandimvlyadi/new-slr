"""RheumaVIT / AIRI hand radiographs (Stolpovsky et al., 2023) as joint ROIs in the RAM-H1200 format (RQ3, RQ4, RQ5-E)."""
import argparse
import json
import os
from pathlib import Path
import statistics

import cv2
import numpy as np
import pandas as pd

from .data import NATIVE_SCORES

DEFAULT_ROOT = os.environ.get(
    'RHEUMAVIT_ROOT',
    '/home/sandimulyadi/Projects/Dataset/AIRINIIReumHands',
)

REF_PHALANX = 246.0
ROI_SIZE = {'R': 202, 'U': 113, 'CMC-T': 117}
FINGER_ROI = 170
JOINTS = {
    ('MCP', 1): 'MCP-T', ('MCP', 2): 'MCP-I', ('MCP', 3): 'MCP-M', ('MCP', 4): 'MCP-R', ('MCP', 5): 'MCP-S',
    ('PIP', 2): 'PIP-I', ('PIP', 3): 'PIP-M', ('PIP', 4): 'PIP-R', ('PIP', 5): 'PIP-S',
    ('DIP', 1): 'IP', ('CMC', 1): 'CMC-T', ('ulna', 0): 'U', ('RC', 0): 'R',
}
SHARED_JOINTS = tuple(JOINTS.values())
READERS = (0, 1, 2)


def map_score(s: int) -> int:
    s = int(s)
    if s < 0 or s > 5:
        raise ValueError(f'Erosion score out of 0..5: {s}')
    return 3 if s == 4 else s


def joint_key(df: pd.DataFrame):
    return list(zip(df['joint'], df['finger'].fillna(0).astype(int)))


def hand_image(img: np.ndarray, hand: str) -> np.ndarray:
    half = img.shape[1] // 2
    return img[:, half:] if hand == 'left' else np.ascontiguousarray(img[:, :half][:, ::-1])


def crop_square(img: np.ndarray, cx: float, cy: float, side: int) -> np.ndarray:
    x0, y0 = int(round(cx - side / 2)), int(round(cy - side / 2))
    pad = max(0, -x0, -y0, x0 + side - img.shape[1], y0 + side - img.shape[0])
    if pad:
        img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
        x0, y0 = x0 + pad, y0 + pad
    return img[y0:y0 + side, x0:x0 + side]


def reference_scores(scores: pd.DataFrame) -> pd.DataFrame:
    e = scores[scores['disease'] == 'erosion'].dropna(subset=['score']).copy()
    e['stem'] = [JOINTS.get(k) for k in joint_key(e)]
    e = e.dropna(subset=['stem'])
    e['expert'] = e['expert'].astype(int)
    w = e.pivot_table(index=['patient_id', 'hand', 'stem'], columns='expert', values='score', aggfunc='first')
    w = w.reindex(columns=list(READERS))
    out = w.rename(columns={r: f'r{r}' for r in READERS}).reset_index()
    out['n_readers'] = w.notna().sum(axis=1).to_numpy()
    raw_med = [int(statistics.median_low(v[~np.isnan(v)])) for v in w.to_numpy()]
    out['raw_score'] = raw_med
    out['score'] = [map_score(s) for s in raw_med]
    return out


def prepare(src: str | Path = DEFAULT_ROOT, out: str | Path = 'data/derived/rheumavit') -> pd.DataFrame:
    src, out = Path(src), Path(out)
    boxes = pd.read_csv(src / 'bboxes.csv', index_col=0)
    ref = reference_scores(pd.read_csv(src / 'scores.csv', index_col=0))
    split = {int(p): s for s, ps in json.loads((src / 'train_val_test_split.json').read_text()).items() for p in ps}
    boxes['stem'] = [JOINTS.get(k) for k in joint_key(boxes)]
    rows = []
    for pid, pb in boxes.groupby('patient_id'):
        pid = int(pid)
        img = cv2.imread(str(src / 'jpeg' / f'{pid}.jpeg'), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(src / 'jpeg' / f'{pid}.jpeg')
        for hand, hb in pb.groupby('hand'):
            centre = {k: np.array([r.xcenter, r.ycenter], float) for k, r in zip(joint_key(hb), hb.itertuples())}
            scale = float(REF_PHALANX / np.linalg.norm(centre[('MCP', 3)] - centre[('PIP', 3)]))
            himg = hand_image(img, hand)
            case = f'RV_{pid}_{hand[0].upper()}'
            (out / 'rois' / case).mkdir(parents=True, exist_ok=True)
            for r in hb.dropna(subset=['stem']).itertuples():
                size = ROI_SIZE.get(r.stem, FINGER_ROI)
                roi = crop_square(himg, r.xcenter, r.ycenter, int(round(size / scale)))
                rel_path = Path('rois') / case / f'{r.stem}.png'
                cv2.imwrite(str(out / rel_path), cv2.resize(roi, (size, size), interpolation=cv2.INTER_AREA))
                rows.append({
                    'case': case,
                    'patient': f'RV_{pid}',
                    'patient_id': pid,
                    'hand': hand,
                    'joint': r.stem,
                    'img_path': str(rel_path),
                    'scale': scale,
                    'split': split[pid],
                    'e4_split': 'pool' if split[pid] == 'train' else 'test',
                })
    m = pd.DataFrame(rows).merge(
        ref, left_on=['patient_id', 'hand', 'joint'], right_on=['patient_id', 'hand', 'stem'], how='inner'
    ).drop(columns='stem')
    m.to_csv(out / 'manifest.csv', index=False)
    return m


def _resolve_img_path(raw_path: str, manifest_dir: Path) -> str:
    p = Path(raw_path)
    if p.exists():
        return str(p)
    cand1 = manifest_dir / p
    if cand1.exists():
        return str(cand1)
    parts = p.parts
    if 'rois' in parts:
        idx = parts.index('rois')
        cand2 = manifest_dir / Path(*parts[idx:])
        if cand2.exists():
            return str(cand2)
    return str(cand1)


def load_records(
    manifest_path: str | Path = 'data/derived/rheumavit/manifest.csv',
    e4_split: str | None = 'test',
    target: str = 'consensus',
) -> list[dict]:
    manifest_path = Path(manifest_path)
    manifest_dir = manifest_path.parent
    df = pd.read_csv(manifest_path)
    if e4_split and e4_split != 'all':
        df = df[df['e4_split'] == e4_split]
    if target in ('r0', 'r1', 'r2'):
        df = df[df[target].notna() & (df[target] >= 0)].copy()
        scores = df[target].map(map_score)
    elif target == 'consensus':
        scores = df['score'].map(map_score)
    else:
        raise ValueError(f'Unknown target: {target}')

    records = []
    for row, sc in zip(df.itertuples(index=False), scores):
        records.append({
            'case': row.case,
            'patient': row.patient,
            'date': '00000000',
            'side': 'L' if row.hand == 'left' else 'R',
            'joint': row.joint,
            'score': int(sc),
            'img_path': _resolve_img_path(row.img_path, manifest_dir),
            'r0': int(row.r0) if pd.notna(row.r0) and row.r0 >= 0 else -1,
            'r1': int(row.r1) if pd.notna(row.r1) and row.r1 >= 0 else -1,
            'r2': int(row.r2) if pd.notna(row.r2) and row.r2 >= 0 else -1,
        })
    assert set(r['score'] for r in records) <= set(NATIVE_SCORES)
    return records


def sample_few_shot(pool_records: list[dict], k_patients: int, seed: int) -> tuple[list[dict], list[dict]]:
    patients = sorted({r['patient'] for r in pool_records})
    if k_patients > len(patients):
        raise ValueError(f'Requested {k_patients} patients, pool has {len(patients)}')
    by_p: dict[str, list[dict]] = {}
    for r in pool_records:
        by_p.setdefault(r['patient'], []).append(r)

    rng = np.random.default_rng(seed)
    tr_p = None
    for _ in range(200):
        cand = list(rng.choice(patients, size=k_patients, replace=False))
        sub = [r for p in cand for r in by_p[p]]
        if any(r['score'] >= 1 for r in sub):
            tr_p = cand
            break
    if tr_p is None:
        tr_p = cand
    rem = [p for p in patients if p not in set(tr_p)]
    k_val = min(k_patients, len(rem))
    val_p = list(rng.choice(rem, size=k_val, replace=False))
    tr = sorted([r for p in tr_p for r in by_p[p]], key=lambda r: (r['case'], r['joint']))
    va = sorted([r for p in val_p for r in by_p[p]], key=lambda r: (r['case'], r['joint']))
    return tr, va
