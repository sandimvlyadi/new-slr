"""RAM-H1200 joint-level bone-erosion (BE) scoring data.

Layout (per split): <root>/<split>/_annotation_be_scores.json and one folder
per hand, <root>/<split>/<case>/<joint>.bmp, where <case> is
<country>_<centre>_P<nnnn>_<YYYYMMDD>_<id>_<L|R>. Dates are shifted per patient.
"""
from collections import Counter
from dataclasses import dataclass
import json
import os
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

DEFAULT_ROOT = os.environ.get(
    'RAM_H1200_BE_ROOT',
    '/home/sandimulyadi/Projects/Dataset/RAM-H1200/SvdH_Scoring/SvdH_BE_Scoring',
)

# ROI file stem -> score key, as in the official benchmark (datasets/scorecls.py)
BE_JOINTS = {
    'R': 'BE_R', 'U': 'BE_U', 'IP': 'BE_IP', 'L': 'BE_L',
    'MCP-T': 'BE_MCP-T', 'MCP-I': 'BE_MCP-I', 'MCP-M': 'BE_MCP-M', 'MCP-R': 'BE_MCP-R', 'MCP-S': 'BE_MCP-S',
    'CMC-T': 'BE_CMC-T',
    'PIP-I': 'BE_PIP-I', 'PIP-M': 'BE_PIP-M', 'PIP-R': 'BE_PIP-R', 'PIP-S': 'BE_PIP-S',
    'S': 'BE_S', 'Tm': 'BE_Tm',
}
SCORE_FILE = '_annotation_be_scores.json'

# Native SvdH erosion grades present in RAM-H1200 (grade 4 never occurs)
NATIVE_SCORES = (0, 1, 2, 3, 5)
# Primary-endpoint grouping (proposal section 6): 0, 1, 2, >=3
LEVEL4 = {0: 0, 1: 1, 2: 2, 3: 3, 4: 3, 5: 3}


@dataclass(frozen=True)
class Case:
    centre: str
    patient: str     # centre-qualified, e.g. 'HMCRD_P0050'
    date: str        # YYYYMMDD (shifted)
    side: str        # 'L' or 'R'


def parse_case(name: str) -> Case:
    parts = Path(name).stem.split('_')
    if len(parts) != 6 or parts[-1] not in ('L', 'R'):
        raise ValueError(f'Unexpected case name: {name}')
    _, centre, pid, date, _, side = parts
    return Case(centre, f'{centre}_{pid}', date, side)


def load_records(root: str | Path = DEFAULT_ROOT, split: str = 'train') -> list[dict]:
    """One record per (hand, joint) that has both an ROI image and a score."""
    split_root = Path(root) / split
    scores = json.loads((split_root / SCORE_FILE).read_text())
    records = []
    for image_name, joint_scores in scores.items():
        case_name = Path(image_name).stem
        case = parse_case(case_name)
        for stem, key in BE_JOINTS.items():
            img_path = split_root / case_name / f'{stem}.bmp'
            if not img_path.exists():
                png_path = split_root / case_name / f'{stem}.png'
                if png_path.exists():
                    img_path = png_path
            if key in joint_scores and img_path.exists():
                records.append({
                    'img_path': str(img_path),
                    'case': case_name,
                    'patient': case.patient,
                    'centre': case.centre,
                    'date': case.date,
                    'side': case.side,
                    'joint': stem,
                    'score': int(joint_scores[key]),
                })
    if not records:
        raise RuntimeError(f'No BE records under {split_root}')
    return records


AUGMENTATIONS = ('base', 'none', 'geom', 'intensity', 'strong', 'rheumavit_match')


def get_transform(split: str, aug: str = 'base', seed: int | None = None):
    """Augmentation of the benchmark's utils.get_cls_transform, plus RQ3 domain-robust augmentations."""
    if aug not in AUGMENTATIONS:
        raise ValueError(f'Unknown augmentation {aug}; one of {AUGMENTATIONS}')
    if split != 'train' or aug == 'none':
        return None
    steps = []
    if aug in ('base', 'geom'):
        steps.append(A.ShiftScaleRotate(rotate_limit=10))
    if aug in ('base', 'intensity'):
        steps.append(A.RandomBrightnessContrast(p=0.5))
    if aug == 'strong':
        steps.extend([
            A.ShiftScaleRotate(rotate_limit=15, p=0.8),
            A.RandomBrightnessContrast(brightness_limit=0.35, contrast_limit=0.35, p=0.9),
            A.RandomGamma(gamma_limit=(70, 140), p=0.5),
            A.GaussianBlur(sigma_limit=(0.1, 1.5), p=0.3),
        ])
    if aug == 'rheumavit_match':
        steps.extend([
            A.ShiftScaleRotate(rotate_limit=10),
            # RheumaVIT crops are brighter and lower-contrast than RAM-H1200
            A.RandomBrightnessContrast(brightness_limit=(0.0, 0.25), contrast_limit=(-0.30, 0.05), p=0.85),
        ])
    return A.Compose(steps, seed=seed)


def seed_augmentation_worker(seed: int, worker_id: int) -> None:
    ds = torch.utils.data.get_worker_info().dataset
    transform = getattr(ds, 'transform', None)
    if transform is None and hasattr(ds, 'dataset'):
        transform = getattr(ds.dataset, 'transform', None)
    if transform is not None:
        transform.set_random_seed(seed + worker_id)


def read_roi(path: str, image_size: int) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_LINEAR)


def read_mask(path: str, image_size: int) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_NEAREST)


class BEDataset(Dataset):
    def __init__(self, records, score_values=NATIVE_SCORES, image_size=224, transform=None, mask_root=None, views=1, preload=True):
        self.records = records
        self.score_values = tuple(score_values)
        self.score_to_class = {s: i for i, s in enumerate(self.score_values)}
        missing = {r['score'] for r in records} - set(self.score_values)
        if missing:
            raise ValueError(f'Scores {sorted(missing)} not in score_values {self.score_values}')
        self.labels = np.array([self.score_to_class[r['score']] for r in records], dtype=np.int64)
        self.weights = np.array([float(r.get('weight', 1.0)) for r in records], dtype=np.float32)
        self.image_size = image_size
        self.transform = transform
        self.cache = np.stack([read_roi(r['img_path'], image_size) for r in records]) if (preload and records) else None
        self.has_mask = bool(records) and ('mask_path' in records[0] or mask_root is not None)
        if preload and self.has_mask:
            mpaths = [
                r.get('mask_path') or str(Path(mask_root) / Path(r['img_path']).parent.parent.name / r['case'] / f"{r['joint']}.png")
                for r in records
            ]
            self.mask_cache = np.stack([read_mask(p, image_size) for p in mpaths])
        else:
            self.mask_cache = None
        self.views = views

    @property
    def num_classes(self):
        return len(self.score_values)

    def __len__(self):
        return len(self.records)

    def to_tensor(self, img):
        if self.transform is not None:
            img = self.transform(image=img)['image']
        return torch.from_numpy(img.astype(np.float32) / 255.0)[None]

    def __getitem__(self, idx):
        rec = self.records[idx]
        img = self.cache[idx] if self.cache is not None else read_roi(rec['img_path'], self.image_size)
        if self.has_mask:
            mask = self.mask_cache[idx]
            if self.transform is not None:
                out_t = self.transform(image=img, mask=mask)
                img, mask = out_t['image'], out_t['mask']
            img_t = torch.from_numpy(np.stack([img.astype(np.float32) / 255.0, (mask > 127).astype(np.float32)]))
        else:
            raw = img
            img_t = self.to_tensor(raw)
        label = int(self.labels[idx])
        out = {
            'img': img_t,
            'label': torch.tensor(label, dtype=torch.long),
            'ordinal_target': torch.tensor([float(label > k) for k in range(self.num_classes - 1)]),
            'score': torch.tensor(rec['score'], dtype=torch.long),
            'weight': torch.tensor(self.weights[idx], dtype=torch.float32),
            'case': rec['case'],
            'joint': rec['joint'],
            'patient': rec['patient'],
        }
        if self.views == 2:
            out['img2'] = self.to_tensor(raw)
        return out


def subsample_patients(records: list[dict], fraction: float, seed: int) -> list[dict]:
    """Keep a patient-level fraction, stratified by each patient's maximum erosion grade (exact match to E2/E3/RQ0/RQ1)."""
    if fraction >= 1.0:
        return list(records)
    by_patient = {}
    for r in records:
        by_patient[r['patient']] = max(by_patient.get(r['patient'], 0), r['score'])
    strata = {}
    for patient, worst in sorted(by_patient.items()):
        strata.setdefault(worst, []).append(patient)
    rng = np.random.default_rng(seed)
    keep = set()
    for patients in strata.values():
        n = max(1, round(fraction * len(patients)))
        keep.update(rng.choice(patients, size=n, replace=False).tolist())
    return [r for r in records if r['patient'] in keep]


def zero_quotas(records: list[dict]) -> dict[str, int]:
    """Grade-0 joints to keep per joint type for 1:1 against all other grades (E7 / RQ2-08)."""
    zeros = Counter(r['joint'] for r in records if r['score'] == 0)
    n_zero = sum(zeros.values())
    total = min(len(records) - n_zero, n_zero)
    quotas = {j: c * total // n_zero for j, c in sorted(zeros.items())}
    by_remainder = sorted(zeros, key=lambda j: (-(zeros[j] * total % n_zero), j))
    for j in by_remainder[:total - sum(quotas.values())]:
        quotas[j] += 1
    return quotas


def _zeros_by_joint(records: list[dict]) -> dict[str, np.ndarray]:
    pools = {}
    for i, r in enumerate(records):
        if r['score'] == 0:
            pools.setdefault(r['joint'], []).append(i)
    return {j: np.array(pools[j], dtype=np.int64) for j in sorted(pools)}


def undersample_grade0(records: list[dict], seed: int, ratio: float = 1.0) -> list[dict]:
    """Drop grade-0 joints down to zero_quotas; every other joint stays (E7 / RQ2-08)."""
    rng = np.random.default_rng(seed)
    quotas = zero_quotas(records)
    drop = set()
    for joint, pool in _zeros_by_joint(records).items():
        keep_n = min(len(pool), max(1, round(quotas[joint] * ratio)))
        drop.update(rng.permutation(pool)[keep_n:].tolist())
    return [r for i, r in enumerate(records) if i not in drop]


class RotatingZeroSampler(Sampler):
    """Every epoch: all joints above grade 0 and the next zero_quotas grade-0 joints of each joint type (E7 / RQ2-08)."""

    def __init__(self, records: list[dict], seed: int):
        self.rng = np.random.default_rng(seed)
        self.quotas = zero_quotas(records)
        self.pools = _zeros_by_joint(records)
        self.queues = {j: self.rng.permutation(pool) for j, pool in self.pools.items()}
        self.always = np.array([i for i, r in enumerate(records) if r['score'] != 0], dtype=np.int64)

    def __len__(self):
        return len(self.always) + sum(self.quotas.values())

    def _take(self, joint: str) -> np.ndarray:
        n, queue = self.quotas[joint], self.queues[joint]
        head, rest = queue[:n], queue[n:]
        if len(head) < n:
            fresh = self.rng.permutation(self.pools[joint])
            fresh = fresh[~np.isin(fresh, head)]
            missing = n - len(head)
            head, rest = np.concatenate([head, fresh[:missing]]), np.concatenate([fresh[missing:], head])
        self.queues[joint] = rest
        return head

    def __iter__(self):
        zeros = [self._take(j) for j in self.quotas]
        return iter(self.rng.permutation(np.concatenate([self.always, *zeros])).tolist())


def class_counts(records: list[dict], classes: tuple[int, ...] = NATIVE_SCORES) -> np.ndarray:
    idx = {s: i for i, s in enumerate(classes)}
    counts = np.zeros(len(classes), dtype=np.int64)
    for r in records:
        counts[idx[r['score']]] += 1
    return counts


def joint_class_counts(
    records: list[dict],
    joints=tuple(BE_JOINTS.keys()),
    classes: tuple[int, ...] = NATIVE_SCORES,
) -> dict[str, np.ndarray]:
    idx = {s: i for i, s in enumerate(classes)}
    out = {j: np.zeros(len(classes), dtype=np.int64) for j in joints}
    for r in records:
        if r['joint'] in out:
            out[r['joint']][idx[r['score']]] += 1
    return out
