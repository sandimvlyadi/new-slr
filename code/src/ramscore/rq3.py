"""RQ3: External Validation & Domain Shift Robustness on RheumaVIT (RQ3-01..07)."""
import argparse
import copy
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .data import DEFAULT_ROOT, NATIVE_SCORES, BEDataset, class_counts, get_transform, load_records
from .metrics import compute_metrics, patient_bootstrap_ci
from .models import build_model, logit_adjustment, logits_to_preds, logits_to_probs
from .rheumavit import SHARED_JOINTS, load_records as load_rv_records, sample_few_shot
from .train import make_class_weights, make_sampler, predict, seed_everything


def _load_ckpt_model(run_dir: Path, device: torch.device):
    args = json.loads((run_dir / 'args.json').read_text())
    model = build_model(args['model'], len(NATIVE_SCORES), args['head'], pretrained=False,
                        drop_rate=args.get('drop_rate', 0.0))
    ckpt = torch.load(run_dir / 'model_best.pt', map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model'])
    model.to(device).eval()
    return model, args


# --------------------------------------------------------- RQ3-01: Direct Transfer & Drop

@torch.no_grad()
def eval_direct(run_dir: Path, manifest_path: str | Path, out_dir: Path, device: torch.device) -> dict:
    """Evaluate a RAM-H1200 checkpoint zero-shot on RheumaVIT (pool, test, all) and save logits + predictions."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / 'metrics.json').exists() and (out_dir / 'test_logits.npz').exists():
        return json.loads((out_dir / 'metrics.json').read_text())

    model, args = _load_ckpt_model(run_dir, device)
    head = args['head']
    records_all = load_rv_records(manifest_path, e4_split='all')
    ds = BEDataset(records_all, NATIVE_SCORES, image_size=args['image_size'], transform=None)
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=2)
    df, logits_np, labels_np = predict(model, loader, head, device, NATIVE_SCORES, return_logits=True)

    meta = pd.read_csv(manifest_path)[['case', 'joint', 'e4_split', 'r0', 'r1', 'r2']].drop_duplicates()
    df = df.merge(meta, on=['case', 'joint'], how='left')
    df.to_csv(out_dir / 'predictions.csv', index=False)

    test_mask = (df['e4_split'] == 'test').to_numpy()
    pool_mask = (df['e4_split'] == 'pool').to_numpy()
    df[test_mask][['case', 'patient', 'joint', 'gt', 'pred']].to_csv(out_dir / 'test_predictions.csv', index=False)

    for tag, mask in (('val', pool_mask), ('test', test_mask)):
        sub = df[mask]
        np.savez_compressed(
            out_dir / f'{tag}_logits.npz',
            logits=logits_np[mask],
            labels=labels_np[mask],
            gt=sub['gt'].to_numpy(),
            pred=sub['pred'].to_numpy(),
            case=sub['case'].to_numpy(dtype=str),
            patient=sub['patient'].to_numpy(dtype=str),
            joint=sub['joint'].to_numpy(dtype=str),
        )

    ram_pred = pd.read_csv(run_dir / 'test_predictions.csv')
    ram_13 = ram_pred[ram_pred['joint'].isin(SHARED_JOINTS)]
    m_int_13 = compute_metrics(ram_13['gt'].to_numpy(), ram_13['pred'].to_numpy())
    m_ext_test = compute_metrics(df[test_mask]['gt'].to_numpy(), df[test_mask]['pred'].to_numpy())
    m_ext_all = compute_metrics(df['gt'].to_numpy(), df['pred'].to_numpy())

    res = {
        'source_run': run_dir.name,
        'internal_test_13joints': m_int_13,
        'rheumavit_test': m_ext_test,
        'rheumavit_all': m_ext_all,
        'drop_bacc4': float(m_int_13['bacc4'] - m_ext_test['bacc4']),
        'drop_qwk': float(m_int_13['qwk'] - m_ext_test['qwk']),
    }
    (out_dir / 'metrics.json').write_text(json.dumps(res, indent=2))
    return res


# --------------------------------------------------------- RQ3-02: Test-Time BatchNorm Adaptation (AdaBN)

@torch.no_grad()
def eval_adabn(run_dir: Path, manifest_path: str | Path, out_dir: Path, device: torch.device) -> dict:
    """Update BatchNorm2d running statistics on unlabelled RheumaVIT test batches before scoring (M095)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    model, args = _load_ckpt_model(run_dir, device)
    model = copy.deepcopy(model)
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.train()
            m.reset_running_stats()
            m.momentum = None  # cumulative moving average

    test_rec = load_rv_records(manifest_path, e4_split='test')
    ds = BEDataset(test_rec, NATIVE_SCORES, image_size=args['image_size'], transform=None)
    adapt_loader = DataLoader(ds, batch_size=64, shuffle=True, generator=torch.Generator().manual_seed(0), num_workers=2)
    for b in adapt_loader:
        _ = model(b['img'].to(device))

    model.eval()
    eval_loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=2)
    df = predict(model, eval_loader, args['head'], device, NATIVE_SCORES)
    df.to_csv(out_dir / 'test_predictions.csv', index=False)
    m = compute_metrics(df['gt'].to_numpy(), df['pred'].to_numpy())
    res = {'source_run': run_dir.name, 'method': 'adabn', 'rheumavit_test': m}
    (out_dir / 'metrics.json').write_text(json.dumps(res, indent=2))
    return res


# --------------------------------------------------------- RQ3-03: Histogram Matching to RAM-H1200

def build_reference_cdf(ram_root: str | Path = DEFAULT_ROOT, sample_limit: int = 2000) -> np.ndarray:
    """Compute cumulative intensity histogram over RAM-H1200 train ROIs."""
    recs = load_records(ram_root, 'train')
    if len(recs) > sample_limit:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(recs), size=sample_limit, replace=False)
        recs = [recs[i] for i in idx]
    hist = np.zeros(256, dtype=np.float64)
    for r in recs:
        im = cv2.imread(r['img_path'], cv2.IMREAD_GRAYSCALE)
        if im is not None:
            hist += np.bincount(im.ravel(), minlength=256)
    cdf = np.cumsum(hist)
    return cdf / max(cdf[-1], 1.0)


def match_histogram(img: np.ndarray, ref_cdf: np.ndarray) -> np.ndarray:
    src_hist = np.bincount(img.ravel(), minlength=256).astype(np.float64)
    src_cdf = np.cumsum(src_hist)
    src_cdf /= max(src_cdf[-1], 1.0)
    lut = np.interp(src_cdf, ref_cdf, np.arange(256)).round().astype(np.uint8)
    return lut[img]


@torch.no_grad()
def eval_histmatch(run_dir: Path, manifest_path: str | Path, out_dir: Path, device: torch.device,
                   ram_root: str | Path = DEFAULT_ROOT) -> dict:
    """Match RheumaVIT test ROI intensity histograms to RAM-H1200 reference CDF before inference."""
    out_dir.mkdir(parents=True, exist_ok=True)
    model, args = _load_ckpt_model(run_dir, device)
    ref_cdf = build_reference_cdf(ram_root)
    test_rec = load_rv_records(manifest_path, e4_split='test')
    ds = BEDataset(test_rec, NATIVE_SCORES, image_size=args['image_size'], transform=None)
    for i in range(len(ds.cache)):
        ds.cache[i] = match_histogram(ds.cache[i], ref_cdf)
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=2)
    df = predict(model, loader, args['head'], device, NATIVE_SCORES)
    df.to_csv(out_dir / 'test_predictions.csv', index=False)
    m = compute_metrics(df['gt'].to_numpy(), df['pred'].to_numpy())
    res = {'source_run': run_dir.name, 'method': 'histmatch', 'rheumavit_test': m}
    (out_dir / 'metrics.json').write_text(json.dumps(res, indent=2))
    return res


# --------------------------------------------------------- RQ3-06: Few-Shot Adaptation (5 / 20 patients)

def eval_fewshot(run_dir: Path, manifest_path: str | Path, k_patients: int, draw: int, out_dir: Path,
                 device: torch.device, epochs: int = 15, min_steps: int = 150, lr: float = 1e-4) -> dict:
    """Fine-tune a RAM-H1200 checkpoint on k_patients from RheumaVIT pool and evaluate on RheumaVIT test."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / 'metrics.json').exists():
        return json.loads((out_dir / 'metrics.json').read_text())

    seed = 1000 * k_patients + draw
    seed_everything(seed)
    model, args = _load_ckpt_model(run_dir, device)

    pool_rec = load_rv_records(manifest_path, e4_split='pool')
    test_rec = load_rv_records(manifest_path, e4_split='test')
    tr_rec, val_rec = sample_few_shot(pool_rec, k_patients=k_patients, seed=draw)

    train_ds = BEDataset(tr_rec, NATIVE_SCORES, args['image_size'], get_transform('train', 'base', seed))
    val_ds = BEDataset(val_rec, NATIVE_SCORES, args['image_size'], None)
    test_ds = BEDataset(test_rec, NATIVE_SCORES, args['image_size'], None)

    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=2)

    counts = class_counts(tr_rec, NATIVE_SCORES)
    tau_vec = logit_adjustment(counts, tau=float(args.get('logit_adjust', 0.0))).to(device) if args.get('logit_adjust', 0.0) > 0 else None
    cls_w = make_class_weights(counts).to(device) if args.get('class_weight') else None

    bs = min(32, len(train_ds))
    steps_per_ep = max(1, len(train_ds) // max(bs, 1))
    n_epochs = max(epochs, (min_steps + steps_per_ep - 1) // steps_per_ep)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=lr * 0.1)
    scaler = torch.amp.GradScaler(device.type, enabled=(device.type == 'cuda'))

    best_qwk = -2.0
    best_state = copy.deepcopy(model.state_dict())
    for ep in range(n_epochs):
        model.train()
        gen = torch.Generator().manual_seed(seed + ep)
        loader = DataLoader(train_ds, batch_size=bs, shuffle=True, generator=gen, num_workers=2)
        for b in loader:
            optimizer.zero_grad(set_to_none=True)
            img = b['img'].to(device)
            labels = b['label'].to(device)
            with torch.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                logits = model(img)
                adj = logits + tau_vec[None, :] if (tau_vec is not None and args['head'] == 'softmax') else logits
                loss = torch.nn.functional.cross_entropy(adj, labels, weight=cls_w)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        scheduler.step()
        val_df = predict(model, val_loader, args['head'], device, NATIVE_SCORES)
        vm = compute_metrics(val_df['gt'].to_numpy(), val_df['pred'].to_numpy())
        if vm['qwk'] > best_qwk:
            best_qwk = vm['qwk']
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    test_df = predict(model, test_loader, args['head'], device, NATIVE_SCORES)
    test_df.to_csv(out_dir / 'test_predictions.csv', index=False)
    res = {
        'source_run': run_dir.name,
        'k_patients': k_patients,
        'draw': draw,
        'rheumavit_test': compute_metrics(test_df['gt'].to_numpy(), test_df['pred'].to_numpy()),
    }
    (out_dir / 'metrics.json').write_text(json.dumps(res, indent=2))
    return res


# --------------------------------------------------------- RQ3-07: Reader Anchor

def reader_anchor(manifest_path: str | Path, pred_csv_paths: list[Path]) -> dict:
    """Compare model predictions on RheumaVIT test against each of the 3 expert rheumatologists (r0, r1, r2)."""
    from .rheumavit import map_score

    manifest = pd.read_csv(manifest_path)
    test_m = manifest[manifest['e4_split'] == 'test'].copy()
    for r in ('r0', 'r1', 'r2'):
        test_m = test_m[test_m[r] >= 0]
        test_m[f'{r}_mapped'] = test_m[r].map(map_score)

    inter_reader = {}
    for a, b in (('r0', 'r1'), ('r0', 'r2'), ('r1', 'r2')):
        m = compute_metrics(test_m[f'{a}_mapped'].to_numpy(), test_m[f'{b}_mapped'].to_numpy())
        inter_reader[f'{a}_vs_{b}'] = {'bacc4': m['bacc4'], 'qwk': m['qwk'], 'mae': m['mae']}

    vs_consensus, vs_readers = [], {r: [] for r in ('r0', 'r1', 'r2')}
    for p in pred_csv_paths:
        df = pd.read_csv(p)
        merged = test_m.merge(df[['case', 'joint', 'pred']], on=['case', 'joint'], how='inner')
        vs_consensus.append(compute_metrics(merged['score'].to_numpy(), merged['pred'].to_numpy()))
        for r in ('r0', 'r1', 'r2'):
            vs_readers[r].append(compute_metrics(merged[f'{r}_mapped'].to_numpy(), merged['pred'].to_numpy()))

    def avg(ms):
        return {k: float(np.mean([m[k] for m in ms])) for k in ('bacc4', 'qwk', 'mae', 'recall_ge2')} if ms else {}

    return {
        'n_test_joints_3readers': int(len(test_m)),
        'inter_reader_pairs': inter_reader,
        'inter_reader_mean': {
            k: float(np.mean([v[k] for v in inter_reader.values()])) for k in ('bacc4', 'qwk', 'mae')
        },
        'model_vs_consensus': avg(vs_consensus),
        'model_vs_readers': {r: avg(ms) for r, ms in vs_readers.items()},
        'model_vs_individual_mean': {
            k: float(np.mean([avg(vs_readers[r])[k] for r in ('r0', 'r1', 'r2')])) for k in ('bacc4', 'qwk', 'mae', 'recall_ge2')
        } if pred_csv_paths else {},
    }
