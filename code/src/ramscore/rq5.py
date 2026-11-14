"""RQ5: Label Noise, Longitudinal Visit-Consistency Reliability Weighting, Mean-Teacher, & Multi-Reader Evaluation."""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import DEFAULT_ROOT, NATIVE_SCORES, BEDataset, class_counts, get_transform, load_records, subsample_patients
from .metrics import compute_metrics
from .models import build_model, logit_adjustment, logits_to_preds, logits_to_probs
from .rheumavit import SHARED_JOINTS, map_score
from .train import predict, seed_everything


# --------------------------------------------------------- RQ5-00..03 & RQ5-06/07: OOF & CleanLab / Longitudinal Overrides

def compute_longitudinal_reliability(records: list[dict], floor_weight: float = 0.35) -> dict[tuple[str, str], float]:
    """Identify clinically implausible negative score transitions (Delta S < 0) across serial visits of (patient, side, joint)
    and assign down-weighted reliability w_long(i) in [floor_weight, 1.0] (RQ5-07 novel modification).
    """
    series: dict[tuple[str, str, str], list[dict]] = {}
    for r in records:
        series.setdefault((r['patient'], r['side'], r['joint']), []).append(r)

    weights: dict[tuple[str, str], float] = {}
    for _, recs in series.items():
        recs_sorted = sorted(recs, key=lambda x: int(x['date']))
        if len(recs_sorted) < 2:
            for r in recs_sorted:
                weights[(r['case'], r['joint'])] = 1.0
            continue
        scores = [r['score'] for r in recs_sorted]
        for idx, r in enumerate(recs_sorted):
            w = 1.0
            # Check if this visit is followed by a lower erosion score (healing is rare; 1->0 is 47% noise in RAM-H1200)
            if idx + 1 < len(scores) and scores[idx] > scores[idx + 1]:
                drop = scores[idx] - scores[idx + 1]
                w = max(floor_weight, 1.0 - 0.35 * drop)
            elif idx > 0 and scores[idx - 1] > scores[idx]:
                drop = scores[idx - 1] - scores[idx]
                w = max(floor_weight, 1.0 - 0.25 * drop)
            weights[(r['case'], r['joint'])] = float(w)
    return weights


def run_oof_and_overrides(
    data_root: str | Path = DEFAULT_ROOT,
    out_dir: str | Path = 'runs/rq5/oof',
    n_folds: int = 5,
    epochs: int = 30,
    min_steps: int = 1000,
    seed: int = 42,
    limit: int = 0,
) -> dict:
    """5-Fold Patient-Grouped Out-of-Fold (OOF) probability estimation on RAM-H1200 train split (RQ5-00),
    generating override CSVs for:
      - RQ5-01: cleanlab_relabel.csv
      - RQ5-02: cleanlab_prune.csv
      - RQ5-03: cleanlab_downweight.csv
      - RQ5-06: conf_split.csv
      - RQ5-07: longitudinal_reliability.csv
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_rec = load_records(data_root, 'train')
    if limit > 0:
        train_rec = train_rec[:limit]
    patients = sorted({r['patient'] for r in train_rec})
    rng = np.random.default_rng(seed)
    perm_p = rng.permutation(patients)
    p_to_fold = {p: i % n_folds for i, p in enumerate(perm_p)}

    oof_probs = np.zeros((len(train_rec), len(NATIVE_SCORES)), dtype=np.float32)
    classes_arr = np.asarray(NATIVE_SCORES, dtype=np.int64)
    score_to_idx = {s: i for i, s in enumerate(NATIVE_SCORES)}
    labels = np.array([score_to_idx[r['score']] for r in train_rec], dtype=np.int64)

    for fold in range(n_folds):
        tr_idx = [i for i, r in enumerate(train_rec) if p_to_fold[r['patient']] != fold]
        va_idx = [i for i, r in enumerate(train_rec) if p_to_fold[r['patient']] == fold]
        if not tr_idx or not va_idx:
            continue
        tr_sub = [train_rec[i] for i in tr_idx]
        va_sub = [train_rec[i] for i in va_idx]

        tr_ds = BEDataset(tr_sub, NATIVE_SCORES, 224, get_transform('train', 'base', seed + fold))
        va_ds = BEDataset(va_sub, NATIVE_SCORES, 224, None)
        tr_loader = DataLoader(tr_ds, batch_size=64, shuffle=True,
                               generator=torch.Generator().manual_seed(seed + fold), num_workers=2)
        va_loader = DataLoader(va_ds, batch_size=64, shuffle=False, num_workers=2)

        counts = class_counts(tr_sub, NATIVE_SCORES)
        tau_vec = logit_adjustment(counts, tau=1.0).to(device)
        model = build_model('resnet34', len(NATIVE_SCORES), 'softmax', pretrained=True).to(device)
        steps_ep = max(1, len(tr_loader))
        n_ep = max(epochs, (min_steps + steps_ep - 1) // steps_ep)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_ep, eta_min=1e-6)
        scaler = torch.amp.GradScaler(device.type, enabled=(device.type == 'cuda'))

        for _ in range(n_ep):
            model.train()
            for b in tr_loader:
                opt.zero_grad(set_to_none=True)
                img = b['img'].to(device)
                y = b['label'].to(device)
                with torch.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                    loss = F.cross_entropy(model(img) + tau_vec[None, :], y)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            sched.step()

        _, lg_va, _ = predict(model, va_loader, 'softmax', device, NATIVE_SCORES, return_logits=True)
        p_va = logits_to_probs(torch.from_numpy(lg_va), 'softmax').numpy()
        oof_probs[va_idx] = p_va
        print(f'[RQ5-00 OOF] Fold {fold+1}/{n_folds} complete ({len(va_idx)} joints)', flush=True)

    np.savez_compressed(out_dir / 'oof_probs.npz', probs=oof_probs, labels=labels)

    # Confident Learning (Northcutt et al., JMLR 2021; CleanLab formulation):
    # Class threshold t_c = mean predicted probability p(y=c | x) over samples labelled c
    thresholds = np.array([
        float(oof_probs[labels == c, c].mean()) if (labels == c).any() else 0.5
        for c in range(len(NATIVE_SCORES))
    ])
    self_conf = oof_probs[np.arange(len(labels)), labels]
    pred_cls = oof_probs.argmax(axis=1)
    pred_conf = oof_probs.max(axis=1)
    # Flag as confident label issue when self-confidence < class threshold AND another class exceeds its threshold
    is_issue = (self_conf < thresholds[labels] * 0.75) & (pred_cls != labels) & (pred_conf >= thresholds[pred_cls])

    base_df = pd.DataFrame({
        'case': [r['case'] for r in train_rec],
        'joint': [r['joint'] for r in train_rec],
        'orig_score': [r['score'] for r in train_rec],
        'self_conf': self_conf,
        'pred_score': classes_arr[pred_cls],
        'is_issue': is_issue,
    })
    base_df.to_csv(out_dir / 'oof_summary.csv', index=False)

    # RQ5-01: Relabel confident issues to predicted class
    relabel_df = base_df[['case', 'joint']].copy()
    relabel_df['keep'] = True
    relabel_df['score'] = np.where(is_issue, classes_arr[pred_cls], base_df['orig_score'])
    relabel_df['weight'] = 1.0
    relabel_df.to_csv(out_dir / 'cleanlab_relabel.csv', index=False)

    # RQ5-02: Prune confident issues
    prune_df = base_df[['case', 'joint']].copy()
    prune_df['keep'] = ~is_issue
    prune_df['score'] = base_df['orig_score']
    prune_df['weight'] = 1.0
    prune_df.to_csv(out_dir / 'cleanlab_prune.csv', index=False)

    # RQ5-03: Downweight by self-confidence clipped to [0.2, 1.0]
    dw_df = base_df[['case', 'joint']].copy()
    dw_df['keep'] = True
    dw_df['score'] = base_df['orig_score']
    dw_df['weight'] = np.clip(self_conf / np.maximum(thresholds[labels], 1e-4), 0.20, 1.00)
    dw_df.to_csv(out_dir / 'cleanlab_downweight.csv', index=False)

    # RQ5-06: Confidence sample splitting (down-weight lowest 15% self-confidence samples per class to 0.25)
    cs_df = base_df[['case', 'joint']].copy()
    cs_df['keep'] = True
    cs_df['score'] = base_df['orig_score']
    w_cs = np.ones(len(labels), dtype=np.float32)
    for c in range(len(NATIVE_SCORES)):
        m = labels == c
        if m.any():
            q15 = np.quantile(self_conf[m], 0.15)
            w_cs[m & (self_conf <= q15)] = 0.25
    cs_df['weight'] = w_cs
    cs_df.to_csv(out_dir / 'conf_split.csv', index=False)

    # RQ5-07: Longitudinal Visit-Consistency Reliability Weighting combined with confidence splitting
    long_map = compute_longitudinal_reliability(train_rec)
    w_long = np.array([long_map[(r['case'], r['joint'])] for r in train_rec], dtype=np.float32)
    lr_df = base_df[['case', 'joint']].copy()
    lr_df['keep'] = True
    lr_df['score'] = base_df['orig_score']
    lr_df['weight'] = np.clip(w_cs * w_long, 0.15, 1.00)
    lr_df.to_csv(out_dir / 'longitudinal_reliability.csv', index=False)

    summary = {
        'n_train_joints': int(len(train_rec)),
        'n_issues_flagged': int(is_issue.sum()),
        'issue_rate': float(is_issue.mean()),
        'n_longitudinal_downweighted': int((w_long < 1.0).sum()),
    }
    (out_dir / 'summary.json').write_text(json.dumps(summary, indent=2))
    return summary


# --------------------------------------------------------- RQ5-11: Mean-Teacher Semi-Supervised Consistency

def train_mean_teacher(
    name: str,
    out: str | Path = 'runs/rq5',
    data_root: str | Path = DEFAULT_ROOT,
    label_fraction: float = 0.1,
    fraction_seed: int = 0,
    seed: int = 42,
    epochs: int = 50,
    min_steps: int = 2000,
    ema_decay: float = 0.996,
    cons_weight: float = 2.0,
    limit: int = 0,
) -> dict:
    """Mean-Teacher semi-supervised consistency training (Tarvainen & Valpola, 2017; M061, M185; RQ5-11)."""
    out_dir = Path(out) / name
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    all_train = load_records(data_root, 'train')
    val_rec = load_records(data_root, 'val')
    test_rec = load_records(data_root, 'test')
    labelled_rec = subsample_patients(all_train, label_fraction, seed=fraction_seed)
    lab_keys = {(r['case'], r['joint']) for r in labelled_rec}
    unlabelled_rec = [r for r in all_train if (r['case'], r['joint']) not in lab_keys] or labelled_rec

    if limit > 0:
        labelled_rec = labelled_rec[:limit]
        unlabelled_rec = unlabelled_rec[:limit]
        val_rec = val_rec[:limit]
        test_rec = test_rec[:limit]

    lab_ds = BEDataset(labelled_rec, NATIVE_SCORES, 224, get_transform('train', 'base', seed))
    unlab_ds = BEDataset(unlabelled_rec, NATIVE_SCORES, 224, get_transform('train', 'base', seed + 100), views=2)
    val_ds = BEDataset(val_rec, NATIVE_SCORES, 224, None)
    test_ds = BEDataset(test_rec, NATIVE_SCORES, 224, None)

    lab_loader = DataLoader(lab_ds, batch_size=32, shuffle=True, generator=torch.Generator().manual_seed(seed), num_workers=2)
    unlab_loader = DataLoader(unlab_ds, batch_size=32, shuffle=True, generator=torch.Generator().manual_seed(seed + 1), num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=2)

    counts = class_counts(labelled_rec, NATIVE_SCORES)
    tau_vec = logit_adjustment(counts, tau=1.0).to(device)

    student = build_model('resnet34', len(NATIVE_SCORES), 'softmax', pretrained=True).to(device)
    teacher = copy.deepcopy(student).to(device)
    for p in teacher.parameters():
        p.requires_grad = False

    steps_ep = max(1, len(lab_loader))
    n_epochs = max(epochs, (min_steps + steps_ep - 1) // steps_ep)
    opt = torch.optim.AdamW(student.parameters(), lr=1e-4, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler(device.type, enabled=(device.type == 'cuda'))

    def inf_unlab():
        while True:
            yield from unlab_loader

    unlab_iter = inf_unlab()
    best_qwk = -2.0
    best_ep = 0
    ckpt_path = out_dir / 'model_best.pt'

    for ep in range(n_epochs):
        student.train()
        teacher.train()
        ramp = min(1.0, (ep + 1) / max(1, n_epochs // 4))
        for b_lab in lab_loader:
            b_un = next(unlab_iter)
            opt.zero_grad(set_to_none=True)
            x_l = b_lab['img'].to(device)
            y_l = b_lab['label'].to(device)
            x_u1 = b_un['img'].to(device)
            x_u2 = b_un['img2'].to(device)
            with torch.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                lg_l = student(x_l)
                loss_sup = F.cross_entropy(lg_l + tau_vec[None, :], y_l)
                p_u_s = F.softmax(student(x_u1), dim=1)
                with torch.no_grad():
                    p_u_t = F.softmax(teacher(x_u2), dim=1)
                loss_cons = F.mse_loss(p_u_s, p_u_t)
                loss = loss_sup + (cons_weight * ramp) * loss_cons
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            with torch.no_grad():
                for ps, pt in zip(student.parameters(), teacher.parameters()):
                    pt.mul_(ema_decay).add_(ps.detach(), alpha=1.0 - ema_decay)
        sched.step()
        val_df = predict(teacher, val_loader, 'softmax', device, NATIVE_SCORES)
        vm = compute_metrics(val_df['gt'].to_numpy(), val_df['pred'].to_numpy())
        if vm['qwk'] > best_qwk:
            best_qwk = vm['qwk']
            best_ep = ep
            torch.save({'model': teacher.state_dict(), 'epoch': ep, 'val': vm}, ckpt_path)

    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    teacher.load_state_dict(state['model'])
    val_df, val_lg, val_lb = predict(teacher, val_loader, 'softmax', device, NATIVE_SCORES, return_logits=True)
    test_df, test_lg, test_lb = predict(teacher, test_loader, 'softmax', device, NATIVE_SCORES, return_logits=True)
    test_df.to_csv(out_dir / 'test_predictions.csv', index=False)

    args_meta = {
        'name': name, 'model': 'resnet34', 'head': 'softmax', 'label_fraction': label_fraction,
        'fraction_seed': fraction_seed, 'seed': seed, 'image_size': 224, 'logit_adjust': 1.0,
    }
    (out_dir / 'args.json').write_text(json.dumps(args_meta, indent=2))
    res = {
        'best_epoch': best_ep,
        'val': compute_metrics(val_df['gt'].to_numpy(), val_df['pred'].to_numpy()),
        'test': compute_metrics(test_df['gt'].to_numpy(), test_df['pred'].to_numpy()),
    }
    (out_dir / 'metrics.json').write_text(json.dumps(res, indent=2))
    return res


# --------------------------------------------------------- RQ5-E: Multi-Reader Distributional Evaluation on RheumaVIT

def evaluate_multireader_distribution(manifest_path: str | Path, test_logits_npz_path: str | Path, head: str = 'softmax') -> dict:
    """Compare model predicted distribution p(y|x) against the empirical 3-reader distribution on RheumaVIT test (RQ5-E)."""
    manifest = pd.read_csv(manifest_path)
    test_m = manifest[manifest['e4_split'] == 'test'].copy()
    npz = np.load(test_logits_npz_path)
    probs = logits_to_probs(torch.from_numpy(npz['logits']), head).numpy()

    score_to_idx = {s: i for i, s in enumerate(NATIVE_SCORES)}
    pred_df = pd.DataFrame({
        'case': npz['case'],
        'joint': npz['joint'],
        'pred': npz['pred'],
        'idx': np.arange(len(npz['case'])),
    })
    merged = test_m.merge(pred_df, on=['case', 'joint'], how='inner')
    valid = merged[(merged['r0'] >= 0) & (merged['r1'] >= 0) & (merged['r2'] >= 0)].copy()

    k = len(NATIVE_SCORES)
    q_emp = np.zeros((len(valid), k), dtype=np.float64)
    unanimous = np.zeros(len(valid), dtype=bool)
    for row_i, row in enumerate(valid.itertuples(index=False)):
        r_scores = [map_score(int(row.r0)), map_score(int(row.r1)), map_score(int(row.r2))]
        unanimous[row_i] = (len(set(r_scores)) == 1)
        for s in r_scores:
            q_emp[row_i, score_to_idx[s]] += 1.0 / 3.0

    p_mod = np.clip(probs[valid['idx'].to_numpy()], 1e-8, 1.0)
    p_mod = p_mod / p_mod.sum(axis=1, keepdims=True)

    # KL(q_emp || p_mod) = sum_c q_emp * log(q_emp / p_mod)
    kl = np.sum(np.where(q_emp > 0, q_emp * (np.log(np.maximum(q_emp, 1e-12)) - np.log(p_mod)), 0.0), axis=1)
    gt_all = valid['score'].to_numpy(dtype=np.int64)
    pr_all = valid['pred'].to_numpy(dtype=np.int64)

    return {
        'n_joints': int(len(valid)),
        'unanimous_share': float(unanimous.mean()),
        'mean_kl_to_readers': float(kl.mean()),
        'mean_kl_unanimous': float(kl[unanimous].mean()) if unanimous.any() else float('nan'),
        'mean_kl_disagreement': float(kl[~unanimous].mean()) if (~unanimous).any() else float('nan'),
        'bacc4_unanimous': compute_metrics(gt_all[unanimous], pr_all[unanimous])['bacc4'] if unanimous.any() else float('nan'),
        'bacc4_disagreement': compute_metrics(gt_all[~unanimous], pr_all[~unanimous])['bacc4'] if (~unanimous).any() else float('nan'),
    }
