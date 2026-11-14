"""Single-run training & evaluation entrypoint for RAM-H1200 BE grading (RQ0-RQ6).

Writes to `<out>/<name>/`:
  args.json              CLI arguments + resolved class counts
  history.csv            per-epoch train_loss, val_bacc4, val_bacc5, val_qwk, val_macro_f1, lr, sec
  model_best.pt          checkpoint with best validation monitor metric
  metrics.json           best validation metrics + final test metrics (+ patient-bootstrap 95% CIs)
  test_predictions.csv   per-joint test predictions (case, patient, joint, gt, pred)
  val_logits.npz         saved validation logits, labels, cases, patients, joints (for RQ2-P / RQ4)
  test_logits.npz        saved test logits, labels, cases, patients, joints (for RQ2-P / RQ4)
"""
import argparse
from functools import partial
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from .data import (
    BE_JOINTS,
    BEDataset,
    DEFAULT_ROOT,
    NATIVE_SCORES,
    class_counts,
    get_transform,
    joint_class_counts,
    load_records,
    seed_augmentation_worker,
    subsample_patients,
    undersample_grade0,
)
from .metrics import compute_metrics, patient_bootstrap_ci
from .models import (
    CumulativeLinkWrapper,
    HEAD_OUT_DIM,
    ProjectionHead,
    build_model,
    corn_logit_adjustment,
    corn_loss,
    effective_number_weights,
    focal_ce_loss,
    gce_loss,
    hierarchical_loss,
    joint_logit_adjustment,
    ldam_loss,
    ldam_margins,
    load_backbone,
    load_full,
    logit_adjustment,
    logits_to_preds,
    ordinal_loss,
    pooled_and_logits,
    soft_ordinal_loss,
    supcon_loss,
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_sampler(labels: np.ndarray, num_classes: int, power: float = 1.0, seed: int = 0) -> WeightedRandomSampler:
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    inv = np.where(counts > 0, (1.0 / np.maximum(counts, 1.0)) ** power, 0.0)
    weights = torch.from_numpy(inv[labels]).double()
    gen = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(weights, num_samples=len(labels), replacement=True, generator=gen)


def make_class_weights(counts: np.ndarray, power: float = 0.5) -> torch.Tensor:
    c = np.maximum(counts.astype(np.float64), 1.0)
    w = (c.sum() / (len(c) * c)) ** power
    w = w / w.mean()
    return torch.from_numpy(w.astype(np.float32))


def head_parameters(model: nn.Module) -> list[nn.Parameter]:
    if isinstance(model, CumulativeLinkWrapper):
        return list(model.head.parameters())
    head_name = model.pretrained_cfg['classifier']
    return list(model.get_submodule(head_name).parameters())


def freeze_backbone(model: nn.Module) -> None:
    head = set(head_parameters(model))
    for p in model.parameters():
        p.requires_grad = p in head


def set_train_mode(model: nn.Module, frozen_backbone: bool) -> None:
    if not frozen_backbone:
        model.train()
        return
    model.eval()
    if isinstance(model, CumulativeLinkWrapper):
        model.head.train()
    else:
        model.get_submodule(model.pretrained_cfg['classifier']).train()


def vit_layer_decay_groups(model: nn.Module, base_lr: float, weight_decay: float, decay: float) -> list[dict]:
    if not hasattr(model, 'blocks'):
        raise ValueError(f'--layer-decay requires a ViT with .blocks, got {type(model).__name__}')
    n_blocks = len(model.blocks)
    n_layers = n_blocks + 2

    def layer_id(name: str) -> int:
        if name in ('cls_token', 'pos_embed') or name.startswith('patch_embed'):
            return 0
        if name.startswith('blocks.'):
            return int(name.split('.')[1]) + 1
        return n_layers - 1

    groups: dict[tuple[int, bool], list] = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        lid = layer_id(name)
        no_wd = p.ndim <= 1 or name in ('cls_token', 'pos_embed')
        groups.setdefault((lid, no_wd), []).append(p)

    return [
        {
            'params': params,
            'lr': base_lr * (decay ** (n_layers - 1 - lid)),
            'weight_decay': 0.0 if no_wd else weight_decay,
        }
        for (lid, no_wd), params in sorted(groups.items())
    ]


def apply_label_overrides(records: list[dict], override_path: str | Path) -> list[dict]:
    """Apply CleanLab / reliability overrides (prune, relabel, or sample weight) to training records."""
    df = pd.read_csv(override_path)
    lookup = {}
    for row in df.itertuples(index=False):
        lookup[(str(row.case), str(row.joint))] = row
    out = []
    for r in records:
        key = (r['case'], r['joint'])
        if key not in lookup:
            out.append(r)
            continue
        ov = lookup[key]
        if hasattr(ov, 'keep') and not bool(ov.keep):
            continue
        r_new = dict(r)
        if hasattr(ov, 'score') and pd.notna(ov.score):
            r_new['score'] = int(ov.score)
        if hasattr(ov, 'weight') and pd.notna(ov.weight):
            r_new['weight'] = float(ov.weight)
        out.append(r_new)
    return out


def compute_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    head: str,
    num_classes: int,
    cls_w: torch.Tensor | None = None,
    pos_w: torch.Tensor | None = None,
    sev_w: torch.Tensor | None = None,
    tau_vec: torch.Tensor | None = None,
    joint_tau_map: dict[str, torch.Tensor] | None = None,
    joints: list[str] | None = None,
    corn_tau: torch.Tensor | None = None,
    sample_weights: torch.Tensor | None = None,
    focal_gamma: float = 0.0,
    soft_label_sigma: float = 0.0,
    disagreement_soft_label: bool = False,
    ldam_m: torch.Tensor | None = None,
    corn_reg_weight: float = 0.0,
    label_smoothing: float = 0.0,
    gce_q: float = 0.0,
) -> torch.Tensor:
    if head == 'cumulative_link':
        # logits are log-probabilities from CumulativeLinkHead
        loss_vec = F.nll_loss(logits, labels, weight=cls_w, reduction='none')
        return (loss_vec * sample_weights).sum() / sample_weights.sum().clamp_min(1e-8) if sample_weights is not None else loss_vec.mean()

    if head == 'softmax_corn':
        s_logits = logits[:, :num_classes]
        c_logits = logits[:, num_classes:]
        if tau_vec is not None:
            s_logits = s_logits + tau_vec[None, :]
        l_ce = F.cross_entropy(s_logits, labels, weight=cls_w, reduction='none')
        l_main = (l_ce * sample_weights).sum() / sample_weights.sum().clamp_min(1e-8) if sample_weights is not None else l_ce.mean()
        l_corn = corn_loss(c_logits, labels, num_classes, tau=corn_tau, sample_weights=sample_weights)
        return l_main + corn_reg_weight * l_corn

    if head == 'softmax':
        adj = logits
        if joint_tau_map is not None and joints is not None:
            offsets = torch.stack([joint_tau_map[j] for j in joints]).to(logits.device)
            adj = logits + offsets
        elif tau_vec is not None:
            adj = logits + tau_vec[None, :]

        if ldam_m is not None:
            return ldam_loss(adj, labels, ldam_m, weight=cls_w, sample_weights=sample_weights)
        if gce_q > 0.0:
            return gce_loss(adj, labels, q=gce_q, sample_weights=sample_weights)
        if disagreement_soft_label:
            # Grade 0 has higher inter-reader agreement (sigma=0.25); grades >=1 have wider disagreement (sigma=0.65)
            sigmas = torch.where(labels == 0, 0.25, 0.65)
            return soft_ordinal_loss(adj, labels, num_classes, sigma=sigmas, sample_weights=sample_weights)
        if soft_label_sigma > 0.0:
            return soft_ordinal_loss(adj, labels, num_classes, sigma=soft_label_sigma, sample_weights=sample_weights)
        if focal_gamma > 0.0:
            return focal_ce_loss(adj, labels, gamma=focal_gamma, weight=cls_w, sample_weights=sample_weights)

        loss_vec = F.cross_entropy(adj, labels, weight=cls_w, label_smoothing=label_smoothing, reduction='none')
        return (loss_vec * sample_weights).sum() / sample_weights.sum().clamp_min(1e-8) if sample_weights is not None else loss_vec.mean()

    if head == 'ordinal':
        sw = cls_w[labels] if cls_w is not None else None
        if sample_weights is not None:
            sw = sample_weights if sw is None else sw * sample_weights
        return ordinal_loss(logits, labels, num_classes, sample_weights=sw, focal_gamma=focal_gamma)

    if head == 'corn':
        return corn_loss(logits, labels, num_classes, tau=corn_tau, sample_weights=sample_weights)

    if head == 'hierarchical':
        return hierarchical_loss(logits, labels, pos_weight=pos_w, sev_weight=sev_w, sample_weights=sample_weights)

    raise ValueError(head)


def supcon_step(
    model: nn.Module,
    proj: nn.Module,
    batch: dict,
    device: torch.device,
    amp: bool,
    loss_fn,
    weight: float,
    temperature: float = 0.1,
    alpha: float | None = None,
    num_classes: int = 5,
) -> torch.Tensor:
    n = len(batch['img'])
    x = torch.cat([batch['img'], batch['img2']], dim=0).to(device, non_blocking=True)
    y = batch['label'].to(device, non_blocking=True)
    w = batch['weight'].to(device, non_blocking=True) if 'weight' in batch else None
    with torch.autocast(device_type=device.type, enabled=amp):
        feats, logits = pooled_and_logits(model, x)
        z = proj(feats)
        loss_cls = loss_fn(logits[:n], y, batch['joint'], w)
        loss_sc = supcon_loss(z, y.repeat(2), temperature=temperature, alpha=alpha, num_classes=num_classes)
        return loss_cls + weight * loss_sc


@torch.no_grad()
def predict(
    model: nn.Module,
    loader: DataLoader,
    head: str,
    device: torch.device,
    classes: tuple[int, ...] = NATIVE_SCORES,
    return_logits: bool = False,
):
    model.eval()
    rows = []
    all_logits = []
    all_labels = []
    classes_arr = np.asarray(classes, dtype=np.int64)
    for batch in loader:
        img = batch['img'].to(device, non_blocking=True)
        logits = model(img)
        preds = logits_to_preds(logits, head=head).cpu().numpy()
        pred_scores = classes_arr[preds]
        gt_scores = batch['score'].numpy()
        if return_logits:
            all_logits.append(logits.float().cpu().numpy())
            all_labels.append(batch['label'].numpy())
        for c, p, j, g, pr in zip(batch['case'], batch['patient'], batch['joint'], gt_scores, pred_scores):
            rows.append({'case': c, 'patient': p, 'joint': j, 'gt': int(g), 'pred': int(pr)})
    df = pd.DataFrame(rows)
    if not return_logits:
        return df
    logits_np = np.concatenate(all_logits, axis=0) if all_logits else np.empty((0, len(classes)), dtype=np.float32)
    labels_np = np.concatenate(all_labels, axis=0) if all_labels else np.empty((0,), dtype=np.int64)
    return df, logits_np, labels_np


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description='Train SvdH BE joint grader on RAM-H1200')
    ap.add_argument('--name', required=True, help='Run directory name under --out')
    ap.add_argument('--out', default='runs', help='Output parent directory')
    ap.add_argument('--data-root', default=DEFAULT_ROOT)
    ap.add_argument('--model', default='resnet34')
    ap.add_argument('--pretrained', action='store_true', default=False)
    ap.add_argument('--init-from', default=None, help='Path to encoder.pt or model_best.pt')
    ap.add_argument('--keep-head', action='store_true', default=False)
    ap.add_argument('--freeze-backbone', action='store_true', default=False)
    ap.add_argument('--grad-checkpointing', action='store_true', default=False)
    ap.add_argument('--layer-decay', type=float, default=1.0)
    ap.add_argument('--warmup', type=float, default=0.0)
    ap.add_argument('--head', choices=list(HEAD_OUT_DIM), default='ordinal')
    ap.add_argument('--oversample', type=float, default=0.0)
    ap.add_argument('--undersample-zero', choices=['none', 'permanent', 'rotate'], default='none')
    ap.add_argument('--undersample-ratio', type=float, default=1.0)
    ap.add_argument('--class-weight', action='store_true', default=False)
    ap.add_argument('--logit-adjust', type=float, default=0.0)
    ap.add_argument('--joint-logit-adjust', type=float, default=0.0, help='Per-joint prior logit adjustment (RQ2-17)')
    ap.add_argument('--focal-gamma', type=float, default=0.0, help='Focal exponent gamma (RQ2-09/10/11)')
    ap.add_argument('--soft-label-sigma', type=float, default=0.0, help='Soft ordinal Gaussian sigma (RQ2-13)')
    ap.add_argument('--disagreement-soft-label', action='store_true', default=False, help='Reader-disagreement soft labels (RQ5-05)')
    ap.add_argument('--ldam-drw', action='store_true', default=False, help='LDAM margin + DRW schedule (RQ2-14)')
    ap.add_argument('--effective-beta', type=float, default=0.0, help='Class-balanced effective number beta (RQ2-15)')
    ap.add_argument('--corn-reg-weight', type=float, default=0.0, help='CORN regularizer weight on softmax_corn head (RQ2-16)')
    ap.add_argument('--drop-rate', type=float, default=0.0, help='Classifier head dropout rate (RQ4-01)')
    ap.add_argument('--label-smoothing', type=float, default=0.0, help='Cross-entropy label smoothing (RQ5-04)')
    ap.add_argument('--gce-q', type=float, default=0.0, help='Generalized cross-entropy q in (0,1] (RQ5-08)')
    ap.add_argument('--label-overrides', default=None, help='CSV with case,joint,[keep,score,weight] for RQ5')
    ap.add_argument('--supcon-weight', type=float, default=0.0)
    ap.add_argument('--supcon-temp', type=float, default=0.1)
    ap.add_argument('--supcon-alpha', type=float, default=None)
    ap.add_argument('--label-fraction', type=float, default=1.0)
    ap.add_argument('--fraction-seed', type=int, default=0)
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--min-steps', type=int, default=1500)
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--weight-decay', type=float, default=1e-4)
    ap.add_argument('--image-size', type=int, default=224)
    ap.add_argument('--mask-root', default=None)
    ap.add_argument('--aug', choices=['base', 'geom', 'intensity', 'strong', 'rheumavit_match', 'none'], default='base')
    ap.add_argument('--seed-aug', action='store_true', default=False)
    ap.add_argument('--monitor', choices=['bacc4', 'qwk'], default='qwk')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--amp', action='store_true', default=False)
    ap.add_argument('--n-boot', type=int, default=2000)
    ap.add_argument('--limit', type=int, default=0)
    return ap.parse_args(argv)


def check_init_args(args: argparse.Namespace) -> None:
    if args.pretrained and args.init_from:
        raise ValueError('--pretrained and --init-from are mutually exclusive')
    if args.keep_head and not args.init_from:
        raise ValueError('--keep-head requires --init-from')
    if args.freeze_backbone and args.layer_decay != 1.0:
        raise ValueError('--freeze-backbone and --layer-decay are mutually exclusive')


def check_supcon_args(args: argparse.Namespace) -> None:
    if args.supcon_alpha is not None and args.supcon_weight <= 0:
        raise ValueError('--supcon-alpha requires --supcon-weight > 0')
    if args.supcon_weight > 0 and args.freeze_backbone:
        raise ValueError('--supcon-weight > 0 is incompatible with --freeze-backbone')


def main(argv=None) -> dict:
    args = parse_args(argv)
    check_init_args(args)
    check_supcon_args(args)
    seed_everything(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    out_dir = Path(args.out) / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    train_rec = load_records(args.data_root, 'train')
    val_rec = load_records(args.data_root, 'val')
    test_rec = load_records(args.data_root, 'test')

    if args.label_overrides:
        train_rec = apply_label_overrides(train_rec, args.label_overrides)

    assert not ({r['patient'] for r in train_rec} & {r['patient'] for r in val_rec})
    assert not ({r['patient'] for r in train_rec} & {r['patient'] for r in test_rec})
    assert not ({r['patient'] for r in val_rec} & {r['patient'] for r in test_rec})

    train_rec = subsample_patients(train_rec, args.label_fraction, seed=args.fraction_seed)
    if args.undersample_zero == 'permanent':
        train_rec = undersample_grade0(train_rec, seed=args.seed, ratio=args.undersample_ratio)
    if args.limit > 0:
        train_rec = train_rec[:args.limit]
        val_rec = val_rec[:args.limit]
        test_rec = test_rec[:args.limit]

    classes = NATIVE_SCORES
    num_classes = len(classes)
    counts = class_counts(train_rec, classes)
    j_counts = joint_class_counts(train_rec, BE_JOINTS, classes)

    aug_seed = args.seed if args.seed_aug else None
    worker_init = partial(seed_augmentation_worker, args.seed) if (args.seed_aug and args.workers > 0) else None
    n_views = 2 if args.supcon_weight > 0 else 1
    train_ds = BEDataset(train_rec, classes, args.image_size, get_transform('train', args.aug, aug_seed), args.mask_root,
                         views=n_views)
    val_ds = BEDataset(val_rec, classes, args.image_size, get_transform('val', args.aug), args.mask_root)
    test_ds = BEDataset(test_rec, classes, args.image_size, get_transform('test', args.aug), args.mask_root)

    if args.undersample_zero == 'rotate':
        sample_keys = {(r['case'], r['joint']): i for i, r in enumerate(train_rec)}

    def build_train_loader(epoch: int = 0) -> DataLoader:
        if args.undersample_zero == 'rotate':
            sub_rec = undersample_grade0(train_rec, seed=args.seed + epoch, ratio=args.undersample_ratio)
            sub_idx = [sample_keys[(r['case'], r['joint'])] for r in sub_rec]
            ds = torch.utils.data.Subset(train_ds, sub_idx)
            sub_labels = train_ds.labels[sub_idx]
        else:
            ds = train_ds
            sub_labels = train_ds.labels
        if args.oversample > 0:
            sampler = make_sampler(sub_labels, num_classes, power=args.oversample, seed=args.seed + epoch)
            return DataLoader(ds, batch_size=args.batch_size, sampler=sampler,
                              num_workers=args.workers, pin_memory=True, drop_last=False, worker_init_fn=worker_init)
        gen = torch.Generator().manual_seed(args.seed + epoch)
        return DataLoader(ds, batch_size=args.batch_size, shuffle=True, generator=gen,
                          num_workers=args.workers, pin_memory=True, drop_last=False, worker_init_fn=worker_init)

    train_loader = build_train_loader(0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    steps_per_epoch = max(1, len(train_loader))
    epochs = max(args.epochs, math.ceil(args.min_steps / steps_per_epoch))

    meta = {
        **vars(args),
        'resolved_epochs': epochs,
        'steps_per_epoch': steps_per_epoch,
        'n_train_joints': len(train_ds),
        'n_train_patients': len({r['patient'] for r in train_rec}),
        'n_val_joints': len(val_ds),
        'n_test_joints': len(test_ds),
        'classes': list(classes),
        'train_class_counts': counts.tolist(),
    }
    with open(out_dir / 'args.json', 'w') as f:
        json.dump(meta, f, indent=2)

    in_chans = 2 if args.mask_root is not None else 1
    model = build_model(args.model, num_classes=num_classes, head=args.head, pretrained=args.pretrained,
                        in_chans=in_chans, drop_rate=args.drop_rate)
    if args.init_from:
        if args.keep_head:
            load_full(model, args.init_from)
        else:
            load_backbone(model, args.init_from)
    if args.freeze_backbone:
        freeze_backbone(model)
    if args.grad_checkpointing:
        if hasattr(model, 'set_grad_checkpointing'):
            model.set_grad_checkpointing(True)
        elif isinstance(model, CumulativeLinkWrapper) and hasattr(model.backbone, 'set_grad_checkpointing'):
            model.backbone.set_grad_checkpointing(True)
    model = model.to(device)
    proj = ProjectionHead(model.num_features).to(device) if args.supcon_weight > 0 else None

    cls_w = None
    if args.effective_beta > 0.0:
        cls_w = effective_number_weights(counts, beta=args.effective_beta).to(device)
    elif args.class_weight:
        cls_w = make_class_weights(counts).to(device)

    drw_cls_w = effective_number_weights(counts, beta=0.9999).to(device) if args.ldam_drw else None
    ldam_m = ldam_margins(counts).to(device) if args.ldam_drw else None

    pos_w = torch.tensor([counts[0] / max(counts[1:].sum(), 1)], dtype=torch.float32, device=device) if args.class_weight else None
    sev_w = make_class_weights(counts[1:]).to(device) if args.class_weight else None
    tau_vec = logit_adjustment(counts, tau=args.logit_adjust).to(device) if args.logit_adjust > 0 else None
    joint_tau_map = joint_logit_adjustment(j_counts, tau=args.joint_logit_adjust) if args.joint_logit_adjust > 0 else None
    corn_tau = corn_logit_adjustment(counts, tau=args.logit_adjust).to(device) if (args.logit_adjust > 0 and args.head in ('corn', 'softmax_corn')) else None

    if args.freeze_backbone:
        param_groups = head_parameters(model)
    elif args.layer_decay < 1.0:
        param_groups = vit_layer_decay_groups(model, args.lr, args.weight_decay, args.layer_decay)
    else:
        param_groups = list(model.parameters()) + (list(proj.parameters()) if proj is not None else [])
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)
    warmup_epochs = int(round(epochs * args.warmup))
    if warmup_epochs > 0:
        warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup_epochs)
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs - warmup_epochs), eta_min=args.lr * 0.01)
        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [warmup, cosine], milestones=[warmup_epochs])
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=args.lr * 0.01)
    scaler = torch.amp.GradScaler(device.type, enabled=(args.amp and device.type == 'cuda'))

    best_score = -math.inf
    best_epoch = -1
    best_val_metrics = None
    history = []
    ckpt_path = out_dir / 'model_best.pt'

    for epoch in range(epochs):
        t0 = time.time()
        set_train_mode(model, args.freeze_backbone)
        if proj is not None:
            proj.train()
        if epoch > 0 and (args.undersample_zero == 'rotate' or args.oversample > 0):
            train_loader = build_train_loader(epoch)

        # For LDAM-DRW (RQ2-14): enable deferred re-weighting in the second half of training
        active_cls_w = (drw_cls_w if epoch >= epochs // 2 else None) if args.ldam_drw else cls_w

        def batch_loss_fn(logits_b, labels_b, joints_b, weights_b):
            return compute_loss(
                logits_b, labels_b, args.head, num_classes,
                cls_w=active_cls_w, pos_w=pos_w, sev_w=sev_w,
                tau_vec=tau_vec, joint_tau_map=joint_tau_map, joints=joints_b, corn_tau=corn_tau,
                sample_weights=weights_b, focal_gamma=args.focal_gamma,
                soft_label_sigma=args.soft_label_sigma, disagreement_soft_label=args.disagreement_soft_label,
                ldam_m=ldam_m, corn_reg_weight=args.corn_reg_weight,
                label_smoothing=args.label_smoothing, gce_q=args.gce_q,
            )

        running_loss = 0.0
        n_seen = 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            if proj is not None:
                loss = supcon_step(model, proj, batch, device, args.amp and device.type == 'cuda', batch_loss_fn,
                                   args.supcon_weight, args.supcon_temp, args.supcon_alpha, num_classes)
            else:
                img = batch['img'].to(device, non_blocking=True)
                labels = batch['label'].to(device, non_blocking=True)
                w = batch['weight'].to(device, non_blocking=True) if 'weight' in batch else None
                with torch.autocast(device_type=device.type, enabled=(args.amp and device.type == 'cuda')):
                    logits = model(img)
                    loss = batch_loss_fn(logits, labels, batch['joint'], w)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            bs = len(batch['label'])
            running_loss += float(loss.item()) * bs
            n_seen += bs

        scheduler.step()
        train_loss = running_loss / max(n_seen, 1)

        val_df = predict(model, val_loader, args.head, device, classes)
        val_m = compute_metrics(val_df['gt'].to_numpy(), val_df['pred'].to_numpy(), classes)
        dt = time.time() - t0

        row = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_bacc4': val_m['bacc4'],
            'val_bacc5': val_m['bacc5'],
            'val_qwk': val_m['qwk'],
            'val_macro_f1': val_m['macro_f1'],
            'val_recall_ge2': val_m['recall_ge2'],
            'lr': optimizer.param_groups[-1]['lr'],
            'sec': dt,
        }
        history.append(row)

        score = val_m[args.monitor]
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_val_metrics = val_m
            torch.save({'model': model.state_dict(), 'epoch': epoch, 'val': val_m, 'args': meta}, ckpt_path)

        print(
            f"[{args.name}] ep {epoch:02d}/{epochs-1:02d} "
            f"loss={train_loss:.4f} val_bacc4={val_m['bacc4']:.4f} val_qwk={val_m['qwk']:.4f} "
            f"val_rec>=2={val_m['recall_ge2']:.4f} ({dt:.1f}s)"
            + (' *' if epoch == best_epoch else ''),
            flush=True,
        )

    pd.DataFrame(history).to_csv(out_dir / 'history.csv', index=False)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model'])

    val_df, val_logits, val_labels = predict(model, val_loader, args.head, device, classes, return_logits=True)
    test_df, test_logits, test_labels = predict(model, test_loader, args.head, device, classes, return_logits=True)
    test_df.to_csv(out_dir / 'test_predictions.csv', index=False)

    np.savez_compressed(
        out_dir / 'val_logits.npz',
        logits=val_logits,
        labels=val_labels,
        gt=val_df['gt'].to_numpy(),
        pred=val_df['pred'].to_numpy(),
        case=val_df['case'].to_numpy(dtype=str),
        patient=val_df['patient'].to_numpy(dtype=str),
        joint=val_df['joint'].to_numpy(dtype=str),
    )
    np.savez_compressed(
        out_dir / 'test_logits.npz',
        logits=test_logits,
        labels=test_labels,
        gt=test_df['gt'].to_numpy(),
        pred=test_df['pred'].to_numpy(),
        case=test_df['case'].to_numpy(dtype=str),
        patient=test_df['patient'].to_numpy(dtype=str),
        joint=test_df['joint'].to_numpy(dtype=str),
    )

    test_m = compute_metrics(test_df['gt'].to_numpy(), test_df['pred'].to_numpy(), classes)
    ci = patient_bootstrap_ci(
        test_df['patient'].to_numpy(),
        test_df['gt'].to_numpy(),
        test_df['pred'].to_numpy(),
        n_boot=args.n_boot,
        seed=args.seed,
    ) if args.n_boot > 0 else {}

    result = {
        'best_epoch': best_epoch,
        'val': best_val_metrics,
        'test': test_m,
        'test_ci95_patient_bootstrap': ci,
    }
    with open(out_dir / 'metrics.json', 'w') as f:
        json.dump(result, f, indent=2)

    print(
        f"[{args.name}] BEST ep={best_epoch} | "
        f"VAL bacc4={best_val_metrics['bacc4']:.4f} qwk={best_val_metrics['qwk']:.4f} | "
        f"TEST bacc4={test_m['bacc4']:.4f} bacc5={test_m['bacc5']:.4f} "
        f"qwk={test_m['qwk']:.4f} f1={test_m['macro_f1']:.4f} rec>=2={test_m['recall_ge2']:.4f}",
        flush=True,
    )
    return result


if __name__ == '__main__':
    main()
