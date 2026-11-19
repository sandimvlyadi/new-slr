"""RQ1-14 / RQ1-15: SS-FewSOME normality model (M197 stage 1) and normality gate."""
import argparse
import json
import math
import time
from functools import partial
from pathlib import Path

import albumentations as A
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .data import DEFAULT_ROOT, NATIVE_SCORES, BEDataset, load_records, subsample_patients
from .metrics import LEVEL4, compute_metrics
from .models import build_model

LEVEL_GRADES = np.array([0, 1, 2, 3])


def cutpaste(img, rng, area=(0.02, 0.15), aspect=(0.3, 3.3)):
    h, w = img.shape
    a = rng.uniform(*area) * h * w
    r = math.exp(rng.uniform(math.log(aspect[0]), math.log(aspect[1])))
    ph, pw = max(1, min(h, round(math.sqrt(a * r)))), max(1, min(w, round(math.sqrt(a / r))))
    y0, x0 = rng.integers(0, h - ph + 1), rng.integers(0, w - pw + 1)
    y1, x1 = rng.integers(0, h - ph + 1), rng.integers(0, w - pw + 1)
    out = img.copy()
    out[y1:y1 + ph, x1:x1 + pw] = img[y0:y0 + ph, x0:x0 + pw]
    mask = np.zeros((h, w), np.float32)
    mask[y1:y1 + ph, x1:x1 + pw] = 1.0
    return out, mask


def _tensor(img):
    return torch.from_numpy(img.astype(np.float32) / 255.0)[None]


class FewSOMEPairs(Dataset):
    def __init__(self, images, seed, p_anom=0.5):
        if len(images) < 2:
            raise ValueError('need at least two normal joints')
        self.images, self.p_anom = images, p_anom
        self.norm = A.Compose([A.RandomBrightnessContrast(0.1, 0.1, p=0.5), A.Sharpen(p=0.5)], seed=seed)
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        j = int(self.rng.integers(len(self.images) - 1))
        j += j >= i
        xi = self.norm(image=self.images[i])['image']
        xj, mask = self.images[j], np.zeros(self.images[j].shape, np.float32)
        if self.rng.random() < self.p_anom:
            xj, mask = cutpaste(xj, self.rng)
        return {'xi': _tensor(xi), 'xj': _tensor(xj), 'mask': torch.from_numpy(mask)[None]}


def seed_pair_worker(seed, worker_id):
    ds = torch.utils.data.get_worker_info().dataset
    ds.rng = np.random.default_rng(seed + worker_id)
    ds.norm.set_random_seed(seed + worker_id)


class PatchNet(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.net = timm.create_model('resnet34', pretrained=pretrained, in_chans=1, features_only=True, out_indices=(2,))

    def forward(self, x):
        return self.net(x)[0]


def patch_targets(mask, size):
    return F.adaptive_max_pool2d(mask, size)[:, 0]


def fewsome_loss(fi, fj, target, eps=1e-6):
    with torch.autocast(fi.device.type, enabled=False):
        y_hat = (1 - F.cosine_similarity(fi.float(), fj.float(), dim=1)).clamp(eps, 1 - eps)
        return F.binary_cross_entropy(y_hat, target.float())


@torch.no_grad()
def patch_maps(net, images, device, batch=64):
    net.eval()
    for k in range(0, len(images), batch):
        x = torch.stack([_tensor(im) for im in images[k:k + batch]]).to(device)
        yield net(x).float()


def centre_of(net, images, device):
    total, n = None, 0
    for f in patch_maps(net, images, device):
        s = f.sum(dim=(0, 2, 3))
        total = s if total is None else total + s
        n += f.shape[0] * f.shape[2] * f.shape[3]
    return total / n


def scores_of(net, images, centre, device):
    out = []
    for f in patch_maps(net, images, device):
        cos = F.cosine_similarity(f, centre[None, :, None, None], dim=1)
        out.append((1 - cos).mean(dim=(1, 2)).cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)


def level4(scores):
    return np.vectorize(LEVEL4.get)(np.asarray(scores, dtype=int))


def score_levels(s, cuts):
    return np.searchsorted(np.asarray(cuts), s, side='left')


def choose_cutpoints(s, gt_scores, n_grid=99):
    s, g = np.asarray(s, dtype=float), level4(gt_scores)
    q = np.unique(np.quantile(s, np.linspace(0.01, 0.99, n_grid)))
    present = [lv for lv in range(4) if (g == lv).any()]
    below = {lv: np.array([(s[g == lv] <= t).sum() for t in q], dtype=float) for lv in present}
    n = {lv: float((g == lv).sum()) for lv in present}
    a, b, c = np.meshgrid(np.arange(len(q)), np.arange(len(q)), np.arange(len(q)), indexing='ij')
    recall = []
    for lv in present:
        cnt = below[lv]
        hit = {0: cnt[a], 1: cnt[b] - cnt[a], 2: cnt[c] - cnt[b], 3: n[lv] - cnt[c]}[lv]
        recall.append(hit / n[lv])
    bacc = np.mean(recall, axis=0)
    bacc[(a > b) | (b > c)] = -1.0
    i = np.unravel_index(int(np.argmax(bacc)), bacc.shape)
    return [float(q[k]) for k in i], float(bacc[i])


def grades_from_scores(s, cuts):
    return LEVEL_GRADES[score_levels(s, cuts)]


def choose_gate(s, gt_scores, positive_grades, n_grid=99):
    q = np.unique(np.quantile(np.asarray(s, dtype=float), np.linspace(0.01, 0.99, n_grid)))
    g = level4(gt_scores)
    best = (-1.0, None)
    for t in q:
        pred = np.where(s <= t, 0, positive_grades)
        bacc = float(np.mean([np.mean(level4(pred)[g == lv] == lv) for lv in range(4) if (g == lv).any()]))
        if bacc > best[0]:
            best = (bacc, float(t))
    return best[1], best[0]


def predictions_frame(records, pred):
    return pd.DataFrame({'case': [r['case'] for r in records], 'patient': [r['patient'] for r in records],
                         'joint': [r['joint'] for r in records], 'gt': [r['score'] for r in records],
                         'pred': np.asarray(pred, dtype=int)})


def write_run(run_dir, args, records, pred, extra):
    m = {s: compute_metrics([r['score'] for r in records[s]], pred[s], NATIVE_SCORES) for s in ('val', 'test')}
    predictions_frame(records['test'], pred['test']).to_csv(run_dir / 'test_predictions.csv', index=False)
    result = {'best_epoch': extra.pop('best_epoch'), 'val': m['val'], 'test': m['test'],
              'test_ci95_patient_bootstrap': None, **extra}
    (run_dir / 'metrics.json').write_text(json.dumps(result, indent=2))


def load_splits(args):
    splits = {s: load_records(args.data_root, s) for s in ('train', 'val', 'test')}
    splits['train'] = subsample_patients(splits['train'], args.label_fraction, args.fraction_seed)
    if args.limit:
        splits = {s: r[:args.limit] for s, r in splits.items()}
    return splits


def images_of(records):
    return list(BEDataset(records, NATIVE_SCORES, 224, None).cache) if records else []


def train(args):
    from .train import seed_everything

    run_dir = Path(args.out) / args.name
    run_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    splits = load_splits(args)
    normal = [r for r in splits['train'] if r['score'] == 0]
    normal_images = images_of(normal)
    pairs = FewSOMEPairs(normal_images, args.seed, args.p_anom)
    loader = DataLoader(pairs, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.workers,
                        pin_memory=True, persistent_workers=args.workers > 0,
                        generator=torch.Generator().manual_seed(args.seed),
                        worker_init_fn=partial(seed_pair_worker, args.seed) if args.workers else None)
    args.epochs = max(args.epochs, math.ceil(args.min_steps / max(len(loader), 1)))
    (run_dir / 'args.json').write_text(json.dumps(vars(args), indent=2))
    net = PatchNet(pretrained=not args.no_pretrained).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler(enabled=args.amp)
    history = []
    for epoch in range(args.epochs):
        net.train()
        losses = []
        for batch in loader:
            xi, xj = batch['xi'].to(device, non_blocking=True), batch['xj'].to(device, non_blocking=True)
            with torch.autocast(device.type, enabled=args.amp):
                f = net(torch.cat([xi, xj]))
                fi, fj = f.chunk(2)
                loss = fewsome_loss(fi, fj, patch_targets(batch['mask'].to(device), f.shape[-2:]))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            losses.append(loss.item())
        history.append({'epoch': epoch, 'train_loss': float(np.mean(losses))})
        pd.DataFrame(history).to_csv(run_dir / 'history.csv', index=False)
    centre = centre_of(net, normal_images, device)
    torch.save({'model': net.state_dict(), 'centre': centre.cpu(), 'epoch': args.epochs - 1}, run_dir / 'normality.pt')
    score = {s: scores_of(net, images_of(splits[s]), centre, device) for s in ('val', 'test')}
    for s in ('val', 'test'):
        predictions_frame(splits[s], np.zeros(len(splits[s]))).drop(columns='pred').assign(score=score[s]) \
            .to_csv(run_dir / f'scores_{s}.csv', index=False)
    cuts, val_bacc4 = choose_cutpoints(score['val'], [r['score'] for r in splits['val']])
    (run_dir / 'cutpoints.json').write_text(json.dumps({'cutpoints': cuts, 'val_bacc4': val_bacc4}, indent=2))
    pred = {s: grades_from_scores(score[s], cuts) for s in ('val', 'test')}
    write_run(run_dir, args, splits, pred, {'best_epoch': args.epochs - 1, 'cutpoints': cuts})


@torch.no_grad()
def classifier_logits(run, records, device, workers, split=None):
    if split is not None:
        npz_path = Path(run) / f'{split}_logits.npz'
        if npz_path.exists():
            return np.load(npz_path)['logits']
    net = build_model('resnet34', len(NATIVE_SCORES), 'softmax', pretrained=False)
    net.load_state_dict(torch.load(Path(run) / 'model_best.pt', map_location='cpu', weights_only=False)['model'])
    net.to(device).eval()
    ds = BEDataset(records, NATIVE_SCORES, 224, None)
    out = [net(b['img'].to(device)).float().cpu() for b in DataLoader(ds, batch_size=64, num_workers=workers)]
    return torch.cat(out).numpy()


def gate(args):
    run_dir = Path(args.out) / args.name
    run_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    (run_dir / 'args.json').write_text(json.dumps(vars(args), indent=2))
    splits = load_splits(args)
    pred, score, positive = {}, {}, {}
    for s in ('val', 'test'):
        df = pd.read_csv(Path(args.normality) / f'scores_{s}.csv')
        score[s] = df['score'].to_numpy()
        logits = classifier_logits(args.classifier, splits[s], device, args.workers, split=s)
        positive[s] = np.asarray(NATIVE_SCORES)[logits[:, 1:].argmax(axis=1) + 1]
    t, val_bacc4 = choose_gate(score['val'], [r['score'] for r in splits['val']], positive['val'])
    (run_dir / 'cutpoints.json').write_text(json.dumps({'gate': t, 'val_bacc4': val_bacc4}, indent=2))
    for s in ('val', 'test'):
        pred[s] = np.where(score[s] <= t, 0, positive[s])
    best_epoch = json.loads((Path(args.classifier) / 'metrics.json').read_text())['best_epoch']
    write_run(run_dir, args, splits, pred, {'best_epoch': best_epoch, 'gate': t})


def get_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('command', choices=['train', 'gate'])
    p.add_argument('--name', required=True)
    p.add_argument('--out', default='runs')
    p.add_argument('--data-root', default=DEFAULT_ROOT)
    p.add_argument('--label-fraction', type=float, default=1.0)
    p.add_argument('--fraction-seed', type=int, default=0)
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--workers', type=int, default=6)
    p.add_argument('--amp', action='store_true')
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--min-steps', type=int, default=2000)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--lr', type=float, default=1e-5)
    p.add_argument('--weight-decay', type=float, default=0.1)
    p.add_argument('--p-anom', type=float, default=0.5)
    p.add_argument('--no-pretrained', action='store_true')
    p.add_argument('--normality')
    p.add_argument('--classifier')
    args = p.parse_args(argv)
    if args.command == 'gate' and not (args.normality and args.classifier):
        p.error('gate needs --normality and --classifier')
    return args


def main(argv=None):
    args = get_args(argv)
    (train if args.command == 'train' else gate)(args)


if __name__ == '__main__':
    main()
