"""RQ2: Imbalance & Ordinality strategies (RQ2-01..18) and post-hoc inference methods (RQ2-P)."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import DEFAULT_ROOT, NATIVE_SCORES, BEDataset, load_records, subsample_patients
from .e6abl import add_holm, contrast, load, paired, prediction_stats
from .metrics import compute_metrics
from .models import build_model, logits_to_probs, pooled_and_logits

FRACTIONS = ('1.0', '0.1')
MIN_DIFF = 0.05
ALPHA = 0.05
BEST = 'best_existing'
BEST_QWK = 'best_existing_qwk'

LABELS = {
    'rq2-01': 'CE polos (sumbu)',
    'rq0-01': 'Balanced softmax (τ=1, rujukan)',
    'rq2-02': 'B1: Ordinal BCE + class weight + oversample',
    'rq2-03': 'Oversampling saja',
    'rq2-04': 'Class weight saja',
    'rq2-05': 'B3: CORN',
    'rq2-06': 'B4: CORN + logit adjustment (modifikasi)',
    'rq2-07': 'B6: Decoupled cRT',
    'rq2-08': 'Undersampling grade 0 bergilir per sendi (modifikasi)',
    'rq2-09': 'Focal loss (γ=2)',
    'rq2-10': 'Focal (γ=2) + oversampling',
    'rq2-11': 'Focal ordinal loss (FOL, M132)',
    'rq2-12': 'Cumulative link / proportional odds (ORM, M089)',
    'rq2-13': 'Soft label ordinal Gaussian (σ=0.5)',
    'rq2-14': 'LDAM-DRW',
    'rq2-15': 'Class-balanced loss (effective number β=0.9999)',
    'rq2-16': 'Balanced softmax + CORN regularizer (modifikasi)',
    'rq2-17': 'Logit adjustment dengan prior per jenis sendi (modifikasi)',
}

BEST_CANDIDATES = (
    'rq0-01', 'rq2-01', 'rq2-02', 'rq2-03', 'rq2-04', 'rq2-05',
    'rq2-07', 'rq2-09', 'rq2-10', 'rq2-12', 'rq2-13', 'rq2-14', 'rq2-15',
)

FAMILIES = {
    'F1': [(a, 'rq2-01') for a in ('rq0-01', 'rq2-02', 'rq2-03', 'rq2-04', 'rq2-05', 'rq2-07', 'rq2-09', 'rq2-12', 'rq2-13')],
    'F2': [('rq2-10', 'rq2-09'), ('rq2-12', 'rq2-05'), ('rq2-14', 'rq0-01'), ('rq2-15', 'rq2-04')],
    'F3': [('rq2-06', 'rq2-05'), ('rq2-06', BEST), ('rq2-08', 'rq2-01'), ('rq2-08', BEST),
           ('rq2-17', 'rq0-01'), ('rq2-17', BEST)],
}


def arm_glob(root, arm, frac):
    sub = 'rq0' if arm.startswith('rq0') else 'rq2'
    return str(Path(root) / sub / f'{arm}_*_f{frac}_d*_s*')


# --------------------------------------------------------- RQ2-P: Post-hoc methods

def tune_class_multipliers(val_probs: np.ndarray, val_gt: np.ndarray, classes=NATIVE_SCORES, n_iter: int = 3) -> np.ndarray:
    """Coordinate search on validation probabilities for class multipliers m_c maximizing BACC4 (0-GPU)."""
    classes_arr = np.asarray(classes, dtype=np.int64)
    mult = np.ones(len(classes), dtype=np.float64)
    grid = np.geomspace(0.25, 4.0, 25)
    best_bacc = compute_metrics(val_gt, classes_arr[(val_probs * mult).argmax(axis=1)], classes)['bacc4']
    for _ in range(n_iter):
        for c in range(1, len(classes)):
            cur_best = mult[c]
            for g in grid:
                cand = mult.copy()
                cand[c] = g
                pred = classes_arr[(val_probs * cand).argmax(axis=1)]
                bacc = compute_metrics(val_gt, pred, classes)['bacc4']
                if bacc > best_bacc + 1e-6:
                    best_bacc = bacc
                    cur_best = g
            mult[c] = cur_best
    return mult


def run_posthoc_thresholds(run_dir: Path, out_dir: Path) -> dict:
    """Apply validation-tuned class threshold multipliers to a saved run's val_logits.npz / test_logits.npz."""
    args = json.loads((run_dir / 'args.json').read_text())
    head = args['head']
    val_npz = np.load(run_dir / 'val_logits.npz')
    test_npz = np.load(run_dir / 'test_logits.npz')

    val_p = logits_to_probs(torch.from_numpy(val_npz['logits']), head).numpy()
    test_p = logits_to_probs(torch.from_numpy(test_npz['logits']), head).numpy()

    mult = tune_class_multipliers(val_p, val_npz['gt'])
    classes_arr = np.asarray(NATIVE_SCORES, dtype=np.int64)
    val_pred = classes_arr[(val_p * mult).argmax(axis=1)]
    test_pred = classes_arr[(test_p * mult).argmax(axis=1)]

    out_dir.mkdir(parents=True, exist_ok=True)
    df_test = pd.DataFrame({
        'case': test_npz['case'],
        'patient': test_npz['patient'],
        'joint': test_npz['joint'],
        'gt': test_npz['gt'],
        'pred': test_pred,
    })
    df_test.to_csv(out_dir / 'test_predictions.csv', index=False)
    res = {
        'multipliers': mult.tolist(),
        'val': compute_metrics(val_npz['gt'], val_pred),
        'test': compute_metrics(test_npz['gt'], test_pred),
    }
    (out_dir / 'metrics.json').write_text(json.dumps(res, indent=2))
    return res


@torch.no_grad()
def run_posthoc_knn_and_tta(run_dir: Path, out_knn_dir: Path, out_tta_dir: Path, data_root: str = DEFAULT_ROOT,
                            k: int = 15, tau: float = 0.1) -> dict:
    """Evaluate cosine k-NN over pooled backbone features (M161) and 4-view TTA (C093) for a checkpoint."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args = json.loads((run_dir / 'args.json').read_text())
    head = args['head']
    model = build_model(args['model'], len(NATIVE_SCORES), head, pretrained=False)
    state = torch.load(run_dir / 'model_best.pt', map_location='cpu', weights_only=False)
    model.load_state_dict(state['model'])
    model.to(device).eval()

    train_rec = subsample_patients(load_records(data_root, 'train'), args['label_fraction'], args['fraction_seed'])
    val_rec = load_records(data_root, 'val')
    test_rec = load_records(data_root, 'test')

    def extract(records, tta: bool = False):
        ds = BEDataset(records, NATIVE_SCORES, args['image_size'], None)
        loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=2)
        feats_list, probs_list, tta_probs_list = [], [], []
        for b in loader:
            x = b['img'].to(device)
            f, lg = pooled_and_logits(model, x)
            p = logits_to_probs(lg, head)
            feats_list.append(F.normalize(f.float(), dim=1).cpu())
            probs_list.append(p.float().cpu())
            if tta:
                views = [
                    x,
                    torch.flip(x, dims=[-1]),
                    (x * 1.10).clamp(0.0, 1.0),
                    (x * 0.90).clamp(0.0, 1.0),
                ]
                p_tta = sum(logits_to_probs(model(v), head).float() for v in views) / len(views)
                tta_probs_list.append(p_tta.cpu())
        return (
            torch.cat(feats_list, dim=0),
            torch.cat(probs_list, dim=0).numpy(),
            torch.cat(tta_probs_list, dim=0).numpy() if tta else None,
            ds.labels,
        )

    tr_f, _, _, tr_y = extract(train_rec, tta=False)
    va_f, va_p, _, _ = extract(val_rec, tta=False)
    te_f, te_p, te_tta_p, _ = extract(test_rec, tta=True)

    tr_onehot = F.one_hot(torch.from_numpy(tr_y), num_classes=len(NATIVE_SCORES)).float()

    def knn_probs(query_f: torch.Tensor) -> np.ndarray:
        sim = query_f @ tr_f.T  # (N, N_train)
        vals, idx = sim.topk(min(k, len(tr_f)), dim=1)
        w = F.softmax(vals / tau, dim=1)  # (N, k)
        picked = tr_onehot[idx]  # (N, k, K)
        return (w[:, :, None] * picked).sum(dim=1).numpy()

    va_knn = knn_probs(va_f)
    te_knn = knn_probs(te_f)

    classes_arr = np.asarray(NATIVE_SCORES, dtype=np.int64)
    val_gt = np.array([r['score'] for r in val_rec])
    test_gt = np.array([r['score'] for r in test_rec])

    # Pick interpolation weight alpha in {0.0, 0.1, 0.2, 0.3, 0.4, 0.5} on validation BACC4
    best_alpha, best_val = 0.0, -1.0
    for alpha in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5):
        comb = (1.0 - alpha) * va_p + alpha * va_knn
        bacc = compute_metrics(val_gt, classes_arr[comb.argmax(axis=1)])['bacc4']
        if bacc > best_val:
            best_val, best_alpha = bacc, alpha

    te_comb = (1.0 - best_alpha) * te_p + best_alpha * te_knn
    knn_pred = classes_arr[te_comb.argmax(axis=1)]
    tta_pred = classes_arr[te_tta_p.argmax(axis=1)]

    for dest, pred_arr, extra in (
        (out_knn_dir, knn_pred, {'alpha': best_alpha, 'k': k}),
        (out_tta_dir, tta_pred, {'views': 4}),
    ):
        dest.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({
            'case': [r['case'] for r in test_rec],
            'patient': [r['patient'] for r in test_rec],
            'joint': [r['joint'] for r in test_rec],
            'gt': test_gt,
            'pred': pred_arr,
        }).to_csv(dest / 'test_predictions.csv', index=False)
        res = {'test': compute_metrics(test_gt, pred_arr), **extra}
        (dest / 'metrics.json').write_text(json.dumps(res, indent=2))

    return {'knn_alpha': best_alpha}
