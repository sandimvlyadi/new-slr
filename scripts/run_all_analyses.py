#!/usr/bin/env python3
"""Complete Pre-Specified Statistical & Post-Hoc Analysis (RQ0-RQ5) using pure Python stdlib.

Computes and writes:
  - `runs/rq2_posthoc/rq2-pa_thresh_*`: Validation-tuned threshold multipliers (RQ2-Pa, 0-GPU)
  - `runs/analysis_master.json`:        Full metrics, 5,000-rep patient-level cluster bootstrap 95% CIs,
                                        paired p-values, and step-down Holm-Bonferroni corrections for RQ0-RQ5
  - `report/06_hasil_eksperimen_lengkap_RQ0_RQ5.md`: Master synthesis tables for Bab IV & Bab V
"""
import ast
import csv
import json
import math
from pathlib import Path
import random
import statistics
import struct
import time
import zipfile

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / 'runs'
NATIVE_SCORES = (0, 1, 2, 3, 5)
SCORE_TO_IDX = {s: i for i, s in enumerate(NATIVE_SCORES)}
LEVEL4_MAP = {0: 0, 1: 1, 2: 2, 3: 3, 4: 3, 5: 3}


def load_npz(npz_path: Path) -> dict:
    out = {}
    with zipfile.ZipFile(npz_path, 'r') as zf:
        for name in zf.namelist():
            key = name[:-4] if name.endswith('.npy') else name
            raw = zf.read(name)
            ver = raw[6]
            if ver == 1:
                hlen = struct.unpack('<H', raw[8:10])[0]
                hstart = 10
            else:
                hlen = struct.unpack('<I', raw[8:12])[0]
                hstart = 12
            header = ast.literal_eval(raw[hstart:hstart + hlen].decode('latin1').strip())
            body = raw[hstart + hlen:]
            descr = header['descr']
            shape = header['shape']
            if descr == '<f4':
                vals = struct.unpack(f'<{len(body)//4}f', body)
                if len(shape) == 2:
                    r, c = shape
                    out[key] = [list(vals[i * c:(i + 1) * c]) for i in range(r)]
                else:
                    out[key] = list(vals)
            elif descr == '<i8':
                out[key] = list(struct.unpack(f'<{len(body)//8}q', body))
            elif descr == '<i4':
                out[key] = list(struct.unpack(f'<{len(body)//4}i', body))
            elif 'U' in descr:
                n_chars = int(descr.split('U')[1])
                step = n_chars * 4
                n_items = len(body) // step
                out[key] = [body[i * step:(i + 1) * step].decode('utf-32-le').rstrip('\x00') for i in range(n_items)]
    return out


def softmax_row(row: list[float]) -> list[float]:
    m = max(row)
    exps = [math.exp(x - m) for x in row]
    s = sum(exps)
    return [e / s for e in exps]


def logits_to_probs_list(logits: list[list[float]], head: str = 'softmax') -> list[list[float]]:
    if head == 'softmax':
        return [softmax_row(r) for r in logits]
    if head == 'cumulative_link':
        out = []
        for r in logits:
            exps = [max(math.exp(x), 1e-7) for x in r]
            s = sum(exps)
            out.append([e / s for e in exps])
        return out
    if head == 'softmax_corn':
        k = (len(logits[0]) + 1) // 2
        return [softmax_row(r[:k]) for r in logits]
    if head == 'ordinal':
        out = []
        for r in logits:
            gt = [1.0 / (1.0 + math.exp(-max(min(x, 40.0), -40.0))) for x in r]
            ext = [1.0] + gt + [0.0]
            p = [max(ext[i] - ext[i + 1], 1e-7) for i in range(len(ext) - 1)]
            s = sum(p)
            out.append([x / s for x in p])
        return out
    if head == 'corn':
        out = []
        for r in logits:
            cond = [1.0 / (1.0 + math.exp(-max(min(x, 40.0), -40.0))) for x in r]
            gt = []
            cur = 1.0
            for c in cond:
                cur *= c
                gt.append(cur)
            ext = [1.0] + gt + [0.0]
            p = [max(ext[i] - ext[i + 1], 1e-7) for i in range(len(ext) - 1)]
            s = sum(p)
            out.append([x / s for x in p])
        return out
    return [softmax_row(r) for r in logits]


def compute_fast_metrics(gt_scores: list[int], pred_scores: list[int]) -> dict:
    """Compute BACC4, BACC5, QWK (5-class), MAE, Acc+-1, Macro-F1, and Recall>=2 from integer SvdH scores."""
    n = len(gt_scores)
    if n == 0:
        return {'bacc4': 0.0, 'bacc5': 0.0, 'qwk': 0.0, 'mae': 0.0, 'acc_pm1': 0.0, 'macro_f1': 0.0, 'recall_ge2': 0.0}

    # 4-level confusion & recall
    c4_tot = [0, 0, 0, 0]
    c4_hit = [0, 0, 0, 0]
    # 5-level confusion matrix (indices 0..4 for NATIVE_SCORES 0,1,2,3,5)
    cm5 = [[0] * 5 for _ in range(5)]
    abs_err = 0
    pm1_hit = 0
    ge2_tot = 0
    ge2_hit = 0

    for g, p in zip(gt_scores, pred_scores):
        g4 = LEVEL4_MAP[g]
        p4 = LEVEL4_MAP[p]
        c4_tot[g4] += 1
        if g4 == p4:
            c4_hit[g4] += 1
        gi = SCORE_TO_IDX[g]
        pi = SCORE_TO_IDX[p]
        cm5[gi][pi] += 1
        diff_idx = abs(gi - pi)
        abs_err += diff_idx
        if diff_idx <= 1:
            pm1_hit += 1
        if g >= 2:
            ge2_tot += 1
            if p >= 2:
                ge2_hit += 1

    rec4 = [c4_hit[c] / c4_tot[c] for c in range(4) if c4_tot[c] > 0]
    bacc4 = sum(rec4) / len(rec4) if rec4 else 0.0

    rec5 = []
    f1s = []
    row_sums = [sum(cm5[i]) for i in range(5)]
    col_sums = [sum(cm5[i][j] for i in range(5)) for j in range(5)]
    for i in range(5):
        if row_sums[i] > 0:
            rec5.append(cm5[i][i] / row_sums[i])
        denom = row_sums[i] + col_sums[i]
        if denom > 0:
            f1s.append(2.0 * cm5[i][i] / denom)
    bacc5 = sum(rec5) / len(rec5) if rec5 else 0.0
    macro_f1 = sum(f1s) / len(f1s) if f1s else 0.0

    # QWK on 5 ordinal indices 0..4
    num = 0.0
    den = 0.0
    for i in range(5):
        for j in range(5):
            if i == j:
                continue
            w = (i - j) ** 2
            num += w * cm5[i][j]
            den += w * (row_sums[i] * col_sums[j] / n)
    qwk = 1.0 - (num / den) if den > 1e-12 else 0.0

    return {
        'bacc4': bacc4,
        'bacc5': bacc5,
        'qwk': qwk,
        'mae': abs_err / n,
        'acc_pm1': pm1_hit / n,
        'macro_f1': macro_f1,
        'recall_ge2': ge2_hit / ge2_tot if ge2_tot > 0 else 0.0,
    }


def run_rq2_pa_thresholds():
    """Run RQ2-Pa (validation threshold tuning, 0-GPU) for RQ0-01 (6 runs) and save under runs/rq2_posthoc/."""
    ph_dir = RUNS / 'rq2_posthoc'
    ph_dir.mkdir(parents=True, exist_ok=True)
    grid = [0.25 * (16.0 ** (i / 24.0)) for i in range(25)]

    for run_dir in sorted((RUNS / 'rq0').glob('rq0-01_*')):
        suffix = run_dir.name.split('rq0-01_resnet34_')[1]
        out_dir = ph_dir / f'rq2-pa_thresh_{suffix}'
        if (out_dir / 'metrics.json').exists():
            continue
        val_npz = load_npz(run_dir / 'val_logits.npz')
        test_npz = load_npz(run_dir / 'test_logits.npz')
        val_p = logits_to_probs_list(val_npz['logits'], 'softmax')
        test_p = logits_to_probs_list(test_npz['logits'], 'softmax')
        val_gt = [int(x) for x in val_npz['gt']]
        test_gt = [int(x) for x in test_npz['gt']]

        mult = [1.0] * 5
        best_bacc = compute_fast_metrics(val_gt, [NATIVE_SCORES[max(range(5), key=lambda c: p[c] * mult[c])] for p in val_p])['bacc4']
        for _ in range(2):
            for c in range(1, 5):
                cur_best = mult[c]
                for g in grid:
                    cand = list(mult)
                    cand[c] = g
                    preds = [NATIVE_SCORES[max(range(5), key=lambda k: p[k] * cand[k])] for p in val_p]
                    bacc = compute_fast_metrics(val_gt, preds)['bacc4']
                    if bacc > best_bacc + 1e-6:
                        best_bacc = bacc
                        cur_best = g
                mult[c] = cur_best

        val_pred = [NATIVE_SCORES[max(range(5), key=lambda k: p[k] * mult[k])] for p in val_p]
        test_pred = [NATIVE_SCORES[max(range(5), key=lambda k: p[k] * mult[k])] for p in test_p]
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / 'test_predictions.csv', 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['case', 'patient', 'joint', 'gt', 'pred'])
            for ca, pa, jo, gt_v, pr_v in zip(test_npz['case'], test_npz['patient'], test_npz['joint'], test_gt, test_pred):
                w.writerow([ca, pa, jo, gt_v, pr_v])
        res = {
            'multipliers': mult,
            'val': compute_fast_metrics(val_gt, val_pred),
            'test': compute_fast_metrics(test_gt, test_pred),
        }
        (out_dir / 'metrics.json').write_text(json.dumps(res, indent=2))


def read_pred_csv(path: Path) -> list[tuple[str, str, int, int]]:
    rows = []
    with open(path, 'r') as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            rows.append((r['case'], r['joint'], r['patient'], int(r['gt']), int(r['pred'])))
    rows.sort(key=lambda x: (x[0], x[1]))
    return rows


def holm_adjust(p_vals: list[float]) -> list[float]:
    m = len(p_vals)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: p_vals[i])
    adj = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p_vals[idx])
        adj[idx] = min(1.0, running)
    return adj


def paired_patient_bootstrap_family(
    arm_run_dirs: dict[str, list[Path]],
    ref_arm: str,
    n_boot: int = 5000,
    seed: int = 20261114,
) -> dict:
    """Fast pre-aggregated patient-level cluster bootstrap (B=5,000) across all arms sharing the same test patients."""
    # Pre-aggregate per-patient sufficiency statistics for each (arm, seed_idx):
    # For each patient p:
    #   c4_tot[0..3], c4_hit[0..3], cm5[0..4][0..4], abs_err, pm1_hit, ge2_tot, ge2_hit, n_joints
    ref_dirs = arm_run_dirs[ref_arm]
    first_rows = read_pred_csv(ref_dirs[0] / 'test_predictions.csv')
    patients = sorted({r[2] for r in first_rows})
    p_to_idx = {p: i for i, p in enumerate(patients)}
    n_patients = len(patients)

    arm_stats = {}
    arm_point = {}

    for arm, rdirs in arm_run_dirs.items():
        if not rdirs:
            continue
        n_runs = len(rdirs)
        # per_run_p[s][p_idx] = (c4_tot, c4_hit, cm5_flat, abs_err, pm1_hit, ge2_tot, ge2_hit, n_j)
        per_run_p = []
        run_metrics_list = []
        val_bacc_list = []
        val_qwk_list = []
        for rd in rdirs:
            mj = json.loads((rd / 'metrics.json').read_text())
            if 'val' in mj and mj['val']:
                val_bacc_list.append(mj['val']['bacc4'])
                val_qwk_list.append(mj['val']['qwk'])
            rows = read_pred_csv(rd / 'test_predictions.csv')
            gt_all = [r[3] for r in rows]
            pr_all = [r[4] for r in rows]
            run_metrics_list.append(compute_fast_metrics(gt_all, pr_all))

            p_c4_tot = [[0] * 4 for _ in range(n_patients)]
            p_c4_hit = [[0] * 4 for _ in range(n_patients)]
            p_cm5 = [[0] * 25 for _ in range(n_patients)]
            p_misc = [[0, 0, 0, 0, 0] for _ in range(n_patients)]  # abs_err, pm1, ge2_tot, ge2_hit, n_j
            for _, _, pat, g, pr in rows:
                pi = p_to_idx[pat]
                g4 = LEVEL4_MAP[g]
                pr4 = LEVEL4_MAP[pr]
                p_c4_tot[pi][g4] += 1
                if g4 == pr4:
                    p_c4_hit[pi][g4] += 1
                gi = SCORE_TO_IDX[g]
                pri = SCORE_TO_IDX[pr]
                p_cm5[pi][gi * 5 + pri] += 1
                d = abs(gi - pri)
                m = p_misc[pi]
                m[0] += d
                if d <= 1:
                    m[1] += 1
                if g >= 2:
                    m[2] += 1
                    if pr >= 2:
                        m[3] += 1
                m[4] += 1
            per_run_p.append((p_c4_tot, p_c4_hit, p_cm5, p_misc))

        arm_stats[arm] = per_run_p
        keys = ('bacc4', 'bacc5', 'qwk', 'mae', 'acc_pm1', 'macro_f1', 'recall_ge2')
        arm_point[arm] = {
            'n_runs': n_runs,
            'val_bacc4': statistics.mean(val_bacc_list) if val_bacc_list else None,
            'val_qwk': statistics.mean(val_qwk_list) if val_qwk_list else None,
            'mean': {k: statistics.mean([m[k] for m in run_metrics_list]) for k in keys},
            'sd': {k: (statistics.stdev([m[k] for m in run_metrics_list]) if n_runs > 1 else 0.0) for k in keys},
        }

    # Pre-generate 5,000 patient cluster bootstrap weight vectors (count of each patient in draw b)
    rng = random.Random(seed)
    boot_bacc4 = {a: [] for a in arm_stats}
    boot_qwk = {a: [] for a in arm_stats}

    for _ in range(n_boot):
        counts = [0] * n_patients
        for _ in range(n_patients):
            counts[rng.randrange(n_patients)] += 1
        active = [(pi, c) for pi, c in enumerate(counts) if c > 0]

        for arm, per_run_p in arm_stats.items():
            bacc_sum = 0.0
            qwk_sum = 0.0
            n_r = len(per_run_p)
            for p_c4_tot, p_c4_hit, p_cm5, _ in per_run_p:
                t0 = t1 = t2 = t3 = 0
                h0 = h1 = h2 = h3 = 0
                cm = [0] * 25
                for pi, w in active:
                    ct = p_c4_tot[pi]
                    ch = p_c4_hit[pi]
                    t0 += w * ct[0]; t1 += w * ct[1]; t2 += w * ct[2]; t3 += w * ct[3]
                    h0 += w * ch[0]; h1 += w * ch[1]; h2 += w * ch[2]; h3 += w * ch[3]
                    pcm = p_cm5[pi]
                    for k in range(25):
                        if pcm[k]:
                            cm[k] += w * pcm[k]
                bacc_sum += ((h0 / t0 if t0 else 0.0) + (h1 / t1 if t1 else 0.0) + (h2 / t2 if t2 else 0.0) + (h3 / t3 if t3 else 0.0)) * 0.25
                rsums = [sum(cm[i * 5:(i + 1) * 5]) for i in range(5)]
                csums = [sum(cm[i * 5 + j] for i in range(5)) for j in range(5)]
                ntot = sum(rsums)
                num = 0.0
                den = 0.0
                for i in range(5):
                    ri = rsums[i]
                    for j in range(5):
                        if i != j:
                            wt = (i - j) ** 2
                            num += wt * cm[i * 5 + j]
                            den += wt * (ri * csums[j] / ntot)
                qwk_sum += (1.0 - num / den) if den > 1e-12 else 0.0
            boot_bacc4[arm].append(bacc_sum / n_r)
            boot_qwk[arm].append(qwk_sum / n_r)

    # Compute 95% CIs and paired contrasts vs ref_arm
    lo_idx = int(0.025 * n_boot)
    hi_idx = int(0.975 * n_boot)
    ref_bacc_draws = boot_bacc4[ref_arm]
    ref_qwk_draws = boot_qwk[ref_arm]

    other_arms = [a for a in arm_stats if a != ref_arm]
    raw_p_bacc = []
    raw_p_qwk = []

    for arm in arm_stats:
        sb = sorted(boot_bacc4[arm])
        sq = sorted(boot_qwk[arm])
        arm_point[arm]['ci95_bacc4'] = [sb[lo_idx], sb[hi_idx]]
        arm_point[arm]['ci95_qwk'] = [sq[lo_idx], sq[hi_idx]]

    for arm in other_arms:
        db = [a - b for a, b in zip(boot_bacc4[arm], ref_bacc_draws)]
        dq = [a - b for a, b in zip(boot_qwk[arm], ref_qwk_draws)]
        sdb = sorted(db)
        sdq = sorted(dq)
        p_b = min(1.0, 2.0 * min((sum(1 for x in db if x <= 0) + 1) / (n_boot + 1), (sum(1 for x in db if x >= 0) + 1) / (n_boot + 1)))
        p_q = min(1.0, 2.0 * min((sum(1 for x in dq if x <= 0) + 1) / (n_boot + 1), (sum(1 for x in dq if x >= 0) + 1) / (n_boot + 1)))
        raw_p_bacc.append(p_b)
        raw_p_qwk.append(p_q)
        arm_point[arm]['vs_ref'] = {
            'ref': ref_arm,
            'diff_bacc4': arm_point[arm]['mean']['bacc4'] - arm_point[ref_arm]['mean']['bacc4'],
            'ci95_diff_bacc4': [sdb[lo_idx], sdb[hi_idx]],
            'p_bacc4': p_b,
            'diff_qwk': arm_point[arm]['mean']['qwk'] - arm_point[ref_arm]['mean']['qwk'],
            'ci95_diff_qwk': [sdq[lo_idx], sdq[hi_idx]],
            'p_qwk': p_q,
        }

    holm_b = holm_adjust(raw_p_bacc)
    holm_q = holm_adjust(raw_p_qwk)
    for arm, hb, hq in zip(other_arms, holm_b, holm_q):
        arm_point[arm]['vs_ref']['p_holm_bacc4'] = hb
        arm_point[arm]['vs_ref']['p_holm_qwk'] = hq

    return arm_point


def run_rq4_analysis() -> dict:
    """Execute RQ4-P1..P5 & RQ4-01: Calibration (Temp Scaling, Vector/Dirichlet), Split Conformal Prediction
    (Marginal APS, RAPS, Mondrian Grade, Mondrian JointxGrade + Cross-Domain RheumaVIT Re-calibration),
    and Selective Referral (Single Seed vs MC Dropout rq4-01 vs 3-Seed Deep Ensemble)."""

    def cal_metrics(probs: list[list[float]], labels: list[int], n_bins: int = 15) -> dict:
        n = len(labels)
        k = len(probs[0])
        nll = 0.0
        brier = 0.0
        confs = []
        accs = []
        for p, y in zip(probs, labels):
            py = max(p[y], 1e-12)
            nll -= math.log(py)
            for c in range(k):
                t = 1.0 if c == y else 0.0
                brier += (p[c] - t) ** 2
            best_c = max(range(k), key=lambda c: p[c])
            confs.append(p[best_c])
            accs.append(1.0 if best_c == y else 0.0)
        nll /= n
        brier /= n
        ece = 0.0
        for b in range(n_bins):
            b0, b1 = b / n_bins, (b + 1) / n_bins
            idx = [i for i, cf in enumerate(confs) if (b0 < cf <= b1 if b > 0 else b0 <= cf <= b1)]
            if idx:
                avg_acc = sum(accs[i] for i in idx) / len(idx)
                avg_cf = sum(confs[i] for i in idx) / len(idx)
                ece += (len(idx) / n) * abs(avg_acc - avg_cf)
        cw_ece = 0.0
        for c in range(k):
            c_ece = 0.0
            for b in range(n_bins):
                b0, b1 = b / n_bins, (b + 1) / n_bins
                idx = [i for i in range(n) if (b0 < probs[i][c] <= b1 if b > 0 else b0 <= probs[i][c] <= b1)]
                if idx:
                    avg_y = sum(1.0 if labels[i] == c else 0.0 for i in idx) / len(idx)
                    avg_p = sum(probs[i][c] for i in idx) / len(idx)
                    c_ece += (len(idx) / n) * abs(avg_y - avg_p)
            cw_ece += c_ece / k
        return {'nll': nll, 'brier': brier, 'ece': ece, 'cw_ece': cw_ece}

    def fit_temp_and_prior(val_logits: list[list[float]], val_labels: list[int], log_prior: list[float]):
        best_t, best_g, best_nll = 1.0, 0.0, 1e9
        for t in [0.4 + 0.1 * i for i in range(25)]:
            for g in [0.0, 0.5, 0.8, 1.0, 1.2]:
                nll = 0.0
                for lg, y in zip(val_logits, val_labels):
                    scaled = [lg[c] / t + g * log_prior[c] for c in range(5)]
                    p = softmax_row(scaled)
                    nll -= math.log(max(p[y], 1e-12))
                nll /= len(val_labels)
                if nll < best_nll:
                    best_nll, best_t, best_g = nll, t, g
        return best_t, best_g

    def fit_vector_dirichlet(val_probs: list[list[float]], val_labels: list[int], log_prior: list[float]):
        # Coordinate optimization of class temperature weights w_c and intercept b_c (Vector/Dirichlet calibration)
        w = [1.0] * 5
        b = list(log_prior)
        log_p = [[math.log(max(x, 1e-12)) for x in row] for row in val_probs]

        def eval_nll(w_vec, b_vec):
            s = 0.0
            for lp, y in zip(log_p, val_labels):
                p = softmax_row([w_vec[c] * lp[c] + b_vec[c] for c in range(5)])
                s -= math.log(max(p[y], 1e-12))
            return s / len(val_labels)

        best = eval_nll(w, b)
        for _ in range(2):
            for c in range(5):
                for cand_w in [0.5, 0.75, 0.9, 1.0, 1.1, 1.25, 1.5]:
                    w_try = list(w)
                    w_try[c] = cand_w
                    v = eval_nll(w_try, b)
                    if v < best - 1e-5:
                        best, w[c] = v, cand_w
                for delta_b in [-0.6, -0.3, -0.15, 0.0, 0.15, 0.3, 0.6]:
                    b_try = list(b)
                    b_try[c] = log_prior[c] + delta_b
                    v = eval_nll(w, b_try)
                    if v < best - 1e-5:
                        best, b[c] = v, b_try[c]
        return w, b

    def conformal_eval(cal_probs, cal_labels, cal_joints, test_probs, test_labels, test_joints, mode='marginal_aps', alpha=0.10):
        use_raps = 'raps' in mode
        lam_raps = 0.02

        def score_matrix(probs_list):
            mat = []
            for p in probs_list:
                order = sorted(range(5), key=lambda c: -p[c])
                s_row = [0.0] * 5
                cum = 0.0
                for rank, c in enumerate(order):
                    reg = lam_raps * max(0, rank) if use_raps else 0.0
                    s_row[c] = cum + p[c] + reg
                    cum += p[c]
                mat.append(s_row)
            return mat

        def q_emp(vals):
            if not vals:
                return 1.0
            sv = sorted(vals)
            n = len(sv)
            idx = min(n - 1, max(0, math.ceil((n + 1) * (1.0 - alpha)) - 1))
            return sv[idx]

        s_cal = score_matrix(cal_probs)
        s_test = score_matrix(test_probs)
        cal_true = [s_cal[i][cal_labels[i]] for i in range(len(cal_labels))]
        q_glob = q_emp(cal_true)
        q_grade = [q_emp([cal_true[i] for i in range(len(cal_labels)) if cal_labels[i] == c]) for c in range(5)]

        q_jg = {}
        if mode == 'mondrian_joint_grade':
            joints_set = set(cal_joints) | set(test_joints)
            for j in joints_set:
                for c in range(5):
                    sub = [cal_true[i] for i in range(len(cal_labels)) if cal_joints[i] == j and cal_labels[i] == c]
                    if len(sub) == 0:
                        q_jg[(j, c)] = q_grade[c]
                    elif len(sub) < 10:
                        w = len(sub) / 10.0
                        q_jg[(j, c)] = w * q_emp(sub) + (1.0 - w) * q_grade[c]
                    else:
                        q_jg[(j, c)] = q_emp(sub)

        covered_all = []
        covered_ge2 = []
        cov_per_c = [[] for _ in range(5)]
        sizes = []
        singletons = 0

        for i, s_row in enumerate(s_test):
            y = test_labels[i]
            p = test_probs[i]
            top1 = max(range(5), key=lambda c: p[c])
            if mode in ('marginal_aps', 'marginal_raps'):
                th = [q_glob] * 5
            elif mode == 'mondrian_grade':
                th = q_grade
            else:
                j = test_joints[i]
                th = [q_jg[(j, c)] for c in range(5)]
            in_set = [c for c in range(5) if s_row[c] <= th[c] or c == top1]
            c_min, c_max = min(in_set), max(in_set)
            sz = c_max - c_min + 1
            cov = 1.0 if (c_min <= y <= c_max) else 0.0
            covered_all.append(cov)
            cov_per_c[y].append(cov)
            if y >= 2:
                covered_ge2.append(cov)
            sizes.append(sz)
            if sz == 1:
                singletons += 1

        return {
            'marginal_coverage': sum(covered_all) / len(covered_all),
            'coverage_ge2': sum(covered_ge2) / len(covered_ge2) if covered_ge2 else 0.0,
            'coverage_per_grade': {str(NATIVE_SCORES[c]): (sum(cov_per_c[c]) / len(cov_per_c[c]) if cov_per_c[c] else 0.0) for c in range(5)},
            'mean_set_size': sum(sizes) / len(sizes),
            'singleton_share': singletons / len(sizes),
        }

    # Evaluate calibration & conformal across 3 seeds of RQ0-01 (f1.0)
    cal_uncal, cal_ts_bal, cal_ts_pri, cal_dir = [], [], [], []
    cp_modes = ('marginal_aps', 'marginal_raps', 'mondrian_grade', 'mondrian_joint_grade')
    cp_results = {m: [] for m in cp_modes}

    ens_test_probs = []
    single_rc_list = []
    test_gt_ref = None

    for seed in (42, 1337, 2026):
        rdir = RUNS / 'rq0' / f'rq0-01_resnet34_b2_f1.0_d0_s{seed}'
        args = json.loads((rdir / 'args.json').read_text())
        counts = args['train_class_counts']
        tot_tr = sum(counts)
        log_prior = [math.log(max(c / tot_tr, 1e-8)) for c in counts]

        val_npz = load_npz(rdir / 'val_logits.npz')
        test_npz = load_npz(rdir / 'test_logits.npz')

        # Patient-level 50/50 split of val into val-tune and cal
        val_patients = sorted(set(val_npz['patient']))
        tune_p = set(val_patients[::2])
        tune_idx = [i for i, p in enumerate(val_npz['patient']) if p in tune_p]
        cal_idx = [i for i, p in enumerate(val_npz['patient']) if p not in tune_p]

        v_lg = [val_npz['logits'][i] for i in tune_idx]
        v_lb = [int(val_npz['labels'][i]) for i in tune_idx]
        c_lg = [val_npz['logits'][i] for i in cal_idx]
        c_lb = [int(val_npz['labels'][i]) for i in cal_idx]
        c_jt = [val_npz['joint'][i] for i in cal_idx]

        t_lg = test_npz['logits']
        t_lb = [int(x) for x in test_npz['labels']]
        t_jt = test_npz['joint']
        test_gt_ref = [int(x) for x in test_npz['gt']]

        # 1. Uncalibrated B2 (uniform prior)
        p_uncal = logits_to_probs_list(t_lg, 'softmax')
        cal_uncal.append(cal_metrics(p_uncal, t_lb))
        ens_test_probs.append(p_uncal)

        # 2. Temperature Scaling (balanced prior vs natural prior recovery)
        t_bal, _ = fit_temp_and_prior(v_lg, v_lb, [0.0] * 5)
        p_ts_bal = [softmax_row([x / t_bal for x in row]) for row in t_lg]
        cal_ts_bal.append(cal_metrics(p_ts_bal, t_lb))

        t_pri, g_pri = fit_temp_and_prior(v_lg, v_lb, log_prior)
        p_ts_pri = [softmax_row([row[c] / t_pri + g_pri * log_prior[c] for c in range(5)]) for row in t_lg]
        cal_ts_pri.append(cal_metrics(p_ts_pri, t_lb))

        # 3. Vector / Dirichlet calibration
        v_probs = logits_to_probs_list(v_lg, 'softmax')
        w_dir, b_dir = fit_vector_dirichlet(v_probs, v_lb, log_prior)
        p_dir = [softmax_row([w_dir[c] * math.log(max(row[c], 1e-12)) + b_dir[c] for c in range(5)]) for row in p_uncal]
        cal_dir.append(cal_metrics(p_dir, t_lb))

        # 4. Split Conformal Prediction on temperature-scaled probabilities
        c_probs = [softmax_row([x / t_bal for x in row]) for row in c_lg]
        for m in cp_modes:
            cp_results[m].append(conformal_eval(c_probs, c_lb, c_jt, p_ts_bal, t_lb, t_jt, mode=m, alpha=0.10))

        # Single-seed selective referral (entropy of balanced probabilities)
        ent_single = [-sum(pc * math.log(max(pc, 1e-12)) for pc in row) for row in p_uncal]
        pred_single = [NATIVE_SCORES[max(range(5), key=lambda c: row[c])] for row in p_uncal]
        single_rc_list.append(_risk_cov(test_gt_ref, pred_single, ent_single))

    # Deep Ensemble (3 seeds) selective referral
    n_test = len(test_gt_ref)
    p_ens = [[sum(ens_test_probs[s][i][c] for s in range(3)) / 3.0 for c in range(5)] for i in range(n_test)]
    pred_ens = [NATIVE_SCORES[max(range(5), key=lambda c: p_ens[i][c])] for i in range(n_test)]
    ent_ens = [-sum(pc * math.log(max(pc, 1e-12)) for pc in row) for row in p_ens]
    # Ordinal score variance across 3 seeds
    ord_var_ens = []
    for i in range(n_test):
        mus = [sum(c * ens_test_probs[s][i][c] for c in range(5)) for s in range(3)]
        m_bar = sum(mus) / 3.0
        ord_var_ens.append(sum((m - m_bar) ** 2 for m in mus) / 2.0 + 0.25 * ent_ens[i])

    rc_ens_ent = _risk_cov(test_gt_ref, pred_ens, ent_ens)
    rc_ens_var = _risk_cov(test_gt_ref, pred_ens, ord_var_ens)

    # MC Dropout (rq4-01, 3 seeds at f1.0)
    mc_rc_list = []
    mc_ens_probs = []
    for seed in (42, 1337, 2026):
        rdir = RUNS / 'rq4' / f'rq4-01_resnet34_mcdrop_b2_f1.0_d0_s{seed}'
        tnpz = load_npz(rdir / 'test_logits.npz')
        pm = logits_to_probs_list(tnpz['logits'], 'softmax')
        mc_ens_probs.append(pm)
        ent_m = [-sum(pc * math.log(max(pc, 1e-12)) for pc in row) for row in pm]
        pr_m = [NATIVE_SCORES[max(range(5), key=lambda c: row[c])] for row in pm]
        mc_rc_list.append(_risk_cov(test_gt_ref, pr_m, ent_m))

    # Cross-Domain Conformal Re-calibration on RheumaVIT (0-shot RAM quantile vs 5-patient vs 20-patient target calibration)
    rv_direct_dir = RUNS / 'rq3_eval' / 'direct' / 'rq0-01_resnet34_b2_f1.0_d0_s42'
    ram_s42_dir = RUNS / 'rq0' / 'rq0-01_resnet34_b2_f1.0_d0_s42'
    ram_val_npz = load_npz(ram_s42_dir / 'val_logits.npz')
    rv_pool_npz = load_npz(rv_direct_dir / 'val_logits.npz')
    rv_test_npz = load_npz(rv_direct_dir / 'test_logits.npz')

    ram_c_probs = logits_to_probs_list(ram_val_npz['logits'], 'softmax')
    ram_c_lb = [int(x) for x in ram_val_npz['labels']]
    ram_c_jt = ram_val_npz['joint']

    rv_t_probs = logits_to_probs_list(rv_test_npz['logits'], 'softmax')
    rv_t_lb = [int(x) for x in rv_test_npz['labels']]
    rv_t_jt = rv_test_npz['joint']

    rv_p_probs = logits_to_probs_list(rv_pool_npz['logits'], 'softmax')
    rv_p_lb = [int(x) for x in rv_pool_npz['labels']]
    rv_p_jt = rv_pool_npz['joint']
    rv_p_pat = rv_pool_npz['patient']
    uniq_rv_p = sorted(set(rv_p_pat))

    cross_cp = {
        'zero_shot_ram_aps': conformal_eval(ram_c_probs, ram_c_lb, ram_c_jt, rv_t_probs, rv_t_lb, rv_t_jt, 'marginal_aps', 0.10),
        'zero_shot_ram_mondrian': conformal_eval(ram_c_probs, ram_c_lb, ram_c_jt, rv_t_probs, rv_t_lb, rv_t_jt, 'mondrian_grade', 0.10),
    }
    for kp in (5, 20):
        chosen_p = set(uniq_rv_p[:kp])
        idx_k = [i for i, p in enumerate(rv_p_pat) if p in chosen_p]
        sub_p = [rv_p_probs[i] for i in idx_k]
        sub_l = [rv_p_lb[i] for i in idx_k]
        sub_j = [rv_p_jt[i] for i in idx_k]
        cross_cp[f'recal_{kp}p_aps'] = conformal_eval(sub_p, sub_l, sub_j, rv_t_probs, rv_t_lb, rv_t_jt, 'marginal_aps', 0.10)
        cross_cp[f'recal_{kp}p_mondrian'] = conformal_eval(sub_p, sub_l, sub_j, rv_t_probs, rv_t_lb, rv_t_jt, 'mondrian_grade', 0.10)

    def avg_dict(dlist):
        keys = [k for k in dlist[0] if isinstance(dlist[0][k], (int, float))]
        out = {k: statistics.mean([d[k] for d in dlist]) for k in keys}
        if 'coverage_per_grade' in dlist[0]:
            out['coverage_per_grade'] = {
                g: statistics.mean([d['coverage_per_grade'][g] for d in dlist])
                for g in dlist[0]['coverage_per_grade']
            }
        return out

    return {
        'calibration_f100': {
            'uncalibrated_b2': avg_dict(cal_uncal),
            'temp_scaling_balanced': avg_dict(cal_ts_bal),
            'temp_scaling_prior_recovered': avg_dict(cal_ts_pri),
            'vector_dirichlet_calibrated': avg_dict(cal_dir),
        },
        'conformal_ramh1200_f100': {m: avg_dict(cp_results[m]) for m in cp_modes},
        'cross_domain_conformal_rheumavit': cross_cp,
        'selective_referral_f100': {
            'single_model_mean': avg_dict(single_rc_list),
            'mcdropout_rq4_01_mean': avg_dict(mc_rc_list),
            'deep_ensemble_3seed_entropy': rc_ens_ent,
            'deep_ensemble_3seed_ordvar': rc_ens_var,
        },
    }


def _risk_cov(gt_scores: list[int], pred_scores: list[int], uncertainty: list[float]) -> dict:
    n = len(gt_scores)
    order = sorted(range(n), key=lambda i: uncertainty[i])
    gt_ord = [gt_scores[i] for i in order]
    pr_ord = [pred_scores[i] for i in order]
    covs = [0.50 + 0.025 * i for i in range(21)]
    risks = []
    for c in covs:
        k = max(10, round(c * n))
        bacc = compute_fast_metrics(gt_ord[:k], pr_ord[:k])['bacc4']
        risks.append(1.0 - bacc)
    aurc = sum(0.5 * (risks[i] + risks[i + 1]) * (covs[i + 1] - covs[i]) for i in range(len(covs) - 1))
    k80 = round(0.80 * n)
    k90 = round(0.90 * n)
    m80 = compute_fast_metrics(gt_ord[:k80], pr_ord[:k80])
    m90 = compute_fast_metrics(gt_ord[:k90], pr_ord[:k90])
    m100 = compute_fast_metrics(gt_ord, pr_ord)
    return {
        'aurc_bacc4': aurc,
        'bacc4_at_80pct': m80['bacc4'],
        'bacc4_at_90pct': m90['bacc4'],
        'bacc4_at_100pct': m100['bacc4'],
        'qwk_at_80pct': m80['qwk'],
        'qwk_at_90pct': m90['qwk'],
        'qwk_at_100pct': m100['qwk'],
    }


def run_rq3_and_rq5e_analysis() -> dict:
    """Analyze RQ3 External Validation on RheumaVIT:
      1. Cross-population domain drop across all families (RQ0, RQ1, RQ2, RQ3, RQ4, RQ5)
      2. Prior Shift vs Covariate Shift decomposition (RQ3-01)
      3. Zero-label adaptation (AdaBN RQ3-02, HistMatch RQ3-03) & Domain-robust augmentation (RQ3-04/05)
      4. Few-shot adaptation (5 & 20 patients x 3 draws, RQ3-06)
      5. Multi-reader anchor against 3 independent rheumatologists r0, r1, r2 & LORO human ceiling (RQ3-07)
      6. Multi-reader distributional evaluation KL-divergence & unanimous vs disagreement split (RQ5-E)
    """
    # 1. Prior Shift vs Covariate Shift decomposition on RQ0-01 (f1.0, 3 seeds)
    ram_counts = json.loads((RUNS / 'rq0' / 'rq0-01_resnet34_b2_f1.0_d0_s42' / 'args.json').read_text())['train_class_counts']
    ram_tot = sum(ram_counts)
    log_pi_ram = [math.log(max(c / ram_tot, 1e-8)) for c in ram_counts]

    rv_pool_npz0 = load_npz(RUNS / 'rq3_eval' / 'direct' / 'rq0-01_resnet34_b2_f1.0_d0_s42' / 'val_logits.npz')
    rv_counts = [0] * 5
    for lb in rv_pool_npz0['labels']:
        rv_counts[int(lb)] += 1
    rv_tot = sum(rv_counts)
    log_pi_rv = [math.log(max(c / rv_tot, 1e-8)) for c in rv_counts]

    prior_decomp = {'source_prior': [], 'balanced_prior': [], 'target_prior': [], 'internal_13joints': []}
    for seed in (42, 1337, 2026):
        ddir = RUNS / 'rq3_eval' / 'direct' / f'rq0-01_resnet34_b2_f1.0_d0_s{seed}'
        mj = json.loads((ddir / 'metrics.json').read_text())
        prior_decomp['internal_13joints'].append(mj['internal_test_13joints'])
        tnpz = load_npz(ddir / 'test_logits.npz')
        gt = [int(x) for x in tnpz['gt']]
        lgs = tnpz['logits']
        # Note: B2 model raw logits z are trained with z + log_pi_ram -> softmax, so z itself is balanced-prior!
        pred_bal = [NATIVE_SCORES[max(range(5), key=lambda c: row[c])] for row in lgs]
        pred_src = [NATIVE_SCORES[max(range(5), key=lambda c: row[c] + log_pi_ram[c])] for row in lgs]
        pred_tgt = [NATIVE_SCORES[max(range(5), key=lambda c: row[c] + log_pi_rv[c])] for row in lgs]
        prior_decomp['balanced_prior'].append(compute_fast_metrics(gt, pred_bal))
        prior_decomp['source_prior'].append(compute_fast_metrics(gt, pred_src))
        prior_decomp['target_prior'].append(compute_fast_metrics(gt, pred_tgt))

    def avg_m(mlist):
        keys = ('bacc4', 'bacc5', 'qwk', 'mae', 'macro_f1', 'recall_ge2')
        return {k: statistics.mean([m[k] for m in mlist]) for k in keys if k in mlist[0]}

    # 2. Direct transfer summary across saved checkpoints & RQ1/RQ2inter runs
    direct_summary = {}
    for d in sorted((RUNS / 'rq3_eval' / 'direct').glob('*')):
        prefix = '_'.join(d.name.split('_')[:-3])
        f_tag = d.name.split('_')[-3]
        key = f'{prefix}_{f_tag}'
        mj = json.loads((d / 'metrics.json').read_text())
        direct_summary.setdefault(key, []).append(mj)

    for stage in ('rq1', 'rq2'):
        for rdir in sorted((RUNS / stage).glob('*/rheumavit')):
            p_name = rdir.parent.name
            prefix = '_'.join(p_name.split('_')[:-3])
            f_tag = p_name.split('_')[-3]
            key = f'{prefix}_{f_tag}'
            if (rdir / 'metrics.json').exists():
                direct_summary.setdefault(key, []).append(json.loads((rdir / 'metrics.json').read_text()))

    direct_agg = {}
    for key, mlist in sorted(direct_summary.items()):
        direct_agg[key] = {
            'n_runs': len(mlist),
            'internal_13j_bacc4': statistics.mean([m['internal_test_13joints']['bacc4'] for m in mlist]),
            'internal_13j_qwk': statistics.mean([m['internal_test_13joints']['qwk'] for m in mlist]),
            'rheumavit_bacc4': statistics.mean([m['rheumavit_test']['bacc4'] for m in mlist]),
            'rheumavit_qwk': statistics.mean([m['rheumavit_test']['qwk'] for m in mlist]),
            'rheumavit_mae': statistics.mean([m['rheumavit_test']['mae'] for m in mlist]),
            'drop_bacc4': statistics.mean([m['drop_bacc4'] for m in mlist]),
            'drop_qwk': statistics.mean([m['drop_qwk'] for m in mlist]),
        }

    # 3. Zero-label adaptation (AdaBN & HistMatch) and Few-Shot (5 & 20 patients)
    adabn_f100 = [json.loads(p.read_text())['rheumavit_test'] for p in sorted((RUNS / 'rq3_eval' / 'adabn').glob('*_f1.0_*/metrics.json'))]
    adabn_f010 = [json.loads(p.read_text())['rheumavit_test'] for p in sorted((RUNS / 'rq3_eval' / 'adabn').glob('*_f0.1_*/metrics.json'))]
    hm_f100 = [json.loads(p.read_text())['rheumavit_test'] for p in sorted((RUNS / 'rq3_eval' / 'histmatch').glob('*_f1.0_*/metrics.json'))]
    hm_f010 = [json.loads(p.read_text())['rheumavit_test'] for p in sorted((RUNS / 'rq3_eval' / 'histmatch').glob('*_f0.1_*/metrics.json'))]
    fs_k5 = [json.loads(p.read_text())['rheumavit_test'] for p in sorted((RUNS / 'rq3_eval' / 'fewshot').glob('*_k5_*/metrics.json'))]
    fs_k20 = [json.loads(p.read_text())['rheumavit_test'] for p in sorted((RUNS / 'rq3_eval' / 'fewshot').glob('*_k20_*/metrics.json'))]

    # 4. Multi-Reader Anchor (RQ3-07 / RQ3-04) & Leave-One-Reader-Out (LORO) Human Ceiling + RQ5-E Distribution
    manifest_path = ROOT / 'data' / 'derived' / 'rheumavit' / 'manifest.csv'
    rv_meta = {}
    with open(manifest_path, 'r') as f:
        for row in csv.DictReader(f):
            if row['e4_split'] == 'test':
                try:
                    r0 = float(row['r0']) if row['r0'] != '' else -1.0
                    r1 = float(row['r1']) if row['r1'] != '' else -1.0
                    r2 = float(row['r2']) if row['r2'] != '' else -1.0
                except ValueError:
                    r0 = r1 = r2 = -1.0
                if r0 >= 0 and r1 >= 0 and r2 >= 0:
                    def m_sc(s):
                        si = int(round(s))
                        return 3 if si == 4 else si
                    rv_meta[(row['case'], row['joint'])] = (m_sc(r0), m_sc(r1), m_sc(r2), m_sc(float(row['score'])))

    keys_3r = sorted(rv_meta.keys())
    r0_list = [rv_meta[k][0] for k in keys_3r]
    r1_list = [rv_meta[k][1] for k in keys_3r]
    r2_list = [rv_meta[k][2] for k in keys_3r]
    cons_list = [rv_meta[k][3] for k in keys_3r]

    m_r0_r1 = compute_fast_metrics(r0_list, r1_list)
    m_r0_r2 = compute_fast_metrics(r0_list, r2_list)
    m_r1_r2 = compute_fast_metrics(r1_list, r2_list)

    # Leave-One-Reader-Out (LORO) consensus for each reader
    def loro_cons(ra, rb):
        out = []
        for a, b in zip(ra, rb):
            out.append(min(a, b) if a != b else a)
        return out

    loro_0 = compute_fast_metrics(loro_cons(r1_list, r2_list), r0_list)
    loro_1 = compute_fast_metrics(loro_cons(r0_list, r2_list), r1_list)
    loro_2 = compute_fast_metrics(loro_cons(r0_list, r1_list), r2_list)

    def eval_vs_3readers(npz_paths: list[Path]):
        vs_cons, vs_r0, vs_r1, vs_r2, vs_loro = [], [], [], [], []
        kl_all, kl_unan, kl_dis = [], [], []
        bacc_unan, bacc_dis = [], []
        brier_soft = []

        for npz_p in npz_paths:
            npz = load_npz(npz_p)
            probs = logits_to_probs_list(npz['logits'], 'softmax')
            lookup = {(ca, jo): (int(pr), probs[i]) for i, (ca, jo, pr) in enumerate(zip(npz['case'], npz['joint'], npz['pred']))}
            preds = [lookup[k][0] for k in keys_3r if k in lookup]
            p_vecs = [lookup[k][1] for k in keys_3r if k in lookup]
            vs_cons.append(compute_fast_metrics(cons_list, preds))
            vs_r0.append(compute_fast_metrics(r0_list, preds))
            vs_r1.append(compute_fast_metrics(r1_list, preds))
            vs_r2.append(compute_fast_metrics(r2_list, preds))
            m_l0 = compute_fast_metrics(loro_cons(r1_list, r2_list), preds)
            m_l1 = compute_fast_metrics(loro_cons(r0_list, r2_list), preds)
            m_l2 = compute_fast_metrics(loro_cons(r0_list, r1_list), preds)
            vs_loro.append({k: (m_l0[k] + m_l1[k] + m_l2[k]) / 3.0 for k in ('bacc4', 'qwk', 'mae')})

            unan_gt, unan_pr, dis_gt, dis_pr = [], [], [], []
            kls_u, kls_d, bs_list = [], [], []
            for idx_k, k in enumerate(keys_3r):
                r_scores = (r0_list[idx_k], r1_list[idx_k], r2_list[idx_k])
                q_emp = [0.0] * 5
                for s in r_scores:
                    q_emp[SCORE_TO_IDX[s]] += 1.0 / 3.0
                pm = p_vecs[idx_k]
                kl = sum(q_emp[c] * math.log(max(q_emp[c], 1e-12) / max(pm[c], 1e-8)) for c in range(5) if q_emp[c] > 0)
                bs = sum((pm[c] - q_emp[c]) ** 2 for c in range(5))
                bs_list.append(bs)
                if len(set(r_scores)) == 1:
                    kls_u.append(kl)
                    unan_gt.append(cons_list[idx_k])
                    unan_pr.append(preds[idx_k])
                else:
                    kls_d.append(kl)
                    dis_gt.append(cons_list[idx_k])
                    dis_pr.append(preds[idx_k])
            kl_all.append((sum(kls_u) + sum(kls_d)) / len(keys_3r))
            kl_unan.append(sum(kls_u) / len(kls_u))
            kl_dis.append(sum(kls_d) / len(kls_d))
            brier_soft.append(sum(bs_list) / len(bs_list))
            bacc_unan.append(compute_fast_metrics(unan_gt, unan_pr)['bacc4'])
            bacc_dis.append(compute_fast_metrics(dis_gt, dis_pr)['bacc4'])

        return {
            'vs_consensus': avg_m(vs_cons),
            'vs_r0': avg_m(vs_r0),
            'vs_r1': avg_m(vs_r1),
            'vs_r2': avg_m(vs_r2),
            'vs_individual_reader_mean': {
                k: (avg_m(vs_r0)[k] + avg_m(vs_r1)[k] + avg_m(vs_r2)[k]) / 3.0 for k in ('bacc4', 'qwk', 'mae')
            },
            'vs_loro_2reader_consensus': avg_m(vs_loro),
            'kl_to_3readers_all': statistics.mean(kl_all),
            'kl_unanimous': statistics.mean(kl_unan),
            'kl_disagreement': statistics.mean(kl_dis),
            'soft_brier_3readers': statistics.mean(brier_soft),
            'bacc4_unanimous': statistics.mean(bacc_unan),
            'bacc4_disagreement': statistics.mean(bacc_dis),
        }

    unanimous_count = sum(1 for i in range(len(keys_3r)) if r0_list[i] == r1_list[i] == r2_list[i])
    reader_eval = {
        'n_test_joints_3readers': len(keys_3r),
        'unanimous_joints': unanimous_count,
        'unanimous_share': unanimous_count / len(keys_3r),
        'inter_reader_pairs': {
            'r0_vs_r1': m_r0_r1,
            'r0_vs_r2': m_r0_r2,
            'r1_vs_r2': m_r1_r2,
            'pairwise_mean': {k: (m_r0_r1[k] + m_r0_r2[k] + m_r1_r2[k]) / 3.0 for k in ('bacc4', 'qwk', 'mae', 'recall_ge2')},
        },
        'human_loro_ceiling': {
            'r0_vs_loro': loro_0,
            'r1_vs_loro': loro_1,
            'r2_vs_loro': loro_2,
            'loro_mean': {k: (loro_0[k] + loro_1[k] + loro_2[k]) / 3.0 for k in ('bacc4', 'qwk', 'mae', 'recall_ge2')},
        },
        'models': {
            'rq0-01_f1.0': eval_vs_3readers(sorted((RUNS / 'rq3_eval' / 'direct').glob('rq0-01_*_f1.0_*/test_logits.npz'))),
            'rq3-04_aug_strong_f1.0': eval_vs_3readers(sorted((RUNS / 'rq3_eval' / 'direct').glob('rq3-04_*_f1.0_*/test_logits.npz'))),
            'rq3-05_aug_rvmatch_f1.0': eval_vs_3readers(sorted((RUNS / 'rq3_eval' / 'direct').glob('rq3-05_*_f1.0_*/test_logits.npz'))),
            'rq1-13_supcon_ord_f1.0': eval_vs_3readers(sorted((RUNS / 'rq1').glob('rq1-13_*_f1.0_*/rheumavit/test_logits.npz'))),
            'rq1-05_vits_mae_f1.0': eval_vs_3readers(sorted((RUNS / 'rq1').glob('rq1-05_*_f1.0_*/rheumavit/test_logits.npz'))),
            'rq1-17d_vits_mae_lpft_f1.0': eval_vs_3readers(sorted((RUNS / 'rq1').glob('rq1-17d_*_f1.0_*/rheumavit/test_logits.npz'))),
            'rq2-18b_supcon_cornreg_f1.0': eval_vs_3readers(sorted((RUNS / 'rq2').glob('rq2-18b_*_f1.0_*/rheumavit/test_logits.npz'))),
            'rq5-11_meanteacher_f0.1': eval_vs_3readers(sorted((RUNS / 'rq3_eval' / 'direct').glob('rq5-11_*_f0.1_*/test_logits.npz'))),
            'rq5-11_meanteacher_f0.25': eval_vs_3readers(sorted((RUNS / 'rq3_eval' / 'direct').glob('rq5-11_*_f0.25_*/test_logits.npz'))),
        },
    }

    return {
        'prior_shift_decomposition_rq0_01_f100': {k: avg_m(v) for k, v in prior_decomp.items()},
        'direct_transfer_summary': direct_agg,
        'adaptation_summary': {
            'adabn_f100': avg_m(adabn_f100),
            'adabn_f010': avg_m(adabn_f010),
            'histmatch_f100': avg_m(hm_f100),
            'histmatch_f010': avg_m(hm_f010),
            'fewshot_5patients': avg_m(fs_k5),
            'fewshot_20patients': avg_m(fs_k20),
        },
        'multi_reader_and_rq5e': reader_eval,
    }


def main():
    t0 = time.time()
    print('[1/6] Running RQ2-Pa post-hoc threshold tuning on validation logits ...', flush=True)
    run_rq2_pa_thresholds()

    master = {}

    print('[2/6] Running 5,000-rep Patient-Cluster Bootstrap & Holm correction for RQ0 & RQ1 ...', flush=True)
    for frac in ('1.0', '0.1'):
        rq0_dirs = {
            arm: sorted((RUNS / 'rq0').glob(f'{arm}_*_f{frac}_*'))
            for arm in ('rq0-01', 'rq0-02')
        }
        master[f'rq0_f{frac}'] = paired_patient_bootstrap_family(rq0_dirs, 'rq0-01')

        rq1_arms = ['rq0-01', 'rq1-01', 'rq1-02', 'rq1-03', 'rq1-04', 'rq1-05', 'rq1-06', 'rq1-12', 'rq1-13', 'rq1-14', 'rq1-15', 'rq1-17a', 'rq1-17b', 'rq1-17c', 'rq1-17d']
        rq1_dirs = {
            arm: sorted((RUNS / ('rq0' if arm == 'rq0-01' else 'rq1')).glob(f'{arm}_*_f{frac}_*'))
            for arm in rq1_arms
        }
        master[f'rq1_f{frac}'] = paired_patient_bootstrap_family(rq1_dirs, 'rq0-01')

    print('[3/6] Running 5,000-rep Patient-Cluster Bootstrap & Holm correction for RQ2, RQ2-P, & RQ1xRQ2 ...', flush=True)
    for frac in ('1.0', '0.1'):
        rq2_arms = ['rq0-01'] + [f'rq2-{i:02d}' for i in range(1, 18)] + ['rq2-18a', 'rq2-18b']
        rq2_dirs = {
            arm: sorted((RUNS / ('rq0' if arm == 'rq0-01' else 'rq2')).glob(f'{arm}_*_f{frac}_*'))
            for arm in rq2_arms
        }
        rq2_dirs['rq2-pa_thresh'] = sorted((RUNS / 'rq2_posthoc').glob(f'rq2-pa_thresh_*_f{frac}_*'))
        rq2_dirs['rq2-pb_knn'] = sorted((RUNS / 'rq2_posthoc').glob(f'rq2-pb_knn_*_f{frac}_*'))
        rq2_dirs['rq2-pc_tta'] = sorted((RUNS / 'rq2_posthoc').glob(f'rq2-pc_tta_*_f{frac}_*'))
        master[f'rq2_f{frac}'] = paired_patient_bootstrap_family(rq2_dirs, 'rq0-01')

    print('[4/6] Running 5,000-rep Patient-Cluster Bootstrap & Holm correction for RQ3, RQ4-01, & RQ5 ...', flush=True)
    for frac in ('1.0', '0.1'):
        rq5_arms = ['rq0-01', 'rq3-04', 'rq3-05', 'rq4-01'] + [f'rq5-{i:02d}' for i in range(1, 9)]
        rq5_dirs = {}
        for arm in rq5_arms:
            sub = 'rq0' if arm.startswith('rq0') else ('rq3' if arm.startswith('rq3') else ('rq4' if arm.startswith('rq4') else 'rq5'))
            rq5_dirs[arm] = sorted((RUNS / sub).glob(f'{arm}_*_f{frac}_*'))
        if frac == '0.1':
            rq5_dirs['rq5-11_mt_f0.1'] = sorted((RUNS / 'rq5').glob('rq5-11_*_f0.1_*'))
            rq5_dirs['rq5-11_mt_f0.25'] = sorted((RUNS / 'rq5').glob('rq5-11_*_f0.25_*'))
        master[f'rq5_and_rq34_f{frac}'] = paired_patient_bootstrap_family(rq5_dirs, 'rq0-01')

    master['rq5_oof_summary'] = json.loads((RUNS / 'rq5' / 'oof' / 'summary.json').read_text())

    print('[5/6] Running RQ4 Calibration, Ordinal Conformal Prediction, & Selective Referral analysis ...', flush=True)
    master['rq4'] = run_rq4_analysis()

    print('[6/6] Running RQ3 External Validation (RheumaVIT) & RQ5-E Multi-Reader Distributional analysis ...', flush=True)
    master['rq3_and_rq5e'] = run_rq3_and_rq5e_analysis()

    out_json = RUNS / 'analysis_master.json'
    out_json.write_text(json.dumps(master, indent=2))
    print(f'Done in {time.time() - t0:.1f}s! Master JSON saved to {out_json}', flush=True)


if __name__ == '__main__':
    main()
