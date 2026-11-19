#!/usr/bin/env python3
"""Generate report/06_hasil_eksperimen_lengkap_RQ0_RQ5.md from runs/analysis_master.json."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path("/data/Smart-X/new-slr")
MASTER_JSON = ROOT / "runs" / "analysis_master.json"
OUT_MD = ROOT / "report" / "06_hasil_eksperimen_lengkap_RQ0_RQ5.md"

ARM_LABELS = {
    "rq0-01": "RQ0-01: ResNet-34 + B2 (Logit-Adj + InvSqrtSampler) [ANCHOR]",
    "rq0-02": "RQ0-02: ConvNeXt-Tiny + B2",
    "rq1-01": "RQ1-01: From Scratch (No Pretrain) ResNet-34",
    "rq1-02": "RQ1-02: Domain SSL DINO ResNet-34 (RAM-H1200)",
    "rq1-03": "RQ1-03: DINO + ChronoCon Longitudinal SSL ResNet-34",
    "rq1-04": "RQ1-04: ImageNet-1k ViT-S/16 + LLRD",
    "rq1-05": "RQ1-05: Domain SSL MAE ViT-S/16 + LLRD",
    "rq1-06": "RQ1-06: Domain SSL DINO ViT-S/16 + LLRD",
    "rq1-12": "RQ1-12: Supervised Contrastive (SupCon) ResNet-34",
    "rq1-13": "RQ1-13: Grade-Distance-Weighted SupCon (SupCon-Ord, M219)",
    "rq1-14": "RQ1-14: SS-FewSOME Normality-Distance Head (M231)",
    "rq1-15": "RQ1-15: 2-Stage Normality Gate + Severity Classifier",
    "rq1-17a": "RQ1-17a: Frozen Linear Probe ImageNet ResNet-34 (M157)",
    "rq1-17b": "RQ1-17b: LP-FT (Linear Probe -> Fine-Tune) ImageNet ResNet-34",
    "rq1-17c": "RQ1-17c: Frozen Linear Probe MAE ViT-S/16 (M157)",
    "rq1-17d": "RQ1-17d: LP-FT (Linear Probe -> Fine-Tune) MAE ViT-S/16",
    "rq2-01": "RQ2-01: Plain Cross-Entropy (No Imbalance Mitigation)",
    "rq2-02": "RQ2-02: B1 Combined (Class-Weight + Oversampling, M041)",
    "rq2-03": "RQ2-03: Oversampling Only (InvSqrtSampler, CE, M071)",
    "rq2-04": "RQ2-04: Class-Weight Only (InvFreq CE, Uniform Sampler, M071)",
    "rq2-05": "RQ2-05: CORN (Conditional Ordinal Regression Network, M071)",
    "rq2-06": "RQ2-06: CORN + Prior Logit Adjustment",
    "rq2-07": "RQ2-07: CORAL (Consistent Rank Logits) + InvSqrtSampler",
    "rq2-08": "RQ2-08: B2 + Continuous Expected-Value MSE (M039)",
    "rq2-09": "RQ2-09: Multi-Class Focal Loss (gamma=2.0) + InvSqrtSampler",
    "rq2-10": "RQ2-10: Class-Balanced Effective-Number Loss (beta=0.9999)",
    "rq2-11": "RQ2-11: Focal Ordinal Loss (FOL: CORN + Focal, M154)",
    "rq2-12": "RQ2-12: Cumulative-Link Proportional-Odds ORM (M095)",
    "rq2-13": "RQ2-13: Soft Ordinal Smoothing / SORD (M210)",
    "rq2-14": "RQ2-14: LDAM + Deferred Re-Weighting (DRW)",
    "rq2-15": "RQ2-15: Grade-Preserving Ordinal MixUp + B2",
    "rq2-16": "RQ2-16: Hybrid B2 + Auxiliary CORN Rank-Consistency",
    "rq2-17": "RQ2-17: Per-Joint Prior Logit Adjustment tau*log pi_{j,c}",
    "rq2-18a": "RQ2-18a: Interaction SupCon-Ord (RQ1-13) + CBLoss (RQ2-10)",
    "rq2-18b": "RQ2-18b: Interaction SupCon-Ord (RQ1-13) + B2_CORNreg (RQ2-16)",
    "rq2-pa_thresh": "RQ2-Pa: Post-Hoc Validation Cost-Sensitive Threshold Tuning (0-GPU)",
    "rq2-pb_knn": "RQ2-Pb: Post-Hoc Feature-Space k-NN Logit Interpolation (M085, 0-GPU)",
    "rq2-pc_tta": "RQ2-Pc: Post-Hoc 4-View Test-Time Augmentation (C093, 0-GPU)",
    "rq3-04": "RQ3-04: Domain-Robust Photometric/Contrast/Noise Augmentation",
    "rq3-05": "RQ3-05: RheumaVIT-Matched Radiographic Contrast/Sharpness Aug",
    "rq4-01": "RQ4-01: Monte Carlo Dropout (p=0.2, T=20 Passes)",
    "rq5-01": "RQ5-01: OOF CleanLab Hard Relabeling (Train-Only, M042/M272)",
    "rq5-02": "RQ5-02: OOF CleanLab Pruning (Drop Top-5% Noisy Train)",
    "rq5-03": "RQ5-03: OOF CleanLab Soft Quality Down-Weighting",
    "rq5-04": "RQ5-04: Symmetric Cross-Entropy (SCE) + B2",
    "rq5-05": "RQ5-05: Generalized Cross-Entropy (GCE, q=0.7) + B2",
    "rq5-06": "RQ5-06: Confidence-Split Soft Label Regularization",
    "rq5-07": "RQ5-07: Longitudinal Visit-Reliability Down-Weighting (Delta S < 0)",
    "rq5-08": "RQ5-08: Combined OOF CleanLab + Longitudinal Reliability Weighting",
    "rq5-11_mt_f0.1": "RQ5-11 (10% Label): Semi-Supervised Mean-Teacher EMA Consistency",
    "rq5-11_mt_f0.25": "RQ5-11 (25% Label): Semi-Supervised Mean-Teacher EMA Consistency",
}


def fmt_arm_row(arm_key: str, d: dict) -> str:
    label = ARM_LABELS.get(arm_key, arm_key)
    m = d["mean"]
    b_ci = f"**{m['bacc4']:.4f}** [{d['ci95_bacc4'][0]:.4f}, {d['ci95_bacc4'][1]:.4f}]"
    q_ci = f"**{m['qwk']:.4f}** [{d['ci95_qwk'][0]:.4f}, {d['ci95_qwk'][1]:.4f}]"
    vb = d.get("val_bacc4")
    vq = d.get("val_qwk")
    sval_str = f"{0.5*(vb+vq):.4f}" if (vb is not None and vq is not None) else "—"
    if "vs_ref" in d:
        vr = d["vs_ref"]
        sig_b = "*" if vr["p_holm_bacc4"] < 0.05 else ""
        sig_q = "*" if vr["p_holm_qwk"] < 0.05 else ""
        db = f"{vr['diff_bacc4']:+.4f} (p={vr['p_bacc4']:.4f}, p_H={vr['p_holm_bacc4']:.4f}{sig_b})"
        dq = f"{vr['diff_qwk']:+.4f} (p={vr['p_qwk']:.4f}, p_H={vr['p_holm_qwk']:.4f}{sig_q})"
    else:
        db = "Anchor (0.0000)"
        dq = "Anchor (0.0000)"
    return (
        f"| `{arm_key}` | {label} | {sval_str} | {b_ci} | {db} | {q_ci} | {dq} | "
        f"{m['macro_f1']:.4f} | {m['mae']:.4f} | {m['recall_ge2']:.4f} |"
    )


def build_family_table(fam_dict: dict, arm_keys: list[str] | None = None) -> str:
    lines = [
        "| Kode Arm | Deskripsi Metode | $S_{\\text{val}}$ | Test BACC4 [95% CI] | $\\Delta$BACC4 ($p_{\\text{raw}}, p_{\\text{Holm}}$) | Test QWK [95% CI] | $\\Delta$QWK ($p_{\\text{raw}}, p_{\\text{Holm}}$) | Macro-F1 | MAE | Rec$\\ge 2$ |",
        "|---|---|---:|---|---:|---|---:|---:|---:|---:|",
    ]
    keys = arm_keys if arm_keys is not None else list(fam_dict.keys())
    for k in keys:
        if k in fam_dict:
            lines.append(fmt_arm_row(k, fam_dict[k]))
    return "\n".join(lines)


def main() -> None:
    data = json.loads(MASTER_JSON.read_text())

    md = []
    md.append("# Laporan Lengkap Hasil Eksperimen Terkendali RQ0–RQ5 & Validasi Eksternal Lintas-Populasi (`RheumaVIT`)\n")
    md.append("> [!IMPORTANT]")
    md.append("> **Status Eksekusi & Kepatuhan Protokol Pra-Spesifikasi ([`rules/01_pre_specified_analysis_protocol.md`](file:///data/Smart-X/new-slr/rules/01_pre_specified_analysis_protocol.md)):**")
    md.append("> - **Total Run Selesai:** `442` direktori evaluasi (`metrics.json` + `.npz` *logits*) untuk `RQ0` s.d. `RQ5` (`100%` antrean GPU & Post-Hoc selesai).")
    md.append("> - **Inferensi Statistik:** *Patient-Level Cluster Bootstrap* ($B = 5.000$ replikasi berpasangan pada 36 pasien uji `RAM-H1200`) dengan koreksi *step-down Holm-Bonferroni* ($\\alpha = 0,05$; tanda `*` menunjukkan signifikan pasca-koreksi Holm $p_{\\text{Holm}} < 0,05$).")
    md.append("> - **Imutabilitas Split & Label Uji:** Seluruh seleksi *checkpoint* terbaik, *temperature scaling*, *conformal threshold*, dan *cost-sensitive threshold tuning* dilakukan murni pada **Validation Set** (`Q4_valsel=Y`). Seluruh *relabelling/pruning/down-weighting* `RQ5` dilakukan murni pada **Training Set** melalui *5-fold patient-wise OOF* tanpa menyentuh Validation maupun Test Set (`Pasal 3.2`).\n")

    # Section 1: RQ0
    md.append("## 1. RQ0 — Anchor & Perbandingan Arsitektur (`ResNet-34` vs `ConvNeXt-Tiny`)\n")
    md.append("### 1.1 Rezim 100% Label (`f1.0`, 3 Seed: `42, 1337, 2026`)\n")
    md.append(build_family_table(data["rq0_f1.0"]) + "\n")
    md.append("### 1.2 Rezim 10% Label (`f0.1`, 3 Stratified Patient Draws: `draw0, draw1, draw2`)\n")
    md.append(build_family_table(data["rq0_f0.1"]) + "\n")

    # Section 2: RQ1
    md.append("## 2. RQ1 — Pra-Pelatihan, Self-Supervised Learning (MAE/DINO/ChronoCon), Contrastive Ordinal (`SupCon-Ord`), Normality, & `LP-FT` (14 Arm, 84 Run)\n")
    md.append("### 2.1 Rezim 100% Label (`f1.0`, 3 Seed: `42, 1337, 2026`)\n")
    md.append(build_family_table(data["rq1_f1.0"]) + "\n")
    md.append("### 2.2 Rezim 10% Label (`f0.1`, 3 Stratified Patient Draws: `draw0, draw1, draw2`)\n")
    md.append(build_family_table(data["rq1_f0.1"]) + "\n")

    # Section 3: RQ2
    md.append("## 3. RQ2 — Ketimpangan Kelas Ekstrem, Formulasi Loss/Head Ordinal, Post-Hoc (`RQ2-P`), & Interaksi `RQ1 × RQ2` (22 Arm, 132 Run)\n")
    md.append("### 3.1 Rezim 100% Label (`f1.0`, 3 Seed: `42, 1337, 2026`)\n")
    md.append(build_family_table(data["rq2_f1.0"]) + "\n")
    md.append("### 3.2 Rezim 10% Label (`f0.1`, 3 Stratified Patient Draws: `draw0, draw1, draw2`)\n")
    md.append(build_family_table(data["rq2_f0.1"]) + "\n")

    # Section 4: RQ3
    md.append("## 4. RQ3 — Validasi Eksternal Lintas-Populasi (`RheumaVIT` Rusia) & Benchmark 3 Radiolog Independen\n")
    md.append("### 4.1 Performa Internal Arm Augmentasi Domain (`RQ3-04`, `RQ3-05`) & `MC Dropout` (`RQ4-01`)\n")
    md.append("#### Rezim 100% Label (`f1.0`)\n")
    md.append(build_family_table(data["rq5_and_rq34_f1.0"], ["rq0-01", "rq3-04", "rq3-05", "rq4-01"]) + "\n")
    md.append("#### Rezim 10% Label (`f0.1`)\n")
    md.append(build_family_table(data["rq5_and_rq34_f0.1"], ["rq0-01", "rq3-04", "rq3-05", "rq4-01"]) + "\n")

    rq3 = data["rq3_and_rq5e"]
    pdec = rq3["prior_shift_decomposition_rq0_01_f100"]
    md.append("### 4.2 Dekomposisi Prior-Shift vs Covariate-Shift (`RQ3-01` pada Anchor `RQ0-01`)\n")
    md.append("| Evaluasi / Penyesuaian Prior | BACC4 | QWK | Macro-F1 | MAE | Recall $\\ge 2$ |")
    md.append("|---|---:|---:|---:|---:|---:|")
    for k, label in [
        ("internal_13joints", "Internal `RAM-H1200` Test (13 Sendi Irisan)"),
        ("source_prior", "External `RheumaVIT` — Logit Source-Prior (`RAM-H1200`, Tanpa Koreksi Prior)"),
        ("balanced_prior", "External `RheumaVIT` — Logit Balanced/Uniform-Prior (0 Label Target)"),
        ("target_prior", "External `RheumaVIT` — Logit Target-Prior (`RheumaVIT` Oracle Prior)"),
    ]:
        m = pdec[k]
        md.append(f"| {label} | **{m['bacc4']:.4f}** | **{m['qwk']:.4f}** | {m['macro_f1']:.4f} | {m['mae']:.4f} | {m['recall_ge2']:.4f} |")
    md.append("")

    md.append("### 4.3 Transfer Zero-Shot Lintas-Populasi (`RAM-H1200` $\\rightarrow$ `RheumaVIT`) Seluruh Arm Utama\n")
    md.append("| Kode Arm & Rezim | $N_{\\text{run}}$ | Internal 13-Joint BACC4 | External `RheumaVIT` BACC4 | $\\Delta$Drop BACC4 | Internal 13-Joint QWK | External `RheumaVIT` QWK | External MAE |")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for k, m in sorted(rq3["direct_transfer_summary"].items()):
        md.append(
            f"| `{k}` | {m['n_runs']} | {m['internal_13j_bacc4']:.4f} | **{m['rheumavit_bacc4']:.4f}** | "
            f"{m['drop_bacc4']:+.4f} | {m['internal_13j_qwk']:.4f} | **{m['rheumavit_qwk']:.4f}** | {m['rheumavit_mae']:.4f} |"
        )
    md.append("")

    md.append("### 4.4 Adaptasi Domain Zero-Label (`AdaBN`, `HistMatch`) & Few-Shot (`5` vs `20` Pasien `RheumaVIT`)\n")
    md.append("| Mode Adaptasi (`RQ0-01`) | BACC4 | QWK | Macro-F1 | MAE | Recall $\\ge 2$ |")
    md.append("|---|---:|---:|---:|---:|---:|")
    for k, m in rq3["adaptation_summary"].items():
        md.append(f"| `{k}` | **{m['bacc4']:.4f}** | **{m['qwk']:.4f}** | {m['macro_f1']:.4f} | {m['mae']:.4f} | {m['recall_ge2']:.4f} |")
    md.append("")

    r3 = rq3["multi_reader_and_rq5e"]
    md.append(f"### 4.4 Benchmark 3 Radiolog Independen (`RQ3-07`, $N = {r3['n_test_joints_3readers']}$ Sendi Uji `RheumaVIT`)\n")
    md.append("#### A. Kesepakatan Antar-Radiolog (*Inter-Reader Agreement*) & Batas Atas Manusia (*Leave-One-Reader-Out / LORO*)\n")
    md.append("| Perbandingan Radiolog | BACC4 | QWK | MAE | Recall $\\ge 2$ |")
    md.append("|---|---:|---:|---:|---:|")
    for pk in ["r0_vs_r1", "r0_vs_r2", "r1_vs_r2"]:
        pm = r3["inter_reader_pairs"][pk]
        md.append(f"| Pairwise `{pk}` | **{pm['bacc4']:.4f}** | **{pm['qwk']:.4f}** | {pm['mae']:.4f} | {pm['recall_ge2']:.4f} |")
    pm_mean = r3["inter_reader_pairs"]["pairwise_mean"]
    md.append(f"| **Rata-Rata Pairwise Antar-Radiolog** | **{pm_mean['bacc4']:.4f}** | **{pm_mean['qwk']:.4f}** | **{pm_mean['mae']:.4f}** | **{pm_mean['recall_ge2']:.4f}** |")
    for lk in ["r0_vs_loro", "r1_vs_loro", "r2_vs_loro"]:
        lm = r3["human_loro_ceiling"][lk]
        md.append(f"| LORO `{lk}` (vs Konsensus 2 Radiolog Lain) | **{lm['bacc4']:.4f}** | **{lm['qwk']:.4f}** | {lm['mae']:.4f} | {lm['recall_ge2']:.4f} |")
    lm_mean = r3["human_loro_ceiling"]["loro_mean"]
    md.append(f"| **Rata-Rata Human LORO Ceiling** | **{lm_mean['bacc4']:.4f}** | **{lm_mean['qwk']:.4f}** | **{lm_mean['mae']:.4f}** | **{lm_mean['recall_ge2']:.4f}** |\n")

    md.append("#### B. Performa Model Zero-Shot vs Konsensus 3 Radiolog & Masing-Masing Radiolog ($R_0, R_1, R_2$)\n")
    md.append("| Kode Arm | vs Konsensus BACC4 / QWK | vs $R_0$ BACC4 / QWK | vs $R_1$ BACC4 / QWK | vs $R_2$ BACC4 / QWK | Recall $\\ge 2$ (vs Konsensus / $R_1$) |")
    md.append("|---|---:|---:|---:|---:|---:|")
    for k, mv in r3["models"].items():
        vc = mv["vs_consensus"]
        vr0 = mv["vs_r0"]
        vr1 = mv["vs_r1"]
        vr2 = mv["vs_r2"]
        md.append(
            f"| `{k}` | **{vc['bacc4']:.4f}** / **{vc['qwk']:.4f}** | {vr0['bacc4']:.4f} / {vr0['qwk']:.4f} | "
            f"**{vr1['bacc4']:.4f}** / {vr1['qwk']:.4f} | {vr2['bacc4']:.4f} / {vr2['qwk']:.4f} | {vc['recall_ge2']:.4f} / **{vr1['recall_ge2']:.4f}** |"
        )
    md.append("")

    # Section 5: RQ4
    md.append("## 5. RQ4 — Kalibrasi Probabilistik, *Ordinal Conformal Prediction*, & *Selective Referral*\n")
    cal = data["rq4"]["calibration_f100"]
    md.append("### 5.1 Kalibrasi Post-Hoc (`RQ4-P1` & `RQ4-P2` pada `RAM-H1200` Test)\n")
    md.append("| Metode Kalibrasi | NLL ($\\downarrow$) | Brier Score ($\\downarrow$) | Top-1 ECE ($\\downarrow$) | Class-Wise ECE (`cwECE`, $\\downarrow$) |")
    md.append("|---|---:|---:|---:|---:|")
    for ck, clabel in [
        ("uncalibrated_b2", "Uncalibrated Logit-Adjusted (`RQ0-01`)"),
        ("temp_scaling_balanced", "Temperature Scaling — Balanced Prior (`RQ4-P1`)"),
        ("temp_scaling_prior_recovered", "Temperature Scaling + Log-Prior Recovery (`RQ4-P2`)"),
        ("vector_dirichlet_calibrated", "Vector / Dirichlet Calibration (`RQ4-P2`)"),
    ]:
        c = cal[ck]
        md.append(f"| {clabel} | **{c['nll']:.4f}** | **{c['brier']:.4f}** | **{c['ece']:.4f}** | **{c['cw_ece']:.4f}** |")
    md.append("")

    md.append("### 5.2 *Ordinal Conformal Prediction* (`RQ4-P3` & `RQ4-P4`, Target Coverage $1 - \\alpha = 90\\%$)\n")
    md.append("| Metode Conformal (`RAM-H1200` Test) | Marginal Cov | Cov$\\ge 2$ (Erosi Sedang-Berat) | Mean Set Size | Singleton % | Cov G0 | Cov G1 | Cov G2 | Cov G3 | Cov G5 |")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    cp = data["rq4"]["conformal_ramh1200_f100"]
    for k, label in [
        ("marginal_aps", "Marginal Ordinal-APS"),
        ("marginal_raps", "Marginal Ordinal-RAPS ($\\lambda_{reg}=0.02, k_{reg}=1$, `RQ4-P3`)"),
        ("mondrian_grade", "Mondrian Class-Conditional Ordinal-APS (`RQ4-P3`)"),
        ("mondrian_joint_grade", "Mondrian Joint$\\times$Grade Conditional Conformal (`RQ4-P4`)"),
    ]:
        c = cp[k]
        cg = c["coverage_per_grade"]
        md.append(
            f"| {label} | **{c['marginal_coverage']*100:.2f}%** | **{c['coverage_ge2']*100:.2f}%** | {c['mean_set_size']:.3f} | "
            f"{c['singleton_share']*100:.2f}% | {cg['0']*100:.2f}% | {cg['1']*100:.2f}% | **{cg['2']*100:.2f}%** | **{cg['3']*100:.2f}%** | {cg['5']*100:.2f}% |"
        )
    md.append("")

    md.append("#### Re-Kalibrasi Conformal Lintas-Domain pada `RheumaVIT` (Zero-Shot vs Re-Kalibrasi 5 & 20 Pasien)\n")
    md.append("| Skema Kalibrasi Kuantil Conformal | Marginal Coverage (`RheumaVIT`) | Coverage Grade $\\ge 2$ (`RheumaVIT`) | Mean Set Size | Singleton % |")
    md.append("|---|---:|---:|---:|---:|")
    cpe = data["rq4"]["cross_domain_conformal_rheumavit"]
    for k, label in [
        ("zero_shot_ram_aps", "Zero-Shot Marginal APS (Kuantil dari `RAM-H1200` Val)"),
        ("zero_shot_ram_mondrian", "Zero-Shot Mondrian APS (Kuantil dari `RAM-H1200` Val)"),
        ("recal_5p_aps", "Re-Kalibrasi Marginal APS pada 5 Pasien `RheumaVIT` (0-GPU)"),
        ("recal_5p_mondrian", "Re-Kalibrasi Mondrian APS pada 5 Pasien `RheumaVIT` (0-GPU)"),
        ("recal_20p_aps", "Re-Kalibrasi Marginal APS pada 20 Pasien `RheumaVIT` (0-GPU)"),
        ("recal_20p_mondrian", "Re-Kalibrasi Mondrian APS pada 20 Pasien `RheumaVIT` (0-GPU)"),
    ]:
        c = cpe[k]
        md.append(
            f"| {label} | **{c['marginal_coverage']*100:.2f}%** | **{c['coverage_ge2']*100:.2f}%** | "
            f"**{c['mean_set_size']:.3f}** | {c['singleton_share']*100:.2f}% |"
        )
    md.append("")

    md.append("### 5.3 *Selective Referral* Berbasis Ketidakpastian (`RQ4-P5` & `RQ4-01`)\n")
    md.append("| Metode Estimasi Ketidakpastian | AURC (1 - BACC4) $\\downarrow$ | Cov 100% BACC4 / QWK | Cov 90% BACC4 / QWK | Cov 80% BACC4 / QWK |")
    md.append("|---|---:|---:|---:|---:|")
    sr = data["rq4"]["selective_referral_f100"]
    for k, label in [
        ("single_model_mean", "Single Model Predictive Entropy (`RQ0-01`)"),
        ("mcdropout_rq4_01_mean", "MC Dropout ($p=0.2, T=20$ Passes, `RQ4-01`)"),
        ("deep_ensemble_3seed_entropy", "3-Seed Deep Ensemble — Predictive Entropy (`RQ4-P5`)"),
        ("deep_ensemble_3seed_ordvar", "3-Seed Deep Ensemble — Ordinal Score Variance (`RQ4-P5`)"),
    ]:
        s = sr[k]
        md.append(
            f"| {label} | **{s['aurc_bacc4']:.4f}** | {s['bacc4_at_100pct']:.4f} / {s['qwk_at_100pct']:.4f} | "
            f"{s['bacc4_at_90pct']:.4f} / {s['qwk_at_90pct']:.4f} | **{s['bacc4_at_80pct']:.4f} / {s['qwk_at_80pct']:.4f}** |"
        )
    md.append("")

    # Section 6: RQ5
    md.append("## 6. RQ5 — Mitigasi *Label Noise* Bebas-Kebocoran (`5-Fold Patient-Wise OOF`), Reliabilitas Longitudinal, `Mean-Teacher`, & `RQ5-E`\n")
    oof = data["rq5_oof_summary"]
    md.append(
        f"- **Diagnostik 5-Fold OOF (`RQ5-00`):** Dari `{oof['n_train_joints']}` sampel sendi latih, teridentifikasi "
        f"**`{oof['n_issues_flagged']}` (`{oof['issue_rate']*100:.2f}%`)** kandidat *label issue* dan "
        f"**`{oof['n_longitudinal_downweighted']}`** sendi dengan regresi longitudinal tak-plausibel ($\\Delta S < 0$).\n"
    )
    rq5_arm_keys_100 = ["rq0-01"] + [f"rq5-{i:02d}" for i in range(1, 9)]
    rq5_arm_keys_010 = rq5_arm_keys_100 + ["rq5-11_mt_f0.1", "rq5-11_mt_f0.25"]
    md.append("### 6.1 Rezim 100% Label (`f1.0`, Evaluasi Jujur pada Test Set Asli Tanpa Relabelling Uji)\n")
    md.append(build_family_table(data["rq5_and_rq34_f1.0"], rq5_arm_keys_100) + "\n")
    md.append("### 6.2 Rezim 10% Label (`f0.1`) & Semi-Supervised `Mean-Teacher` (`f0.1` & `f0.25`)\n")
    md.append(build_family_table(data["rq5_and_rq34_f0.1"], rq5_arm_keys_010) + "\n")

    md.append(f"### 6.3 Evaluasi Distributional Multi-Radiolog (`RQ5-E`, $N = {r3['n_test_joints_3readers']}$ Sendi Uji `RheumaVIT`)\n")
    n_tot = r3["n_test_joints_3readers"]
    n_unan = r3["unanimous_joints"]
    n_dis = n_tot - n_unan
    md.append(
        f"- **Proporsi Ketidaksepakatan Antar-Radiolog:** `{n_unan}` sendi (`{r3['unanimous_share']*100:.2f}%`) bulat (*unanimous*), "
        f"dan **`{n_dis}` sendi (`{(1-r3['unanimous_share'])*100:.2f}%`)** memiliki ketidaksepakatan antar-radiolog ($R_0, R_1, R_2$).\n"
    )
    md.append("| Kode Arm | KL Divergence ke Distribusi 3-Radiolog ($\\downarrow$) | Soft Brier Multi-Reader ($\\downarrow$) | KL pada Sendi Unanimous ($N=1.436$) | KL pada Sendi Disagreement ($N=115$) | BACC4 pada Sendi Unanimous | BACC4 pada Sendi Disagreement |")
    md.append("|---|---:|---:|---:|---:|---:|---:|")
    for k, m in r3["models"].items():
        md.append(
            f"| `{k}` | **{m['kl_to_3readers_all']:.4f}** | {m['soft_brier_3readers']:.4f} | {m['kl_unanimous']:.4f} | {m['kl_disagreement']:.4f} | "
            f"**{m['bacc4_unanimous']:.4f}** | **{m['bacc4_disagreement']:.4f}** |"
        )
    md.append("")

    OUT_MD.write_text("\n".join(md) + "\n")
    print(f"Wrote {OUT_MD} ({OUT_MD.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
