# Laporan Lengkap Hasil Eksperimen Terkendali RQ0–RQ5 & Validasi Eksternal Lintas-Populasi (`RheumaVIT`)

> [!IMPORTANT]
> **Status Eksekusi & Kepatuhan Protokol Pra-Spesifikasi ([`rules/01_pre_specified_analysis_protocol.md`](file:///data/Smart-X/new-slr/rules/01_pre_specified_analysis_protocol.md)):**
> - **Total Run Selesai:** `442` direktori evaluasi (`metrics.json` + `.npz` *logits*) untuk `RQ0` s.d. `RQ5` (`100%` antrean GPU & Post-Hoc selesai).
> - **Inferensi Statistik:** *Patient-Level Cluster Bootstrap* ($B = 5.000$ replikasi berpasangan pada 36 pasien uji `RAM-H1200`) dengan koreksi *step-down Holm-Bonferroni* ($\alpha = 0,05$; tanda `*` menunjukkan signifikan pasca-koreksi Holm $p_{\text{Holm}} < 0,05$).
> - **Imutabilitas Split & Label Uji:** Seluruh seleksi *checkpoint* terbaik, *temperature scaling*, *conformal threshold*, dan *cost-sensitive threshold tuning* dilakukan murni pada **Validation Set** (`Q4_valsel=Y`). Seluruh *relabelling/pruning/down-weighting* `RQ5` dilakukan murni pada **Training Set** melalui *5-fold patient-wise OOF* tanpa menyentuh Validation maupun Test Set (`Pasal 3.2`).

## 1. RQ0 — Anchor & Perbandingan Arsitektur (`ResNet-34` vs `ConvNeXt-Tiny`)

### 1.1 Rezim 100% Label (`f1.0`, 3 Seed: `42, 1337, 2026`)

| Kode Arm | Deskripsi Metode | $S_{\text{val}}$ | Test BACC4 [95% CI] | $\Delta$BACC4 ($p_{\text{raw}}, p_{\text{Holm}}$) | Test QWK [95% CI] | $\Delta$QWK ($p_{\text{raw}}, p_{\text{Holm}}$) | Macro-F1 | MAE | Rec$\ge 2$ |
|---|---|---:|---|---:|---|---:|---:|---:|---:|
| `rq0-01` | RQ0-01: ResNet-34 + B2 (Logit-Adj + InvSqrtSampler) [ANCHOR] | 0.4584 | **0.4855** [0.4362, 0.5141] | Anchor (0.0000) | **0.4832** [0.3822, 0.5683] | Anchor (0.0000) | 0.4043 | 0.3940 | 0.2998 |
| `rq0-02` | RQ0-02: ConvNeXt-Tiny + B2 | 0.4908 | **0.4964** [0.4445, 0.5255] | +0.0109 (p=0.3243, p_H=0.3243) | **0.5515** [0.4529, 0.6296] | +0.0682 (p=0.0012, p_H=0.0012*) | 0.4242 | 0.3387 | 0.3150 |

### 1.2 Rezim 10% Label (`f0.1`, 3 Stratified Patient Draws: `draw0, draw1, draw2`)

| Kode Arm | Deskripsi Metode | $S_{\text{val}}$ | Test BACC4 [95% CI] | $\Delta$BACC4 ($p_{\text{raw}}, p_{\text{Holm}}$) | Test QWK [95% CI] | $\Delta$QWK ($p_{\text{raw}}, p_{\text{Holm}}$) | Macro-F1 | MAE | Rec$\ge 2$ |
|---|---|---:|---|---:|---|---:|---:|---:|---:|
| `rq0-01` | RQ0-01: ResNet-34 + B2 (Logit-Adj + InvSqrtSampler) [ANCHOR] | 0.2783 | **0.3619** [0.3346, 0.3836] | Anchor (0.0000) | **0.2352** [0.1537, 0.3222] | Anchor (0.0000) | 0.2653 | 0.5515 | 0.2647 |
| `rq0-02` | RQ0-02: ConvNeXt-Tiny + B2 | 0.3385 | **0.3653** [0.3405, 0.3844] | +0.0034 (p=0.8058, p_H=0.8058) | **0.3164** [0.2235, 0.4062] | +0.0812 (p=0.0004, p_H=0.0004*) | 0.2820 | 0.4241 | 0.1991 |

## 2. RQ1 — Pra-Pelatihan, Self-Supervised Learning (MAE/DINO/ChronoCon), Contrastive Ordinal (`SupCon-Ord`), Normality, & `LP-FT` (14 Arm, 84 Run)

### 2.1 Rezim 100% Label (`f1.0`, 3 Seed: `42, 1337, 2026`)

| Kode Arm | Deskripsi Metode | $S_{\text{val}}$ | Test BACC4 [95% CI] | $\Delta$BACC4 ($p_{\text{raw}}, p_{\text{Holm}}$) | Test QWK [95% CI] | $\Delta$QWK ($p_{\text{raw}}, p_{\text{Holm}}$) | Macro-F1 | MAE | Rec$\ge 2$ |
|---|---|---:|---|---:|---|---:|---:|---:|---:|
| `rq0-01` | RQ0-01: ResNet-34 + B2 (Logit-Adj + InvSqrtSampler) [ANCHOR] | 0.4584 | **0.4855** [0.4362, 0.5141] | Anchor (0.0000) | **0.4832** [0.3822, 0.5683] | Anchor (0.0000) | 0.4043 | 0.3940 | 0.2998 |
| `rq1-01` | RQ1-01: From Scratch (No Pretrain) ResNet-34 | 0.4509 | **0.4984** [0.4538, 0.5345] | +0.0129 (p=0.3071, p_H=1.0000) | **0.3980** [0.2764, 0.5099] | -0.0852 (p=0.0004, p_H=0.0056*) | 0.3834 | 0.5009 | 0.3806 |
| `rq1-02` | RQ1-02: Domain SSL DINO ResNet-34 (RAM-H1200) | 0.4412 | **0.4348** [0.3956, 0.4648] | -0.0507 (p=0.0008, p_H=0.0088*) | **0.4276** [0.3132, 0.5211] | -0.0556 (p=0.0004, p_H=0.0056*) | 0.3707 | 0.4086 | 0.2387 |
| `rq1-03` | RQ1-03: DINO + ChronoCon Longitudinal SSL ResNet-34 | 0.4106 | **0.4358** [0.3873, 0.4764] | -0.0497 (p=0.0004, p_H=0.0056*) | **0.4078** [0.2894, 0.5086] | -0.0755 (p=0.0004, p_H=0.0056*) | 0.3523 | 0.4309 | 0.2700 |
| `rq1-04` | RQ1-04: ImageNet-1k ViT-S/16 + LLRD | 0.4279 | **0.4749** [0.4436, 0.4978] | -0.0106 (p=0.4327, p_H=1.0000) | **0.4681** [0.3687, 0.5493] | -0.0152 (p=0.2240, p_H=0.5951) | 0.4033 | 0.3972 | 0.2891 |
| `rq1-05` | RQ1-05: Domain SSL MAE ViT-S/16 + LLRD | 0.4326 | **0.4548** [0.4017, 0.4919] | -0.0307 (p=0.0160, p_H=0.1600) | **0.4500** [0.3150, 0.5635] | -0.0332 (p=0.1488, p_H=0.5951) | 0.3930 | 0.4149 | 0.3166 |
| `rq1-06` | RQ1-06: Domain SSL DINO ViT-S/16 + LLRD | 0.4109 | **0.5070** [0.4633, 0.5381] | +0.0215 (p=0.0704, p_H=0.5631) | **0.4268** [0.3127, 0.5280] | -0.0565 (p=0.0040, p_H=0.0280*) | 0.4076 | 0.4771 | 0.3463 |
| `rq1-12` | RQ1-12: Supervised Contrastive (SupCon) ResNet-34 | 0.4579 | **0.4733** [0.4325, 0.4941] | -0.0122 (p=0.2056, p_H=1.0000) | **0.4959** [0.3967, 0.5787] | +0.0127 (p=0.3779, p_H=0.5951) | 0.4182 | 0.3667 | 0.2937 |
| `rq1-13` | RQ1-13: Grade-Distance-Weighted SupCon (SupCon-Ord, M219) | 0.4775 | **0.5034** [0.4615, 0.5291] | +0.0179 (p=0.0504, p_H=0.4535) | **0.5111** [0.4163, 0.5912] | +0.0278 (p=0.0064, p_H=0.0384*) | 0.4271 | 0.3837 | 0.3509 |
| `rq1-14` | RQ1-14: SS-FewSOME Normality-Distance Head (M231) | 0.1609 | **0.2883** [0.2517, 0.3195] | -0.1972 (p=0.0004, p_H=0.0056*) | **-0.0419** [-0.0514, -0.0309] | -0.5251 (p=0.0004, p_H=0.0056*) | 0.0879 | 2.1919 | 0.6690 |
| `rq1-15` | RQ1-15: 2-Stage Normality Gate + Severity Classifier | 0.2815 | **0.4200** [0.3718, 0.4500] | -0.0655 (p=0.0004, p_H=0.0056*) | **0.1119** [0.0268, 0.2023] | -0.3713 (p=0.0004, p_H=0.0056*) | 0.2108 | 1.0468 | 0.3356 |
| `rq1-17a` | RQ1-17a: Frozen Linear Probe ImageNet ResNet-34 (M157) | 0.3522 | **0.4754** [0.4439, 0.5015] | -0.0101 (p=0.6767, p_H=1.0000) | **0.2307** [0.1576, 0.3040] | -0.2525 (p=0.0004, p_H=0.0056*) | 0.2981 | 0.7715 | 0.4500 |
| `rq1-17b` | RQ1-17b: LP-FT (Linear Probe -> Fine-Tune) ImageNet ResNet-34 | 0.4332 | **0.5012** [0.4620, 0.5261] | +0.0157 (p=0.1980, p_H=1.0000) | **0.5023** [0.4138, 0.5771] | +0.0191 (p=0.1792, p_H=0.5951) | 0.4288 | 0.3897 | 0.3600 |
| `rq1-17c` | RQ1-17c: Frozen Linear Probe MAE ViT-S/16 (M157) | 0.4018 | **0.4932** [0.4423, 0.5237] | +0.0077 (p=0.6019, p_H=1.0000) | **0.3367** [0.2119, 0.4549] | -0.1465 (p=0.0004, p_H=0.0056*) | 0.3612 | 0.5711 | 0.3638 |
| `rq1-17d` | RQ1-17d: LP-FT (Linear Probe -> Fine-Tune) MAE ViT-S/16 | 0.4604 | **0.5011** [0.4520, 0.5367] | +0.0156 (p=0.2232, p_H=1.0000) | **0.4376** [0.3189, 0.5427] | -0.0457 (p=0.0244, p_H=0.1220) | 0.4061 | 0.4698 | 0.3799 |

### 2.2 Rezim 10% Label (`f0.1`, 3 Stratified Patient Draws: `draw0, draw1, draw2`)

| Kode Arm | Deskripsi Metode | $S_{\text{val}}$ | Test BACC4 [95% CI] | $\Delta$BACC4 ($p_{\text{raw}}, p_{\text{Holm}}$) | Test QWK [95% CI] | $\Delta$QWK ($p_{\text{raw}}, p_{\text{Holm}}$) | Macro-F1 | MAE | Rec$\ge 2$ |
|---|---|---:|---|---:|---|---:|---:|---:|---:|
| `rq0-01` | RQ0-01: ResNet-34 + B2 (Logit-Adj + InvSqrtSampler) [ANCHOR] | 0.2783 | **0.3619** [0.3346, 0.3836] | Anchor (0.0000) | **0.2352** [0.1537, 0.3222] | Anchor (0.0000) | 0.2653 | 0.5515 | 0.2647 |
| `rq1-01` | RQ1-01: From Scratch (No Pretrain) ResNet-34 | 0.2245 | **0.3123** [0.2815, 0.3418] | -0.0496 (p=0.0164, p_H=0.1968) | **0.0058** [-0.0308, 0.0496] | -0.2294 (p=0.0004, p_H=0.0056*) | 0.1808 | 1.1085 | 0.2571 |
| `rq1-02` | RQ1-02: Domain SSL DINO ResNet-34 (RAM-H1200) | 0.3332 | **0.3812** [0.3502, 0.4074] | +0.0193 (p=0.2535, p_H=1.0000) | **0.2492** [0.1640, 0.3366] | +0.0140 (p=0.5659, p_H=1.0000) | 0.2673 | 0.5194 | 0.2197 |
| `rq1-03` | RQ1-03: DINO + ChronoCon Longitudinal SSL ResNet-34 | 0.2929 | **0.3642** [0.3268, 0.3979] | +0.0023 (p=0.8918, p_H=1.0000) | **0.2735** [0.1727, 0.3711] | +0.0383 (p=0.0864, p_H=0.5183) | 0.2893 | 0.5339 | 0.2090 |
| `rq1-04` | RQ1-04: ImageNet-1k ViT-S/16 + LLRD | 0.2903 | **0.3755** [0.3262, 0.4249] | +0.0136 (p=0.7067, p_H=1.0000) | **0.2664** [0.1553, 0.3797] | +0.0312 (p=0.3623, p_H=1.0000) | 0.2964 | 0.4968 | 0.2067 |
| `rq1-05` | RQ1-05: Domain SSL MAE ViT-S/16 + LLRD | 0.3248 | **0.4148** [0.3689, 0.4502] | +0.0529 (p=0.0216, p_H=0.2376) | **0.3300** [0.2240, 0.4369] | +0.0948 (p=0.0016, p_H=0.0160*) | 0.3351 | 0.4927 | 0.2166 |
| `rq1-06` | RQ1-06: Domain SSL DINO ViT-S/16 + LLRD | 0.2692 | **0.3848** [0.3299, 0.4260] | +0.0229 (p=0.4475, p_H=1.0000) | **0.2833** [0.1768, 0.3969] | +0.0480 (p=0.1400, p_H=0.6999) | 0.3185 | 0.4751 | 0.1739 |
| `rq1-12` | RQ1-12: Supervised Contrastive (SupCon) ResNet-34 | 0.2811 | **0.3633** [0.3382, 0.3866] | +0.0014 (p=0.8426, p_H=1.0000) | **0.2463** [0.1627, 0.3347] | +0.0110 (p=0.3919, p_H=1.0000) | 0.2616 | 0.5351 | 0.2639 |
| `rq1-13` | RQ1-13: Grade-Distance-Weighted SupCon (SupCon-Ord, M219) | 0.2839 | **0.3600** [0.3382, 0.3792] | -0.0019 (p=0.8518, p_H=1.0000) | **0.2346** [0.1566, 0.3193] | -0.0007 (p=0.9598, p_H=1.0000) | 0.2588 | 0.5635 | 0.2677 |
| `rq1-14` | RQ1-14: SS-FewSOME Normality-Distance Head (M231) | 0.1922 | **0.2712** [0.2309, 0.3103] | -0.0907 (p=0.0048, p_H=0.0624) | **-0.0570** [-0.0693, -0.0422] | -0.2922 (p=0.0004, p_H=0.0056*) | 0.1023 | 1.9281 | 0.6423 |
| `rq1-15` | RQ1-15: 2-Stage Normality Gate + Severity Classifier | 0.1552 | **0.2576** [0.2359, 0.2906] | -0.1043 (p=0.0008, p_H=0.0112*) | **-0.0222** [-0.0301, -0.0130] | -0.2574 (p=0.0004, p_H=0.0056*) | 0.1030 | 2.3122 | 0.7757 |
| `rq1-17a` | RQ1-17a: Frozen Linear Probe ImageNet ResNet-34 (M157) | 0.2595 | **0.3540** [0.3369, 0.3704] | -0.0079 (p=0.6195, p_H=1.0000) | **0.1676** [0.1233, 0.2166] | -0.0677 (p=0.0144, p_H=0.1296) | 0.2445 | 0.7489 | 0.2906 |
| `rq1-17b` | RQ1-17b: LP-FT (Linear Probe -> Fine-Tune) ImageNet ResNet-34 | 0.2955 | **0.3554** [0.3340, 0.3758] | -0.0065 (p=0.5779, p_H=1.0000) | **0.2719** [0.1878, 0.3607] | +0.0367 (p=0.0312, p_H=0.2496) | 0.2700 | 0.5290 | 0.2761 |
| `rq1-17c` | RQ1-17c: Frozen Linear Probe MAE ViT-S/16 (M157) | 0.2004 | **0.3215** [0.2990, 0.3438] | -0.0404 (p=0.0272, p_H=0.2719) | **0.1000** [0.0780, 0.1202] | -0.1352 (p=0.0004, p_H=0.0056*) | 0.1955 | 1.0325 | 0.3608 |
| `rq1-17d` | RQ1-17d: LP-FT (Linear Probe -> Fine-Tune) MAE ViT-S/16 | 0.3194 | **0.4062** [0.3513, 0.4442] | +0.0443 (p=0.1000, p_H=0.8998) | **0.3090** [0.1796, 0.4345] | +0.0738 (p=0.0592, p_H=0.4143) | 0.3423 | 0.4782 | 0.2197 |

## 3. RQ2 — Ketimpangan Kelas Ekstrem, Formulasi Loss/Head Ordinal, Post-Hoc (`RQ2-P`), & Interaksi `RQ1 × RQ2` (22 Arm, 132 Run)

### 3.1 Rezim 100% Label (`f1.0`, 3 Seed: `42, 1337, 2026`)

| Kode Arm | Deskripsi Metode | $S_{\text{val}}$ | Test BACC4 [95% CI] | $\Delta$BACC4 ($p_{\text{raw}}, p_{\text{Holm}}$) | Test QWK [95% CI] | $\Delta$QWK ($p_{\text{raw}}, p_{\text{Holm}}$) | Macro-F1 | MAE | Rec$\ge 2$ |
|---|---|---:|---|---:|---|---:|---:|---:|---:|
| `rq0-01` | RQ0-01: ResNet-34 + B2 (Logit-Adj + InvSqrtSampler) [ANCHOR] | 0.4584 | **0.4855** [0.4362, 0.5141] | Anchor (0.0000) | **0.4832** [0.3822, 0.5683] | Anchor (0.0000) | 0.4043 | 0.3940 | 0.2998 |
| `rq2-01` | RQ2-01: Plain Cross-Entropy (No Imbalance Mitigation) | 0.3895 | **0.4132** [0.3722, 0.4383] | -0.0723 (p=0.0004, p_H=0.0088*) | **0.4241** [0.3235, 0.5071] | -0.0591 (p=0.0012, p_H=0.0204*) | 0.3983 | 0.3532 | 0.2105 |
| `rq2-02` | RQ2-02: B1 Combined (Class-Weight + Oversampling, M041) | 0.3950 | **0.4347** [0.3888, 0.4735] | -0.0508 (p=0.0004, p_H=0.0088*) | **0.4510** [0.3684, 0.5292] | -0.0323 (p=0.0616, p_H=0.8006) | 0.3681 | 0.4521 | 0.2014 |
| `rq2-03` | RQ2-03: Oversampling Only (InvSqrtSampler, CE, M071) | 0.4360 | **0.4493** [0.4029, 0.4859] | -0.0362 (p=0.0016, p_H=0.0208*) | **0.4652** [0.3661, 0.5508] | -0.0180 (p=0.1860, p_H=1.0000) | 0.4093 | 0.4047 | 0.2182 |
| `rq2-04` | RQ2-04: Class-Weight Only (InvFreq CE, Uniform Sampler, M071) | 0.4451 | **0.4463** [0.4008, 0.4769] | -0.0392 (p=0.0004, p_H=0.0088*) | **0.4571** [0.3634, 0.5334] | -0.0261 (p=0.0192, p_H=0.2879) | 0.3963 | 0.3820 | 0.2731 |
| `rq2-05` | RQ2-05: CORN (Conditional Ordinal Regression Network, M071) | 0.4060 | **0.3775** [0.3375, 0.4107] | -0.1080 (p=0.0004, p_H=0.0088*) | **0.3700** [0.2582, 0.4651] | -0.1133 (p=0.0004, p_H=0.0088*) | 0.3602 | 0.3681 | 0.1693 |
| `rq2-06` | RQ2-06: CORN + Prior Logit Adjustment | 0.4828 | **0.4686** [0.4182, 0.5065] | -0.0169 (p=0.2172, p_H=1.0000) | **0.4915** [0.3830, 0.5813] | +0.0083 (p=0.4895, p_H=1.0000) | 0.4076 | 0.3852 | 0.2708 |
| `rq2-07` | RQ2-07: CORAL (Consistent Rank Logits) + InvSqrtSampler | 0.4412 | **0.4618** [0.4184, 0.4870] | -0.0237 (p=0.0448, p_H=0.4031) | **0.4804** [0.3782, 0.5629] | -0.0029 (p=0.7794, p_H=1.0000) | 0.4227 | 0.3757 | 0.3028 |
| `rq2-08` | RQ2-08: B2 + Continuous Expected-Value MSE (M039) | 0.4534 | **0.4263** [0.3863, 0.4572] | -0.0593 (p=0.0004, p_H=0.0088*) | **0.4643** [0.3600, 0.5511] | -0.0190 (p=0.1460, p_H=1.0000) | 0.3742 | 0.3845 | 0.2456 |
| `rq2-09` | RQ2-09: Multi-Class Focal Loss (gamma=2.0) + InvSqrtSampler | 0.4181 | **0.3973** [0.3533, 0.4259] | -0.0882 (p=0.0004, p_H=0.0088*) | **0.3982** [0.2966, 0.4826] | -0.0850 (p=0.0004, p_H=0.0088*) | 0.3705 | 0.3630 | 0.1976 |
| `rq2-10` | RQ2-10: Class-Balanced Effective-Number Loss (beta=0.9999) | 0.4641 | **0.4603** [0.4220, 0.4920] | -0.0252 (p=0.1164, p_H=0.8146) | **0.4556** [0.3665, 0.5326] | -0.0276 (p=0.1412, p_H=1.0000) | 0.4029 | 0.4246 | 0.2807 |
| `rq2-11` | RQ2-11: Focal Ordinal Loss (FOL: CORN + Focal, M154) | 0.4272 | **0.3863** [0.3552, 0.4101] | -0.0992 (p=0.0004, p_H=0.0088*) | **0.4229** [0.3193, 0.5109] | -0.0603 (p=0.0004, p_H=0.0088*) | 0.3525 | 0.3573 | 0.2037 |
| `rq2-12` | RQ2-12: Cumulative-Link Proportional-Odds ORM (M095) | 0.3425 | **0.3610** [0.3273, 0.3854] | -0.1245 (p=0.0004, p_H=0.0088*) | **0.3217** [0.1949, 0.4239] | -0.1615 (p=0.0004, p_H=0.0088*) | 0.2691 | 0.3858 | 0.1625 |
| `rq2-13` | RQ2-13: Soft Ordinal Smoothing / SORD (M210) | 0.3963 | **0.4107** [0.3798, 0.4335] | -0.0748 (p=0.0004, p_H=0.0088*) | **0.4226** [0.3274, 0.5023] | -0.0606 (p=0.0012, p_H=0.0204*) | 0.3952 | 0.3555 | 0.2258 |
| `rq2-14` | RQ2-14: LDAM + Deferred Re-Weighting (DRW) | 0.4501 | **0.4640** [0.4202, 0.4909] | -0.0215 (p=0.0368, p_H=0.3827) | **0.4548** [0.3643, 0.5315] | -0.0285 (p=0.0556, p_H=0.7782) | 0.4016 | 0.4084 | 0.2982 |
| `rq2-15` | RQ2-15: Grade-Preserving Ordinal MixUp + B2 | 0.4558 | **0.4662** [0.4191, 0.4971] | -0.0193 (p=0.0588, p_H=0.4703) | **0.4707** [0.3732, 0.5504] | -0.0125 (p=0.2444, p_H=1.0000) | 0.4105 | 0.3896 | 0.2754 |
| `rq2-16` | RQ2-16: Hybrid B2 + Auxiliary CORN Rank-Consistency | 0.4734 | **0.4843** [0.4347, 0.5125] | -0.0012 (p=0.9602, p_H=1.0000) | **0.4973** [0.3944, 0.5823] | +0.0141 (p=0.3771, p_H=1.0000) | 0.4268 | 0.3904 | 0.3059 |
| `rq2-17` | RQ2-17: Per-Joint Prior Logit Adjustment tau*log pi_{j,c} | 0.4192 | **0.4663** [0.4207, 0.4952] | -0.0192 (p=0.0348, p_H=0.3827) | **0.4986** [0.3945, 0.5848] | +0.0154 (p=0.2040, p_H=1.0000) | 0.3967 | 0.3730 | 0.3043 |
| `rq2-18a` | RQ2-18a: Interaction SupCon-Ord (RQ1-13) + CBLoss (RQ2-10) | 0.4576 | **0.4590** [0.4074, 0.4912] | -0.0265 (p=0.0016, p_H=0.0208*) | **0.4825** [0.3717, 0.5740] | -0.0008 (p=0.9454, p_H=1.0000) | 0.4193 | 0.3738 | 0.2876 |
| `rq2-18b` | RQ2-18b: Interaction SupCon-Ord (RQ1-13) + B2_CORNreg (RQ2-16) | 0.4663 | **0.4801** [0.4291, 0.5148] | -0.0054 (p=0.6155, p_H=1.0000) | **0.5066** [0.4088, 0.5869] | +0.0234 (p=0.0680, p_H=0.8158) | 0.4221 | 0.3771 | 0.3021 |
| `rq2-pa_thresh` | RQ2-Pa: Post-Hoc Validation Cost-Sensitive Threshold Tuning (0-GPU) | 0.4718 | **0.4882** [0.4400, 0.5174] | +0.0027 (p=0.4919, p_H=1.0000) | **0.4577** [0.3656, 0.5383] | -0.0256 (p=0.0008, p_H=0.0144*) | 0.3735 | 0.4295 | 0.3768 |
| `rq2-pb_knn` | RQ2-Pb: Post-Hoc Feature-Space k-NN Logit Interpolation (M085, 0-GPU) | — | **0.4855** [0.4362, 0.5141] | +0.0000 (p=1.0000, p_H=1.0000) | **0.4831** [0.3820, 0.5679] | -0.0002 (p=0.8526, p_H=1.0000) | 0.4042 | 0.3941 | 0.2998 |
| `rq2-pc_tta` | RQ2-Pc: Post-Hoc 4-View Test-Time Augmentation (C093, 0-GPU) | — | **0.4889** [0.4357, 0.5188] | +0.0034 (p=0.4967, p_H=1.0000) | **0.4901** [0.3892, 0.5735] | +0.0069 (p=0.1336, p_H=1.0000) | 0.4085 | 0.3835 | 0.2975 |

### 3.2 Rezim 10% Label (`f0.1`, 3 Stratified Patient Draws: `draw0, draw1, draw2`)

| Kode Arm | Deskripsi Metode | $S_{\text{val}}$ | Test BACC4 [95% CI] | $\Delta$BACC4 ($p_{\text{raw}}, p_{\text{Holm}}$) | Test QWK [95% CI] | $\Delta$QWK ($p_{\text{raw}}, p_{\text{Holm}}$) | Macro-F1 | MAE | Rec$\ge 2$ |
|---|---|---:|---|---:|---|---:|---:|---:|---:|
| `rq0-01` | RQ0-01: ResNet-34 + B2 (Logit-Adj + InvSqrtSampler) [ANCHOR] | 0.2783 | **0.3619** [0.3346, 0.3836] | Anchor (0.0000) | **0.2352** [0.1537, 0.3222] | Anchor (0.0000) | 0.2653 | 0.5515 | 0.2647 |
| `rq2-01` | RQ2-01: Plain Cross-Entropy (No Imbalance Mitigation) | 0.1820 | **0.2844** [0.2740, 0.2908] | -0.0775 (p=0.0004, p_H=0.0088*) | **0.1452** [0.0989, 0.2000] | -0.0900 (p=0.0020, p_H=0.0300*) | 0.2208 | 0.3998 | 0.0282 |
| `rq2-02` | RQ2-02: B1 Combined (Class-Weight + Oversampling, M041) | 0.1890 | **0.2757** [0.2657, 0.2900] | -0.0862 (p=0.0004, p_H=0.0088*) | **0.0960** [0.0561, 0.1428] | -0.1392 (p=0.0004, p_H=0.0088*) | 0.1014 | 0.8678 | 0.1724 |
| `rq2-03` | RQ2-03: Oversampling Only (InvSqrtSampler, CE, M071) | 0.2841 | **0.3292** [0.3127, 0.3456] | -0.0327 (p=0.0064, p_H=0.0576) | **0.2538** [0.1908, 0.3164] | +0.0186 (p=0.2967, p_H=1.0000) | 0.2682 | 0.4616 | 0.1442 |
| `rq2-04` | RQ2-04: Class-Weight Only (InvFreq CE, Uniform Sampler, M071) | 0.2573 | **0.3027** [0.2891, 0.3161] | -0.0592 (p=0.0004, p_H=0.0088*) | **0.2251** [0.1635, 0.2934] | -0.0101 (p=0.6219, p_H=1.0000) | 0.2502 | 0.4141 | 0.0999 |
| `rq2-05` | RQ2-05: CORN (Conditional Ordinal Regression Network, M071) | 0.1626 | **0.2731** [0.2640, 0.2818] | -0.0888 (p=0.0004, p_H=0.0088*) | **0.0775** [0.0499, 0.1088] | -0.1578 (p=0.0004, p_H=0.0088*) | 0.2082 | 0.3999 | 0.0221 |
| `rq2-06` | RQ2-06: CORN + Prior Logit Adjustment | 0.2779 | **0.3268** [0.3148, 0.3405] | -0.0351 (p=0.0024, p_H=0.0264*) | **0.2504** [0.1983, 0.3043] | +0.0152 (p=0.5103, p_H=1.0000) | 0.2487 | 0.4516 | 0.0870 |
| `rq2-07` | RQ2-07: CORAL (Consistent Rank Logits) + InvSqrtSampler | 0.2390 | **0.3207** [0.3046, 0.3330] | -0.0412 (p=0.0004, p_H=0.0088*) | **0.2465** [0.1823, 0.3169] | +0.0112 (p=0.6095, p_H=1.0000) | 0.2560 | 0.4576 | 0.1480 |
| `rq2-08` | RQ2-08: B2 + Continuous Expected-Value MSE (M039) | 0.2517 | **0.2942** [0.2810, 0.3033] | -0.0677 (p=0.0004, p_H=0.0088*) | **0.2038** [0.1437, 0.2716] | -0.0314 (p=0.1100, p_H=1.0000) | 0.2364 | 0.4325 | 0.1076 |
| `rq2-09` | RQ2-09: Multi-Class Focal Loss (gamma=2.0) + InvSqrtSampler | 0.2138 | **0.2848** [0.2747, 0.2942] | -0.0771 (p=0.0004, p_H=0.0088*) | **0.1448** [0.1026, 0.1910] | -0.0904 (p=0.0012, p_H=0.0216*) | 0.2265 | 0.4098 | 0.0610 |
| `rq2-10` | RQ2-10: Class-Balanced Effective-Number Loss (beta=0.9999) | 0.2766 | **0.3188** [0.3041, 0.3323] | -0.0431 (p=0.0036, p_H=0.0360*) | **0.2363** [0.1828, 0.2916] | +0.0010 (p=0.9642, p_H=1.0000) | 0.2544 | 0.4550 | 0.1121 |
| `rq2-11` | RQ2-11: Focal Ordinal Loss (FOL: CORN + Focal, M154) | 0.2193 | **0.2912** [0.2789, 0.3038] | -0.0707 (p=0.0004, p_H=0.0088*) | **0.1924** [0.1323, 0.2588] | -0.0428 (p=0.0240, p_H=0.3119) | 0.2260 | 0.4039 | 0.0572 |
| `rq2-12` | RQ2-12: Cumulative-Link Proportional-Odds ORM (M095) | 0.1426 | **0.2606** [0.2510, 0.2707] | -0.1013 (p=0.0004, p_H=0.0088*) | **0.0342** [0.0059, 0.0671] | -0.2010 (p=0.0004, p_H=0.0088*) | 0.1880 | 0.4042 | 0.0107 |
| `rq2-13` | RQ2-13: Soft Ordinal Smoothing / SORD (M210) | 0.1978 | **0.2777** [0.2702, 0.2872] | -0.0842 (p=0.0004, p_H=0.0088*) | **0.1509** [0.1017, 0.2077] | -0.0844 (p=0.0004, p_H=0.0088*) | 0.2176 | 0.4071 | 0.0420 |
| `rq2-14` | RQ2-14: LDAM + Deferred Re-Weighting (DRW) | 0.2251 | **0.3031** [0.2867, 0.3224] | -0.0588 (p=0.0004, p_H=0.0088*) | **0.1447** [0.1163, 0.1773] | -0.0905 (p=0.0012, p_H=0.0216*) | 0.2349 | 0.5201 | 0.1335 |
| `rq2-15` | RQ2-15: Grade-Preserving Ordinal MixUp + B2 | 0.2807 | **0.3429** [0.3258, 0.3581] | -0.0190 (p=0.0896, p_H=0.5375) | **0.2291** [0.1619, 0.3016] | -0.0061 (p=0.7379, p_H=1.0000) | 0.2680 | 0.5609 | 0.1998 |
| `rq2-16` | RQ2-16: Hybrid B2 + Auxiliary CORN Rank-Consistency | 0.2821 | **0.3493** [0.3280, 0.3659] | -0.0126 (p=0.1492, p_H=0.7459) | **0.2394** [0.1608, 0.3265] | +0.0042 (p=0.7255, p_H=1.0000) | 0.2729 | 0.5040 | 0.2113 |
| `rq2-17` | RQ2-17: Per-Joint Prior Logit Adjustment tau*log pi_{j,c} | 0.3081 | **0.3496** [0.3304, 0.3661] | -0.0123 (p=0.2699, p_H=0.8050) | **0.2069** [0.1171, 0.3061] | -0.0283 (p=0.1480, p_H=1.0000) | 0.2467 | 0.5746 | 0.2555 |
| `rq2-18a` | RQ2-18a: Interaction SupCon-Ord (RQ1-13) + CBLoss (RQ2-10) | 0.3045 | **0.3390** [0.3247, 0.3526] | -0.0229 (p=0.0260, p_H=0.1820) | **0.2386** [0.1703, 0.3110] | +0.0034 (p=0.8186, p_H=1.0000) | 0.2804 | 0.5182 | 0.1953 |
| `rq2-18b` | RQ2-18b: Interaction SupCon-Ord (RQ1-13) + B2_CORNreg (RQ2-16) | 0.3008 | **0.3459** [0.3242, 0.3613] | -0.0160 (p=0.1532, p_H=0.7459) | **0.2606** [0.1887, 0.3335] | +0.0254 (p=0.1988, p_H=1.0000) | 0.2805 | 0.4884 | 0.2410 |
| `rq2-pa_thresh` | RQ2-Pa: Post-Hoc Validation Cost-Sensitive Threshold Tuning (0-GPU) | 0.2916 | **0.3799** [0.3547, 0.4041] | +0.0180 (p=0.0120, p_H=0.0960) | **0.1957** [0.1291, 0.2716] | -0.0395 (p=0.0048, p_H=0.0672) | 0.2469 | 0.7321 | 0.3738 |
| `rq2-pb_knn` | RQ2-Pb: Post-Hoc Feature-Space k-NN Logit Interpolation (M085, 0-GPU) | — | **0.3619** [0.3346, 0.3836] | +0.0000 (p=1.0000, p_H=1.0000) | **0.2351** [0.1532, 0.3223] | -0.0001 (p=0.9386, p_H=1.0000) | 0.2653 | 0.5513 | 0.2639 |
| `rq2-pc_tta` | RQ2-Pc: Post-Hoc 4-View Test-Time Augmentation (C093, 0-GPU) | — | **0.3674** [0.3377, 0.3873] | +0.0055 (p=0.2683, p_H=0.8050) | **0.2581** [0.1706, 0.3484] | +0.0229 (p=0.0012, p_H=0.0216*) | 0.2727 | 0.5124 | 0.2433 |

## 4. RQ3 — Validasi Eksternal Lintas-Populasi (`RheumaVIT` Rusia) & Benchmark 3 Radiolog Independen

### 4.1 Performa Internal Arm Augmentasi Domain (`RQ3-04`, `RQ3-05`) & `MC Dropout` (`RQ4-01`)

#### Rezim 100% Label (`f1.0`)

| Kode Arm | Deskripsi Metode | $S_{\text{val}}$ | Test BACC4 [95% CI] | $\Delta$BACC4 ($p_{\text{raw}}, p_{\text{Holm}}$) | Test QWK [95% CI] | $\Delta$QWK ($p_{\text{raw}}, p_{\text{Holm}}$) | Macro-F1 | MAE | Rec$\ge 2$ |
|---|---|---:|---|---:|---|---:|---:|---:|---:|
| `rq0-01` | RQ0-01: ResNet-34 + B2 (Logit-Adj + InvSqrtSampler) [ANCHOR] | 0.4584 | **0.4855** [0.4362, 0.5141] | Anchor (0.0000) | **0.4832** [0.3822, 0.5683] | Anchor (0.0000) | 0.4043 | 0.3940 | 0.2998 |
| `rq3-04` | RQ3-04: Domain-Robust Photometric/Contrast/Noise Augmentation | 0.4734 | **0.5190** [0.4695, 0.5462] | +0.0335 (p=0.0052, p_H=0.0468*) | **0.5267** [0.4389, 0.5961] | +0.0435 (p=0.0160, p_H=0.0800) | 0.4365 | 0.3885 | 0.3829 |
| `rq3-05` | RQ3-05: RheumaVIT-Matched Radiographic Contrast/Sharpness Aug | 0.4562 | **0.4686** [0.4187, 0.5011] | -0.0169 (p=0.0860, p_H=0.4439) | **0.4816** [0.3767, 0.5688] | -0.0016 (p=0.8822, p_H=1.0000) | 0.3964 | 0.3773 | 0.3127 |
| `rq4-01` | RQ4-01: Monte Carlo Dropout (p=0.2, T=20 Passes) | 0.4571 | **0.4705** [0.4281, 0.4945] | -0.0151 (p=0.2248, p_H=0.8990) | **0.4785** [0.3800, 0.5582] | -0.0047 (p=0.6807, p_H=1.0000) | 0.4100 | 0.3720 | 0.3005 |

#### Rezim 10% Label (`f0.1`)

| Kode Arm | Deskripsi Metode | $S_{\text{val}}$ | Test BACC4 [95% CI] | $\Delta$BACC4 ($p_{\text{raw}}, p_{\text{Holm}}$) | Test QWK [95% CI] | $\Delta$QWK ($p_{\text{raw}}, p_{\text{Holm}}$) | Macro-F1 | MAE | Rec$\ge 2$ |
|---|---|---:|---|---:|---|---:|---:|---:|---:|
| `rq0-01` | RQ0-01: ResNet-34 + B2 (Logit-Adj + InvSqrtSampler) [ANCHOR] | 0.2783 | **0.3619** [0.3346, 0.3836] | Anchor (0.0000) | **0.2352** [0.1537, 0.3222] | Anchor (0.0000) | 0.2653 | 0.5515 | 0.2647 |
| `rq3-04` | RQ3-04: Domain-Robust Photometric/Contrast/Noise Augmentation | 0.3052 | **0.3759** [0.3521, 0.3937] | +0.0140 (p=0.1048, p_H=0.7335) | **0.2774** [0.1903, 0.3733] | +0.0422 (p=0.0036, p_H=0.0288*) | 0.2930 | 0.5693 | 0.3303 |
| `rq3-05` | RQ3-05: RheumaVIT-Matched Radiographic Contrast/Sharpness Aug | 0.2651 | **0.3586** [0.3308, 0.3743] | -0.0033 (p=0.6655, p_H=1.0000) | **0.2284** [0.1480, 0.3152] | -0.0068 (p=0.5487, p_H=1.0000) | 0.2638 | 0.5722 | 0.2777 |
| `rq4-01` | RQ4-01: Monte Carlo Dropout (p=0.2, T=20 Passes) | 0.2867 | **0.3568** [0.3323, 0.3726] | -0.0051 (p=0.4203, p_H=1.0000) | **0.2374** [0.1484, 0.3303] | +0.0022 (p=0.8590, p_H=1.0000) | 0.2686 | 0.5230 | 0.2449 |

### 4.2 Dekomposisi Prior-Shift vs Covariate-Shift (`RQ3-01` pada Anchor `RQ0-01`)

| Evaluasi / Penyesuaian Prior | BACC4 | QWK | Macro-F1 | MAE | Recall $\ge 2$ |
|---|---:|---:|---:|---:|---:|
| Internal `RAM-H1200` Test (13 Sendi Irisan) | **0.4794** | **0.4895** | 0.4065 | 0.4121 | 0.2707 |
| External `RheumaVIT` — Logit Source-Prior (`RAM-H1200`, Tanpa Koreksi Prior) | **0.3213** | **0.1978** | 0.2510 | 0.2207 | 0.1714 |
| External `RheumaVIT` — Logit Balanced/Uniform-Prior (0 Label Target) | **0.4036** | **0.1619** | 0.2165 | 0.6381 | 0.5810 |
| External `RheumaVIT` — Logit Target-Prior (`RheumaVIT` Oracle Prior) | **0.3209** | **0.2048** | 0.2650 | 0.1915 | 0.1571 |

### 4.3 Transfer Zero-Shot Lintas-Populasi (`RAM-H1200` $\rightarrow$ `RheumaVIT`) Seluruh Arm Utama

| Kode Arm & Rezim | $N_{\text{run}}$ | Internal 13-Joint BACC4 | External `RheumaVIT` BACC4 | $\Delta$Drop BACC4 | Internal 13-Joint QWK | External `RheumaVIT` QWK | External MAE |
|---|---:|---:|---:|---:|---:|---:|---:|
| `rq0-01_resnet34_b2_f0.1` | 3 | 0.3576 | **0.2566** | +0.1010 | 0.2356 | **0.0374** | 1.2986 |
| `rq0-01_resnet34_b2_f1.0` | 3 | 0.4794 | **0.4036** | +0.0758 | 0.4895 | **0.1619** | 0.6558 |
| `rq1-01_resnet34_scratch_b2_f0.1` | 3 | 0.3085 | **0.2105** | +0.0981 | 0.0551 | **-0.0066** | 1.1693 |
| `rq1-01_resnet34_scratch_b2_f1.0` | 3 | 0.4979 | **0.2807** | +0.2173 | 0.4343 | **0.0117** | 1.9432 |
| `rq1-02_resnet34_dino_b2_f0.1` | 3 | 0.3820 | **0.2624** | +0.1196 | 0.2493 | **0.0356** | 1.2185 |
| `rq1-02_resnet34_dino_b2_f1.0` | 3 | 0.4392 | **0.2758** | +0.1634 | 0.4372 | **0.0266** | 1.3229 |
| `rq1-03_resnet34_dinochrono_b2_f0.1` | 3 | 0.3704 | **0.2530** | +0.1174 | 0.2861 | **0.0223** | 1.3035 |
| `rq1-03_resnet34_dinochrono_b2_f1.0` | 3 | 0.4436 | **0.3087** | +0.1349 | 0.4195 | **0.0455** | 1.1940 |
| `rq1-04_vits_imagenet_b2_f0.1` | 3 | 0.3891 | **0.2764** | +0.1127 | 0.2704 | **0.0537** | 0.4768 |
| `rq1-04_vits_imagenet_b2_f1.0` | 3 | 0.4739 | **0.3852** | +0.0888 | 0.4759 | **0.2098** | 0.4581 |
| `rq1-05_vits_mae_b2_f0.1` | 3 | 0.4229 | **0.2644** | +0.1585 | 0.3469 | **0.0070** | 1.0149 |
| `rq1-05_vits_mae_b2_f1.0` | 3 | 0.4540 | **0.3274** | +0.1265 | 0.4570 | **0.0340** | 1.9655 |
| `rq1-06_vits_dino_b2_f0.1` | 3 | 0.3917 | **0.2612** | +0.1305 | 0.3023 | **0.0038** | 1.1637 |
| `rq1-06_vits_dino_b2_f1.0` | 3 | 0.5033 | **0.2848** | +0.2185 | 0.4494 | **0.0090** | 3.2673 |
| `rq1-12_resnet34_supcon_b2_f0.1` | 3 | 0.3685 | **0.2949** | +0.0736 | 0.2560 | **0.0469** | 0.8846 |
| `rq1-12_resnet34_supcon_b2_f1.0` | 3 | 0.4686 | **0.4031** | +0.0655 | 0.4901 | **0.1974** | 0.5295 |
| `rq1-13_resnet34_supcon_ord_b2_f0.1` | 3 | 0.3556 | **0.2718** | +0.0838 | 0.2474 | **0.0387** | 1.0417 |
| `rq1-13_resnet34_supcon_ord_b2_f1.0` | 3 | 0.4960 | **0.3946** | +0.1015 | 0.5026 | **0.1783** | 0.6240 |
| `rq1-17a_resnet34_lp_b2_f0.1` | 3 | 0.3427 | **0.2773** | +0.0654 | 0.2133 | **0.0401** | 1.0626 |
| `rq1-17a_resnet34_lp_b2_f1.0` | 3 | 0.4679 | **0.3299** | +0.1380 | 0.2605 | **0.0855** | 1.0920 |
| `rq1-17b_resnet34_lpft_b2_f0.1` | 3 | 0.3541 | **0.3087** | +0.0454 | 0.2826 | **0.0596** | 1.2104 |
| `rq1-17b_resnet34_lpft_b2_f1.0` | 3 | 0.5013 | **0.4222** | +0.0791 | 0.5011 | **0.1486** | 0.8057 |
| `rq1-17c_vits_mae_lp_b2_f0.1` | 3 | 0.3173 | **0.2234** | +0.0939 | 0.0821 | **0.0003** | 0.9574 |
| `rq1-17c_vits_mae_lp_b2_f1.0` | 3 | 0.4956 | **0.2751** | +0.2205 | 0.3652 | **0.0089** | 2.6805 |
| `rq1-17d_vits_mae_lpft_b2_f0.1` | 3 | 0.4184 | **0.2631** | +0.1552 | 0.3306 | **0.0018** | 0.9795 |
| `rq1-17d_vits_mae_lpft_b2_f1.0` | 3 | 0.4996 | **0.2902** | +0.2094 | 0.4558 | **0.0196** | 2.2878 |
| `rq2-01_resnet34_ce_b2_f0.1` | 3 | 0.2857 | **0.2650** | +0.0207 | 0.1562 | **0.0382** | 0.5136 |
| `rq2-01_resnet34_ce_b2_f1.0` | 3 | 0.4138 | **0.3417** | +0.0720 | 0.4106 | **0.1972** | 0.3160 |
| `rq2-18a_resnet34_supcon_ord_cbloss_b2_f0.1` | 3 | 0.3371 | **0.3020** | +0.0351 | 0.2529 | **0.0824** | 0.8985 |
| `rq2-18a_resnet34_supcon_ord_cbloss_b2_f1.0` | 3 | 0.4602 | **0.4193** | +0.0409 | 0.4718 | **0.2025** | 0.6015 |
| `rq2-18b_resnet34_supcon_ord_b2_cornreg_b2_f0.1` | 3 | 0.3488 | **0.3310** | +0.0178 | 0.2732 | **0.0721** | 1.0567 |
| `rq2-18b_resnet34_supcon_ord_b2_cornreg_b2_f1.0` | 3 | 0.4796 | **0.3831** | +0.0965 | 0.5036 | **0.1575** | 0.6762 |
| `rq3-04_resnet34_aug_strong_b2_f0.1` | 3 | 0.3828 | **0.3044** | +0.0785 | 0.2975 | **0.0541** | 1.5405 |
| `rq3-04_resnet34_aug_strong_b2_f1.0` | 3 | 0.5175 | **0.4310** | +0.0865 | 0.5338 | **0.2054** | 0.7221 |
| `rq3-05_resnet34_aug_rvmatch_b2_f0.1` | 3 | 0.3579 | **0.2772** | +0.0807 | 0.2342 | **0.0327** | 1.3201 |
| `rq3-05_resnet34_aug_rvmatch_b2_f1.0` | 3 | 0.4705 | **0.4324** | +0.0381 | 0.4817 | **0.2005** | 0.5328 |
| `rq4-01_resnet34_mcdrop_b2_f0.1` | 3 | 0.3583 | **0.3008** | +0.0576 | 0.2523 | **0.0436** | 1.3382 |
| `rq4-01_resnet34_mcdrop_b2_f1.0` | 3 | 0.4647 | **0.4225** | +0.0422 | 0.4771 | **0.2193** | 0.5216 |
| `rq5-11_resnet34_meanteacher_b2_f0.1` | 3 | 0.3528 | **0.2557** | +0.0971 | 0.1748 | **0.0119** | 1.3952 |
| `rq5-11_resnet34_meanteacher_b2_f0.25` | 3 | 0.4288 | **0.3508** | +0.0779 | 0.2855 | **0.0568** | 1.2850 |

### 4.4 Adaptasi Domain Zero-Label (`AdaBN`, `HistMatch`) & Few-Shot (`5` vs `20` Pasien `RheumaVIT`)

| Mode Adaptasi (`RQ0-01`) | BACC4 | QWK | Macro-F1 | MAE | Recall $\ge 2$ |
|---|---:|---:|---:|---:|---:|
| `adabn_f100` | **0.3700** | **0.1710** | 0.2188 | 0.5583 | 0.4905 |
| `adabn_f010` | **0.2839** | **0.0668** | 0.1877 | 0.8096 | 0.4381 |
| `histmatch_f100` | **0.3827** | **0.0836** | 0.1704 | 1.2068 | 0.6762 |
| `histmatch_f010` | **0.2627** | **0.0155** | 0.1357 | 1.4040 | 0.4952 |
| `fewshot_5patients` | **0.4079** | **0.2269** | 0.2541 | 0.4294 | 0.4667 |
| `fewshot_20patients` | **0.4411** | **0.2721** | 0.2671 | 0.4070 | 0.4857 |

### 4.4 Benchmark 3 Radiolog Independen (`RQ3-07`, $N = 1551$ Sendi Uji `RheumaVIT`)

#### A. Kesepakatan Antar-Radiolog (*Inter-Reader Agreement*) & Batas Atas Manusia (*Leave-One-Reader-Out / LORO*)

| Perbandingan Radiolog | BACC4 | QWK | MAE | Recall $\ge 2$ |
|---|---:|---:|---:|---:|
| Pairwise `r0_vs_r1` | **0.6997** | **0.8338** | 0.0625 | 0.5625 |
| Pairwise `r0_vs_r2` | **0.8316** | **0.8888** | 0.0438 | 0.8125 |
| Pairwise `r1_vs_r2` | **0.8216** | **0.8671** | 0.0522 | 0.9677 |
| **Rata-Rata Pairwise Antar-Radiolog** | **0.7843** | **0.8632** | **0.0529** | **0.7809** |
| LORO `r0_vs_loro` (vs Konsensus 2 Radiolog Lain) | **0.8126** | **0.8536** | 0.0503 | 0.9000 |
| LORO `r1_vs_loro` (vs Konsensus 2 Radiolog Lain) | **0.7608** | **0.8600** | 0.0503 | 0.6923 |
| LORO `r2_vs_loro` (vs Konsensus 2 Radiolog Lain) | **0.8702** | **0.8487** | 0.0509 | 1.0000 |
| **Rata-Rata Human LORO Ceiling** | **0.8145** | **0.8541** | **0.0505** | **0.8641** |

#### B. Performa Model Zero-Shot vs Konsensus 3 Radiolog & Masing-Masing Radiolog ($R_0, R_1, R_2$)

| Kode Arm | vs Konsensus BACC4 / QWK | vs $R_0$ BACC4 / QWK | vs $R_1$ BACC4 / QWK | vs $R_2$ BACC4 / QWK | Recall $\ge 2$ (vs Konsensus / $R_1$) |
|---|---:|---:|---:|---:|---:|
| `rq0-01_f1.0` | **0.4134** / **0.1599** | 0.4256 / 0.1630 | **0.4319** / 0.1472 | 0.4125 / 0.1669 | 0.5159 / **0.5699** |
| `rq3-04_aug_strong_f1.0` | **0.4485** / **0.2067** | 0.4477 / 0.2119 | **0.4638** / 0.1938 | 0.4306 / 0.2163 | 0.7540 / **0.8065** |
| `rq3-05_aug_rvmatch_f1.0` | **0.4325** / **0.1916** | 0.4436 / 0.1996 | **0.4641** / 0.1731 | 0.4483 / 0.2015 | 0.5317 / **0.6022** |
| `rq1-13_supcon_ord_f1.0` | **0.4039** / **0.1696** | 0.4087 / 0.1816 | **0.4066** / 0.1420 | 0.4078 / 0.1804 | 0.5714 / **0.6022** |
| `rq1-05_vits_mae_f1.0` | **0.3501** / **0.0356** | 0.3533 / 0.0340 | **0.3464** / 0.0304 | 0.3859 / 0.0404 | 0.7063 / **0.7419** |
| `rq1-17d_vits_mae_lpft_f1.0` | **0.3210** / **0.0228** | 0.3244 / 0.0214 | **0.3321** / 0.0193 | 0.3256 / 0.0259 | 0.9365 / **0.9355** |
| `rq2-18b_supcon_cornreg_f1.0` | **0.3871** / **0.1477** | 0.3889 / 0.1597 | **0.4067** / 0.1318 | 0.3963 / 0.1571 | 0.4841 / **0.5269** |
| `rq5-11_meanteacher_f0.1` | **0.2597** / **0.0111** | 0.2533 / 0.0102 | **0.2489** / 0.0109 | 0.2564 / 0.0158 | 0.4603 / **0.4839** |
| `rq5-11_meanteacher_f0.25` | **0.3654** / **0.0584** | 0.3739 / 0.0621 | **0.3692** / 0.0560 | 0.3729 / 0.0640 | 0.6429 / **0.6989** |

## 5. RQ4 — Kalibrasi Probabilistik, *Ordinal Conformal Prediction*, & *Selective Referral*

### 5.1 Kalibrasi Post-Hoc (`RQ4-P1` & `RQ4-P2` pada `RAM-H1200` Test)

| Metode Kalibrasi | NLL ($\downarrow$) | Brier Score ($\downarrow$) | Top-1 ECE ($\downarrow$) | Class-Wise ECE (`cwECE`, $\downarrow$) |
|---|---:|---:|---:|---:|
| Uncalibrated Logit-Adjusted (`RQ0-01`) | **1.0096** | **0.4626** | **0.1361** | **0.0738** |
| Temperature Scaling — Balanced Prior (`RQ4-P1`) | **0.8635** | **0.4348** | **0.0197** | **0.0634** |
| Temperature Scaling + Log-Prior Recovery (`RQ4-P2`) | **0.7627** | **0.3994** | **0.0947** | **0.0394** |
| Vector / Dirichlet Calibration (`RQ4-P2`) | **0.7879** | **0.4035** | **0.1023** | **0.0455** |

### 5.2 *Ordinal Conformal Prediction* (`RQ4-P3` & `RQ4-P4`, Target Coverage $1 - \alpha = 90\%$)

| Metode Conformal (`RAM-H1200` Test) | Marginal Cov | Cov$\ge 2$ (Erosi Sedang-Berat) | Mean Set Size | Singleton % | Cov G0 | Cov G1 | Cov G2 | Cov G3 | Cov G5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Marginal Ordinal-APS | **93.48%** | **81.31%** | 2.645 | 19.51% | 96.06% | 89.73% | **83.00%** | **73.33%** | 74.86% |
| Marginal Ordinal-RAPS ($\lambda_{reg}=0.02, k_{reg}=1$, `RQ4-P3`) | **90.83%** | **67.35%** | 2.305 | 23.15% | 94.85% | 87.68% | **66.48%** | **66.67%** | 72.68% |
| Mondrian Class-Conditional Ordinal-APS (`RQ4-P3`) | **95.18%** | **93.75%** | 3.391 | 11.67% | 95.87% | 93.03% | **96.20%** | **90.67%** | 80.87% |
| Mondrian Joint$\times$Grade Conditional Conformal (`RQ4-P4`) | **95.33%** | **91.76%** | 3.367 | 7.88% | 95.70% | 95.91% | **94.49%** | **81.33%** | 80.33% |

#### Re-Kalibrasi Conformal Lintas-Domain pada `RheumaVIT` (Zero-Shot vs Re-Kalibrasi 5 & 20 Pasien)

| Skema Kalibrasi Kuantil Conformal | Marginal Coverage (`RheumaVIT`) | Coverage Grade $\ge 2$ (`RheumaVIT`) | Mean Set Size | Singleton % |
|---|---:|---:|---:|---:|
| Zero-Shot Marginal APS (Kuantil dari `RAM-H1200` Val) | **93.94%** | **98.57%** | **3.864** | 4.89% |
| Zero-Shot Mondrian APS (Kuantil dari `RAM-H1200` Val) | **93.86%** | **98.57%** | **4.090** | 1.86% |
| Re-Kalibrasi Marginal APS pada 5 Pasien `RheumaVIT` (0-GPU) | **96.19%** | **98.57%** | **4.399** | 0.00% |
| Re-Kalibrasi Mondrian APS pada 5 Pasien `RheumaVIT` (0-GPU) | **94.81%** | **94.29%** | **3.566** | 0.00% |
| Re-Kalibrasi Marginal APS pada 20 Pasien `RheumaVIT` (0-GPU) | **93.68%** | **98.57%** | **3.817** | 5.84% |
| Re-Kalibrasi Mondrian APS pada 20 Pasien `RheumaVIT` (0-GPU) | **93.86%** | **97.14%** | **3.924** | 0.00% |

### 5.3 *Selective Referral* Berbasis Ketidakpastian (`RQ4-P5` & `RQ4-01`)

| Metode Estimasi Ketidakpastian | AURC (1 - BACC4) $\downarrow$ | Cov 100% BACC4 / QWK | Cov 90% BACC4 / QWK | Cov 80% BACC4 / QWK |
|---|---:|---:|---:|---:|
| Single Model Predictive Entropy (`RQ0-01`) | **0.2452** | 0.4855 / 0.4832 | 0.5053 / 0.5225 | **0.5104 / 0.5227** |
| MC Dropout ($p=0.2, T=20$ Passes, `RQ4-01`) | **0.2579** | 0.4705 / 0.4785 | 0.4787 / 0.4837 | **0.4882 / 0.5009** |
| 3-Seed Deep Ensemble — Predictive Entropy (`RQ4-P5`) | **0.2427** | 0.5012 / 0.5313 | 0.5047 / 0.5124 | **0.5228 / 0.5236** |
| 3-Seed Deep Ensemble — Ordinal Score Variance (`RQ4-P5`) | **0.2409** | 0.5012 / 0.5313 | 0.4988 / 0.5197 | **0.5190 / 0.5331** |

## 6. RQ5 — Mitigasi *Label Noise* Bebas-Kebocoran (`5-Fold Patient-Wise OOF`), Reliabilitas Longitudinal, `Mean-Teacher`, & `RQ5-E`

- **Diagnostik 5-Fold OOF (`RQ5-00`):** Dari `12688` sampel sendi latih, teridentifikasi **`3084` (`24.31%`)** kandidat *label issue* dan **`1638`** sendi dengan regresi longitudinal tak-plausibel ($\Delta S < 0$).

### 6.1 Rezim 100% Label (`f1.0`, Evaluasi Jujur pada Test Set Asli Tanpa Relabelling Uji)

| Kode Arm | Deskripsi Metode | $S_{\text{val}}$ | Test BACC4 [95% CI] | $\Delta$BACC4 ($p_{\text{raw}}, p_{\text{Holm}}$) | Test QWK [95% CI] | $\Delta$QWK ($p_{\text{raw}}, p_{\text{Holm}}$) | Macro-F1 | MAE | Rec$\ge 2$ |
|---|---|---:|---|---:|---|---:|---:|---:|---:|
| `rq0-01` | RQ0-01: ResNet-34 + B2 (Logit-Adj + InvSqrtSampler) [ANCHOR] | 0.4584 | **0.4855** [0.4362, 0.5141] | Anchor (0.0000) | **0.4832** [0.3822, 0.5683] | Anchor (0.0000) | 0.4043 | 0.3940 | 0.2998 |
| `rq5-01` | RQ5-01: OOF CleanLab Hard Relabeling (Train-Only, M042/M272) | 0.4520 | **0.4889** [0.4413, 0.5202] | +0.0034 (p=0.8054, p_H=1.0000) | **0.4656** [0.3656, 0.5475] | -0.0177 (p=0.1764, p_H=0.5291) | 0.3896 | 0.4211 | 0.3432 |
| `rq5-02` | RQ5-02: OOF CleanLab Pruning (Drop Top-5% Noisy Train) | 0.4759 | **0.4865** [0.4452, 0.5118] | +0.0010 (p=0.8714, p_H=1.0000) | **0.5097** [0.4056, 0.5959] | +0.0265 (p=0.0120, p_H=0.0720) | 0.4168 | 0.3670 | 0.3272 |
| `rq5-03` | RQ5-03: OOF CleanLab Soft Quality Down-Weighting | 0.4733 | **0.5001** [0.4520, 0.5282] | +0.0146 (p=0.0300, p_H=0.2400) | **0.5194** [0.4161, 0.6020] | +0.0362 (p=0.0008, p_H=0.0056*) | 0.4242 | 0.3682 | 0.3326 |
| `rq5-04` | RQ5-04: Symmetric Cross-Entropy (SCE) + B2 | 0.1304 | **0.2508** [0.2467, 0.2537] | -0.2347 (p=0.0004, p_H=0.0044*) | **0.0125** [0.0052, 0.0220] | -0.4708 (p=0.0004, p_H=0.0044*) | 0.0894 | 2.6305 | 0.9969 |
| `rq5-05` | RQ5-05: Generalized Cross-Entropy (GCE, q=0.7) + B2 | 0.4155 | **0.5028** [0.4678, 0.5219] | +0.0173 (p=0.3475, p_H=1.0000) | **0.4603** [0.3720, 0.5405] | -0.0229 (p=0.1212, p_H=0.4847) | 0.3413 | 0.4527 | 0.5118 |
| `rq5-06` | RQ5-06: Confidence-Split Soft Label Regularization | 0.4697 | **0.5111** [0.4598, 0.5415] | +0.0256 (p=0.0584, p_H=0.4087) | **0.5299** [0.4320, 0.6093] | +0.0467 (p=0.0004, p_H=0.0044*) | 0.4431 | 0.3820 | 0.3654 |
| `rq5-07` | RQ5-07: Longitudinal Visit-Reliability Down-Weighting (Delta S < 0) | 0.4654 | **0.4987** [0.4552, 0.5253] | +0.0132 (p=0.0740, p_H=0.4439) | **0.5264** [0.4349, 0.5994] | +0.0432 (p=0.0004, p_H=0.0044*) | 0.4209 | 0.3748 | 0.3402 |
| `rq5-08` | RQ5-08: Combined OOF CleanLab + Longitudinal Reliability Weighting | 0.3106 | **0.3247** [0.3110, 0.3369] | -0.1608 (p=0.0004, p_H=0.0044*) | **0.2985** [0.2501, 0.3418] | -0.1847 (p=0.0004, p_H=0.0044*) | 0.2681 | 0.3960 | 0.1838 |

### 6.2 Rezim 10% Label (`f0.1`) & Semi-Supervised `Mean-Teacher` (`f0.1` & `f0.25`)

| Kode Arm | Deskripsi Metode | $S_{\text{val}}$ | Test BACC4 [95% CI] | $\Delta$BACC4 ($p_{\text{raw}}, p_{\text{Holm}}$) | Test QWK [95% CI] | $\Delta$QWK ($p_{\text{raw}}, p_{\text{Holm}}$) | Macro-F1 | MAE | Rec$\ge 2$ |
|---|---|---:|---|---:|---|---:|---:|---:|---:|
| `rq0-01` | RQ0-01: ResNet-34 + B2 (Logit-Adj + InvSqrtSampler) [ANCHOR] | 0.2783 | **0.3619** [0.3346, 0.3836] | Anchor (0.0000) | **0.2352** [0.1537, 0.3222] | Anchor (0.0000) | 0.2653 | 0.5515 | 0.2647 |
| `rq5-01` | RQ5-01: OOF CleanLab Hard Relabeling (Train-Only, M042/M272) | 0.3755 | **0.4435** [0.4194, 0.4645] | +0.0816 (p=0.0004, p_H=0.0052*) | **0.3477** [0.2474, 0.4425] | +0.1125 (p=0.0004, p_H=0.0052*) | 0.3005 | 0.4984 | 0.3089 |
| `rq5-02` | RQ5-02: OOF CleanLab Pruning (Drop Top-5% Noisy Train) | 0.3465 | **0.3620** [0.3300, 0.3967] | +0.0001 (p=0.9738, p_H=1.0000) | **0.2283** [0.1527, 0.3060] | -0.0069 (p=0.8118, p_H=1.0000) | 0.2791 | 0.5400 | 0.2525 |
| `rq5-03` | RQ5-03: OOF CleanLab Soft Quality Down-Weighting | 0.3057 | **0.3314** [0.3174, 0.3464] | -0.0305 (p=0.0184, p_H=0.1656) | **0.2565** [0.1780, 0.3436] | +0.0212 (p=0.2112, p_H=1.0000) | 0.2606 | 0.5438 | 0.3173 |
| `rq5-04` | RQ5-04: Symmetric Cross-Entropy (SCE) + B2 | 0.1238 | **0.2488** [0.2446, 0.2527] | -0.1131 (p=0.0004, p_H=0.0052*) | **-0.0001** [-0.0016, 0.0017] | -0.2354 (p=0.0004, p_H=0.0052*) | 0.0127 | 2.6751 | 0.9969 |
| `rq5-05` | RQ5-05: Generalized Cross-Entropy (GCE, q=0.7) + B2 | 0.2351 | **0.3807** [0.3449, 0.4045] | +0.0188 (p=0.1784, p_H=0.8918) | **0.1746** [0.1138, 0.2448] | -0.0606 (p=0.0008, p_H=0.0080*) | 0.1699 | 0.9200 | 0.5111 |
| `rq5-06` | RQ5-06: Confidence-Split Soft Label Regularization | 0.2840 | **0.3677** [0.3457, 0.3846] | +0.0058 (p=0.4119, p_H=1.0000) | **0.2508** [0.1753, 0.3370] | +0.0156 (p=0.2328, p_H=1.0000) | 0.2594 | 0.5592 | 0.3005 |
| `rq5-07` | RQ5-07: Longitudinal Visit-Reliability Down-Weighting (Delta S < 0) | 0.3059 | **0.3457** [0.3260, 0.3637] | -0.0162 (p=0.0748, p_H=0.5983) | **0.2491** [0.1671, 0.3385] | +0.0139 (p=0.3719, p_H=1.0000) | 0.2612 | 0.5467 | 0.3089 |
| `rq5-08` | RQ5-08: Combined OOF CleanLab + Longitudinal Reliability Weighting | 0.1977 | **0.2842** [0.2741, 0.2895] | -0.0777 (p=0.0004, p_H=0.0052*) | **0.1307** [0.1062, 0.1564] | -0.1046 (p=0.0012, p_H=0.0108*) | 0.2169 | 0.4142 | 0.0000 |
| `rq5-11_mt_f0.1` | RQ5-11 (10% Label): Semi-Supervised Mean-Teacher EMA Consistency | 0.2341 | **0.3488** [0.3271, 0.3645] | -0.0131 (p=0.1372, p_H=0.8230) | **0.1687** [0.1247, 0.2203] | -0.0665 (p=0.0004, p_H=0.0052*) | 0.2404 | 0.7321 | 0.3135 |
| `rq5-11_mt_f0.25` | RQ5-11 (25% Label): Semi-Supervised Mean-Teacher EMA Consistency | 0.3252 | **0.4289** [0.4115, 0.4516] | +0.0670 (p=0.0004, p_H=0.0052*) | **0.2559** [0.1901, 0.3270] | +0.0207 (p=0.3931, p_H=1.0000) | 0.2935 | 0.7004 | 0.4500 |

### 6.3 Evaluasi Distributional Multi-Radiolog (`RQ5-E`, $N = 1551$ Sendi Uji `RheumaVIT`)

- **Proporsi Ketidaksepakatan Antar-Radiolog:** `1436` sendi (`92.59%`) bulat (*unanimous*), dan **`115` sendi (`7.41%`)** memiliki ketidaksepakatan antar-radiolog ($R_0, R_1, R_2$).

| Kode Arm | KL Divergence ke Distribusi 3-Radiolog ($\downarrow$) | Soft Brier Multi-Reader ($\downarrow$) | KL pada Sendi Unanimous ($N=1.436$) | KL pada Sendi Disagreement ($N=115$) | BACC4 pada Sendi Unanimous | BACC4 pada Sendi Disagreement |
|---|---:|---:|---:|---:|---:|---:|
| `rq0-01_f1.0` | **0.9019** | 0.3906 | 0.8568 | 1.4649 | **0.4333** | **0.2977** |
| `rq3-04_aug_strong_f1.0` | **0.9463** | 0.4425 | 0.9173 | 1.3094 | **0.4206** | **0.3905** |
| `rq3-05_aug_rvmatch_f1.0` | **0.8338** | 0.3579 | 0.7708 | 1.6204 | **0.4857** | **0.3050** |
| `rq1-13_supcon_ord_f1.0` | **0.9125** | 0.3947 | 0.8557 | 1.6215 | **0.4184** | **0.3149** |
| `rq1-05_vits_mae_f1.0` | **2.2511** | 0.8723 | 2.2385 | 2.4085 | **0.3717** | **0.2776** |
| `rq1-17d_vits_mae_lpft_f1.0` | **2.3888** | 0.9606 | 2.4463 | 1.6711 | **0.3172** | **0.3217** |
| `rq2-18b_supcon_cornreg_f1.0` | **1.3208** | 0.3895 | 1.2551 | 2.1418 | **0.4040** | **0.2664** |
| `rq5-11_meanteacher_f0.1` | **1.6530** | 0.7332 | 1.6833 | 1.2734 | **0.2438** | **0.2728** |
| `rq5-11_meanteacher_f0.25` | **1.4401** | 0.5930 | 1.4078 | 1.8443 | **0.3572** | **0.3123** |

