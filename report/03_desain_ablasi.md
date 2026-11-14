# Desain Ablasi per RQ (Pasca-Verifikasi SLR-M, Disetujui untuk Rencana Tesis)

**Dasar:** Kandidat Prioritas A/B di `02_kandidat_ablasi_fulltext.md` dan laporan audit verifikasi `verification/04_verification_report.md`.
**Tanggal Pembaruan:** 2026-11-14 (menggantikan draf 2026-11-10; seluruh rujukan ke paper fabrikasi `M095` telah dihapus dan digantikan oleh bukti terverifikasi dari 44 studi valid SLR-M serta rujukan metodologi asli).

**Prinsip Metodologis (Menjawab Celah `Q1–Q6` SLR-M):**
- Satu sumbu diubah per ablasi; komponen lain dikunci (`Q1_abl = Y`).
- Aturan analisis pra-spesifikasi di-commit sebelum run pertama.
- Partisi *strict patient-level* (`Q3_patient = Y`): seluruh sendi dan kunjungan pasien yang sama tidak pernah lintas split.
- Pemilihan model, ambang keputusan, suhu kalibrasi, dan kuantil konformal dilakukan **100% di validasi** (`Q4_valsel = Y`).
- Label himpunan validasi dan uji **tidak pernah diubah** oleh algoritma pembersih derau label.
- Pelaporan multi-run (`Q2_runs = Y`) dengan *patient-cluster bootstrap 95% CI* dan koreksi Holm dalam setiap RQ.
- Seluruh varian dievaluasi pada kohort eksternal lintas-populasi **`RheumaVIT`** (`Q5_ext = Y`).

---

## 1. Konfigurasi Rujukan dan Unit Biaya

- **Rujukan Bersama (`RQ0-01`):** `ResNet34 + ImageNet (A1) + Balanced Softmax (B2)`, resep augmentasi E8 `base` dengan seed tetap.
- **Satu Arm** = `100% label × 3 seed (42, 1337, 2026)` + `10% label × 3 stratified patient draw (draw 0, 1, 2; seed 42)` = **6 run**.
- **Perkiraan Biaya per Arm (`ResNet34` di RTX 3060 12 GB):** sekitar **1,9 jam GPU** (median E3 A1 B2: 30,9 menit pada 100% dan 5,9 menit pada 10%). Satu arm `ViT-S` $\approx$ **2,9 jam GPU**.
- **RQ3 (Validasi Eksternal `RAM-H1200` $\to$ `RheumaVIT`):** Setiap arm langsung dievaluasi pada `RheumaVIT` (*zero-shot transfer* + dekomposisi *Prior Shift* vs *Covariate Shift* + evaluasi terhadap 3 radiolog independen). Waktu inferensi hanya beberapa menit. Adaptasi *few-shot* (5 dan 20 pasien) dijalankan pada arm terbaik per RQ.

---

## 2. Matriks Arm per RQ

Kolom **Kode**: *ada flag* = cukup memakai flag `train.py` yang sudah ada; *kode baru* = perlu diimplementasikan dan diuji unit terlebih dahulu.

### RQ1: Pre-training, Self-Supervised Learning, dan Strategi Transfer (Sumbu Encoder; Kepala Tetap `B2`)

Sudah ada di tesis: `scratch`, `ImageNet`, `MAE`, `DINO`, `DINO + ChronoCon`, `LLRD`.

| Arm | Isi / Perlakuan | Bukti Terverifikasi | Kode | Biaya (Perkiraan) |
|---|---|---|---|---|
| `R1-1` | **Intermediate Supervised Transfer:** Pre-training antara pada radiograf tangan `RSNA Pediatric Bone Age` ($n=14.036$), lalu fine-tune ke SvdH | **`M005`** (PCC `0,925` setara antar-radiolog) | kode baru (regresi bone age), lalu `--init-from` | $\pm 1,0$ jam pre-train + $1,9$ jam |
| `R1-2` | **SupCon Berbobot Jarak Grade + CE:** Positif = grade sama lintas pasien; bobot tolakan negatif $\propto |y_i - y_j|$ | **`M219`**, **`M082†`**, **`M166†`** | kode baru (loss) | $\pm 2,5$ jam |
| `R1-3` | **SSL Normalitas (`SS-FewSOME`):** Dilatih pada sendi sehat (`Grade 0`), jarak ke pusat normal dipakai sebagai fitur tambahan / *gate* | **`M197`** (AUC `56,7% -> 81,0%`, 5 seed, kode publik) | kode baru | $\pm 2,0$ jam pre-train + $1,9$ jam |
| `R1-4` | **Strategi Fine-Tuning:** `Frozen Linear Probe`, `LP-FT` (*Linear Probe then Fine-Tune*), dan `Gradual Unfreezing` pada `ImageNet` dan encoder SSL terbaik | **`M269`**, **`M005`**, dan koreksi **`M157`** (*frozen FM* kalah dari *end-to-end FT*) | sebagian ada (`--freeze-backbone`, `--init-from`, `--layer-decay`) | $4 \times 1,9 \approx 7,6$ jam |
| `R1-5` | **SKELEX Foundation Model MSK (`ViT-L` MAE 1,29 Juta Radiograf MSK):** `Linear Probe`, lalu `Fine-Tune + LLRD` | **`M015`** (*npj Digit. Med.* 2026: `+6,21%` atas ImageNet-MAE, split pasien, bobot publik) | kode baru (loader bobot) | **ukur 1 epoch dulu** di RTX 3060 |
| `R1-6` (B) | **Penyelarasan Representasi Progresi Longitudinal (`TCN + MMD`):** Menyelaraskan fitur citra tunggal ke sekuens kunjungan tanpa memaksa urutan monotonik per sendi | **`M335`** (CVPR 2022: `BA 0,785 -> 0,822`, `Kappa 0,649 -> 0,736`) | kode baru | $\pm 3,0$ jam |
| `R1-7` (B) | **Foundation Model Medis Umum (`BiomedCLIP` / `RAD-DINO`) Di-fine-tune** | **`M157`** (hanya menguji versi beku) | kode loader | $\pm 6,0$ jam |

---

### RQ2: Class Imbalance dan Ordinalitas (Sumbu Kepala/Loss; Encoder Tetap `ImageNet ResNet34`, Diulang pada Encoder Terbaik `RQ1` untuk 2 Arm Terbaik)

Sudah ada di tesis: `B0–B6`, `E7` undersampling, `E8` augmentasi.

| Arm | Isi / Perlakuan | Bukti Terverifikasi | Kode | Biaya |
|---|---|---|---|---|
| `R2-1` | **Oversampling Murni (`OS-only`):** Memecah `B1` tanpa *class weight* | **`M041`** (`OS-only` paling konsisten pada OA tangan DIP, $p=0,015$), **`M141`**, **`M222`** | **ada flag** (`--oversample`) | 1,9 jam |
| `R2-2` | **Class Weight Murni (`CW-only`):** Memecah `B1` tanpa *oversampling* | **`M071`** (Tabel 4: *class weight* di atas CORAL justru menurunkan Macro-F1 `-5,0%`!) | **ada flag** (`--class-weight`) | 1,9 jam |
| `R2-3` | **Focal Loss ($\gamma = 2$) / Focal Ordinal Loss (`FOL`)** | **`M132`** (`FOL` > CE & *fixed-penalty* di 6 CNN $\times$ 10 run), **`M100`** (erosi RA) | kode baru (kecil) | 1,9 jam |
| `R2-4` | **Focal Loss + Oversampling (`WeightedRandomSampler`)** | **`M041`** (`FL` kuat di recall KL3 `0,510`, saling melengkapi dengan `OS`), **`C093`** (sensitivitas kelas menengah naik `63,3% -> 75,8%`) | kode baru (kecil) + flag | 1,9 jam |
| `R2-5` | **Kepala *Cumulative-Link* (`ORM` dengan $K-1$ *learnable cut-points*)** | **`M089`** (meningkatkan QWK dan menurunkan MAE pada **21/21 arsitektur CNN**; ResNet-34 QWK `+0,061`) | kode baru (kepala) | 1,9 jam |
| `R2-6` | **Distance-Penalty Ordinal Loss & Soft Label Ordinal (Beta/Gaussian/Binomial)** | **`M305`** (unggul di **8/8 arsitektur**), **`M091`**, **`M266†`**, **`C047`** | kode baru (kecil) | 1,9 jam |
| `R2-7` (B) | **B2 + Kepala CORN Tambahan sebagai Regularizer** | **`C015`** (catatan: prioritas B karena verifikasi menunjukkan `Q1_abl = N` di paper aslinya) | kode baru (kecil) | 1,9 jam |
| `R2-P` | **Pasca-Latih (Tanpa Training Baru):** (1) Ambang keputusan per kelas disetel di validasi, (2) `FAISS k-NN` di atas embedding beku, (3) `Test-Time Augmentation (TTA)` | **`M071`** & **`M052†`** (threshold tuning), **`M161`** (`KneeRAC` naik di 13/17 arsitektur), **`C093`** (`TTA` konsisten naik di seluruh model) | kode analisis | **$\pm 0$ GPU** |

---

### RQ3: Ketahanan & Adaptasi Lintas Populasi (`RAM-H1200` Jepang $\to$ `RheumaVIT` Rusia)

Untuk setiap arm di `RQ0, RQ1, RQ2, RQ4, RQ5` dilaporkan *domain drop* ($\Delta\text{BACC4}$, $\Delta\text{QWK}$, $\Delta\text{ECE}$), dilengkapi **5 protokol eksplorasi validasi eksternal**:
- **`R3-1` (Dekomposisi *Covariate Shift* vs *Target Prior Shift*):** Membandingkan prediksi `RheumaVIT` dengan logit mentah (*source-prior*) vs logit *balanced-prior* (`B2`) vs estimasi prior *few-shot*. Biaya: **$\pm 0$ GPU**.
- **`R3-2` (*Zero-Label Test-Time Adaptation*):** Membandingkan inferensi standar vs `TTA` (**`C093`**) vs *Test-Time BatchNorm / Tent Adaptation* (Schneider et al. NeurIPS 2020; Wang et al. ICLR 2021 — menggantikan sitasi `M095` yang dicabut) pada arm terbaik `RQ1` dan `RQ2`. Biaya: **$\pm 0,3$ jam GPU**.
- **`R3-3` (*Few-Shot 5 & 20 Pasien* + *Cross-Domain Contrastive Adaptation*):** Selain *fine-tuning* standar (`E4` / **`M091`**), menguji **Cross-Domain Supervised Contrastive Loss berbobot jarak grade + EMD** (**`M219`**, yang menaikkan QWK lintas-domain `0,678 -> 0,732`). Biaya: **$\pm 3,5$ jam GPU**.
- **`R3-4` (Jangkar 3 Radiolog Independen `RheumaVIT`):** Evaluasi model terhadap masing-masing dari 3 radiolog `RheumaVIT`, *leave-one-reader-out human ceiling*, dan distribusi probabilitas 3 radiolog. Biaya: **$\pm 0$ GPU**.
- **`R3-5` (Uji Diagnostik *Shortcut Learning* via *Pixel-Level BE Masking*):** Mengadopsi ide **`M249`** secara presisi menggunakan masker piksel erosi (`BE mask`) asli di `RAM-H1200` (menghapus piksel erosi vs menghapus latar belakang saat inferensi). Biaya: **$\pm 0$ GPU**.

---

### RQ4: Kalibrasi, Uncertainty Quantification, dan Rujukan Selektif

| Arm | Isi / Perlakuan | Bukti Terverifikasi | Kode | Biaya |
|---|---|---|---|---|
| `R4-P1` | **Pasca-Latih Kalibrasi:** `Temperature Scaling + Log-Prior` vs `Vector / Dirichlet Calibration` (Kull et al. 2019; *sitasi `M095` dihapus*) | Eksperimen tesis + literatur kalibrasi | kode analisis | **$\pm 0$ GPU** |
| `R4-P2` | **Pasca-Latih Conformal:** `Contiguous Ordinal APS` vs **`RAPS`** (**`M110`**) vs **`Mondrian (Class-Conditional) Conformal`** per grade | **`M110`** (*calibration split* 1.000 citra; cakupan `0,91–0,93` pada target `0,90`) | kode analisis | **$\pm 0$ GPU** |
| `R4-P3` | **Pasca-Latih Cross-Domain Conformal Re-calibration:** Uji cakupan konformal *zero-shot* di `RheumaVIT` vs re-kalibrasi kuantil $\hat{q}$ dengan 5 & 20 pasien `RheumaVIT` tanpa melatih ulang bobot | Eksplorasi orisinal `RQ3 × RQ4` | kode analisis | **$\pm 0$ GPU** |
| `R4-P4` | **Pasca-Latih Deep Ensemble UQ:** Varians/entropi antar-3 seed sebagai skor rujukan selektif (*risk-coverage*) | Mengisi celah 0/44 studi di SLR-M | kode analisis | **$\pm 0$ GPU** |
| `R4-1` | **MC Dropout di Head + Kurva *Risk-Coverage* & `AUROC Uncertainty -> Error`** | **`M130`** (15 ulangan), **`M039`** (`-0,040` QWK bila dihapus, $p<0,001$), **`M110`**, **`M061`** | kode baru (kecil) | 1,9 jam |
| `R4-2` (B) | **Evidential Deep Learning (`EDL`)** | **`M270`**, **`M110`** | kode baru | 1,9 jam |

---

### RQ5: Mitigasi Derau Label (*Label Noise*), Konsistensi Longitudinal, dan Semi-Supervised Teacher-Student

| Arm | Isi / Perlakuan | Bukti Terverifikasi | Kode | Biaya |
|---|---|---|---|---|
| `R5-0` | **Prediksi Out-of-Fold (5-Fold Patient-Wise di dalam Train Set, 100%, Seed 42)** sebagai probabilitas masukan `CleanLab` | Protokol OOF bebas-bocoran | kode baru (loop fold) | $\pm 2,6$ jam |
| `R5-1` | **CleanLab `Relabel` di Train Set** (**Label Val & Test Haram Diubah!**) | Menguji secara jujur klaim **`M272`**, **`M042`**, **`M117`** (di mana `M272` pada label uji asli tidak signifikan, $p=0,076$) | kode baru + `cleanlab` | 1,9 jam |
| `R5-2` | **CleanLab `Prune` (Buang Sampel Latih yang Diragukan)** | idem | idem | 1,9 jam |
| `R5-3` | **CleanLab `Down-weight` (Turunkan Bobot Loss Sampel Diragukan)** | idem | idem | 1,9 jam |
| `R5-4` | **Semi-Supervised *Mean-Teacher / Consistency Regularization*** pada sendi tanpa label (rezim 10% dan 25% label) | **`M061`** (`+4,21%` s/d `+11,35%` BA pada ~10% label, uji lintas-situs), **`M077`**, **`M185`**, **`M153`** | kode baru | $\pm 4,0$ jam |
| `R5-5` | **Pemisahan Sampel Berbasis Reliabilitas Kunjungan Longitudinal + Loss Hibrida** (*confidence* + penalti transisi semu grade $1 \to 0$) | **`M200†`** (*IEEE JBHI* 2022) + diagnosis internal `RAM-H1200` (regresi semu grade $1 \to 0$ sebesar 47%) | kode baru | 1,9 jam |
| `R5-6` | **Label Smoothing ($\varepsilon = 0,1$) & *Reader-Agreement Soft Labels*** | **`M238`** (Tabel 6: menghapus label smoothing menurunkan F1 `-1,6%`), **`M044`**, **`M266†`** | kode baru (kecil) | 1,9 jam |
| `R5-7` (B) | **Teacher-Student Knowledge Distillation / Self-Distillation dari Ensemble 3 Seed** | **`M157`** ($T=7, \alpha=0,8$ menaikkan QWK `0,524 -> 0,551`), **`M071`** (*catatan: sitasi `M095` dihapus*) | kode baru | 1,9 jam |
| `R5-E` | **Evaluasi Multi-Pembaca:** Terhadap distribusi probabilitas 3 radiolog `RheumaVIT` di samping label konsensus | Mengatasi bias evaluasi akibat label uji tunggal yang *noisy* | kode analisis | **$\pm 0$ GPU** |

---

## 3. Ringkasan Total Biaya Komputasi (Tanpa `SKELEX`)

| Bagian | Arm Berlatih Baru | Jam GPU (Kasar pada RTX 3060) |
|---|---|---|
| Eksperimen Pasca-Latih & Diagnostik (`R2-P`, `R3-1`, `R3-4`, `R3-5`, `R4-P1..P4`, `R5-E`) | 0 | **$\pm 0$ jam** |
| `RQ2` Prioritas A (`R2-1 … R2-6`) + 2 Arm Terbaik Diulang pada Encoder Terbaik `RQ1` | 8 | $\pm 15$ jam |
| `RQ5` Prioritas A (`R5-0 … R5-6`) | 7 | $\pm 16$ jam |
| `RQ1` Prioritas A Tanpa `SKELEX` (`R1-1 … R1-4`) | 7 | $\pm 16$ jam |
| `RQ4` Aktif (`R4-1` MC Dropout) | 1 | $\pm 2$ jam |
| `RQ3` Adaptasi (`R3-2` TTA-BN + `R3-3` Few-Shot & Cross-Domain SupCon) | 2 | $\pm 4$ jam |
| **Subtotal Arm Baru Prioritas A** *(bila run lama diimpor)* | **25** | **$\pm 53$ jam** ($\approx 6$–$7$ malam GPU) |
| Pelatihan Ulang Baseline/Arm Eksisting dengan Seed Augmentasi Seragam (`RQ0-01`, `RQ1-01..06`, `RQ2-01..02, 05..07`) | 12 | $\pm 25$ jam |
| **Total Keseluruhan Satu Tabel Bersih (Tanpa `SKELEX`)** | **37** | **$\pm 78$–$82$ jam** ($\approx 10$ malam GPU) |
| `SKELEX` (`R1-5`) | 2 | Ukur 1 epoch dulu |
| Arm Prioritas B (`R1-6`, `R1-7`, `R2-7`, `R4-2`, `R5-7`) | 5 | $\pm 14$ jam |
