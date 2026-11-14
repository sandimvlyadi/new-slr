# Urutan Run per RQ di Folder Eksperimen Bersih (Pasca-Verifikasi SLR-M)

**Tanggal Pembaruan:** 2026-11-14.
**Dasar:** `02_kandidat_ablasi_fulltext.md`, `03_desain_ablasi.md`, dan `verification/04_verification_report.md`.
**Perkiraan Waktu:** Kasar, dari median waktu E3. `1 arm = 6 run` (`100% label × seed 42/1337/2026` + `10% label × draw 0/1/2, seed 42`). Satu arm `ResNet34` $\approx 1,9$ jam. Satu arm `ViT-S` $\approx 2,9$ jam.

## 1. Struktur Folder Eksperimen Baru

```
<folder-eksperimen-tesis>/     <- siap dijadikan repositori mandiri yang bersih
  code/        salinan paket ramscore dari experiments @ b6e0135 (dicatat), lalu dikembangkan di sini
  artifacts/   encoder SSL lama yang diimpor, read-only + sha256: DINO R34, DINO ViT-S,
               DINO+ChronoCon R34/ViT-S, MAE ViT-S (hasil E1, tidak di-pre-train ulang)
  data/        symlink read-only ke RAM-H1200 dan RheumaVIT (data tidak disalin)
  rules/       aturan pra-spesifikasi per RQ, di-commit sebelum run pertama RQ itu
  runs/rq0 … rq5/   hasil run per RQ (termasuk simpanan logit, embedding, dan prediksi RheumaVIT)
  reports/     analisis per RQ + tabel ablasi gabungan + kurva kalibrasi/conformal
```

Semua pembanding dilatih di folder baru dengan resep yang sama dan augmentasi yang di-seed agar satu tabel ablasi sepenuhnya sebanding (*apple-to-apple*).
Penamaan run: `rq<n>-<nomor>_<encoder>_<head>_f<fraksi>_d<draw>_s<seed>`.

---

## 2. Daftar Lengkap Run per RQ

### RQ0: Rujukan Bersama (Jalan Pertama)

| Kode | Isi | Run | Jam |
|---|---|---|---|
| `RQ0-01` | `ResNet34` + `ImageNet` + `B2` (*Balanced Softmax*), augmentasi `base` ber-seed | 6 | 1,9 |

Run ini menjadi pembanding utama `RQ1` (sumbu encoder), `RQ2` (sumbu kepala/loss), `RQ4` (kalibrasi/UQ), dan `RQ5` (mitigasi derau label).

---

### RQ1: Pre-training dan Self-Supervised Learning (Kepala Tetap `B2`)

*Pertanyaan:* Metode pre-training/SSL mana yang paling membantu (terutama pada 10% label dan kelas minoritas), dan apakah modifikasi khusus domain menambah *gain*?

| Kode | Encoder / Metode | Asal & Bukti Terverifikasi | Run | Jam |
|---|---|---|---|---|
| `RQ1-01` | `Scratch` (`ResNet34`) | ulang | 6 | 1,9 |
| (`RQ0-01`) | `ImageNet` supervised (`ResNet34`) | rujukan | — | — |
| `RQ1-02` | `DINO` (`ResNet34`) | artefak E1 + fine-tune ulang | 6 | 1,9 |
| `RQ1-03` | `DINO + ChronoCon` (`ResNet34`) | artefak E1 + fine-tune ulang | 6 | 1,9 |
| `RQ1-04` | `ImageNet ViT-S` + `LLRD` | ulang | 6 | 2,9 |
| `RQ1-05` | `MAE ViT-S` + `LLRD` | artefak E1 + fine-tune ulang | 6 | 2,9 |
| `RQ1-06` | `DINO ViT-S` + `LLRD` | artefak E1 + fine-tune ulang | 6 | 2,9 |
| `RQ1-07` | **Baru:** Pre-training antara supervised pada radiograf tangan `RSNA Pediatric Bone Age` ($n=14.036$), lalu fine-tune | **`M005`** (PCC `0,925` setara pembaca) | 6 + pre-train | 2,9 |
| `RQ1-08` | **Baru, modifikasi:** `SupCon + CE` dengan negatif berbobot jarak grade ordinal $|y_i - y_j|$ | **`M219`**, **`M082†`**, **`M166†`** | 6 | 2,5 |
| `RQ1-09` | **Baru:** SSL normalitas (`SS-FewSOME`) pada sendi `Grade 0` + skor jarak ke pusat normal | **`M197`** (AUC `56,7% -> 81,0%`, kode publik) | 6 + pre-train | 3,9 |
| `RQ1-10` | **Ablasi Strategi Fine-Tuning:** `Frozen Linear Probe` dan `LP-FT` (*Linear Probe then Fine-Tune*), masing-masing pada `ImageNet` dan encoder SSL terbaik dari `RQ1-02..06` | **`M269`**, **`M005`**, koreksi **`M157`** (*frozen FM* kalah dari *end-to-end FT*) | $4 \times 6$ | 7,6 |
| `RQ1-11` | **Baru:** `SKELEX` (`ViT-L` foundation model MSK 1,29 juta radiograf), `Linear Probe` lalu `Fine-Tune + LLRD` | **`M015`** (*npj Digit. Med.* 2026, bobot publik; **ukur 1 epoch dulu**) | $2 \times 6$ | ? |
| `RQ1-12` (B) | **Baru, modifikasi:** Penyelarasan progresi antarkunjungan (`TCN + MMD`) sebagai alternatif `ChronoCon` | **`M335`** (CVPR 2022) | 6 | 3,0 |
| `RQ1-13` (B) | **Baru:** `RAD-DINO` / `BiomedCLIP` di-fine-tune (`M157` hanya menguji versi beku) | **`M157`** | $2 \times 6$ | $\pm 6,0$ |

**Subtotal Prioritas A Tanpa `SKELEX`:** sekitar **31 jam**.

---

### RQ2: Class Imbalance dan Ordinalitas (Encoder Tetap `ImageNet ResNet34`)

*Pertanyaan:* Bagaimana kontribusi terpisah maupun gabungan dari *resampling*, *loss weighting/focal*, dan kepala ordinal terhadap `BACC4` dan `QWK`?

| Kode | Isi | Asal & Bukti Terverifikasi | Run | Jam |
|---|---|---|---|---|
| `RQ2-01` | `CE Polos`, tanpa penanganan imbalance (baseline sumbu) | flag ada | 6 | 1,9 |
| `RQ2-02` | `B1 Asli`: `Class Weight + Oversampling`, kepala ordinal BCE | flag ada | 6 | 1,9 |
| `RQ2-03` | **`Oversampling Murni (OS-only)`** (tanpa *class weight*) | flag ada (`--oversample`); **`M041`** ($p=0,015$), **`M141`**, **`M222`** | 6 | 1,9 |
| `RQ2-04` | **`Class Weight Murni (CW-only)`** (tanpa *oversampling*) | flag ada (`--class-weight`); **`M071`** (CW menurunkan F1 `-5,0%`) | 6 | 1,9 |
| (`RQ0-01`) | `B2 Balanced Softmax / Logit Adjustment` | rujukan | — | — |
| `RQ2-05` | `B4 CORN + Logit Adjustment` | flag ada (`M091` CORN > CE di 4/4 ViT) | 6 | 1,9 |
| `RQ2-06` | `B6 Decoupled cRT` | flag ada | 6 | 1,9 |
| `RQ2-07` | `Undersampling Grade 0` (E7 `rotate`) | flag ada (`C007` RA2-DREAM) | 6 | 1,9 |
| `RQ2-08` | **Baru:** `Focal Loss` ($\gamma = 2$) / `Focal Ordinal Loss` (`FOL`) | kode kecil; **`M132`** (10 run $\times$ 6 CNN), **`M100`** | 6 | 1,9 |
| `RQ2-09` | **Baru:** `Focal Loss + Oversampling` (`WeightedRandomSampler`) | kode kecil; **`M041`** (`FL` kuat di KL3, saling melengkapi dgn `OS`), **`C093`** | 6 | 1,9 |
| `RQ2-10` | **Baru:** Kepala *Cumulative-Link* (`ORM` dengan *learnable cut-points*) | kode kecil; **`M089`** (unggul di **21/21 arsitektur CNN**) | 6 | 1,9 |
| `RQ2-11` | **Baru:** `Distance-Penalty Ordinal Loss` & `Soft Label Ordinal` (Beta/Gaussian) | kode kecil; **`M305`** (unggul di **8/8 arsitektur**), **`M266†`**, **`C047`** | 6 | 1,9 |
| `RQ2-12` (B) | **Baru, modifikasi:** `B2` + kepala `CORN` tambahan sebagai regularizer | kode kecil; **`C015`** (prioritas B karena `Q1_abl = N` di paper asli) | 6 | 1,9 |
| `RQ2-13` | **Interaksi `RQ1 × RQ2`:** 2 arm terbaik `RQ2` diulang pada encoder terbaik `RQ1` | menguji sifat aditif | $2 \times 6$ | $\pm 4$–$6$ |
| `RQ2-P` | **Pasca-Latih (0 Jam GPU):** (a) Penyetelan ambang keputusan di validasi (`M071, M052†`), (b) `FAISS k-NN` di atas embedding (`M161`), (c) `Test-Time Augmentation (TTA)` (`C093`) | kode analisis | 0 | **0** |

**Subtotal Prioritas A:** sekitar **25–27 jam**.

---

### RQ3: Ketahanan & Adaptasi Lintas Populasi (`RAM-H1200` Jepang $\to$ `RheumaVIT` Rusia)

| Kode | Isi | Bukti & Tujuan | Run | Jam |
|---|---|---|---|---|
| `RQ3-01` | **Transfer Langsung & Dekomposisi *Prior Shift*:** Setiap run `RQ0, RQ1, RQ2, RQ4, RQ5` memprediksi `RheumaVIT` (logit mentah vs *prior-adjusted* `B2`), lalu dihitung *domain drop* | Acuan `M044, M091, M153, M185, M201` | inferensi | $\pm 1,0$ |
| `RQ3-02` | **Adaptasi Waktu-Uji Tanpa Label (*Zero-Label TTA*):** Bandingkan `TTA` (**`C093`**) vs *Test-Time BN statistics update* (Schneider et al. 2020) vs *Tent* (Wang et al. 2021) pada arm terbaik `RQ1` dan `RQ2` *(sitasi `M095` dihapus)* | Pemulihan kinerja tanpa label target | inferensi | $\pm 0,3$ |
| `RQ3-03` | **Adaptasi *Few-Shot* (5 dan 20 Pasien `RheumaVIT`, 3 Draw):** Bandingkan *fine-tuning* standar (`E4` / **`M091`**) vs **Cross-Domain Supervised Contrastive Adaptation berbobot jarak grade + EMD** (**`M219`**) | **`M091`** (+12% akurasi) & **`M219`** (`QWK 0,678 -> 0,732`) | 36 + 18 run pendek | $\pm 3,5$ |
| `RQ3-04` | **Jangkar Multi-Radiolog (`RheumaVIT` 3 Pembaca):** Model vs masing-masing dari 3 radiolog `RheumaVIT` dan *leave-one-reader-out human ceiling* | Validasi klinis tingkat pakar | analisis | **0** |
| `RQ3-05` | **Uji Diagnostik *Shortcut Learning* via *Pixel-Level BE Masking*:** Menghapus piksel masker erosi vs latar belakang pada himpunan uji `RAM-H1200` | Mengadopsi **`M249`** secara presisi memakai *pixel-level BE masks* `RAM-H1200` | analisis | **0** |

**Subtotal `RQ3`:** sekitar **4,8 jam**.

---

### RQ4: Kalibrasi, Uncertainty Quantification, dan Rujukan Selektif (Di Atas Run `RQ0` dan Arm Terbaik `RQ2`)

| Kode | Isi | Asal & Bukti Terverifikasi | Run | Jam |
|---|---|---|---|---|
| `RQ4-P1` | `Temperature Scaling + Log-Prior` (metode lama, dihitung ulang) | analisis | 0 | **0** |
| `RQ4-P2` | `Vector / Dirichlet Calibration` (Kull et al. 2019; *sitasi `M095` dihapus*) | analisis | 0 | **0** |
| `RQ4-P3` | `Conformal Ordinal APS` (metode lama), **`RAPS`** (**`M110`**), **`Mondrian (Class-Conditional) Conformal`** per grade, serta **Cross-Domain Conformal Re-calibration** di `RheumaVIT` (0 vs 5 vs 20 pasien kalibrasi) | analisis (`M110`) | 0 | **0** |
| `RQ4-P4` | **`Deep Ensemble` Antar-Seed (3 Seed)** sebagai skor ketidakpastian untuk kurva *risk-coverage* / rujukan selektif | analisis (mengisi celah 0/44 studi SLR-M) | 0 | **0** |
| `RQ4-01` | **Baru:** `MC Dropout` di head + kurva *risk-coverage* & `AUROC Uncertainty -> Error` | kode kecil; **`M130`** (15 ulangan), **`M039`**, **`M110`**, **`M061`** | 6 | 1,9 |
| `RQ4-02` (B) | **Baru:** `Evidential Deep Learning` (`EDL`) | kode baru; **`M270`**, **`M110`** | 6 | 1,9 |

Semua dilaporkan pada himpunan uji internal (`RAM-H1200`) dan eksternal (`RheumaVIT`).

---

### RQ5: Label Noise, Konsistensi Longitudinal, dan Semi-Supervised Teacher-Student (Encoder & Kepala = Rujukan `RQ0`)

| Kode | Isi | Asal & Bukti Terverifikasi | Run | Jam |
|---|---|---|---|---|
| `RQ5-00` | Prediksi Out-of-Fold: 5-fold per pasien **di dalam train set**, 100%, seed 42 | kode baru | 5 | 2,6 |
| `RQ5-01` | `CleanLab Relabel` pada label train (**val/test haram diubah!**) | kode baru; menguji jujur **`M272`**, **`M042`**, **`M117`** | 6 | 1,9 |
| `RQ5-02` | `CleanLab Prune` (buang sampel latih yang diragukan) | kode baru | 6 | 1,9 |
| `RQ5-03` | `CleanLab Down-weight` (turunkan bobot sampel latih yang diragukan) | kode baru | 6 | 1,9 |
| `RQ5-04` | `Label Smoothing` ($\varepsilon = 0,1$) & *Reader-Agreement Soft Labels* | kode kecil; **`M238`** (Tabel 6), **`M044`**, **`M266†`** | 6 | 1,9 |
| `RQ5-05` | **Modifikasi:** Pemisahan sampel berdasarkan reliabilitas (*confidence* OOF + inkonsistensi antarkunjungan longitudinal grade $1 \to 0$) | kode baru; **`M200†`** + diagnosis internal `RAM-H1200` | 6 | 1,9 |
| `RQ5-06` | **Semi-Supervised `Mean-Teacher / Consistency Regularization`**, rezim 10% dan 25% label (pasien train di luar draw = tanpa label) | kode baru; **`M061`** (`+4,2%` s/d `+11,4%` BA), **`M077`**, **`M185`**, **`M153`** | $2 \times 3$ | $\pm 4,0$ |
| `RQ5-07` (B) | **Teacher-Student Knowledge Distillation / Self-Distillation** dari ensemble 3 seed ke model tunggal | kode baru; **`M157`** ($T=7, \alpha=0,8$), **`M071`** | 6 | 1,9 |
| `RQ5-E` | **Evaluasi Tambahan:** Terhadap masing-masing dari 3 pembaca `RheumaVIT` dan terhadap distribusi probabilitas 3 pembaca | analisis | 0 | **0** |

**Subtotal Prioritas A:** sekitar **16 jam**.

---

## 3. Urutan Eksekusi Tahap demi Tahap (*Quick-Wins First*)

1. **Tahap 0 (`RQ0-01`, 1,9 jam GPU):** Commit aturan pra-spesifikasi, jalankan `RQ0-01`, simpan seluruh logit/embedding internal (`RAM-H1200`) dan eksternal (`RheumaVIT`).
2. **Tahap 1 (Seluruh Eksperimen `0 Jam GPU`):** Langsung jalankan `RQ2-P` (threshold tuning, FAISS k-NN, TTA), `RQ4-P1..P4` (TS, Dirichlet, Conformal APS/RAPS/Mondrian, Cross-Domain Conformal Re-calibration, Deep Ensemble UQ), `RQ3-04` & `RQ5-E` (evaluasi 3 radiolog `RheumaVIT`), serta `RQ3-05` (uji diagnostik *shortcut* memakai masker piksel erosi `RAM-H1200`).
3. **Tahap 2 (`RQ2`, $\pm 21$ jam GPU):** Jalankan `RQ2-01` s/d `RQ2-11` (dimulai dari `RQ2-03 OS-only` dan `RQ2-04 CW-only` yang sudah memiliki flag).
4. **Tahap 3 (`RQ5`, $\pm 16$ jam GPU):** Jalankan `RQ5-00` s/d `RQ5-06`.
5. **Tahap 4 (`RQ1` + Interaksi `RQ2-13`, $\pm 35$ jam GPU):** Jalankan `RQ1-01` s/d `RQ1-10`, uji 1 epoch `SKELEX` (`RQ1-11`), lalu jalankan interaksi 2 kepala terbaik `RQ2` $\times$ encoder terbaik `RQ1` (`RQ2-13`).
6. **Tahap 5 (`RQ4-01` & `RQ3-02..03`, $\pm 5,7$ jam GPU):** Jalankan `MC Dropout` (`RQ4-01`) dan adaptasi eksternal `RheumaVIT` (`TTA-BN` + `Few-Shot 5/20 Pasien` & `Cross-Domain SupCon`).
