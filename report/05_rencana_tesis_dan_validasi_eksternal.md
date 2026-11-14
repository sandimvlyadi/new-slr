# Rencana Penyusunan Tesis & Eksplorasi Validasi Eksternal Berbasis Hasil SLR-M (Dokumen Induk Tesis)

**Tanggal:** 14 November 2026  
**Status:** Disetujui (*Approved*)  
**Rujukan Dokumen:**
- Protokol PRISMA-P: `protocol/01_protocol_PRISMA-P.md`
- Kandidat Ablasi Terverifikasi: `report/02_kandidat_ablasi_fulltext.md`
- Desain Ablasi & Urutan Run: `report/03_desain_ablasi.md`, `report/04_urutan_run_per_RQ.md`
- Laporan Audit Verifikasi Forensik (13 Nov 2026): `verification/04_verification_report.md`, `verification/03_fulltext_criteria_verification.csv`, `verification/02_extraction_check.csv`

---

## 1. Rekomendasi Judul Tesis (Bahasa Indonesia & Bahasa Inggris)

| Opsi | Fokus Strategis | Judul Bahasa Indonesia | Judul Bahasa Inggris |
|---|---|---|---|
| **Opsi 1 (Utama)** | **Komprehensif & Mencakup Eksplisit RQ1–RQ5 + Validasi Eksternal** | *Pembelajaran Representasi, Penanganan Ketidakseimbangan Ordinal, Kuantifikasi Ketidakpastian, dan Koreksi Derau Label untuk Grading Erosi Sendi Tangan pada Radiografi Artritis Reumatoid dengan Validasi Eksternal Lintas Populasi* | *Representation Learning, Ordinal Imbalance Handling, Uncertainty Quantification, and Label-Noise Mitigation for Joint-Level Rheumatoid Arthritis Erosion Grading on Hand Radiographs with Cross-Population External Validation* |
| **Opsi 2** | **Keandalan Klinis (*Trustworthy & Noise-Robust Ordinal AI*) — Cocok untuk Publikasi Q1** | *Grading Ordinal Erosi Tulang Artritis Reumatoid yang Andal dan Tahan Derau Label pada Radiografi Tangan: Studi Ablasi Sistematis, Prediksi Selektif Terkalibrasi, dan Generalisasi Lintas Pusat* | *Reliable and Noise-Robust Ordinal Grading of Rheumatoid Arthritis Bone Erosions on Hand Radiographs: Systematic Ablation, Calibrated Selective Prediction, and Cross-Center Generalization* |
| **Opsi 3** | **Berbasis Tantangan Data Nyata (*Problem-Driven*)** | *Mengatasi Ketidakseimbangan Ekstrem, Ketidakstabilan Label, dan Pergeseran Domain pada Grading Ordinal Erosi Sendi Artritis Reumatoid berbasis Deep Learning* | *Addressing Extreme Class Imbalance, Label Instability, and Domain Shift in Deep Learning-Based Ordinal Grading of Rheumatoid Arthritis Joint Erosions* |
| **Opsi 4** | **Efisiensi Label & Evaluasi Multi-Radiolog Eksternal** | *Pembelajaran Deep Learning Hemat-Label dan Sadar-Ketidakpastian untuk Skoring Erosi Sharp/van der Heijde pada Radiografi Tangan Artritis Reumatoid dengan Evaluasi Multi-Radiolog Eksternal* | *Label-Efficient and Uncertainty-Aware Deep Learning for Sharp/van der Heijde Erosion Scoring in Hand Radiographs with External Multi-Radiologist Validation* |
| **Opsi 5** | **Ringkas & Lugas (Gaya Tesis Magister ITB)** | *Studi Ablasi Sistematis dan Validasi Eksternal Metode Deep Learning Ordinal untuk Grading Erosi Sendi Tangan pada Artritis Reumatoid* | *Systematic Ablation and External Validation of Ordinal Deep Learning Methods for Hand Joint Erosion Grading in Rheumatoid Arthritis* |

---

## 2. Posisi Kebaruan (*Novelty*) Tesis Berdasarkan Audit 44 Studi Valid SLR-M

Berdasarkan sintesis dan verifikasi teks penuh atas 54 naskah kandidat (`44` studi unik valid setelah mengeluarkan `10` berkas tidak layak/fabrikasi/duplikat pada `verification/04_verification_report.md`), tesis ini menutup **lima celah literatur utama**:

1. **Studi Ablasi Sistematis Pertama G1–G4 pada Grading Ordinal Erosi RA Tingkat Sendi (SvdH 0–5):**
   - Literatur didominasi oleh Kellgren-Lawrence (KL) OA lutut (`39/44` studi). Dari `5` studi RA yang lolos (`M003, M005, M100, M222, C007`), `M003` dan `M005` hanya memprediksi skor total tingkat tangan utuh, `M100` berfokus pada deteksi objek, `M222` menilai JSN 3-kelas, dan `C007` tidak memiliki ablasi komponen (`Q1_abl = N`).
2. **Standar Metodologis Bebas-Bocoran (*Strict Patient-Level Split* & *Pre-Specified Rules*):**
   - Sebanyak **70,5% (`31/44`)** studi di SLR-M tidak memastikan pembagian tingkat pasien (`Q3_patient = N/U`), dan beberapa studi (`M160, M162, M275, C037`) bahkan melakukan *oversampling*/SMOTE sebelum pembagian *train/test*. Tesis ini menerapkan partisi tingkat pasien yang ketat (seluruh sendi dan kunjungan longitudinal pasien yang sama berada dalam satu partisi), 3 seed/draw, *patient-cluster bootstrap 95% CI*, dan koreksi Holm.
3. **Koreksi Metodologis atas Evaluasi *Label Noise* (`RQ5`) yang Bebas Bias Sirkularitas:**
   - Audit forensik SLR-M membuktikan bahwa **seluruh 3 studi yang mengklaim kenaikan akurasi besar dari CleanLab / relabeling kaskade (`M042, M117, M272`) mengubah label himpunan uji**. Pada `M272`, ketika dievaluasi pada label uji asli yang tidak disentuh, peningkatan akurasi dari CleanLab **tidak signifikan secara statistik (McNemar $p = 0,076$)**. Tesis ini mengevaluasi `CleanLab` (`relabel`, `prune`, `down-weight`), pemisahan reliabilitas kunjungan longitudinal, dan *soft label* secara jujur pada label uji asli yang tidak pernah disentuh serta pada 3 radiolog independen di kohort eksternal.
4. **Validasi Eksternal Lintas-Benua Pertama pada Grading Erosi RA (`RAM-H1200` Jepang $\to$ `RheumaVIT` Rusia):**
   - Tidak ada satu pun (`0/5`) studi RA di SLR-M yang melakukan validasi eksternal lintas pusat/populasi (`Q5_ext = N`). Tesis ini menguji seluruh model yang dilatih pada 6 pusat medis di Sapporo, Jepang (`RAM-H1200`) secara langsung maupun adaptif pada kohort V.A. Nasonova Institute di Moskow, Rusia (`RheumaVIT`).
5. **Re-kalibrasi Konformal Lintas Rumah Sakit & Rujukan Selektif Berbasis Ensemble/UQ:**
   - Tidak ada satu pun studi di SLR-M yang menguji *Deep Ensemble* untuk prediksi selektif (*risk-coverage*) ataupun menguji ketahanan jaminan cakupan *Conformal Prediction* (`APS`, `RAPS`, `Mondrian`) ketika dipindahkan ke rumah sakit eksternal.

---

## 3. Arsitektur Eksplorasi Validasi Eksternal (RQ3 & Lintas-RQ)

Untuk menjadikan validasi eksternal sebagai keunggulan utama tesis, evaluasi eksternal dirancang dalam **5 protokol eksplorasi mendalam**:

### 3.1 Protokol Eksternal 1: Dekomposisi *Covariate Shift* vs *Target Prior Shift* (`RQ3-01`, 0 Jam GPU)
- **Rasional:** Antara `RAM-H1200` (Jepang) dan `RheumaVIT` (Rusia), terjadi pergeseran distribusi citra $P(X)$ sekaligus pergeseran prevalensi derajat keparahan erosi $P(Y)$.
- **Implementasi:** Karena kepala rujukan `B2` (*Balanced Softmax / Logit Adjustment*) memisahkan logit menjadi skor seimbang dan $\log \pi_{\text{source}}$, kita mengevaluasi tiga skema inferensi di `RheumaVIT`:
  1. *Source-Prior Logits* (menggunakan distribusi kelas `RAM-H1200`),
  2. *Balanced / Uniform-Prior Logits* (menghilangkan bias prevalensi `RAM-H1200` tanpa memerlukan label `RheumaVIT`), dan
  3. *Target-Prior Adjusted Logits* (menggunakan estimasi frekuensi kelas dari 5 atau 20 pasien *few-shot* `RheumaVIT`).
- **Keluaran:** Persentase kontribusi *prior shift* vs *covariate shift* terhadap penurunan `BACC4`, `QWK`, dan `ECE`.

### 3.2 Protokol Eksternal 2: *Zero-Label Test-Time Adaptation* vs *Few-Shot Cross-Domain Contrastive Adaptation* (`RQ3-02` & `RQ3-03`)
- **Rezim Tanpa Label Target (*Zero-Label*, `RQ3-02`):**
  - Membandingkan inferensi standar vs **Test-Time Augmentation (`TTA`)** (terbukti konsisten di **`C093`**) vs **Test-Time BatchNorm Statistics Adaptation** (Schneider et al., NeurIPS 2020) dan **Tent** (Wang et al., ICLR 2021).
- **Rezim *Few-Shot* Target (5 dan 20 Pasien `RheumaVIT` $\times$ 3 Draw, `RQ3-03`):**
  - Membandingkan *fine-tuning* standar (`E4` / **`M091`** yang menunjukkan pemulihan `+12%` akurasi dari 5 epoch fine-tuning eksternal) melawan **Cross-Domain Supervised Contrastive Adaptation berbobot jarak grade + EMD ordinal loss** yang diadaptasi dari **`M219`** (yang terbukti menaikkan QWK lintas-dataset dari `0,678` ke `0,732`).

### 3.3 Protokol Eksternal 3: *Cross-Domain Conformal Re-calibration* (`RQ4-P3`, 0 Jam GPU)
- **Rasional:** Jaminan cakupan $1-\alpha = 90\%$ dari *Conformal Prediction* (`Ordinal APS`, `RAPS` dari **`M110`**, dan `Mondrian` per grade) bergantung pada asumsi *exchangeability*. Di rumah sakit eksternal, ambang kuantil $\hat{q}_{\text{RAM}}$ dapat mengalami *under-coverage* (cakupan jatuh di bawah 90%) atau *over-coverage* (ukuran himpunan prediksi terlalu lebar).
- **Implementasi:**
  1. Evaluasi cakupan aktual (keseluruhan dan per grade erosi $0..5$) serta rata-rata ukuran himpunan prediksi (*average set size*) di `RheumaVIT` menggunakan $\hat{q}_{\text{RAM}}$ (*Zero-Shot Conformal*).
  2. Gunakan sampel kalibrasi target berukuran sangat kecil (**5 pasien [$\approx 160$ sendi]** dan **20 pasien [$\approx 640$ sendi]** dari `RheumaVIT`) untuk menghitung ulang $\hat{q}_{\text{RheumaVIT}}$ dan suhu $T$ **tanpa mengubah bobot jaringan**.
- **Keluaran:** Kurva konvergensi cakupan 90% dan efisiensi ukuran himpunan prediksi terhadap jumlah pasien kalibrasi lokal.

### 3.4 Protokol Eksternal 4: Evaluasi terhadap Jangkar 3 Radiolog Independen `RheumaVIT` (`RQ3-04` & `RQ5-E`, 0 Jam GPU)
- **Rasional:** `RheumaVIT` menyediakan anotasi dari **3 radiolog ahli secara terpisah** di samping label konsensus.
- **Implementasi:**
  1. **Benchmarking Manusia vs AI (*Turing-style Agreement*):** Hitung `QWK`, `BACC4`, dan `MAE` antara prediksi model vs Radiolog 1, Radiolog 2, dan Radiolog 3, lalu bandingkan dengan kesepakatan antar-radiolog manusia (*Radiolog 1 vs 2, 1 vs 3, 2 vs 3*).
  2. **Evaluasi Distribusi Probabilitas Pembaca (*Soft Reader Distribution*):** Bentuk distribusi empiris $p_{\text{human}}(y|x)$ dari suara ke-3 radiolog untuk setiap sendi, lalu ukur *KL-divergence*, *Cross-Entropy*, dan *Brier Score* antara distribusi probabilitas model (setelah kalibrasi `RQ4` dan mitigasi derau `RQ5`) terhadap $p_{\text{human}}(y|x)$.

### 3.5 Protokol Eksternal 5: Uji Diagnostik *Shortcut Learning* dengan Masker Piksel Erosi `RAM-H1200` (`RQ3-05`, 0 Jam GPU) & Opsi Dataset Sekunder
- **Uji Diagnostik *Lesion-Masking* (Adaptasi Presisi `M249`):**
  - Memanfaatkan **masker segmentasi erosi tingkat piksel (*pixel-level BE masks*)** yang tersedia di `RAM-H1200`: pada citra uji dengan erosi ($\text{grade} \ge 2$), lakukan *inpainting/masking* hanya pada piksel lesi erosi vs hanya pada area di luar lesi. Model yang memiliki *Causal Lesion Sensitivity* tinggi akan menurunkan prediksinya secara tajam saat lesi ditutupi, dan model inilah yang diprediksi memiliki *domain drop* paling kecil di `RheumaVIT`.
- **Opsi Eksplorasi Lintas-Anatomi / Lintas-Skema Tambahan (Opsional Bab V):**
  - **`RAM-W600`** (600 radiograf pergelangan tangan dengan skor erosi SvdH) untuk menguji generalisasi anatomi jari $\to$ karpal/pergelangan tangan.
  - **Dataset Publik `Z. Wang 2022` (`ldy07/RA-Dataset`, $n=2.849$ tangan bilateral dari `M003/M005`)** untuk menguji korelasi agregasi skor erosi sendi $\sum \hat{y}_j$ terhadap skor SvdH total tingkat tangan.

---

## 4. Kerangka Bab Tesis (Outline Bab I – Bab VI) dan Target Tabel/Gambar

### BAB I: PENDAHULUAN
- **1.1 Latar Belakang Masalah** (Penilaian kerusakan struktural RA dengan SvdH; empat sifat simultan tugas: ordinalitas 0–5, *long-tailed* grade 0, lesi kortikal mikro & ketidakstabilan label antarpembaca/antarkunjungan, serta pergeseran domain lintas senter).
- **1.2 Identifikasi Masalah dan Celah Penelitian** (Rangkuman temuan SLR 22 studi RA dan SLR-M 44 studi MSK).
- **1.3 Rumusan Masalah (`RQ1`–`RQ5`) dan Hipotesis Penelitian.**
- **1.4 Tujuan dan Kontribusi Penelitian.**
- **1.5 Ruang Lingkup dan Sistematika Penulisan.**
- *Target Visual Bab I:* Gambar 1.1 (Diagram Empat Tantangan Grading Erosi RA & Pemetaan ke `RQ1–RQ5`).

### BAB II: TINJAUAN PUSTAKA DAN SYSTEMATIC LITERATURE REVIEW METODOLOGIS (SLR-M)
- **2.1 Grading Radiografi Artritis Reumatoid (Skor Sharp/van der Heijde) dan Variabilitas Pengamat.**
- **2.2 Protokol PRISMA-P dan Hasil Seleksi Studi SLR-M (2021–2026)** (Alur PRISMA, verifikasi buta manusia/objektif $\kappa = 0,8880$, eksklusi 10 berkas teks penuh).
- **2.3 Sintesis Metode Kelompok G1 (SSL & Pre-training), G2 (Imbalance & Ordinalitas), G3 (Kalibrasi, UQ, & Conformal), dan G4 (Label Noise & Semi-Supervised).**
- **2.4 Audit Kualitas Bukti (`Q1–Q6`) dan Analisis Forensik Bias Metodologis Literatur** (Kebocoran split tingkat citra, *pre-split oversampling*, dan *test-set label modification circularity* pada `M042, M117, M272`).
- **2.5 Pemilihan dan Rasionalisasi Kandidat Ablasi Tesis.**
- *Target Visual & Tabel Bab II:* Gambar 2.1 (Diagram Alir PRISMA 2020 Terverifikasi); Tabel 2.1 (Distribusi Kualitas Bukti `Q1–Q6` pada 44 Studi Valid); Tabel 2.2–2.5 (Sintesis Efek Terkontrol per Kelompok G1–G4); Tabel 2.6 (Daftar Kandidat Ablasi Terpilih beserta Status & Prioritas).

### BAB III: METODOLOGI PENELITIAN DAN DESAIN EKSPERIMEN
- **3.1 Dataset Penelitian dan Partisi Tingkat Pasien:**
  - `RAM-H1200` (1.200 radiograf tangan, 6 pusat medis Jepang, *patient-level split*, rezim 100% dan 10% label, masker piksel BE, sekuens kunjungan).
  - `RheumaVIT` (330 radiograf bilateral, Moskow, Rusia, 42 sendi, konsensus + 3 radiolog).
- **3.2 Konfigurasi Rujukan (`RQ0-01`) dan Pengendalian Varians Eksperimen.**
- **3.3 Spesifikasi Matematis & Arsitektural Kandidat Ablasi (`RQ1`, `RQ2`, `RQ4`, `RQ5`).**
- **3.4 Protokol Validasi Eksternal Lintas Populasi, Adaptasi, dan Uji Diagnostik (`RQ3`).**
- **3.5 Metrik Evaluasi, Kalibrasi, Conformal, dan Pengujian Statistik Pra-Spesifikasi (Bootstrap Pasien & Koreksi Holm).**
- *Target Visual & Tabel Bab III:* Gambar 3.1 (Alur Pipeline Eksperimen Terpadu `RAM-H1200` $\to$ `RheumaVIT`); Tabel 3.1 (Statistik Demografi & Distribusi Grade 0–5 pada `RAM-H1200` vs `RheumaVIT`); Tabel 3.2 (Matriks Lengkap Run `RQ0` s/d `RQ5`).

### BAB IV: HASIL DAN ANALISIS ABLASI INTERNAL (`RAM-H1200`)
- **4.1 Kinerja Baseline Rujukan (`RQ0-01`) dan Hasil Eksperimen Pasca-Latih (`RQ2-P`).**
- **4.2 Hasil Ablasi `RQ1`: Efek Pre-training Antara (`RSNA Bone Age`), `SupCon` Berbobot Jarak, `SSL Normalitas`, Strategi Fine-Tuning, dan Foundation Model MSK (`100%` vs `10%` Label).**
- **4.3 Hasil Ablasi `RQ2`: Dekomposisi `Oversampling` vs `Class Weight`, `Focal Loss`, Kepala `Cumulative-Link` / `Soft Label`, serta Interaksi `RQ1 × RQ2`.**
- **4.4 Hasil Ablasi `RQ4`: Kalibrasi Probabilitas, `Conformal Prediction` (`APS`, `RAPS`, `Mondrian`), dan Kurva *Risk-Coverage* (`Deep Ensemble` vs `MC Dropout`).**
- **4.5 Hasil Ablasi `RQ5`: Evaluasi Jujur `CleanLab` pada Label Uji Asli, Pemisahan Reliabilitas Kunjungan Longitudinal, dan *Mean-Teacher* pada Rezim 10% Label.**
- *Target Visual & Tabel Bab IV:* Tabel 4.1–4.4 (Tabel Ablasi Utama `RQ1, RQ2, RQ4, RQ5` dengan CI Bootstrap & nilai-$p$ Holm); Gambar 4.1 (Recall per Grade 0–5 antara `OS-only`, `CW-only`, `Focal+OS`, dan `B2`); Gambar 4.2 (Reliability Diagram & Cakupan Conformal per Grade); Gambar 4.3 (Kurva *Risk-Coverage* Rujukan Selektif).

### BAB V: VALIDASI EKSTERNAL LINTAS POPULASI (`RheumaVIT`) DAN ANALISIS DIAGNOSTIK
- **5.1 Analisis *Domain Drop* Lintas-Populasi (`RAM-H1200` $\to$ `RheumaVIT`) pada Seluruh Keluarga Metode.**
- **5.2 Dekomposisi *Covariate Shift* vs *Target Prior Shift* (`RQ3-01`).**
- **5.3 Efektivitas Adaptasi Tanpa Label (`TTA`, `Test-Time BN`) vs *Few-Shot Contrastive Adaptation* (`5 & 20 Pasien`, `RQ3-02` & `RQ3-03`).**
- **5.4 Ketahanan dan Re-kalibrasi *Conformal Prediction* Lintas Rumah Sakit (`RQ4-P3`).**
- **5.5 Evaluasi terhadap Jangkar 3 Radiolog Independen `RheumaVIT` dan Distribusi Soft Label Pembaca (`RQ3-04` & `RQ5-E`).**
- **5.6 Analisis Diagnostik *Shortcut Learning* menggunakan Masker Piksel Erosi (`RQ3-05`).**
- *Target Visual & Tabel Bab V:* Gambar 5.1 (*Scatter Plot* Performa Internal `RAM-H1200` vs Eksternal `RheumaVIT` untuk seluruh arm); Tabel 5.1 (Dekomposisi *Prior Shift* & Hasil Adaptasi *Zero-Shot* vs *Few-Shot*); Gambar 5.2 (Kurva Pemulihan Cakupan 90% Conformal terhadap Jumlah Pasien Kalibrasi `RheumaVIT`); Tabel 5.2 (Perbandingan Model vs 3 Radiolog Independen `RheumaVIT`); Gambar 5.3 (Penurunan Skor Prediksi pada Uji *Pixel-Level BE Lesion Masking*).

### BAB VI: KESIMPULAN DAN SARAN
- **6.1 Kesimpulan per Pertanyaan Penelitian (`RQ1`–`RQ5`).**
- **6.2 Implikasi Metodologis dan Klinis (Pedoman Praktis Desain AI Grading Ordinal Radiografi).**
- **6.3 Keterbatasan Penelitian dan Arah Pengembangan Selanjutnya.**
