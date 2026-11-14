# Laporan Verifikasi Kriteria, Screening, dan Ekstraksi Full-Text (Independent Human/Objective Reviewer Audit)

**Tanggal Verifikasi:** 13 November 2026  
**Protokol Acuan:** `slr-methods/protocol/01_protocol_PRISMA-P.md` (§3 Kriteria Kelayakan `EM1–EM7` & `G1–G4`, §6.6 *Human Verification Plan*, §7 Checklist Kualitas `Q1–Q6`)  
**Cakupan Audit:**
1. **Verifikasi Buta (*Blind Verification*) Screening Judul/Abstrak** pada **87 rekaman** (`01_screening_form.csv`: 68 sampel acak 20% dari 338 rekaman basis data + 19 rekaman *citation searching*) tanpa melihat kunci AI (`01_screening_key_AI.csv`), dilanjutkan dengan perhitungan metrik kesepakatan antar-penilai (*inter-rater reliability*) dan rekonsiliasi berbasis *full-text*.
2. **Verifikasi Ekstraksi & Penilaian Kualitas Bukti** pada **17 Studi Prioritas A × 12 Kolom = 204 Sel** (`02_extraction_check.csv`) terhadap dokumen PDF/XML penuh di `slr-methods/papers/`.
3. **Audit Forensik Kasus Khusus** sesuai `verification/README.md` §3:
   - Studi dengan klaim performa tidak masuk akal / melampaui kesepakatan antar-radiolog (`M095`, `M150`, `M175`, `M188`).
   - Studi yang memodifikasi label himpunan uji (*test-set label modification/leakage*: `M042`, `M272`, serta temuan tambahan `M117`).
   - Status 40 laporan pada `slr-methods/papers/00_TODO_user_download.md`.
4. **Verifikasi Kriteria Kelayakan Full-Text & Checklist Kualitas (`Q1–Q6`) pada Seluruh 54 Paper** di `slr-methods/papers/` (`03_fulltext_criteria_verification.csv`).

---

## 1. Hasil Verifikasi Screening Judul/Abstrak (`01_screening_form.csv` vs `01_screening_key_AI.csv`)

### 1.1 Prosedur *Blinding*
Sesuai `verification/README.md` §1 dan protokol PRISMA-P §6.6, proses peninjauan seluruh 87 rekaman pada `01_screening_form.csv` dilakukan secara **buta (*blinded*)** oleh agen peninjau objektif dengan mengisolasi berkas `01_screening_key_AI.csv`, folder `screening/`, `extraction/`, dan `report/` ke direktori terpisah sebelum peninjauan dimulai. Keputusan `include`/`exclude` dan kode alasan (`EM1–EM7` atau `G1–G4`) baru dibandingkan terhadap `01_screening_key_AI.csv` setelah seluruh 87 baris selesai diisi.

### 1.2 Matriks Konfusi & Statistik Kesepakatan ($N = 87$)

| Keputusan AI (`01_screening_key_AI.csv`) \ Keputusan Reviewer (`01_screening_form.csv`) | Reviewer: **Include** | Reviewer: **Exclude** | **Total AI** |
|---|---:|---:|---:|
| **AI: Include (`include_fulltext`)** | **15** | **3** (`M173`, `M199`, `M222`) | **18** |
| **AI: Exclude (`exclude`)** | **0** | **69** | **69** |
| **Total Reviewer** | **15** | **72** | **87** |

| Metrik Evaluasi Kesepakatan | Nilai (Keseluruhan, $N=87$) | Sampel Acak 20% Basis Data ($n=68$) | Sampel *Citation Searching* ($n=19$) | Interpretasi (Landis & Koch) |
|---|---:|---:|---:|---|
| **Persentase Kesepakatan (*Overall Agreement*, $p_o$)** | **96.55%** (84/87) | **95.59%** (65/68) | **100.00%** (19/19) | Sangat tinggi (melampaui ambang batas protokol $\ge 85\%$) |
| **Cohen's Kappa ($\kappa$)** | **0.8880** | **0.8532** | **1.0000** | *Almost perfect agreement* ($\kappa > 0.80$) |
| **PABAK (*Prevalence-Adjusted Bias-Adjusted Kappa*)** | **0.9310** | **0.9118** | **1.0000** | *Almost perfect agreement* |
| **Sensitivitas AI terhadap Inklusi Reviewer** | **100.00%** (15/15) | **100.00%** (11/11) | **100.00%** (4/4) | **Nol *false negative*** (AI tidak melewatkan satu pun studi relevan) |
| **Spesifisitas AI terhadap Eksklusi Reviewer** | **95.83%** (69/72) | **94.74%** (54/57) | **100.00%** (15/15) | AI bersifat konservatif (*high-recall*) pada tahap judul/abstrak |

- **15 Rekaman yang Disepakati Inklusi (`Both Include`):**
  - **Basis Data ($n=11$):** `M005` (`G1`), `M015` (`G1`), `M041` (`G2`), `M061` (`G4`), `M071` (`G1,G2`), `M091` (`G2`), `M106` (`G2,G4`), `M157` (`G1`), `M165` (`G2`), `M266` (`G2,G4`), `M283` (`G2`).
  - **Sitasi ($n=4$):** `C007` (`G2`), `C015` (`G2,G3`), `C037` (`G2`), `C093` (`G2`).

### 1.3 Analisis & Rekonsiliasi Full-Text terhadap 3 Ketidaksepakatan Biner (`M173`, `M199`, `M222`)

Ketiga ketidaksepakatan biner terjadi pada pola yang sama: **AI memasukkan (`include_fulltext` dengan kode `G2`) sedangkan Reviewer mengeksklusi pada tahap judul/abstrak**. Pemeriksaan terhadap naskah lengkap di `slr-methods/papers/` memberikan resolusi objektif berikut:

| ID | Keputusan AI (T/A) | Keputusan Reviewer (T/A) | Temuan dari Naskah Lengkap (`slr-methods/papers/`) | Keputusan Akhir Rekonsiliasi |
|---|---|---|---|---|
| **`M173`** | `include_fulltext` (`G2`) | `exclude` (`EM2`) | Abstrak dan naskah lengkap (`M173.xml`, hlm. 123) hanya mengevaluasi fidelitas citra sintetis lutut (StyleGAN3 vs Stable Diffusion+CycleGAN vs DCGAN menggunakan FID 10.84, Visual Turing Test oleh 3 klinisi, dan sudut anatomis JLCA). **Tidak ada classifier grading KL yang dilatih atau dievaluasi**; penulis secara eksplisit menyatakan bahwa evaluasi dampak data sintetis terhadap akurasi grading adalah pekerjaan masa depan (*future work*). | **Exclude (`EM2`)** — Keputusan Reviewer terkonfirmasi. |
| **`M199`** | `include_fulltext` (`G2`) | `exclude` (`EM2`) | Pada abstrak, fokus utama adalah *Layer Separation Network* untuk memisahkan lapisan jaringan lunak/tulang dan mensintesis citra sendi jari dengan *joint space width* (JSW) yang dapat diatur. Pemeriksaan `M199.pdf` menunjukkan bahwa §4.4 Tabel 4 memang memuat eksperimen hilir klasifikasi 5-grade JSN (`80.10%` akurasi), **tetapi naskah ini adalah preprint arXiv anonim yang belum melalui peer-review** (`arXiv:2502.01972v1`, penulis `"Anonymous"`, hlm. 9 memuat placeholder *"If a paper is accepted, we will release both the dataset and the code"*). | **Exclude (`EM7` — Preprint belum peer-review)**, dapat dicatat sebagai catatan ide metodologis sintesis lapisan pada diskusi. |
| **`M222`** | `include_fulltext` (`G2`) | `exclude` (`EM3`) | Pada abstrak, penanganan ketidakseimbangan kelas pada skor mTSS sendi tangan RA tampak hanya dikaitkan dengan modifikasi arsitektur perhatian EfficientNetV2 (`EM3`). Namun, pemeriksaan naskah lengkap (`M222.pdf`, hlm. 6 §2.5 & hlm. 9 Tabel 2) membuktikan bahwa penulis **secara eksplisit menguji metode G2** (*complete/partial class downsampling*, *oversampling* dengan `WeightedRandomSampler`, dan *class-weighted cross-entropy loss* untuk grading JSN 3-kelas). | **Include (`G2`)** — Sikap konservatif AI pada tahap judul/abstrak terbukti berhasil menyelamatkan studi G2 valid yang menyembunyikan detail metode penyeimbangannya di dalam teks penuh. |

> **Catatan Perbedaan Kode Alasan Eksklusi (ketika AI dan Reviewer sama-sama memilih `exclude`):**  
> Dari 69 rekaman yang sama-sama dieksklusi oleh AI dan Reviewer, 54 rekaman memiliki kode alasan utama yang identik, sedangkan 15 rekaman berbeda hanya pada urutan prioritas kode eksklusi ketika sebuah studi melanggar lebih dari satu kriteria sekaligus (misalnya studi MRI/US/termografi atau deteksi biner dengan CNN standar yang memenuhi `EM1`/`EM2` sekaligus `EM3`: `M250, M239, M192, M070, M131, M245, M085, M202, M118, M120, C046, C063, C069, C091, C107`). Seluruh rincian terdokumentasi di `01_screening_disagreements.csv`.

---

## 2. Hasil Verifikasi Ekstraksi & Kualitas 17 Studi Prioritas A (`02_extraction_check.csv`)

Seluruh **204 sel** (17 studi Prioritas A × 12 atribut: `dataset, split, external, key_comparison, ablation, runs_ci, code, Q1_abl, Q2_runs, Q3_patient, Q5_ext, Q6_code`) telah diverifikasi langsung terhadap teks penuh PDF/XML di `slr-methods/papers/`. Setiap baris pada `02_extraction_check.csv` kini dilengkapi rujukan halaman/tabel/subbab yang presisi pada kolom `page_or_table`.

### 2.1 Ringkasan Kuantitatif Akurasi Ekstraksi AI

| Kategori Verifikasi Sel (`02_extraction_check.csv`) | Jumlah Sel | Persentase dari Total (204) | Keterangan |
|---|---:|---:|---|
| **Setuju Langsung (`reviewer_value` dikosongkan)** | **181** | **88.73%** | Nilai `AI_value` terverifikasi akurat sesuai naskah lengkap. |
| **Pelengkapan Data yang Sengaja Ditandai `U` / `NR in snippets` / `numbers not extracted` oleh AI** | **14** | **6.86%** | Bukan kesalahan AI, melainkan keterbatasan snippet awal yang berhasil dilengkapi dari PDF penuh (`M041 Q3_patient`, `M061 key_comparison & Q3_patient`, `M071 Q3_patient`, `M089 key_comparison`, `M185` [8 atribut], `C093 key_comparison`). |
| **Koreksi Faktual (`AI_value` keliru dibanding PDF penuh)** | **9** | **4.41%** | Koreksi ketidakakuratan angka/label pada `M041`, `M061`, `M071`, `M110`, `M130` (2 sel), `M132`, dan `C093` (2 sel). |
| **Akurasi Faktual AI pada Sel yang Sudah Diekstraksi ($181 / 190$)** | **181 / 190** | **95.26%** | Sangat akurat; tidak ditemukan halusinasi studi. |

### 2.2 Daftar Lengkap 9 Koreksi Faktual terhadap `AI_value`

| ID | Field | Nilai AI (`AI_value`) | Koreksi Reviewer (`reviewer_value`) | Lokasi Bukti di Full-Text |
|---|---|---|---|---|
| **`M110`** | `dataset` | `OAI baseline (train 10,456 / val 1,614 / test 1,656 + CP split 827)` | Ukuran *calibration split* untuk Conformal Prediction (CP) pada Exp. 3 adalah **1.000 citra** (dipisahkan dari 2.614 citra validasi sehingga menyisakan 1.614 validasi). Angka **`827`** pada Tabel II adalah **`Inference time (s)`** (waktu inferensi dalam detik), bukan jumlah sampel CP. | hlm. 2 §III-A, Fig. 1; hlm. 3 §III-B (baris 162–163), Tabel II |
| **`M132`** | `key_comparison` | `FOL beats CE and fixed-penalty loss on most metrics incl. MSE (ResNet-18-CE acc 63.89 ± 1.30, kappa 0.678)` | AI tertukar antara kolom **MSE** dan **Kappa** pada Tabel 1: untuk `ResNet-18-CEL`, angka **`0.678 ± 0.045` adalah MSE**, sedangkan **Kappa adalah `0.774 ± 0.014`** (dengan `ResNet-18-FOL` mencapai Acc `64.10 ± 0.900%`, MSE `0.598 ± 0.048`, dan Kappa `0.794 ± 0.097`). | hlm. 5212 Tabel 1; hlm. 5213–5220 Tabel 2–8 |
| **`M041`** | `key_comparison` | `... FL-only weak on KL3/KL4` | `FL-only` (Focal Loss) sebenarnya **kuat pada KL3** (`KL3 F1 = 0.427 ± 0.006` vs baseline `0.280`, dan mencapai **recall KL3 tertinggi di antara seluruh lengan tunggal yaitu `0.510`** vs baseline `0.203`), tetapi lemah pada **KL1** (`F1 0.167` vs `0.267`) dan **KL4** (`F1 0.563` vs `0.573`). | hlm. 13 Tabel 4; hlm. 18 Tabel 6–7 |
| **`M061`** | `runs_ci` | `5 runs, mean ± SD` | `5 runs (random seeds), mean ± SE (standard error)` — Teks penuh §3.2 secara eksplisit menyatakan rata-rata dan **standard error (SE)** dari 5 *random seeds*, bukan *standard deviation* (SD). | hlm. 3 §3.2 (baris 277–279), Tabel 2 |
| **`M071`** | `key_comparison` | `ablation from ViT-CORAL baseline accuracy 65.15 / F1 68.60 → full 70.29 / 70.19; KL1 +6.6, KL2 +19.4 vs VGG19` | (1) Selisih F1 kelas KL2 terhadap VGG19 pada Tabel 5 adalah **`+19.2`** (`67.24` vs `48.0`), bukan `+19.4`. (2) Sangat penting untuk sintesis G2: pada ablasi bertahap Tabel 4 (hlm. 15), penambahan *class weighting* `[1.0, 1.5, 1.5, 1.0, 1.0]` di atas ViT-CORAL justru **menurunkan Macro-F1 sebesar `-5.0%`** (`68.60% -> 63.6%`), sebelum akhirnya dipulihkan oleh *CLIP distillation* (`65.0%`), *TTA + threshold tuning* (`67.0%`), dan *5-fold ensemble* (`69.80%` di Tabel 4 / `70.19%` di Tabel 2–3). | hlm. 14–15 §6.1–6.3, Tabel 2–5 |
| **`M130`** | `external` | `Y (52 patients)` | `Same-institution hold-out (52 patients / 104 DRRs from Osaka University Hospital)` — §4.1 (hlm. 8) dan §6 (hlm. 18) menyatakan bahwa kohort pelatihan (197 pasien) dan kohort pengujian (52 pasien) **berasal dari satu institusi yang sama (Osaka University Hospital)** dan sama-sama berupa DRR turunan CT. | hlm. 8 §4.1; hlm. 9 Tabel 1; hlm. 18 §6 |
| **`M130`** | `Q5_ext` | `Y` | **`N`** — Karena kohort 52 pasien berasal dari rumah sakit yang sama (Osaka University Hospital), studi ini merupakan *temporal/hold-out cohort* internal, bukan validasi pusat eksternal independen (penulis sendiri mencatat tidak adanya validasi multi-institusi sebagai keterbatasan di §6 hlm. 18). | hlm. 8 §4.1; hlm. 18 §6 |
| **`C093`** | `code` | `NR` | **Tersedia repositori Google Drive** (`https://drive.google.com/drive/folders/18_Ii7qpgA35PsSx-vsdctCfbqmnrSgbh`) yang memuat kode, dataset hasil kurasi, dan materi reproduktifitas. | hlm. 204 *Declarations (Availability of data and material)* |
| **`C093`** | `Q6_code` | `N` | **`Y`** — Diperbarui dari `N` menjadi `Y` sesuai tautan repositori publik pada bagian *Declarations* hlm. 204. | hlm. 204 *Declarations* |

### 2.3 Pelengkapan 14 Sel yang Sebelumnya Belum Diekstraksi (`numbers not extracted`, `NR in snippets`, `U`)

1. **`M041` (`Q3_patient`: `U` $\to$ `Y`):** §2.2 (hlm. 4, baris 213–215) secara eksplisit memastikan partisi dilakukan pada tingkat pasien (*"First, we partitioned our dataset at the patient level into training, validation, and test subsets..."*, 75%/10%/15%: 8.300 / 919 / 1.626 ROI DIP).
2. **`M061` (`Q3_patient`: `U` $\to$ `Y` & `key_comparison` dilengkapi):** §3.1–3.2 (hlm. 3) memastikan partisi *patient-stratified* pada Sites A–D (`8.050` citra) dan pengujian lintas-situs pada Site E (`953` citra). Angka lengkap Tabel 2 (Balanced Accuracy %, mean ± SE 5 *seeds* pada anggaran 800 label [~10%], dengan vs tanpa *consistency regularization* SSL):
   - `ENT-MCD`: **56.39 ± 0.59%** vs **52.18 ± 1.83%** (+4.21%; pada 250 label: 40.06% vs 37.43%).
   - `LC` (*Least Confidence*): **46.59 ± 1.55%** vs **45.17 ± 4.45%** (pada 250 label: 30.19% vs 25.01% [+5.18%]; pada 500 label: 43.99% vs 36.73% [+7.26%]).
   - `ENT` (*Naive Entropy*): **50.07 ± 1.08%** vs **49.97 ± 1.77%** (pada 250 label: 34.72% vs 23.37% [+11.35%]; pada 500 label: 40.49% vs 36.12% [+4.37%]).
   - `MI-MCD`: **50.91 ± 0.73%** vs **51.97 ± 1.35%** (SSL menghilangkan kejatuhan performa tajam pada iterasi menengah di Fig. 4c). Baseline penuh (100% label): **64.13 ± 0.88%**.
3. **`M071` (`Q3_patient`: `U` $\to$ `N` pada 5-fold CV):** Terungkap kontradiksi internal di dalam PDF: §3.4 (hlm. 7) mengklaim *no patient overlap*, tetapi **§8.5 (hlm. 22) secara eksplisit mengakui** bahwa untuk 5-fold CV, direktori `train` dan `val` dari dataset Kaggle `kneeKL224` digabung terlebih dahulu lalu dibagi dengan *stratified 5-fold CV* pada tingkat citra (karena berkas `kneeKL224` tidak menyertakan ID pasien).
4. **`M089` (`key_comparison` dilengkapi dari Tabel 1–2, hlm. 6–10):** Meskipun §1 menyebut 23 jaringan, §4 dan Tabel 1–2 mengevaluasi **21 varian arsitektur CNN** pada uji OAI ($n=1.656$). Penambahan *Ordinal Regression Module* (ORM) dengan *Cumulative-Link loss* memperbaiki **MAE dan QWK pada seluruh 21/21 jaringan** (rata-rata kenaikan $\Delta\text{QWK} = +0.0441$, rentang `+0.0183` hingga `+0.0825`) serta memperbaiki $\text{ACC}_{\text{Macro}}$ pada 17/21 jaringan:
   - `DenseNet-161` (terbaik): MAE **0.399 $\to$ 0.330** (-0.069), QWK **0.8204 $\to$ 0.8609** (+0.0405), $\text{ACC}_{\text{Macro}}$ **85.92% $\to$ 88.09%** (balanced recall 5-kelas dari matriks konfusi Fig. 3: **64.91% $\to$ 69.11%**).
   - `VGG-16bn`: MAE **0.373 $\to$ 0.338**, QWK **0.8228 $\to$ 0.8575** (+0.0347).
   - `VGG-19`: MAE **0.376 $\to$ 0.345**, QWK **0.8272 $\to$ 0.8532** (+0.0260).
   - `ResNet-34`: MAE **0.444 $\to$ 0.389**, QWK **0.7704 $\to$ 0.8314** (+0.0610).
   - `ResNet-50`: MAE **0.444 $\to$ 0.412**, QWK **0.7858 $\to$ 0.8161** (+0.0303).
   - `MobileNetV2`: MAE **0.479 $\to$ 0.402**, QWK **0.7560 $\to$ 0.8385** (+0.0825).
5. **`M185` (8 atribut dilengkapi dari hlm. 11–19, Tabel 2–9, Fig. 11 & 17):**
   - `dataset`: 5-kelas KL pada OAI (`9.786` asli + `5.400` hasil rekonstruksi MI-Rec = `15.186` citra) dan DKXI (`1.836` asli + `297` rekonstruksi = `2.133` citra), ditambah 2 dataset biner (OP dan KO).
   - `split` & `Q3_patient = N`: Pembagian 80% latih / 20% uji pada tingkat citra (di dalam data latih: 20% berlabel, 80% tanpa label). *Catatan penting:* Matriks konfusi uji OAI pada Fig. 11a berjumlah $n = 3.022 \approx 20\% \times 15.186$, mengindikasikan citra hasil rekonstruksi MI-Rec ikut digabungkan sebelum pembagian 80:20.
   - `external` & `Q5_ext = Y`: Evaluasi lintas-dataset (*cross-dataset generalizability* OAI $\leftrightarrow$ DKXI) dilaporkan pada §5.4 Tabel 9 (hlm. 19).
   - `runs_ci` & `Q2_runs = N`: Hanya melaporkan estimasi titik tunggal tanpa SD/CI multi-seed.
   - `key_comparison`: Pada OAI (20% label), H-SemiS mencapai Acc **84.9%**, F1 **84.7%** (melampaui 13 baseline; supervised terbaik Teoh 2023 Acc 83.1%/F1 80.8%; SSL terbaik Mean-Teacher Acc 80.3%/F1 81.9%). Ablasi pada OAI (§5.3, Tabel 6–8, Fig. 17): (a) *Masking* MI-Rec 75% (Acc 84.9%/F1 84.7%) vs MAE (79.5%/78.6%), CMAE (80.4%/82.2%), GAN-MAE (83.1%/81.8%); (b) Ambang SiRL $\tau_{80}$ (84.9%/84.7%) vs $\tau_{60}$ (78.6%/75.9%), $\tau_{70}$ (83.8%/83.8%), $\tau_{90}$ (79.9%/80.7%); (c) Modul Q-TeSt: Base Network (82.3%/82.1%) $\to$ +QCN (83.7%/83.2%) $\to$ +NL+QCN (84.9%/84.7%); (d) Dekomposisi hierarkis + RecS (84.9%/84.7%) vs OriS-only (80.4%/80.5%) vs Tanpa dekomposisi (77.2%/76.4%) vs Dekomposisi acak (74.1%/75.2%).
6. **`C093` (`key_comparison` dilengkapi dari Tabel VI hlm. 197 & Fig. 7–8 hlm. 199–201):**
   - Perbandingan *Class-Weighted Sampling* (`WeightedRandomSampler`) vs *Standard Unweighted Sampling* pada himpunan uji 4-kelas ($n=1.656$, keduanya memakai Focal Loss $\gamma=2$):
     - Untuk Ensemble (15% EfficientNet-B3 + 85% ConvNeXt-Tiny) + TTA (Tabel VI): *Standard sampling* menghasilkan Acc **82.50%**, Macro-Recall **79.92%**, Macro-F1 **81.06%**, MCC **0.6997**, Kappa **0.6980**; sedangkan *Class-weighted sampling* meningkatkan seluruh metrik menjadi Acc **83.00%** (+0.50%), Macro-Recall **81.26%** (+1.34%), Macro-F1 **81.94%** (+0.88%), MCC **0.7141**, Kappa **0.7135**, serta meningkatkan sensitivitas kelas minoritas/menengah **G2 dari 63.31% menjadi 75.8%** (Fig. 2b) dan PR-AUC G2 dari **0.76 menjadi 0.79** (Fig. 4).
     - Untuk model tunggal & TTA (Fig. 8): Di bawah *Standard sampling* (No-TTA $\to$ TTA): EfficientNet-B3 `77.8% -> 79.2%`, ConvNeXt-Tiny `82.9% -> 83.1%`, Ensemble `81.4% -> 82.5%`. Di bawah *Class-weighted sampling* (No-TTA $\to$ TTA): EfficientNet-B3 `78.4% -> 79.6%`, ConvNeXt-Tiny **`82.6% -> 83.8%`** (tertinggi), Ensemble `82.0% -> 83.0%`. Bootstrap 95% CI ($N=1.000$, Fig. 7) menunjukkan CI antara ConvNeXt-Tiny tunggal (`82.85% [81.10–84.66%]`) dan Ensemble (`82.92% [81.22–84.72%]`) saling tumpang tindih.

---

## 3. Audit Forensik Kasus Khusus (Sesuai `verification/README.md` §3)

### 3.1 Studi dengan Klaim Performa Tidak Masuk Akal / Indikasi Fabrikasi (`M095`, `M150`, `M175`, `M188`)

| ID | Klaim yang Dilaporkan | Bukti Forensik dari Naskah Lengkap (`slr-methods/papers/`) | Rekomendasi Verifikasi |
|---|---|---|---|
| **`M095`** | Macro-F1 `0.982`, QWK `0.991`, ECE `0.009` pada KL 5-kelas | **Terbukti naskah hasil fabrikasi / teks generatif LLM yang belum disunting:**<br>1. **Placeholder Template yang Belum Diganti (hlm. 883, Tabel 11):** Memuat teks literal `"[X] ... Deep learning for knee OA staging on radiography. [ F ]"` dan `"[X] ... Uncertainty-aware medical image classification. [ TMI ]"`.<br>2. **Pengakuan Gambar Mockup (hlm. 882, §6.9):** Menyatakan secara eksplisit *"Saliency and Failure Case Panels (Illustrative)... All panels are synthetic, illustrative mockups summarizing typical behaviors..."*.<br>3. **Teks Prompt Generik (hlm. 876, §5.8):** Menulis *"Institutional data (if present) are governed by local approvals..."* padahal Tabel 3 mengklaim 3 situs rumah sakit dengan ukuran sampel kelipatan bulat 100 (`400, 300, 100` pasien; jumlah kelas `120, 80, 90, 70, 40`).<br>4. **Anomali Parameter (hlm. 882, Tabel 9):** Penambahan *test-time batch-norm adaptation* (yang tidak mengubah arsitektur) justru diklaim menurunkan jumlah parameter model dari `31.8M` menjadi `30.4M`. | **Eksklusi (`exclude: fabricated_paper`)** — Jangan gunakan ukuran efek maupun klaim ablasinya dalam sintesis bukti. |
| **`M175`** | Akurasi `98.97%` / `99.14%` dengan CenterNet + DenseNet201 + KD | **Terbukti tidak memiliki hasil eksperimen (fabrikasi angka di Kesimpulan):**<br>1. **Section V (*"RESULT ANALYSIS"*, hlm. 808) Kosong dari Data:** Hanya terdiri dari 2 paragraf pendek tanpa satu pun tabel hasil, grafik, matriks konfusi, maupun angka metrik, dan ditulis dalam kalimat masa depan (*"Accuracy, precision, recall, and F1-score will be computed..."*).<br>2. **Angka Muncul Tiba-tiba di Kesimpulan (§VI, hlm. 808):** Mengklaim akurasi `98.97%` dan `99.14%` pada "Dataset 1" dan "Dataset 2" padahal §III hanya mendeskripsikan satu dataset.<br>3. **Kesalahan Aritmetika & *Tortured Phrases* (hlm. 805, Tabel I):** Menulis `500 + 400 + 200 + 200 + 200 = 2000` (padahal jumlahnya `1.500`), serta memuat frasa terjemahan mesin (*"deep Gaining knowledge of strategies"*, *"The kneecap (patella) bureaucracy the bony front"*). | **Eksklusi (`exclude: EM6 / fabricated_results`)** — Tidak ada evaluasi eksperimental yang valid. |
| **`M188`** | Tabel 5 mengklaim $\text{AP}_{K2}=98.6\%$, $\text{AP}_{K3}=98.5\%$, $\text{AP}_{K4}=99.6\%$ (`mAP = 98.8%`) | **Terbukti kontradiksi internal langsung antara Tabel 5 (kurva latih) dan Matriks Konfusi Uji (Fig. 10b):**<br>1. **Fig. 10b & §4.4 (hlm. 9):** Melaporkan akurasi per kelas pada himpunan uji hanya **K2 = 48%, K3 = 65%, K4 = 75%**.<br>2. **§5 Keterbatasan (3) (hlm. 13):** Penulis sendiri mengakui *"the proposed TPAFFKnee is still not accurate enough for the detection of KL type 2, with an accuracy of only 48%."*<br>3. **Label Kurva (Fig. 9, 12, 14):** Grafik yang mencapai `0.988 mAP@0.5` diberi judul eksplisit *"Comparative experiment results of **training**"* (dievaluasi pada data latih setelah augmentasi 10× sebelum pemisahan). | **Jangan gunakan angka Tabel 5 (`98.8%`) sebagai performa uji.** Gunakan performa uji sebenarnya di Fig. 10b (`K2 48%, K3 65%, K4 75%`) dengan peringatan *leakage* augmentasi pra-split. |
| **`M150`** | Akurasi `98%`, QWK `0.99`, Macro-F1 `0.98` pada OAI 5-kelas (`92.9%` recall pada KL-1) | **Tidak kredibel (indikasi kuat *data leakage* / duplikasi augmentasi pra-split):**<br>1. Menggunakan CNN standar (ResNet-50, DenseNet-121, MobileNetV2) + CORN loss pada dataset Kaggle/OAI 9.786 citra, di mana model dan fungsi rugi yang sama pada split OAI yang bersih hanya mencapai akurasi `58%–70%` dan recall KL-1 `<35%`.<br>2. Melakukan *image-level random split* 70:20:10 (`Q3_patient = N`) dan **tidak melakukan ablasi CORN terhadap Cross-Entropy** (`Q1_abl = N`). | **Pertahankan hanya sebagai contoh peringatan metodologis (*include_with_caution*)**; keluarkan dari ringkasan efek kuantitatif G2. |

### 3.2 Studi yang Memodifikasi Label Himpunan Uji (`M042`, `M272`, dan Temuan Tambahan `M117`)

| ID | Mekanisme Modifikasi Label Uji (*Test-Set Label Modification*) | Dampak Kuantitatif pada Metrik yang Dilaporkan | Implikasi untuk Sintesis SLR & Eksperimen Tesis |
|---|---|---|---|
| **`M042`** | **Menggabungkan seluruh 8.260 citra (`train 5.778 + val 826 + test 1.656`) menjadi satu kumpulan (§III.A hlm. 34020)**, menjalankan 5-fold *out-of-fold* VGG19 Cascade-I pada seluruh kumpulan tersebut untuk membuang 6 citra dan **mengubah label 286 citra** ke grade sebelahnya, lalu membagi kumpulan yang sudah direlabeli menjadi 70%/10%/20% (`1.594` citra uji) untuk melatih dan menguji VGG19 Cascade-II. | Akurasi pada himpunan uji yang telah direlabeli oleh arsitektur yang sama (VGG19) melonjak secara artifisial sebesar **`+24.13%`** (`57.99% -> 82.12%`, QWK `0.789 -> 0.935`). Penulis **tidak pernah melaporkan performa Cascade-II pada label uji asli**. | **Bias sirkularitas (*circularity bias*):** model Cascade-II hanya sedang memprediksi prediksi VGG19 Cascade-I. Tidak boleh dibandingkan dengan studi yang mengevaluasi pada label uji OAI asli. |
| **`M272`** | Menjalankan *outlier removal* dan *Cleanlab confident learning* **secara terpisah pada himpunan latih, validasi, dan uji** (§3.2.2 hlm. 10, §4.4 hlm. 16), yang **menimpa 364 dari 1.656 label uji (`22.0%`)** dan 214 dari 826 label validasi (`25.9%`). | Pada label uji yang telah diubah oleh Cleanlab, akurasi naik dari `66.36%` menjadi **`82.07%`** ($p = 3.37 \times 10^{-14}$). **Namun, ketika dievaluasi pada label uji asli yang tidak disentuh (§4.3 hlm. 15), perbedaan antara pelatihan dengan label asli vs label Cleanlab TIDAK signifikan secara statistik (McNemar $p = 0.076$).** | **Bukti kunci untuk RQ5:** Hampir seluruh kenaikan `+13%` hingga `+16%` yang diklaim berasal dari penggantian label uji dengan label yang disukai model, bukan peningkatan generalisasi pada *ground truth* klinis asli. |
| **`M117`** *(Temuan Audit)* | Menerapkan *autoencoder + Isolation Forest outlier removal* (~3% citra dibuang) dan *CleanLab-Plus* (384 label diubah, 179 dibuang, sisanya diberi bobot kepercayaan) pada dataset Kaggle 9.786 citra lintas *5-fold CV* (`Q3_patient = N`), serta Tabel 2 (hlm. 11) menggabungkan `Dual-CBAM + CleanLab` dalam satu langkah (`82.4% -> 86.2%`) tanpa mengisolasi kontribusi CleanLab (`Q1_abl = N` untuk G4). | Akurasi CV naik dari `78.9%` (baseline) menjadi `88.5%`, tetapi dipengaruhi oleh pembersihan sampel sulit/berisik dari fold evaluasi dan digabungkan dengan modul atensi. | Konfirmasi langsung atas kecurigaan pada catatan ekstraksi awal AI: pembersihan label sebelum/lintas CV membersihkan lipatan uji. |

### 3.3 Verifikasi Status `papers/00_TODO_user_download.md`

Dari **40 rekaman** yang sebelumnya tercatat di `slr-methods/papers/00_TODO_user_download.md`:
- **21 rekaman telah berhasil diperoleh** dan kini tersedia di `slr-methods/papers/` serta telah diaudit penuh:
  `M003`, `M041`, `M051`, `M061`, `M077`, `M089`, `M091`, `M106`, `M110`, `M132`, `M153`, `M157`, `M161`, `M162`, `M175`, `M180`, `M269`, `M270`, `M275`, `M283`, `M305`.
- **19 rekaman masih belum diperoleh (*reports not retrieved*)**:
  `M001`, `M014`, `M028`, `M052`, `M068`, `M082`, `M083`, `M084`, `M101`, `M148`, `M166`, `M200`, `M224`, `M225`, `M233`, `M266`, `M280`, `M303`, `M329`.

---

## 4. Verifikasi Kriteria Kelayakan Full-Text & Checklist Kualitas Seluruh 54 Paper (`03_fulltext_criteria_verification.csv`)

Pada draf awal `screening/02_prisma_counts.md`, tercatat `Dikeluarkan di tahap full text: 0` sehingga seluruh 54 berkas yang terunduh langsung dimasukkan ke sintesis. **Sebagai reviewer manusia objektif, pemeriksaan naskah lengkap seluruh 54 paper di `slr-methods/papers/` menunjukkan bahwa 10 berkas tidak memenuhi kriteria inklusi penuh (atau merupakan duplikat), sementara 44 studi (40 studi bersih + 4 studi dengan catatan kehati-hatian metodologis) layak dipertahankan.**

### 4.1 Daftar 10 Berkas di `slr-methods/papers/` yang Direkomendasikan Dikeluarkan pada Tahap Full-Text

| Kategori Eksklusi Full-Text | ID | Kode Kriteria | Alasan Objektif Berdasarkan Naskah Lengkap (`slr-methods/papers/`) |
|---|---|---|---|
| **1. Tidak Mengevaluasi Tugas Grading Ordinal Multi-Kelas (`EM2`)** | **`M173`** | `EM2` | Hanya mengevaluasi kualitas visual citra sintetis (FID, Visual Turing Test, JLCA) tanpa melatih/menguji classifier grading KL (hlm. 123). |
| | **`M165`** | `EM2` | Meskipun abstrak menyebut tingkat keparahan KOA, pemeriksaan §2 (hlm. 5: *"last layer is Dense with one unit and sigmoid activation"*) dan Tabel 6 (hlm. 6) membuktikan classifier hilir yang dilatih **hanya melakukan klasifikasi biner (`KL01` vs `KL234`)**, bukan grading multi-derajat. |
| **2. Tidak Menguji Metode G1–G4 (`EM3`) / Evaluasi Tidak Memadai (`EM6`)** | **`M180`** | `EM3` | Meskipun dimasukkan karena dugaan regresi ordinal pada GNN, pemeriksaan Tabel 1 (hlm. 9) membuktikan model menggunakan **Categorical Cross-Entropy standar dari inisialisasi acak** dengan optimasi hiperparameter *Secretary Bird Optimization Algorithm* (SBOA), tanpa komponen G1–G4. |
| | **`M051`** | `EM3 / EM6` | Hanya membandingkan fungsi aktivasi lapisan dense (`Linear, ReLU, Tanh, Sigmoid`) di atas 4 backbone CNN beku tanpa membandingkan strategi *layer freezing/fine-tuning* (`G1`) dan hanya melaporkan kurva akurasi/loss per epoch tanpa tabel metrik himpunan uji. |
| | **`M249`** | `EM3` | Hanya melatih satu model standar EfficientNetV2-M dengan Cross-Entropy pada berbagai pemotongan citra (*masking/cropping*) untuk mendiagnosis *shortcut learning* pada Grad-CAM, bukan mengusulkan/menguji metode G1–G4 untuk meningkatkan grading ordinal (namun berguna sebagai referensi uji diagnostik di diskusi). |
| **3. Preprint Belum Peer-Review (`EM7`)** | **`M191`** | `EM7 + EM2/EM3` | Preprint arXiv (`arXiv:2507.01972v1`) yang menggunakan terjemahan CycleGAN sebagai **serangan adversarial 1-shot** untuk mengecoh classifier pra-latih, bukan untuk meningkatkan performa grading. |
| | **`M199`** | `EM7` | Naskah kiriman konferensi anonim di arXiv (`arXiv:2502.01972v1`, penulis `"Anonymous"`, hlm. 9: *"If a paper is accepted..."*) yang belum melalui peer-review. |
| **4. Indikasi Fabrikasi Data/Hasil (`EM6`)** | **`M095`** | `EM6 / Fabricated` | Memuat placeholder template LLM (`[X]`, `[ F ]`, `[ TMI ]` di Tabel 11 hlm. 883), gambar mockup sintetis (§6.9 hlm. 882), dan angka sempurna yang tidak realistis (`Macro-F1 0.982`). |
| | **`M175`** | `EM6 / Fabricated` | Bagian *Result Analysis* (§V hlm. 808) kosong dari tabel/eksperimen dan ditulis dalam bentuk kalimat masa depan, sementara §VI mengklaim akurasi `98.97%`/`99.14%`. |
| **5. Duplikat Publikasi (`Duplicate`)** | **`M283`** | `Duplicate of M141` | Merupakan laporan konferensi yang identik (*KneeXNeT*: penulis, dataset 8.260 citra, 10 model, *weighted sampling* + ensemble, dan tabel hasil yang sama persis) dengan artikel jurnal **`M141`**. Wajib dihitung satu kali sebagai `M141`. |

### 4.2 Revisi Penilaian Kualitas Bukti (`Q1–Q6`) Berdasarkan Verifikasi Naskah Lengkap

Pemeriksaan PDF/XML penuh berhasil menyelesaikan sebagian besar status `U` (*Unclear*) yang sebelumnya tercatat pada ekstraksi berbasis snippet (`01_fulltext_extraction.tsv`):

| Dimensi Kualitas (`protocol/01_protocol_PRISMA-P.md` §7) | Distribusi Awal AI (54 Berkas) | Distribusi Terverifikasi Reviewer (Seluruh 54 Berkas) | Distribusi Terverifikasi pada **44 Studi Valid yang Dipertahankan** | Perubahan Kunci dari Audit Full-Text |
|---|---|---|---|---|
| **`Q1_abl` (Ablasi terkontrol komponen G1–G4)** | `Y: 39, U: 10, N: 5` | **`Y: 40, N: 14, U: 0`** | **`Y: 36 (81.8%), N: 8 (18.2%)`** | Seluruh 10 status `U` berhasil dipastikan dari PDF penuh (`Y` pada `M106, M162, M269, M306, C007, C037`; `N` pada `M077, M117, M150, M305, C015`). |
| **`Q2_runs` ($\ge 3$ seed / CV / CI dilaporkan)** | `Y: 20, N: 23, U: 11` | **`Y: 20, N: 25, U: 9`** | **`Y: 19 (43.2%), N: 20 (45.5%), U: 5 (11.4%)`** | `M185` dipastikan `N` (estimasi titik tunggal); `M061` dikoreksi menjadi `mean ± SE (5 seeds)`. |
| **`Q3_patient` (Pemisahan tingkat pasien)** | `Y: 8, N: 16, U: 30` | **`Y: 14, N: 26, U: 14`** | **`Y: 13 (29.5%), N: 21 (47.7%), U: 10 (22.7%)`** | **6 studi naik dari `U` ke `Y`** setelah verifikasi teks metode (`M041, M061, M100, M106, M153, M335`), dan **10 studi dipastikan `N`** (`M042, M071, M095, M175, M185, M188, M275, M305, M306, C007`). |
| **`Q5_ext` (Validasi eksternal lintas pusat/kohort)** | `Y: 10, N: 39, U: 5` | **`Y: 11, N: 43, U: 0`** | **`Y: 11 (25.0%), N: 33 (75.0%)`** | Seluruh 5 status `U` terpecahkan (`M091, M153, M185, M335` $\to$ `Y`; `M077` $\to$ `N`), sementara `M130` dikoreksi dari `Y` ke `N` (kohort rumah sakit yang sama). |
| **`Q6_code` (Kode / bobot model publik)** | `Y: 9, P: 4, N: 41` | **`Y: 10, P: 3, N: 41`** | **`Y: 8 (18.2%), P: 2 (4.5%), N: 34 (77.3%)`** | `C093` dinaikkan dari `N` ke `Y` (tautan repositori Google Drive pada hlm. 204). |

### 4.3 Revisi Diagram Alir PRISMA (`screening/02_prisma_counts.md`) yang Direkomendasikan

Berdasarkan hasil verifikasi di atas, angka alir PRISMA pada tahap kelayakan (*eligibility*) dan inklusi akhir (*included*) direkomendasikan untuk diperbarui sebagai berikut:
- **Laporan yang dicari untuk diambil (*Reports sought for retrieval*):** `73` (69 dari basis data + 4 dari *citation searching*).
- **Laporan yang tidak diperoleh (*Reports not retrieved*):** `19` (sesuai audit `00_TODO_user_download.md`).
- **Laporan yang dinilai kelayakannya pada teks penuh (*Reports assessed for eligibility*):** `54` (`50` dari basis data + `4` dari *citation searching*).
- **Laporan yang dikeluarkan pada tahap teks penuh (*Reports excluded at full text*):** **`10`**
  - `EM2` (Bukan grading ordinal multi-kelas): `2` (`M165`, `M173`)
  - `EM3` (Tidak menguji metode G1–G4): `3` (`M051`, `M180`, `M249`)
  - `EM6` (Tanpa hasil eksperimen valid / indikasi fabrikasi): `2` (`M095`, `M175`)
  - `EM7` (Preprint belum peer-review): `2` (`M191`, `M199`)
  - Duplikat publikasi dari studi yang sudah masuk (`M141`): `1` (`M283`)
- **Studi yang dimasukkan dalam sintesis akhir (*Studies included in review*):** **`44` studi unik** (`40` studi utama + `4` studi dengan peringatan metodologis khusus `M042, M117, M150, M188`).
