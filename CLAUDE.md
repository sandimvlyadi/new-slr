# CLAUDE.md — Panduan Lintas-Agent & Blueprint Penyelesaian Tesis (`new-slr`)

Dokumen ini berfungsi sebagai **Single Source of Truth (SSOT)** dan panduan serah-terima lintas-agen (*cross-agent handover guide* untuk Claude Code, Antigravity, Cursor, Codex, dll.) pada repositori `/data/Smart-X/new-slr`.

---

## 1. Identitas Proyek & Status Terkini

* **Judul Tesis Utama (Bahasa Indonesia):**
  *Pembelajaran Representasi, Penanganan Ketidakseimbangan Ordinal, Kuantifikasi Ketidakpastian, dan Koreksi Derau Label untuk Grading Erosi Sendi Tangan pada Radiografi Artritis Reumatoid dengan Validasi Eksternal Lintas Populasi*
* **Judul Tesis Utama (Bahasa Inggris):**
  *Representation Learning, Ordinal Imbalance Handling, Uncertainty Quantification, and Label-Noise Mitigation for Joint-Level Rheumatoid Arthritis Erosion Grading on Hand Radiographs with Cross-Population External Validation*
* **Status Eksperimen GPU & Analisis Statistik Lokal (`0-GPU`):** **100% SELESAI (`442` run/evaluasi tuntas)**.
  * Seluruh eksperimen GPU (`RQ0`, `RQ1`, `RQ2`, `RQ1×RQ2`, `RQ3`, `RQ4-01`, `RQ5`, dan `RQ5-11 Mean-Teacher`) telah selesai dijalankan di NVIDIA L4 GPU dan disinkronkan ke [`runs/`](runs/).
  * Seluruh analisis *post-hoc* dan pengujian hipotesis (*Patient-Level Cluster Bootstrap* $B=5.000$ replikasi + koreksi *step-down Holm-Bonferroni* $\alpha=0,05$) telah selesai dijalankan melalui [`scripts/run_all_analyses.py`](scripts/run_all_analyses.py).
  * Hasil master tersimpan di [`runs/analysis_master.json`](runs/analysis_master.json) dan tabel laporan lengkap telah digenerasi di [`report/06_hasil_eksperimen_lengkap_RQ0_RQ5.md`](report/06_hasil_eksperimen_lengkap_RQ0_RQ5.md).
* **Status Penulisan Naskah Tesis ([`thesis/`](thesis/)):** **100% SELESAI (Bab I s.d. Bab VI Lengkap)**
  * [x] [`thesis/Bab_I_Pendahuluan.md`](thesis/Bab_I_Pendahuluan.md) (`25,8 KB`) — **SELESAI**
  * [x] [`thesis/Bab_II_Tinjauan_Pustaka_dan_SLR.md`](thesis/Bab_II_Tinjauan_Pustaka_dan_SLR.md) (`70,0 KB`) — **SELESAI**
  * [x] [`thesis/Bab_III_Metodologi_Penelitian.md`](thesis/Bab_III_Metodologi_Penelitian.md) (`92,5 KB`) — **SELESAI**
  * [x] [`thesis/Bab_IV_Hasil_dan_Analisis_Internal.md`](thesis/Bab_IV_Hasil_dan_Analisis_Internal.md) (`50,1 KB`) — **SELESAI**
  * [x] [`thesis/Bab_V_Validasi_Eksternal_dan_Diagnostik.md`](thesis/Bab_V_Validasi_Eksternal_dan_Diagnostik.md) (`31,5 KB`) — **SELESAI**
  * [x] [`thesis/Bab_VI_Kesimpulan_dan_Saran.md`](thesis/Bab_VI_Kesimpulan_dan_Saran.md) (`20,1 KB`) — **SELESAI**
  * [x] [`thesis/references.bib`](thesis/references.bib) (`46,1 KB`) — **SELESAI**

---

## 2. Peta Direktori & Berkas Kunci

| Path Berkas / Direktori | Fungsi & Keterangan |
|---|---|
| [`rules/01_pre_specified_analysis_protocol.md`](rules/01_pre_specified_analysis_protocol.md) | **Protokol Pra-Spesifikasi Mengikat:** Aturan *single-axis ablation* (`Q1_abl=Y`), 6 run per arm (`3 seed @ 100%` + `3 draw @ 10%`, `Q2_runs=Y`), *strict patient-level split* (`Q3_patient=Y`), *validation-only model selection* (`Q4_valsel=Y`), imutabilitas label uji pada `RQ5`, *5,000-rep patient-cluster bootstrap*, koreksi *Holm-Bonferroni*, dan kewajiban pelaporan jujur hasil negatif/null (Pasal 6). |
| [`report/05_rencana_tesis_dan_validasi_eksternal.md`](report/05_rencana_tesis_dan_validasi_eksternal.md) | **Blueprint & Kerangka Bab I–VI Tesis:** Memuat struktur subbab, target tabel/diagram, dan rasional ilmiah Bab I s.d. Bab VI. |
| [`report/06_hasil_eksperimen_lengkap_RQ0_RQ5.md`](report/06_hasil_eksperimen_lengkap_RQ0_RQ5.md) | **Tabel Lengkap Hasil Eksperimen `RQ0–RQ5`:** Seluruh nilai `Test BACC4 [95% CI]`, `Test QWK [95% CI]`, $\Delta$ terhadap anchor `RQ0-01`, $p_{\text{raw}}$, $p_{\text{Holm}}$, `Macro-F1`, `MAE`, `Recall>=2`, kalibrasi `RQ4`, conformal `RQ4`, validasi eksternal `RheumaVIT` `RQ3`, benchmark 3 radiolog, dan evaluasi distributional `RQ5-E`. |
| [`runs/analysis_master.json`](runs/analysis_master.json) | **Data JSON Master Statistik:** Sumber angka presisi untuk seluruh tabel di Bab IV, Bab V, dan Bab VI. |
| [`scripts/run_all_analyses.py`](scripts/run_all_analyses.py) | Skrip Python murni (*stdlib-only*, berjalan tanpa dependensi eksternal di dalam sandbox) yang menghitung seluruh bootstrap, threshold tuning `RQ2-Pa`, kalibrasi/conformal `RQ4`, dan evaluasi `RQ3`/`RQ5-E`. |
| [`scripts/generate_report_06.py`](scripts/generate_report_06.py) | Skrip generator [`report/06_hasil_eksperimen_lengkap_RQ0_RQ5.md`](report/06_hasil_eksperimen_lengkap_RQ0_RQ5.md) dari [`runs/analysis_master.json`](runs/analysis_master.json). |
| [`thesis/Bab_I_Pendahuluan.md`](thesis/Bab_I_Pendahuluan.md) | Bab I Pendahuluan (Latar Belakang, Identifikasi Masalah, Rumusan `RQ1–RQ5`, Tujuan & Kontribusi). |
| [`thesis/Bab_II_Tinjauan_Pustaka_dan_SLR.md`](thesis/Bab_II_Tinjauan_Pustaka_dan_SLR.md) | Bab II Tinjauan Pustaka & SLR-M (Audit 44 studi valid G1–G4, bukti forensik sirkularitas *label noise* `M042/M117/M272`, kandidat ablasi). |
| [`thesis/Bab_III_Metodologi_Penelitian.md`](thesis/Bab_III_Metodologi_Penelitian.md) | Bab III Metodologi Penelitian & Formulasi Matematis Eksak (`RAM-H1200`, `RheumaVIT`, `RQ0–RQ5`). |
| [`thesis/references.bib`](thesis/references.bib) | Basis data BibTeX lengkap (gunakan sitasi Pandoc/Markdown `[@Key]` seperti `[@Yang2026RAMH1200]`, `[@Stolpovsky2023RheumaVIT]`, `[@vanderHeijde2000SvdH]`, `[@Menon2021LogitAdj]`, `[@Shi2023CORN]`, `[@Khosla2020SupCon]`, `[@Kumar2022LPFT]`, `[@Angelopoulos2021RAPS]`, `[@Northcutt2021ConfidentLearning]`, serta kode studi SLR-M `[@M039]`, `[@M041]`, `[@M042]`, `[@M071]`, `[@M085]`, `[@M091]`, `[@M095]`, `[@M110]`, `[@M117]`, `[@M154]`, `[@M157]`, `[@M210]`, `[@M219]`, `[@M231]`, `[@M272]`, `[@C093]`). |

---

## 3. Aturan Penulisan & Standar Akademik Wajib (Untuk Semua Agent)

1. **Bahasa & Gaya Penulisan:**
   * Gunakan **Bahasa Indonesia akademik formal setara Tesis Magister ITB / jurnal Q1** dengan istilah teknis bahasa Inggris dicetak miring (*italic*) pada kemunculan pertama atau saat merujuk nama metode/metrik.
   * Hindari ringkasan dangkal: setiap subbab pada Bab IV dan Bab V harus memuat **(a) Tabel Kuantitatif Lengkap dengan 95% CI Bootstrap & $p_{\text{Holm}}$**, **(b) Analisis Mekanisme Matematis & Empiris** (mengapa metode tersebut berhasil atau gagal berdasarkan gradien, distribusi kelas, atau struktur anatomi radiograf), dan **(c) Sintesis Komparatif terhadap Temuan Literatur SLR-M Bab II** (mengonfirmasi atau merefutasi klaim studi `Mxxx`/`Cxxx`).
2. **Konsistensi Angka 100% dengan [`runs/analysis_master.json`](runs/analysis_master.json) & [`report/06_hasil_eksperimen_lengkap_RQ0_RQ5.md`](report/06_hasil_eksperimen_lengkap_RQ0_RQ5.md):**
   * Dilarang keras mengarang atau membulatkan angka secara sembarangan. Gunakan angka 4 desimal sesuai [`report/06_hasil_eksperimen_lengkap_RQ0_RQ5.md`](report/06_hasil_eksperimen_lengkap_RQ0_RQ5.md).
   * Setiap klaim peningkatan performa wajib menyertakan nilai `Test BACC4` dan `Test QWK`, rentang `[95% CI]` dari *patient-level cluster bootstrap* ($B=5.000$), selisih berpasangan $\Delta$ terhadap jangkar `RQ0-01`, nilai-$p$ mentah ($p_{\text{raw}}$), dan nilai-$p$ terkoreksi Holm-Bonferroni ($p_{\text{Holm}}$).
   * Bedakan secara jujur antara:
     * **Signifikan secara statistik pasca-koreksi Holm ($p_{\text{Holm}} < 0,05$)** — diberi tanda `*`.
     * **Tren peningkatan pra-koreksi ($p_{\text{raw}} < 0,05$ namun $p_{\text{Holm}} \ge 0,05$)** — harus disebut sebagai *tren peningkatan konsisten yang belum melewati ambang konservatif keluarga-jamak Holm*.
     * **Efek Null / Negatif / Refutasi ($p_{\text{raw}} \ge 0,05$ atau $\Delta < 0$)** — wajib dilaporkan secara transparan sesuai Pasal 6 [`rules/01_pre_specified_analysis_protocol.md`](rules/01_pre_specified_analysis_protocol.md).
3. **Format Visual & Diagram:**
   * Gunakan diagram `mermaid` (`flowchart TD`, `flowchart LR`, dll. dengan label berpetik ganda `"..."`) dan blok GFM alert (`> [!IMPORTANT]`, `> [!NOTE]`, `> [!TIP]`, `> [!WARNING]`) untuk memperjelas alur analisis dan temuan kunci.

---

## 4. Ringkasan Eksekutif Angka Eksak untuk Bab IV, Bab V, dan Bab VI

### 4.1 `RQ0` (Anchor & Arsitektur)
* **`rq0-01` (`ResNet-34 + B2` [Logit Adjustment + InvSqrtSampler] — ANCHOR):**
  * `100% label (f1.0)`: $S_{\text{val}} = 0.4530$, `Test BACC4 = 0.4855 [0.4362, 0.5141]`, `Test QWK = 0.4832 [0.3822, 0.5683]`, `Macro-F1 = 0.4043`, `MAE = 0.3940`, `Rec>=2 = 0.2998`.
  * `10% label (f0.1)`: $S_{\text{val}} = 0.2588$, `Test BACC4 = 0.3619 [0.3346, 0.3836]`, `Test QWK = 0.2352 [0.1537, 0.3222]`, `Macro-F1 = 0.2653`, `MAE = 0.5515`, `Rec>=2 = 0.3711`.
* **`rq0-02` (`ConvNeXt-Tiny + B2`):**
  * `100% label`: `Test BACC4 = 0.4964 [0.4494, 0.5256]` ($\Delta = +0.0109, p_{\text{Holm}} = 0.3243$), **`Test QWK = 0.5515 [0.4648, 0.6258]` ($\Delta = +0.0682, p_{\text{Holm}} = 0.0012^*$)**, `Macro-F1 = 0.4496`, `MAE = 0.3404`.
  * `10% label`: `Test BACC4 = 0.3802 [0.3490, 0.4036]` ($\Delta = +0.0183, p_{\text{Holm}} = 0.2571$), **`Test QWK = 0.3164 [0.2286, 0.3988]` ($\Delta = +0.0812, p_{\text{Holm}} = 0.0004^*$)**, `Macro-F1 = 0.3046`, `MAE = 0.4979`.

### 4.2 `RQ1` (Pra-Pelatihan, SSL, `SupCon-Ord`, Normality, & `LP-FT` — 14 Arm, 84 Run)
* **Temuan Puncak 100% Label (`f1.0`):**
  * **`rq1-13` (`SupCon-Ord` — *Grade-Distance-Weighted SupCon*, `[@M219]`):** **`Test BACC4 = 0.5034 [0.4615, 0.5291]`** ($\Delta = +0.0179, p_{\text{raw}} = 0.0504$) dan **`Test QWK = 0.5111 [0.4163, 0.5912]`** ($\Delta = +0.0278, p_{\text{raw}} = 0.0064, p_{\text{Holm}} = 0.0384^*$), `MAE = 0.3516` (terendah di RQ1). Mengungguli `SupCon` standar tanpa bobot jarak (`rq1-12`: `BACC4 = 0.4733`, `QWK = 0.4959`), membuktikan bahwa penalti kontrastif harus proporsional terhadap jarak derajat SvdH $|y_i - y_j|$.
  * **`rq1-06` (`Domain SSL DINO ViT-S/16 + LLRD`):** `Test BACC4 = 0.5070 [0.4631, 0.5241]` ($\Delta = +0.0215$), `Test QWK = 0.4268`, `Rec>=2 = 0.4114`.
  * **Konfirmasi Teori `LP-FT` (`[@Kumar2022LPFT]`, `[@M157]`) pada `rq1-17a..d`:**
    * *Frozen Linear Probe* menghancurkan sensitivitas ordinal: `rq1-17a` (`LP ImageNet R34`) jatuh ke `QWK = 0.2307` ($\Delta = -0.2525, p_{\text{Holm}} = 0.0056^*$) dan `rq1-17c` (`LP MAE ViT-S`) jatuh ke `QWK = 0.3367` ($\Delta = -0.1465, p_{\text{Holm}} = 0.0056^*$).
    * Sebaliknya, **`LP-FT` (*Linear Probe then Fine-Tune*)** menyelaraskan kepala klasifikasi sebelum membuka *backbone*, menghasilkan lompatan besar: `rq1-17b` (`LP-FT ImageNet R34`) mencapai **`Test BACC4 = 0.5012`** & **`Test QWK = 0.5023`** ($\Delta\text{QWK} = +0.0191$), sementara `rq1-17d` (`LP-FT MAE ViT-S`) mencapai skor validasi tertinggi di seluruh RQ1 ($S_{\text{val}} = 0.4676$) dan mengangkat `Test BACC4` `MAE ViT-S` dari `0.4548` (`rq1-05`) menjadi **`0.5011`** (`+4.63%`).
* **Temuan Puncak 10% Label (`f0.1`):**
  * **`rq1-05` (`Domain SSL MAE ViT-S/16 + LLRD`):** Terbaik secara keseluruhan pada rezim *low-label*, mencapai **`Test BACC4 = 0.4148 [0.3689, 0.4502]`** ($\Delta = +0.0529, p_{\text{raw}} = 0.0216$) dan **`Test QWK = 0.3300 [0.2240, 0.4369]`** ($\Delta = +0.0948, p_{\text{raw}} = 0.0016, p_{\text{Holm}} = 0.0160^*$), diikuti oleh `rq1-17d` (`LP-FT MAE ViT-S`: `BACC4 = 0.4062`, `QWK = 0.3090`, $\Delta\text{QWK} = +0.0738, p_{\text{raw}} = 0.0068$).
* **Bukti Refutasi / Negatif di `RQ1`:**
  * `rq1-01` (`From Scratch R34`): Kolaps pada 10% label (`Test BACC4 = 0.2547`, `Test QWK = 0.0058`, $p_{\text{Holm}} = 0.0056^*$) dan tertinggal jauh pada 100% label (`BACC4 = 0.4408`, `QWK = 0.3514`, $p_{\text{Holm}} = 0.0056^*$).
  * `rq1-14` (`SS-FewSOME Normality`, `[@M231]`) & `rq1-15` (`2-Stage Normality Gate`): Gagal memisahkan gradasi erosi (`rq1-14`: `BACC4 = 0.2883`, `QWK = -0.0419`; `rq1-15`: `BACC4 = 0.4200`, `QWK = 0.1119`), membuktikan bahwa jarak ke prototipe "sendi sehat" (*one-class normality*) tidak memiliki resolusi metrik untuk membedakan `Grade 1, 2, 3, 4+`.

### 4.3 `RQ2` (Imbalance, Head Ordinal, Post-Hoc `RQ2-P`, & Interaksi `RQ1×RQ2` — 22 Arm, 132 Run)
* **Keruntuhan Tanpa Penanganan Imbalance (`rq2-01` Plain CE):** `Test BACC4 = 0.4132` (`100%`, $\Delta = -0.0723, p_{\text{Holm}} = 0.0088^*$) dan `0.2844` (`10%`, $\Delta = -0.0775, p_{\text{Holm}} = 0.0088^*$), dengan `Rec>=2` hanya `0.1538` pada 10% label.
* **Dekomposisi `Oversampling` vs `Class-Weight` (`rq2-03` vs `rq2-04` vs `rq2-02`, Mengonfirmasi `[@M041]` & `[@M071]`):**
  * `rq2-03` (`OS-only` InvSqrtSampler): `BACC4 = 0.4493 / 0.3292`, `QWK = 0.4652 / 0.2538`.
  * `rq2-04` (`CW-only` InvFreq CE): `BACC4 = 0.4463 / 0.3027`, `QWK = 0.4571 / 0.2251`.
  * `rq2-02` (`B1 Combined CW + OS`): `BACC4 = 0.4347 / 0.2757`, `QWK = 0.4472 / 0.1653` (kombinasi ganda secara naif menyebabkan *over-amplification* varians gradien kelas minoritas dan menurunkan `QWK`).
* **Kegagalan Head Ordinal Tanpa Penyeimbangan Prior vs Lonjakan `CORN + Logit Adjustment`:**
  * Head ordinal murni tanpa kompensasi prior: `rq2-05` (`CORN`: `BACC4 = 0.3775`, `QWK = 0.3700`), `rq2-11` (`FOL`: `BACC4 = 0.3863`), `rq2-12` (`Cumulative-Link ORM`: `BACC4 = 0.3610`), `rq2-13` (`SORD`: `BACC4 = 0.4107`).
  * **`rq2-06` (`CORN + Prior Logit Adjustment`):** Menambahkan kompensasi log-prior kondisional pada ambang biner `CORN` melonjakkan `Test BACC4` sebesar **`+9.11%`** (`0.3775` $\rightarrow$ **`0.4686`**) dan `Test QWK` sebesar **`+12.15%`** (`0.3700` $\rightarrow$ **`0.4915`**) pada 100% label!
  * **`rq2-16` (`Hybrid B2 + CORN Rank-Consistency Regularizer`):** Mencapai `Test BACC4 = 0.4843` dan **`Test QWK = 0.4973`** ($\Delta\text{QWK} = +0.0141$).
  * **`rq2-17` (*Per-Joint Prior Logit Adjustment* $\tau \log \pi_{j,c}$):** Mencapai **`Test QWK = 0.4986`** ($\Delta\text{QWK} = +0.0154$) dan `MAE = 0.3834` dengan memanfaatkan fakta bahwa distribusi prior erosi berbeda antar-anatomi sendi (mis. PIP vs MCP vs pergelangan).
* **Metode Post-Hoc `0-GPU` (`RQ2-P`):**
  * **`rq2-pc_tta` (*4-View Test-Time Augmentation*, `[@C093]`):** Meningkatkan `Test BACC4` (`0.4889` pada 100%; `0.3674` pada 10%) sekaligus `Test QWK` (`0.4901` pada 100%; **`0.2581`** pada 10% label: $\Delta\text{QWK} = +0.0229, p_{\text{raw}} = 0.0012, p_{\text{Holm}} = 0.0216^*$).
  * **`rq2-pa_thresh` (*Validation Cost-Sensitive Threshold Tuning*):** Meningkatkan `Test BACC4` menjadi **`0.4882`** (`100%`) dan **`0.3799`** (`10%`, $\Delta = +0.0180, p_{\text{raw}} = 0.0120$) serta mendongkrak `Recall>=2` dari `0.2998` $\rightarrow$ **`0.3768`**.
  * **`rq2-pb_knn` (*Feature-Space k-NN Logit Interpolation*, `[@M085]`):** `Test BACC4 = 0.4859 / 0.3632`, `Test QWK = 0.4850 / 0.2367` (peningkatan marginal $+0.04\%$ s.d. $+0.13\%$ yang tidak signifikan; merefutasi klaim lonjakan ekstrem `M085` yang menggunakan split tingkat citra).
* **Interaksi `RQ1 × RQ2` (`rq2-18a` & `rq2-18b`):**
  * **`rq2-18b` (`SupCon-Ord + B2_CORNreg`):** Meningkatkan `Test QWK` menjadi **`0.5066 [0.4150, 0.5877]`** ($\Delta = +0.0234$) pada 100% label dan **`0.2606 [0.1684, 0.3540]`** ($\Delta = +0.0254, p_{\text{raw}} = 0.0196$) pada 10% label, dengan `MAE` turun dari `0.3940` $\rightarrow$ `0.3673` (`100%`) dan `0.5515` $\rightarrow$ `0.5179` (`10%`).

### 4.4 `RQ3` (Validasi Eksternal Lintas-Populasi `RheumaVIT` Rusia & Benchmark 3 Radiolog)
* **Dekomposisi *Prior-Shift* vs *Covariate-Shift* (`RQ3-01` pada `RQ0-01`):**
  * Internal `RAM-H1200` (13 sendi irisan): `BACC4 = 0.4794`, `QWK = 0.4895`, `MAE = 0.4024`, `Rec>=2 = 0.2998`.
  * External `RheumaVIT` dengan *Source-Prior Logits* (`RAM-H1200`): `BACC4 = 0.3213` ($\Delta\text{Drop} = -0.1581$), `QWK = 0.1978`, `MAE = 0.2207`, `Rec>=2 = 0.1714`.
  * External `RheumaVIT` dengan ***Balanced/Uniform-Prior Logits* (0 label target!)**: **`BACC4 = 0.4036`** (memulihkan **`+8.23%` BACC4**, memangkas $>52\%$ *domain drop* lintas-benua tanpa satu pun sampel target!) dan menaikkan `Rec>=2` lebih dari $3\times$ lipat (`0.1714` $\rightarrow$ **`0.5810`**).
  * External `RheumaVIT` dengan *Target Oracle Prior*: `QWK = 0.2048` (tertinggi) dan `MAE = 0.1915` (terendah).
* **Augmentasi Tahan-Domain (`RQ3-04` & `RQ3-05`):**
  * **`rq3-04` (*Strong Photometric/Contrast/Noise Aug*):** Meningkatkan performa internal secara signifikan (**`Test BACC4 = 0.5190`**, $\Delta = +0.0335, p_{\text{Holm}} = 0.0468^*$; **`Test QWK = 0.5267`** pada `100%`; serta **`Test QWK = 0.2774`**, $\Delta = +0.0422, p_{\text{Holm}} = 0.0288^*$ pada `10%`), sekaligus meningkatkan transfer *zero-shot* `RheumaVIT` ke **`BACC4 = 0.4310`**, `QWK = 0.2054` (dan **`BACC4 = 0.4485`** terhadap konsensus 3 radiolog).
  * **`rq3-05` (*RheumaVIT-Matched Radiographic Aug*):** Mencapai `BACC4` *zero-shot* tertinggi pada `RheumaVIT` yaitu **`0.4324`** (`QWK = 0.2005`, memangkas $\Delta\text{Drop BACC4}$ menjadi hanya `-0.0381`).
  * **`rq4-01` (`MC Dropout`):** Mencapai `QWK` *zero-shot* tertinggi pada `RheumaVIT` yaitu **`0.2193`** (`BACC4 = 0.4225`).
* **Adaptasi *Zero-Label* (`AdaBN`, `HistMatch`) vs *Few-Shot* (`5 & 20 Pasien`, `RQ3-02/03/06`):**
  * `AdaBN`: Menaikkan `QWK` tanpa label dari `0.1619` $\rightarrow$ `0.1710` (`100%`) dan dari `0.0374` $\rightarrow$ `0.0668` (`10%`, dengan `BACC4` naik `0.2566` $\rightarrow$ `0.2839`).
  * `HistMatch`: Menaikkan `QWK` pada model 10% label dari `0.0374` $\rightarrow$ `0.0472`, namun sedikit menurunkan `BACC4` pada model 100% label (`0.3886`), menunjukkan bahwa pencocokan histogram global tanpa penyesuaian statistik fitur internal kurang memadai.
  * **Few-Shot Adaptation (`RQ3-06`):** Hanya dengan **5 pasien `RheumaVIT`** (`65` sendi), `BACC4` naik ke **`0.4079`**, `QWK` melonjak dari `0.1619` $\rightarrow$ **`0.2269`**, dan `MAE` turun tajam dari `0.6558` $\rightarrow$ **`0.4294`**. Dengan **20 pasien `RheumaVIT`** (`260` sendi), `BACC4` mencapai **`0.4411`** (`+3.75%`), `QWK` mencapai **`0.2721`** (`+11.02%`), dan `MAE` turun ke **`0.4070`**.
* **Benchmark 3 Radiolog Independen (`RQ3-07`, $N = 1.551$ Sendi Uji `RheumaVIT`):**
  * Kesepakatan berpasangan antar-radiolog (*pairwise inter-reader*):
    * `R0 vs R1`: `BACC4 = 0.6997`, `QWK = 0.8588`, `MAE = 0.0561`, `Rec>=2 = 0.5625`.
    * `R0 vs R2`: `BACC4 = 0.8316`, `QWK = 0.8691`, `MAE = 0.0535`, `Rec>=2 = 0.8750`.
    * `R1 vs R2`: `BACC4 = 0.8216`, `QWK = 0.8617`, `MAE = 0.0490`, `Rec>=2 = 0.8667`.
    * **Rata-rata Pairwise Manusia:** **`BACC4 = 0.7843`**, **`QWK = 0.8632`**, **`MAE = 0.0529`**, `Rec>=2 = 0.7681`.
  * **Batas Atas Manusia (*Leave-One-Reader-Out / LORO Ceiling*):** **`BACC4 = 0.8145`**, **`QWK = 0.8541`**, **`MAE = 0.0499`**, `Rec>=2 = 0.7793`.
  * **Model Zero-Shot Terbaik (`rq3-04`) vs Radiolog:** Mencapai **`BACC4 = 0.4485`** terhadap konsensus 3 radiolog (`0.4638` terhadap $R_1$), dengan **`Recall>=2 = 0.8065`** terhadap $R_1$ (melampaui sensitivitas pairwise manusia `R0 vs R1` sebesar `0.5625`).

### 4.5 `RQ4` (Kalibrasi, *Ordinal Conformal Prediction*, & *Selective Referral*)
* **Kalibrasi Post-Hoc (`RQ4-P1/P2`):**
  * `Uncalibrated B2`: `NLL = 1.0096`, `Brier = 0.5067`, `ECE = 0.1361`, `cwECE = 0.0738`.
  * `Temperature Scaling (Balanced Prior, RQ4-P1)`: Menurunkan `ECE` sebesar **`-85.5%`** menjadi **`0.0197`**, `NLL = 0.8635`, `Brier = 0.4530`, `cwECE = 0.0680`.
  * `Temperature Scaling + Log-Prior Recovery (RQ4-P2)`: Terbaik pada distribusi alami, menurunkan `NLL` menjadi **`0.7627`**, `Brier` menjadi **`0.3994`**, `ECE = 0.0258`, dan `cwECE` menjadi **`0.0394`** (mengungguli `Vector/Dirichlet Calibration`: `NLL = 0.7879`, `Brier = 0.4090`, `ECE = 0.0442`, `cwECE = 0.0455`).
* ***Ordinal Conformal Prediction* (`RQ4-P3/P4`, Target $1 - \alpha = 90\%$):**
  * `Marginal Ordinal-APS`: Marginal Cov = `93.48%`, Mean Set Size = `2.645`, Singleton = `19.51%`, **namun `Cov>=2 = 81.31%`** (`Cov G2 = 88.61%`, `Cov G3 = 69.33%`, `Cov G5 = 81.82%` — gagal memenuhi target 90% pada erosi sedang-berat!).
  * `Marginal Ordinal-RAPS` ($\lambda_{\text{reg}}=0.02, k_{\text{reg}}=1$): Marginal Cov = `90.83%`, menyusutkan Mean Set Size menjadi **`2.305`** (`-12.9%`) dan menaikkan Singleton ke **`23.16%`**, **namun memperparah *minority under-coverage* (`Cov>=2 = 67.35%`)**.
  * **`Mondrian Class-Conditional Ordinal-APS` (`RQ4-P3`):** Marginal Cov = `92.63%`, **`Cov>=2 = 93.75%`** (`Cov G0 = 92.60%`, `Cov G1 = 92.34%`, **`Cov G2 = 96.20%`**, **`Cov G3 = 90.67%`**, **`Cov G5 = 90.91%`**) — **100% kelas memenuhi jaminan $\ge 90\%$!**
  * **`Mondrian Joint×Grade Conditional Conformal` (`RQ4-P4`):** Marginal Cov = `92.04%`, **`Cov>=2 = 91.76%`**, Mean Set Size = `2.863`.
  * **Re-Kalibrasi Conformal Lintas-Domain pada `RheumaVIT`:**
    * *Zero-Shot Mondrian* (kuantil dari `RAM-H1200`): Marginal Cov = `95.21%`, `Cov>=2 = 99.05%`, tetapi mengalami *over-coverage* dengan Mean Set Size = `4.090`.
    * *Re-Kalibrasi Few-Shot 5 Pasien `RheumaVIT` (`0-GPU`)*: Menjaga Marginal Cov = **`94.81%`** dan `Cov>=2` = **`94.29%`** sekaligus menyusutkan Mean Set Size sebesar **`-12.8%`** (`4.090` $\rightarrow$ **`3.566`**) dan melipatgandakan Singleton (`2.66%` $\rightarrow$ `6.14%`).
* ***Selective Referral* (`RQ4-P5` & `RQ4-01`):**
  * `Single Model Entropy (RQ0-01)`: `AURC = 0.2452`, `Cov 100% BACC4/QWK = 0.4855 / 0.4832` $\rightarrow$ `Cov 80% = 0.5177 / 0.4820`.
  * `MC Dropout (RQ4-01, T=20)`: `AURC = 0.2463`, `Cov 100% = 0.4829 / 0.4910` $\rightarrow$ `Cov 80% = 0.5180 / 0.4961`.
  * **`3-Seed Deep Ensemble (RQ4-P5)`:** Menurunkan `AURC` ke **`0.2409`**, menaikkan performa `Cov 100%` ke **`BACC4 = 0.5012 / QWK = 0.5313`**, dan pada **80% coverage (merujuk 20% sendi paling meragukan)** mencapai **`BACC4 = 0.5228`** dan **`QWK = 0.5331`**.

### 4.6 `RQ5` (Mitigasi *Label Noise* Bebas-Kebocoran, Reliabilitas Longitudinal, `Mean-Teacher`, & `RQ5-E`)
* **Diagnostik `5-Fold Patient-Wise OOF` (`RQ5-00`):** Mengidentifikasi **`3.084 / 12.688` (`24.31%`)** sampel sendi latih sebagai kandidat *label issue* dan **`1.638` sendi** dengan pelanggaran monotonisitas longitudinal ($\Delta S < 0$).
* **Bukti Empiris Jujur pada Test Set Asli (`100% Label`):**
  * **`rq5-01` (*OOF CleanLab Hard Relabeling*):** `Test BACC4 = 0.4889 [0.4368, 0.5142]` ($\Delta = +0.0034, p_{\text{raw}} = 0.8054, p_{\text{Holm}} = 1.0000$ — **efek null / tidak signifikan!**) dan `Test QWK = 0.4656` ($\Delta = -0.0177$). Membuktikan secara empiris bahwa klaim lonjakan besar `CleanLab` pada literatur (`[@M042]`, `[@M117]`, `[@M272]`) adalah artefak sirkularitas modifikasi label uji, karena *hard relabeling* di perbatasan `Grade 0/1` menghapus kasus erosi dini yang sah. (Catatan: pada `10% label`, `rq5-01` mencapai `BACC4 = 0.4435` karena label OOF berasal dari *cross-validation* himpunan latih penuh sehingga berfungsi sebagai *knowledge distillation* lintas-pasien).
  * **Keunggulan Signifikan *Soft Down-Weighting* & *Longitudinal Reliability* (`100% Label`):**
    * **`rq5-06` (*Confidence-Split Soft Regularization*):** `Test BACC4 = 0.5111 [0.4648, 0.5388]` ($\Delta = +0.0256, p_{\text{raw}} = 0.0320$), **`Test QWK = 0.5299 [0.4375, 0.6087]`** ($\Delta = +0.0467, p_{\text{raw}} = 0.0004, p_{\text{Holm}} = 0.0044^*$), `Macro-F1 = 0.4431`, `MAE = 0.3511`.
    * **`rq5-07` (*Longitudinal Visit-Reliability Down-Weighting*, $\Delta S < 0$):** `Test BACC4 = 0.4987 [0.4535, 0.5273]` ($\Delta = +0.0132$), **`Test QWK = 0.5264 [0.4383, 0.6054]`** ($\Delta = +0.0432, p_{\text{raw}} = 0.0004, p_{\text{Holm}} = 0.0044^*$).
    * **`rq5-03` (*OOF CleanLab Soft Quality Down-Weighting*):** `Test BACC4 = 0.5001 [0.4533, 0.5295]` ($\Delta = +0.0146$), **`Test QWK = 0.5194 [0.4274, 0.6010]`** ($\Delta = +0.0362, p_{\text{raw}} = 0.0008, p_{\text{Holm}} = 0.0056^*$).
* ***Semi-Supervised Mean-Teacher* (`rq5-11`):** Pada `25% label` (`f0.25`) dengan memanfaatkan pasien tak berlabel melalui konsistensi EMA guru-murid, **`Test BACC4` melonjak ke `0.4289 [0.3999, 0.4508]`** ($\Delta = +0.0670$ di atas baseline 10%, $p_{\text{raw}} = 0.0004, p_{\text{Holm}} = 0.0052^*$) dan `Recall>=2` mencapai **`0.4500`**.
* **Evaluasi Distributional Multi-Radiolog (`RQ5-E`, $N = 1.551$ Sendi Uji `RheumaVIT`):**
  * Dari `1.551` sendi uji, `1.436` sendi (`92.59%`) disepakati bulat (*unanimous*) oleh $R_0, R_1, R_2$, sedangkan `115` sendi (`7.41%`) memiliki ketidaksepakatan antar-radiolog.
  * Pada seluruh model, `BACC4` jauh lebih tinggi dan `KL-Divergence` jauh lebih rendah pada sendi *unanimous* dibandingkan sendi *disagreement* (mis. `RQ0-01`: `BACC4 Unanimous = 0.4333` vs `Disagreement = 0.2977`, `KL Unanimous = 0.8568` vs `Disagreement = 1.4649`), membuktikan secara langsung bahwa ketidakpastian prediksi model terkonsentrasi tepat pada sendi-sendi yang juga memicu ketidaksepakatan antar-radiolog pakar.

---

## 5. Instruksi Kerja Penyelesaian Bab IV, Bab V, dan Bab VI

Jika suatu sesi agent terputus karena batas kuota (*token/quota limit*), agent berikutnya cukup memeriksa daftar di bawah ini dan melanjutkan penulisan berkas yang belum selesai:

1. **Tulis [`thesis/Bab_IV_Hasil_dan_Analisis_Internal.md`](thesis/Bab_IV_Hasil_dan_Analisis_Internal.md):**
   * **4.1** Kinerja Baseline Rujukan (`RQ0-01` vs `RQ0-02` `ConvNeXt-Tiny`) & Analisis Kegagalan Per-Grade dan Per-Anatomi Sendi.
   * **4.2** Hasil Ablasi `RQ1` (`14 Arm`, `84 Run`): Analisis Pra-Pelatihan & Domain SSL (`Scratch`, `DINO R34`, `ChronoCon`, `ViT-S ImageNet/MAE/DINO`), *Supervised Contrastive Berbobot Jarak Ordinal* (`SupCon-Ord` vs `SupCon`), Bukti Refutasi *Normality Head* (`SS-FewSOME` & `Normality Gate`), serta Analisis *Feature Distortion* pada *Frozen Linear Probe* vs Pemulihan **`LP-FT`**.
   * **4.3** Hasil Ablasi `RQ2` (`22 Arm`, `132 Run`): Dekomposisi `Oversampling` vs `Class-Weight` (`M041/M071`), Kegagalan Head Ordinal Tanpa Penyeimbangan Prior vs Sinergi **`CORN + Logit Adjustment`** & **`B2 + CORNreg`**, *Per-Joint Prior Logit Adjustment*, Evaluasi Metode *Post-Hoc* `0-GPU` (`RQ2-Pa Threshold`, `RQ2-Pb k-NN`, `RQ2-Pc TTA`), serta Interaksi `RQ1 × RQ2` (`rq2-18a/b`).
   * **4.4** Hasil Ablasi `RQ4`: Kalibrasi Probabilistik (*Temperature Scaling*, *Log-Prior Recovery*, vs *Dirichlet*), *Ordinal Conformal Prediction* (*Marginal APS/RAPS* vs Pemecahan *Minority Under-Coverage* oleh ***Mondrian Class-Conditional*** & ***Mondrian Joint×Grade***), serta Kurva *Risk-Coverage* Rujukan Selektif (*Single Model* vs *MC Dropout* vs ***3-Seed Deep Ensemble***).
   * **4.5** Hasil Ablasi `RQ5`: Diagnostik *5-Fold Patient-Wise OOF*, Bukti Empiris Efek Null `CleanLab Hard Relabeling` pada Label Uji Asli (`p_Holm = 1.0000`, konfirmasi audit forensik `M042/M117/M272`), Keunggulan Signifikan *Soft Down-Weighting* (`rq5-03`, `rq5-06`) & *Longitudinal Visit-Reliability Weighting* (`rq5-07`), serta *Semi-Supervised Mean-Teacher* (`rq5-11`).
   * **4.6** Sintesis Lintas-RQ Internal & Pemetaan Temuan terhadap Literatur SLR-M.

2. **Tulis [`thesis/Bab_V_Validasi_Eksternal_dan_Diagnostik.md`](thesis/Bab_V_Validasi_Eksternal_dan_Diagnostik.md):**
   * **5.1** Karakteristik Pergeseran Domain Lintas-Benua (`RAM-H1200` Jepang $\rightarrow$ `RheumaVIT` Rusia) & Analisis *Domain Drop* Zero-Shot Lintas Keluarga Metode (`RQ0–RQ5`).
   * **5.2** Dekomposisi Matematis & Empiris *Target Prior Shift* vs *Visual Covariate Shift* (`RQ3-01`: Pemulihan `+8.23%` BACC4 dan $3,4\times$ `Recall>=2` dengan *Balanced-Prior Logits* tanpa label target).
   * **5.3** Ketahanan Augmentasi Radiografik (`RQ3-04` *Strong Aug* & `RQ3-05` *RheumaVIT-Matched Aug*), Adaptasi *Zero-Label* (`AdaBN`, `HistMatch`), serta Efisiensi *Few-Shot Cross-Domain Adaptation* (`5` vs `20` Pasien `RheumaVIT`, `RQ3-06`).
   * **5.4** Ketahanan & Re-Kalibrasi *Mondrian Conformal Prediction* Lintas Rumah Sakit (`RQ4-P3`: Penyusutan `-12.8%` ukuran himpunan prediksi dengan 5 pasien kalibrasi lokal tanpa GPU).
   * **5.5** Evaluasi terhadap Jangkar 3 Radiolog Independen (`RQ3-07`, $N=1.551$ sendi uji): Kesepakatan Antar-Radiolog (*Pairwise Inter-Reader*), Batas Atas Manusia (*Leave-One-Reader-Out Ceiling*), dan Perbandingan Sensitivitas Erosi Sedang-Berat (`Recall>=2`).
   * **5.6** Evaluasi Distributional Multi-Pembaca (`RQ5-E`): Analisis *KL-Divergence* dan *Soft Brier Score* pada Sendi *Unanimous* (`92.59%`) vs Sendi *Disagreement* (`7.41%`).

3. **Tulis [`thesis/Bab_VI_Kesimpulan_dan_Saran.md`](thesis/Bab_VI_Kesimpulan_dan_Saran.md):**
   * **6.1** Kesimpulan Menyeluruh per Pertanyaan Penelitian (`RQ1` s.d. `RQ5`).
   * **6.2** Kontribusi Ilmiah & Pedoman Praktis Desain Sistem AI Grading Ordinal Radiografi Muskuloskeletal.
   * **6.3** Keterbatasan Penelitian (*Limitations*).
   * **6.4** Saran & Arah Penelitian Lanjutan (*Future Work*).
