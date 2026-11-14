# SLR-M: Kandidat Ablasi per RQ (Setelah Verifikasi Full-Text)

**Status:** Diperbarui 2026-11-14 (mengintegrasikan hasil verifikasi buta dan audit forensik teks penuh pada `verification/04_verification_report.md` tanggal 2026-11-13). Menggantikan `01_kandidat_ablasi_sementara.md` (berbasis abstrak) dan draf awal 2026-11-10.
- Seluruh **54 berkas teks penuh** di `papers/` (50 dari basis data + 4 dari *citation searching*) telah diaudit kelayakan dan kualitas buktinya (`verification/03_fulltext_criteria_verification.csv`).
- Sampel acak 20% screening judul/abstrak + *citation searching* ($N = 87$) telah diverifikasi secara buta ($\kappa = 0,8880$, PABAK $= 0,9310$, sensitivitas AI $= 100\%$).
- Seluruh 17 studi Prioritas A $\times$ 12 atribut ($204$ sel) telah diverifikasi langsung terhadap PDF/XML penuh (`verification/02_extraction_check.csv`).
- 19 studi yang laporan teks penuhnya tidak diperoleh tetap dicatat secara terpisah dari abstrak dan ditandai †.

**Data & Berkas Terkait:**
- Tabel ekstraksi awal: `extraction/01_fulltext_extraction.tsv`
- Koreksi ekstraksi & verifikasi kriteria: `verification/02_extraction_check.csv`, `verification/03_fulltext_criteria_verification.csv`, `verification/04_verification_report.md`
- Screening: `screening/`
- Protokol: `protocol/01_protocol_PRISMA-P.md`

---

## 1. Alur PRISMA Terverifikasi

| Tahap | n |
|---|---|
| Teridentifikasi dari basis data: Scopus 290, PubMed 96, IEEE 41 (dari 310, setelah filter DL + G1–G4) | 427 |
| Duplikat dihapus | 86 |
| Di-screen judul/abstrak (basis data) | 341 |
| Dikeluarkan pada judul/abstrak (EM3 144, EM2 69, EM1 36, EM4 18, EM5 4, EM6 1) | 272 |
| Laporan basis data yang dicari teks penuhnya | 69 |
| Citation searching (OpenAlex, 1 putaran dari 38 studi berablasi): 1.157 backward + 417 forward; 95 baru setelah filter tugas + radiografi + DL $\to$ 91 dikeluarkan (EM2/EM3/EM4/EM7) | +4 laporan dicari (C007, C015, C037, C093) |
| **Total laporan dicari teks penuhnya (*Reports sought for retrieval*)** | **73** (69 basis data + 4 sitasi) |
| **Laporan tidak diperoleh (*Reports not retrieved*)** (`papers/00_TODO_user_download.md`) | **19** (dicatat dari abstrak dengan tanda †) |
| **Laporan dinilai kelayakannya pada teks penuh (*Reports assessed for eligibility*)** | **54** (50 basis data + 4 sitasi) |
| **Dikeluarkan pada tahap teks penuh (*Excluded at full text*)** (`verification/04_verification_report.md` §4.1) | **10** |
| — `EM2` (Bukan grading ordinal multi-kelas: `M165` [biner KL01 vs KL234], `M173` [sintesis citra murni tanpa classifier grading]) | 2 |
| — `EM3` (Tidak menguji metode G1–G4: `M051` [hanya rotasi aktivasi dense], `M180` [CE standar + inisialisasi acak], `M249` [uji diagnostik cropping/masking pada CNN standar]) | 3 |
| — `EM6` (Tanpa hasil eksperimen valid / terbukti fabrikasi: `M095` [placeholder template LLM `[X]`, `[ F ]`, `[ TMI ]` & gambar mockup sintetis], `M175` [bagian hasil kosong & ditulis dalam kalimat masa depan]) | 2 |
| — `EM7` (Preprint arXiv belum peer-review: `M191` [serangan adversarial 1-shot], `M199` [kiriman konferensi anonim]) | 2 |
| — Duplikat publikasi (`M283` identik verbatim dengan `M141`, dihitung satu kali sebagai `M141`) | 1 |
| **Studi unik terverifikasi dalam sintesis teks penuh** | **44** (40 studi bersih + 4 studi dengan peringatan metodologis khusus: `M042, M117, M150, M188`) |
| *(Catatan tambahan dari abstrak untuk pemetaan kandidat)* | *(+19 rekaman abstrak †)* |

Dicatat di luar PRISMA sebagai referensi metode: `C047` (preprint arXiv 2026, soft labelling KL dengan 4 bentuk distribusi: binomial, beta, triangular, eksponensial; semuanya mengalahkan baseline nominal) dan `M249` (sebagai inspirasi uji diagnostik *shortcut learning* dengan penghapusan ROI/lesi).

---

## 2. Kualitas Bukti Terverifikasi (Checklist 6 Butir pada 44 Studi Valid)

Berdasarkan audit teks penuh pada `verification/03_fulltext_criteria_verification.csv` dan `verification/04_verification_report.md` §4.2:

| Butir Kualitas (§8 Protokol) | Seluruh 54 Berkas Terunduh | **44 Studi Unik Valid yang Dipertahankan** | Catatan Perubahan Kunci dari Verifikasi Full-Text |
|---|---|---|---|
| **Q1. Ablasi terkontrol pada split yang sama** | Ya: 40, Tidak: 14, Tidak jelas: 0 | **Ya: 36 (81,8%), Tidak: 8 (18,2%)** | Seluruh 10 status `U` terpecahkan (`Y` pada `M077, M091, M153, M162, M201, M222, M275, M305, M306, C037`; `N` pada `M106 [untuk loss G2], M117, M150, M269, C007, C015`). |
| **Q2. Lebih dari satu run/seed, atau ada CI** | Ya: 20, Tidak: 25, Tidak jelas: 9 | **Ya: 19 (43,2%), Tidak: 20 (45,5%), Tidak jelas: 5 (11,4%)** | `M185` dipastikan `N` (estimasi titik tunggal); `M061` dikoreksi menjadi `mean ± SE (5 seeds)`. |
| **Q3. Split per pasien** | Ya: 14, Tidak: 26, Tidak jelas: 14 | **Ya: 13 (29,5%), Tidak: 21 (47,7%), Tidak jelas: 10 (22,7%)** | 6 studi naik dari `U` ke `Y` (`M041, M061, M100, M153, M165*, M191*`), sedangkan `M071` terbukti menggabungkan `train+val` untuk 5-fold CV tingkat citra (§8.5 hlm. 22). |
| **Q4. Hiperparameter/ambang dipilih di validasi** | Ya: 7, Tidak jelas: 47 | **Ya: 6 (13,6%), Tidak jelas: 38 (86,4%)** | Dilaporkan eksplisit antara lain pada `M071, M110, M130, M219, C015`. |
| **Q5. Uji eksternal / lintas senter** | Ya: 11, Tidak: 43, Tidak jelas: 0 | **Ya: 11 (25,0%), Tidak: 33 (75,0%)** | `M130` dikoreksi `Y -> N` (kohort 52 pasien berasal dari RS yang sama, Osaka Univ. Hospital); 4 studi naik `U -> Y` (`M061, M091, M153, M185`). **0 dari 5 studi RA memiliki validasi eksternal.** |
| **Q6. Kode atau bobot model tersedia publik** | Ya: 10, Sebagian: 3, Tidak: 41 | **Ya: 8 (18,2%), Sebagian: 2 (4,5%), Tidak: 34 (77,3%)** | `C093` naik `N -> Y` (repositori Google Drive tersedia di hlm. 204). |

### Peringatan Metodologis Utama dari Audit Forensik (`verification/04_verification_report.md` §3):
1. **Paper Fabrikasi yang Telah Dikeluarkan (`M095`, `M175`):**
   - **`M095`** (mengklaim Macro-F1 `0,982`, QWK `0,991`, ECE `0,009`) terbukti memuat placeholder template LLM (`[X]`, `[ F ]`, `[ TMI ]` di Tabel 11 hlm. 883), gambar mockup sintetis (§6.9 hlm. 882), serta anomali jumlah parameter. **Seluruh klaim dan sitasi `M095` dicabut dari desain tesis.**
   - **`M175`** (mengklaim akurasi `98,97%`) tidak memiliki tabel/eksperimen di §V (*Result Analysis* ditulis dalam kalimat masa depan).
2. **Modifikasi Label Himpunan Uji pada Studi *Label Noise* (`M042`, `M272`, `M117`):**
   - **`M042`** menggabungkan seluruh 8.260 citra (`train+val+test`) sebelum relabeling 286 citra oleh VGG19 Cascade-I, lalu menguji Cascade-II pada label yang sudah diubah oleh arsitektur yang sama (+24,13 poin adalah bias sirkularitas).
   - **`M272`** menerapkan CleanLab pada himpunan uji dan menimpa **22,0% (364/1.656) label uji**. Akurasi naik `66,36% -> 82,07%` hanya pada label uji yang telah diubah CleanLab; **ketika dievaluasi pada label uji asli yang tidak disentuh (§4.3 hlm. 15), perbedaannya TIDAK signifikan secara statistik (McNemar $p = 0,076$)**.
   - **`M117`** membersihkan outlier dan merelabeli lintas 5-fold CV termasuk lipatan uji (*held-out CleanLab-Plus test set*, hlm. 6) tanpa ablasi terpisah (`Q1_abl = N`).
3. **Kebocoran Pra-Split & Kontradiksi Metrik (`M150`, `M160`, `M162`, `M188`, `M275`, `C037`):**
   - `M160, M162, M275, C037` melakukan *oversampling* / SMOTE / augmentasi sebelum pembagian *train/val/test*.
   - `M188` melaporkan angka kurva *training* (`98,8% mAP`) di Tabel 5, padahal matriks konfusi uji (Fig. 10b) hanya mencapai `K2 = 48%, K3 = 65%, K4 = 75%`.

---

## 3. Kandidat per RQ (Pasca-Verifikasi)

Kolom **Status**: *ada* = sudah diuji; *mirip* = ada varian dekat; *belum* = belum pernah diuji.
Kolom **Prioritas**: A = utama, B = sekunder, C = opsional.

### RQ1: Pre-training dan Self-Supervised Learning (G1)

Sudah ada di tesis: `scratch`, `ImageNet`, `MAE`, `DINO`, `DINO + ChronoCon`, `LLRD`, `linear probe`.

| # | Kandidat | Bukti Terverifikasi di SLR-M | Status | Prioritas | Catatan untuk RAM-H1200 |
|---|---|---|---|---|---|
| 1.1 | **SKELEX**: Foundation model MSK (`ViT-L` MAE dilanjutkan pada 1,29 juta radiograf MSK) | **`M015`** (*npj Digit. Med.* 2026): rata-rata `+6,21%` relatif terhadap inisialisasi ImageNet-MAE dengan protokol identik; split per pasien (`Q3=Y`), 5.000 bootstrap CI (`Q2=Y`), validasi eksternal (`Q5=Y`), **bobot publik di GitHub (`skhoha/SKELEX`)** | belum | A | `ViT-L` berat untuk RTX 3060 (12 GB). Ukur waktu/memori 1 epoch dulu; uji *linear probe* lalu *fine-tune* + LLRD |
| 1.2 | **Pre-training antara supervised pada radiograf tangan** (`RSNA Pediatric Bone Age`, 14.036 citra) | **`M005`**: `ImageNet -> RSNA bone age -> RA SvdH`; membandingkan skema fine-tuning + ensemble, mencapai PCC `0,925` dan RMSE `18,02` setara kesepakatan antar-radiolog (PCC `0,97`) | belum | A | Data tangan publik, anatomi identik dengan `RAM-H1200`, murah (`ResNet34`); membuka sumbu *domain-adjacent supervised transfer* |
| 1.3 | **Supervised Contrastive (`SupCon`) + CE dengan negatif berbobot jarak grade** | **`M219`** (negatif berbobot jarak KL + EMD menaikkan QWK lintas-domain `0,678 -> 0,732`), **`M082†`**, **`M166†`**, **`M280†`** | belum | A | Positif = sendi dengan grade sama lintas pasien; bobot tolakan negatif proporsional terhadap $|y_i - y_j|$ |
| 1.4 | **SSL "Normalitas"** dari sendi sehat (`Grade 0`) + skor jarak ke pusat normal | **`M197`** (`SS-FewSOME`, *Artif. Intell. Med.* 2025): dilatih dari lutut sehat saja (<3% label), AUC deteksi OA naik `56,7% -> 81,0%`; split per pasien, 5 seed $\pm$ SD, **kode publik di GitHub** | belum | A | Sangat cocok karena sendi `Grade 0` mendominasi `RAM-H1200`; jarak ke pusat normal dipakai sebagai fitur tambahan / *gate* |
| 1.5 | **Strategi Fine-Tuning**: `Frozen Linear Probe` vs `LP-FT` vs `Gradual Unfreezing` | **`M269`** (gradual unfreezing 2 tahap), **`M005`** (skema FT), dan **`M157`** (membuktikan *frozen FM embeddings* `BiomedCLIP`/`RAD-DINO` kalah telak dari *end-to-end fine-tuning* ConvNeXt $\kappa=0,880$ dan ResNet18 $\kappa=0,874$) | mirip (`LLRD`, `linear probe`) | A | Satu sumbu murah dan terkontrol pada encoder `ImageNet` dan encoder SSL terbaik |
| 1.6 | **Penyelarasan representasi progresi longitudinal** (`TCN` atas sekuens + `MMD`, inferensi citra tunggal) | **`M335`** (CVPR 2022): menyelaraskan fitur ViT citra tunggal ke fitur progresi sekuens kunjungan (TCN) via MMD; `BA 0,7850 -> 0,8216`, `Kappa 0,6492 -> 0,7357` (5-fold CV) | mirip (`ChronoCon`, E9) | B | Memanfaatkan data kunjungan `RAM-H1200` melalui penyelarasan distribusi (MMD) tanpa memaksa urutan monotonik per sendi yang gagal di E9 |
| 1.7 | **Autoencoder rekonstruksi + tugas bantu anatomis gratis** | **`M153`** (`DC-AAE` + tugas bantu identifikasi sisi kiri/kanan pada 8.363 citra tak berlabel; ablasi tugas bantu `73,00% -> 75,53%`, split per pasien, uji lintas-dataset OAI $\leftrightarrow$ MOST), **`M185`** (`MI-Rec`) | mirip (`MAE`) | B | Tugas bantu gratis dari ROI `RAM-H1200`: jenis sendi (MCP/PIP/pergelangan) atau sisi tangan |
| 1.8 | **Foundation model medis umum (`BiomedCLIP`, `RAD-DINO`) di-fine-tune** | **`M157`**: versi beku kalah dari CNN supervised (`QWK 0,551` vs `0,880`); versi *fine-tuned* belum diuji di `M157` | belum | B | Hanya diuji dengan *fine-tuning* penuh / LLRD |
| 1.9 | **Vision-Language / CLIP fine-tuning** dengan templat teks deskripsi grade | **`M201`** (CLIP fine-tuned 2 layer teratas + adapter, uji eksternal $n=1.650$ mencapai akurasi `76,94%` mengalahkan ViT `73,7%` dan ResNet50 `68,3%`), **`M306`** (templat laporan + *hard negative captions* menaikkan *zero-shot* `45% -> 55%`), **`M071`** | belum | C | Menggunakan templat deskripsi klinis SvdH per grade (0–5) dan per jenis sendi |

---

### RQ2: Class Imbalance dan Ordinalitas (G2)

Sudah ada di tesis: `B0–B6`, `E7` undersampling grade 0, `E8` augmentasi.

| # | Kandidat | Bukti Terverifikasi di SLR-M | Status | Prioritas | Catatan untuk RAM-H1200 |
|---|---|---|---|---|---|
| 2.1 | **Memecah `B1` menjadi `Oversampling Murni (OS-only)` dan `Class Weight Murni (CW-only)`** | **`M041`** (OA tangan DIP, split pasien, 3 run): `OS-only` paling konsisten (`KL3 F1 0,280 -> 0,447`, $p=0,015$; `KL4 F1 0,573 -> 0,750`; `OA F1 0,416 -> 0,588`, $p=0,031$). **`M071`** (Tabel 4 hlm. 15): penambahan *class weighting* kasar di atas ViT-CORAL justru **menurunkan Macro-F1 sebesar `-5,0%`** (`68,60% -> 63,6%`)! **`M222`** (RA mTSS tangan, 24.690 ROI): `WeightedRandomSampler + CW-CE` menaikkan Macro-Acc dari `48–72%` ke `80–88%`. **`C093`**: `WeightedRandomSampler` menaikkan sensitivitas kelas menengah `63,3% -> 75,8%` | **belum** (`B1` selalu menggabungkan OS + CW) | A | Sangat penting dan langsung tersedia lewat flag `--oversample` dan `--class-weight` |
| 2.2 | **Focal Loss (`FL`) & Focal Ordinal Loss (`FOL`)**, sendiri maupun **`FL + Oversampling`** | **`M132`** (10 run $\times$ 6 CNN: `FOL` menurunkan MSE 12–19% dan menaikkan Kappa `0,774 -> 0,794` atas CE dan *fixed-penalty*). **`M041`** (koreksi full-text Tabel 4 & 6): `FL-only` mencapai **recall KL3 tertinggi (`0,510` vs `0,203`, F1 `0,427` vs `0,280`)** tetapi lemah di KL1/KL4, sehingga **`OS + FL`** saling melengkapi (`KL3 F1 0,527`, $p=0,048$). **`M100`** (erosi tangan RA). **`C093`** (`FL + WeightedRandomSampler`) | belum | A | Uji `Focal` sendiri dan `Focal + Oversampling` |
| 2.3 | **Kepala Ordinal *Cumulative-Link* (`ORM` dengan *learnable cut-points*)** | **`M089`** (Tabel 1–2 hlm. 6–10): dievaluasi pada **21 arsitektur CNN**; `ORM + Cumulative-Link loss` memperbaiki MAE dan QWK pada **seluruh 21/21 jaringan** (rata-rata $\Delta\text{QWK} = +0,0441$; pada `ResNet-34` QWK naik **`0,7704 -> 0,8314`** [+0,0610], MAE turun `0,444 -> 0,389`) | belum (`B0` ambang biner, `B3/B4` CORN) | A | Sangat murah (hanya mengganti proyeksi kepala dengan 1 skalar + $K-1$ *cut-points* terurut) |
| 2.4 | **Distance-Penalty Ordinal Loss & Soft Label Ordinal (Beta/Gaussian/Binomial)** | **`M305`** (Tabel II hlm. 4: *distance-weighted ordinal loss* mengalahkan CE pada **8/8 arsitektur** dalam Macro-F1 dan MAE), **`M091`** (`CORN` mengalahkan CE di 4/4 ViT, $p<0,05$), **`M039`** (`CORAL -> CE` menurunkan QWK `-0,068`, $p<0,001$), **`M266†`** (20 kombinasi, terbaik CE + cumulative link + soft label beta), **`C047`** | belum untuk penalti jarak & soft label | A | Murah: hanya mengubah matriks penalti / distribusi target |
| 2.5 | **Penyetelan Ambang Keputusan (*Decision Threshold Tuning*) di Validasi** | **`M071`** (ambang $\tau$ dioptimalkan di validasi untuk Macro-F1), **`M052†`** (*inverse-frequency weight + decision threshold calibration*) | belum | A | **Tanpa training baru (0 GPU)**: gunakan logit validasi untuk mencari ambang/bias per kelas yang memaksimalkan `BACC4` |
| 2.6 | **Retrieval-Augmented / `FAISS k-NN` Classifier di atas Embedding** | **`M161`** (`KneeRAC`, 5-fold CV pada 17 arsitektur): konkatenasi embedding kueri dengan rata-rata $k=25$ tetangga terdekat per kelas meningkatkan akurasi pada **13 dari 17 backbone** (mis. EfficientNet-B0 `83,51% -> 86,81%`) | mirip (`B6` cRT) | A (Pasca-latih) | **Tanpa training backbone baru**: langsung dijalankan di atas embedding yang tersimpan |
| 2.7 | **Test-Time Augmentation (`TTA`)** | **`C093`** (Fig. 8 & Tabel VI, 1.000 bootstrap CI): `TTA` secara konsisten meningkatkan akurasi di seluruh model dan rezim sampling (mis. ConvNeXt-Tiny `82,6% -> 83,8%`, EfficientNet-B3 `78,4% -> 79,6%`), sedangkan *weighted ensemble* tidak berbeda signifikan dari model tunggal terbaik | belum | A (Pasca-latih) | **Tanpa training baru (0 GPU)** |
| 2.8 | **Uji Diagnostik *Shortcut Learning* via Masking Piksel Erosi (`BE Mask`)** | **`M249`** (menguji apakah model masih memprediksi grade ekstrem ketika ROI sendi dihapus $\to$ mendeteksi *shortcut bias*) | belum | A (Diagnostik) | **Tanpa training baru (0 GPU)**: manfaatkan *pixel-level Bone Erosion (BE) masks* yang tersedia di `RAM-H1200` |
| 2.9 | **Kepala Ordinal Tambahan sebagai Regularizer (`Softmax + CORN/Cumulative`)** | **`C015`**: memakai 3 kepala saat latih dan kalibrasi di validasi (`QWK 0,845`), namun verifikasi teks penuh menunjukkan `Q1_abl = N` (tidak ada ablasi yang mengisolasi kepala tambahan) | belum | B | Prioritas diturunkan dari A ke B karena `C015` tidak memiliki ablasi terkontrol |
| 2.10 | **Mixup Varian (`MixUp`, `DAB-Mix`, `CutMix`) untuk Kelas Minoritas** | **`M269`** (`DAB-Mix` khusus KL3 & KL4, tetapi `Q1_abl = N`), **`M077`** (constrained MixUp), **`M117`** (CutMix), **`M224†`** | belum | B | Lanjutan opsional resep E8 |

---

### RQ3: Ketahanan & Adaptasi Lintas Populasi (`RAM-H1200` Jepang $\to$ `RheumaVIT` Rusia)

Berdasarkan verifikasi penuh, terdapat **11 studi dengan validasi eksternal (`Q5_ext = Y`)** di SLR-M (`M015, M039, M044, M061, M091, M106, M153, M155, M185, M201, M219`), dan **0 di antaranya pada RA**. Lima mekanisme yang diadopsi ke dalam tesis:
1. **Acuan *Zero-Shot Domain Drop*:** `M044` (`0,939 -> 0,862` di MOST), `M201` (`96,9% -> 76,9%` di Gornale), `M091` (`89–93% -> 61,0%` *zero-shot* di CHECK+CGMH+G&A), `M153` (`75,5% -> 71,0–73,8%` OAI $\leftrightarrow$ MOST).
2. **Dekomposisi *Covariate Shift* vs *Target Prior Shift*:** Memanfaatkan sifat dekomposisi logit `B2` (*Balanced Softmax*) untuk memisahkan efek pergeseran citra vs pergeseran distribusi kelas di `RheumaVIT`.
3. **Adaptasi Tanpa Label (*Zero-Label Test-Time Adaptation*):** Menguji `TTA` (**`C093`**) dan *Test-Time BatchNorm / Tent Adaptation* (Schneider et al. 2020; Wang et al. 2021 — menggantikan sitasi `M095` yang dicabut).
4. **Adaptasi *Few-Shot* (5 & 20 Pasien `RheumaVIT`) & *Cross-Domain Contrastive Adaptation*:** **`M091`** membuktikan 5 epoch *fine-tuning* memulihkan akurasi eksternal dari `61%` ke `73%` (+12 poin). **`M219`** membuktikan penggabungan sampel target berlabel menaikkan QWK `+0,21`, dan penambahan **Cross-Domain Supervised Contrastive Loss berbobot jarak grade + EMD** menaikkan lagi QWK `0,678 -> 0,732` (MAE `0,598 -> 0,525`).
5. **Jangkar 3 Radiolog Independen `RheumaVIT`:** Mengevaluasi model terhadap konsensus, kesepakatan antar-radiolog (*leave-one-reader-out*), dan distribusi probabilitas 3 radiolog.

---

### RQ4: Kalibrasi, Uncertainty Quantification, dan Rujukan Selektif (G3)

Sudah ada di tesis: `temperature scaling`, `pemulihan log-prior`, `conformal ordinal APS (contiguous)`.

| # | Kandidat | Bukti Terverifikasi di SLR-M | Status | Prioritas | Catatan untuk RAM-H1200 & RheumaVIT |
|---|---|---|---|---|---|
| 4.1 | **`RAPS` (Regularized Adaptive Prediction Sets) & `Mondrian (Class-Conditional) Conformal`** | **`M110`** (koreksi full-text: *calibration split* = 1.000 citra dari 2.614 val; `RAPS` mencapai cakupan `0,91–0,93` pada target `0,90`) | mirip (`APS`) | A | **Tanpa training baru (0 GPU)**: `Mondrian` menjamin cakupan $\ge 90\%$ pada grade erosi $\ge 2$, bukan hanya didominasi grade 0 |
| 4.2 | **Cross-Domain Conformal Re-calibration di `RheumaVIT`** | Eksplorasi baru `RQ3 × RQ4` berbasis temuan **`M110`** | belum | A | **Tanpa training baru (0 GPU)**: uji pergeseran cakupan konformal di `RheumaVIT` (*zero-shot*) vs re-kalibrasi kuantil $\hat{q}$ memakai 5/20 pasien `RheumaVIT` |
| 4.3 | **`MC Dropout` + Kurva *Risk-Coverage* & Prediksi Kesalahan (`AUROC Uncertainty -> Error`)** | **`M130`** (4-fold patient-wise CV $\times$ 15 repeats: varians MC Dropout tervalidasi sebagai prediktor kesalahan klasifikasi), **`M039`** (menghapus gating MC Dropout menurunkan QWK `-0,040`, $p<0,001$), **`M110`**, **`M061`**, **`M280†`** | belum | A | Tambah dropout di head (1 arm pelatihan, 1,9 jam GPU); dievaluasi di `RAM-H1200` dan `RheumaVIT` |
| 4.4 | **`Deep Ensemble` Antar-Seed (3 Seed) untuk *Selective Prediction*** | Celah literatur SLR-M (`C093` dan `M141` hanya menguji ensemble untuk akurasi titik, belum ada yang menguji varians ensemble untuk rujukan selektif / UQ) | belum untuk UQ | A | **Tanpa training baru (0 GPU)**: langsung memanfaatkan 3 seed yang sudah dilatih per arm |
| 4.5 | **Kalibrasi Multikelas Kaya (`Vector / Dirichlet Calibration`)** | Literatur kalibrasi standar (Kull et al. NeurIPS 2019; *catatan: rujukan `M095` dihapus karena fabrikasi*) | mirip (`TS`) | A (Pasca-latih) | **Tanpa training baru (0 GPU)**: bandingkan terhadap *Temperature Scaling* pada ECE, NLL, dan Brier |
| 4.6 | **Evidential Deep Learning (`EDL` — Parameterisasi Dirichlet & *Subjective Logic*)** | **`M270`** (ketidakpastian $u = K/S$ secara konsisten lebih tinggi pada prediksi salah dibanding benar di 3 eksperimen, meskipun `Q1_abl = N` terhadap CE), **`M110`** | belum | B | 1 arm pelatihan (1,9 jam GPU) |

---

### RQ5: Label Noise, Konsistensi Longitudinal, dan Semi-Supervised Teacher-Student (G4)

Sudah ada di tesis: hanya `E9` (penalti pasangan kunjungan ChronoCon, tidak ada gain).

| # | Kandidat | Bukti Terverifikasi di SLR-M | Status | Prioritas | Catatan untuk RAM-H1200 & RheumaVIT |
|---|---|---|---|---|---|
| 5.1 | **Confident Learning (`CleanLab`) via 5-Fold Out-of-Fold di dalam Train Set: `Relabel` vs `Prune` vs `Down-weight` (Label Uji Haram Diubah!)** | **`M272`** (mengubah 22% label uji $\to$ akurasi semu `82,07%`, namun pada **label uji asli** McNemar $p=0,076$ **tidak signifikan**), **`M042`** (menggabungkan train+val+test sebelum relabeling 286 citra), **`M117`** (membersihkan lipatan uji CV). Eksperimen ini menguji secara jujur apakah CleanLab bermanfaat tanpa korupsi data uji | belum | A | Evaluasi wajib pada label uji asli `RAM-H1200` **dan** pada 3 radiolog eksternal `RheumaVIT` |
| 5.2 | **Semi-Supervised *Consistency Regularization / Mean-Teacher* pada Sendi Tanpa Label (Rezim 10% & 25% Label)** | **`M061`** (split pasien pada Sites A–D, uji eksternal Site E, 5 seed: pada ~10% label [800 citra], *consistency regularization* menaikkan Balanced Accuracy `+4,21%` [ENT-MCD `52,18% -> 56,39%`] hingga `+11,35%` pada 250 label), **`M077`** (50% label SSL mengalahkan supervised `0,66 vs 0,58` di OAI dan `0,71 vs 0,65` di MOST), **`M185`** (`H-SemiS`, kode publik), **`M153`** | belum | A | Sangat relevan untuk rezim 10% label di mana 90% pasien latih `RAM-H1200` bertindak sebagai data tak berlabel |
| 5.3 | **Pemisahan Sampel Berbasis Reliabilitas Kunjungan Longitudinal + Loss Hibrida** | **`M200†`** (*IEEE JBHI* 2022: pemisahan dinamis sampel yakin vs tidak yakin) + temuan diagnosis internal `RAM-H1200` (47% sendi grade 1 berubah menjadi grade 0 pada kunjungan berikutnya) | belum | A | Skor reliabilitas diturunkan dari konsistensi biologis antarkunjungan + *confidence* OOF |
| 5.4 | **Label Smoothing ($\varepsilon=0,1$) & *Reader-Agreement-Informed Ordinal Soft Labels*** | **`M238`** (ablasi *leave-one-out* Tabel 6 hlm. 1306: menghapus label smoothing menurunkan akurasi `93,4% -> 91,8%` dan F1 `92,6% -> 91,0%`), **`M044`**, **`M266†`**, **`C047`** | belum | A | Murah; lebar distribusi soft label disesuaikan dengan tingkat ketidaksepakatan pembaca per grade |
| 5.5 | **Teacher-Student Knowledge Distillation / Self-Distillation dari Ensemble Seed** | **`M157`** (Tabel 2 hlm. 2142: *teacher-student distillation* $T=7, \alpha=0,8$ meningkatkan QWK `0,524 -> 0,551` dan F1 `0,427 -> 0,478` dengan 1.000 patient-bootstrap CI), **`M071`** (distilasi memulihkan Macro-F1 setelah jatuh karena *class weight*) | belum | B | Teacher = ensemble 3 seed, Student = model tunggal |
| 5.6 | **Evaluasi terhadap Distribusi 3 Radiolog Eksternal (`RheumaVIT`)** | Tidak ada di SLR-M (kontribusi evaluasi baru untuk mengatasi masalah label uji yang *noisy*) | belum | A (Evaluasi) | **Tanpa training baru (0 GPU)**: ukur KL-divergence/Brier terhadap distribusi empiris 3 radiolog `RheumaVIT` |

---

## 4. Ringkasan Celah Literatur yang Diangkat sebagai Kebaruan (*Novelty*) Tesis

1. **Ablasi Sistematis Pertama G1–G4 pada Grading Erosi RA per Sendi (SvdH 0–5):** Dari 44 studi valid di SLR-M, 39 berfokus pada OA lutut/pinggul dan hanya 5 pada RA (tidak ada satu pun yang mengablasi SSL, dekomposisi imbalance/ordinal, UQ/conformal, dan label noise secara terpadu pada ROI sendi).
2. **Standar Metodologis Bebas-Bocoran (*Strict Patient-Level Split & Untouched Test Set*):** Mengatasi kelemahan 70,5% literatur yang tidak memakai split pasien serta membongkar ilusi kenaikan akurasi +15%–24% akibat modifikasi label himpunan uji pada studi CleanLab/relabeling (`M042, M117, M272`).
3. **Validasi Eksternal Lintas-Benua Pertama pada Erosi RA (`RAM-H1200` Jepang $\to$ `RheumaVIT` Rusia) dengan Jangkar Multi-Radiolog:** Mengisi kekosongan 0% validasi eksternal pada studi RA di SLR-M, dilengkapi dekomposisi *Prior Shift vs Covariate Shift*, *Cross-Domain Conformal Re-calibration*, dan *Few-Shot Contrastive Domain Adaptation* (`M219`).
