# SLR-M: kandidat ablasi per RQ (sementara, berbasis abstrak)

**Status:** 2026-11-10. Disusun dari **abstrak** 69 studi yang lolos screening judul/abstrak. Full text belum dibaca, jadi angka, ada/tidaknya ablasi, dan kode tersedia **belum diverifikasi**. Semua keputusan berasal dari satu reviewer AI dan wajib dicek manusia.
**Protokol:** `slr-methods/protocol/01_protocol_PRISMA-P.md` (v0.1, disetujui penulis 2026-11-10).

## 1. Alur sementara

| Tahap | n |
|---|---|
| Scopus | 290 |
| PubMed | 96 |
| IEEE Xplore: 310 hasil; 94 sudah ada di Scopus (DOI); 215 sisanya disaring dengan blok DL + G1–G4 yang sama seperti Scopus/PubMed (abstrak lengkap dari OpenAlex/Crossref) | 41 masuk (35 cocok + 6 tanpa abstrak) |
| Total sebelum deduplikasi | 427 |
| Duplikat | 86 |
| Di-screen judul/abstrak | 341 |
| Dikeluarkan: EM3 tidak ada metode G1–G4 (sebagian besar hanya transfer ImageNet atau augmentasi umum) 144, EM2 bukan grading ordinal 69, EM1 bukan radiografi 36, EM4 bukan riset primer 18, EM5 bukan DL 4, EM6 tanpa evaluasi 1 | 272 |
| **Lanjut ke full text** (43 jurnal, 26 conference; 38 open access, 31 berbayar) | **69** |
| Per kelompok (satu studi bisa >1): G1 SSL 16, G2 imbalance/ordinal 38, G3 kalibrasi/uncertainty 9, G4 label noise/teacher-student 16 | |

Catatan penyimpangan dari protokol: IEEE tidak bisa diekspor tanpa akun pribadi, jadi blok DL + G1–G4 diterapkan setelah pengambilan, pada judul + abstrak + keyword dari OpenAlex/Crossref, bukan di dalam query IEEE.

## 2. Gambaran umum literatur

- Mayoritas studi adalah **KL grading OA lutut**. Dataset paling sering dipakai: OAI dan dataset Kaggle/Mendeley turunan OAI (±9.786 citra). Banyak studi memakai split per citra dan hanya melaporkan accuracy. Selisih kinerja yang diklaim kemungkinan optimis.
- Studi yang dekat dengan kasus kita: **OA tangan sendi DIP** (M041), **sintesis citra sendi jari RA** (M199), **deteksi erosi tangan RA dengan focal loss** (M100), **mTSS RA** (M222), **SSL untuk skor SvdH RA** (M003), dan **skema fine-tuning untuk SvdH** (M005).
- Studi yang punya ablasi komponen, split per pasien, atau uji eksternal (diklaim di abstrak): M039, M041, M042, M044, M082, M083, M130, M155, M201, M219, M266. Kelompok ini diprioritaskan saat membaca full text.

## 3. Kandidat per RQ

Kolom **Status tesis**: *ada* = sudah diuji di eksperimen; *mirip* = ada varian dekat; *belum* = belum pernah diuji.
Kolom **Cocok RAM-H1200**: penilaian reviewer (*reader-inferred*), bukan klaim studi.
**Ide modifikasi** yang tidak berasal dari studi ditandai (R) = usulan reviewer.

### RQ1: pre-training dan SSL (G1)

Yang sudah ada di tesis: scratch, ImageNet, MAE, DINO, DINO + ChronoCon, LLRD, frozen linear probe.

| # | Kandidat | Bukti (studi) | Status tesis | Cocok RAM-H1200 | Ide modifikasi |
|---|---|---|---|---|---|
| 1.1 | **Foundation model radiografi muskuloskeletal** (SKELEX, SSL pada 1,2 juta radiograf MSK) sebagai encoder | M015 (npj Digit Med 2026): unggul atas baseline untuk OA grading | belum | Tinggi bila bobotnya tersedia (cek di full text) | Linear probe vs fine-tune vs LLRD, sama seperti E1/E2 |
| 1.2 | **Foundation model medis umum** (RAD-DINO, BiomedCLIP, MedSAM, DINOv2) sebagai encoder | M157: embedding beku + XGBoost **kalah** dari CNN supervised; M083: fitur MedSAM/DINOv2 dipakai | belum | Tinggi (bobot publik) | Uji fine-tune, bukan hanya embedding beku, karena bukti beku negatif |
| 1.3 | **Supervised contrastive (SupCon) + CE** | M082 (multikohort, KL1 vs KL2 membaik), M166, M280 (SupCon sadar usia/jenis kelamin) | belum (tesis memakai SSL tanpa label) | Tinggi, murah | (R) SupCon dengan bobot jarak grade ordinal; positif = sendi sama-grade lintas pasien; varian sadar jenis sendi |
| 1.4 | **SSL berbasis "normalitas"/anomaly detection** (dilatih dari sendi sehat saja, jarak ke pusat normal = skor keparahan) | M197 (SS-FewSOME + pseudo label, < 3% label) | belum | Tinggi: sendi grade 0 sangat banyak | (R) Latih pada sendi grade 0, pakai jarak sebagai fitur tambahan atau gate grade 0 vs ≥ 1 |
| 1.5 | **SSL rekonstruksi/adversarial autoencoder** + tugas bantu | M153 (DC-AAE + tugas bantu identifikasi sisi kaki), M185 (rekonstruksi adversarial di H-SemiS) | mirip (MAE) | Sedang | (R) Tugas bantu jenis sendi atau sisi tangan (label gratis dari ROI) |
| 1.6 | **Representasi progresi temporal** diselaraskan ke model citra tunggal (MMD) | M335 (SSL ViT + TCN atas sekuens + rekalibrasi MMD; inferensi citra tunggal) | mirip (ChronoCon, E9) | Tinggi: RAM-H1200 longitudinal | (R) Penyelarasan distribusi alih-alih kontras berurutan; mungkin menghindari masalah ChronoCon pada sendi yang stabil |
| 1.7 | **Vision-language / CLIP-style** dengan teks templat dari skor | M201 (CLIP fine-tuned, uji eksternal), M306 (templat laporan dari skor tabular + caption kontras), M014, M071 (selaras teks + ordinal) | belum | Sedang: tidak ada laporan, tetapi templat dari deskripsi grade SvdH bisa dibuat | (R) Templat teks per grade dan per jenis sendi |
| 1.8 | **Strategi fine-tuning**: layer freezing, gradual unfreezing, beberapa skema fine-tuning | M051, M269, M005 (RA SvdH: beberapa skema fine-tuning + ensemble, setara pembaca) | mirip (LLRD, linear probe) | Tinggi, murah | Ablasi full FT vs LLRD vs gradual unfreeze vs LP-FT pada encoder yang sama |
| 1.9 | **Transfer bertahap dari domain dekat** (pre-train pada data KL lutut publik, lalu RA) | M005 (memakai dataset publik), M260 (pre-training diagnostik untuk prognosis; di luar kelayakan, dicatat) | mirip (pool RAM-W600 di E6abl) | Sedang | (R) OAI KL sebagai tahap supervised antara |
| 1.10 | **Attention + SSL untuk SvdH RA** | M003 (ECBANet, dataset Z. Wang, skor total) | belum | Rendah-sedang (skor total, bukan per sendi) | — |

### RQ2: class imbalance dan ordinalitas (G2)

Yang sudah ada di tesis: B0 ordinal BCE, B1 class weight + oversampling, B2 balanced softmax / logit adjustment, B3 CORN, B4 CORN + LA, B5 gate hierarkis, B6 cRT, E7 undersampling grade 0, E8 ablasi augmentasi (none/geom/intensity).

| # | Kandidat | Bukti | Status tesis | Cocok | Ide modifikasi |
|---|---|---|---|---|---|
| 2.1 | **Focal loss** dan **focal ordinal loss** | M132 (focal ordinal loss), M041 (hand OA DIP), M224 (+ mixup), M100 (erosi tangan RA, RetinaNet) | belum | Tinggi | Focal pada CE dan pada kepala ordinal; γ ∈ {1, 2} |
| 2.2 | **Oversampling murni** (tanpa class weight) | M041: strategi dengan oversampling paling konsisten (hand OA, 6 strategi diablasi); M141/M283 weighted sampling | **belum**: B1 selalu menggabungkan oversampling + class weight | Tinggi, murah | Pisahkan B1 jadi "hanya OS" dan "hanya CW" (celah yang sudah tercatat di Notion) |
| 2.3 | **Kombinasi weighted sampling + class-weighted CE + label smoothing** | M044 (split per pasien, ablasi komponen, CI bootstrap, uji eksternal), M238 | mirip (B1) | Sedang | Ablasi per komponen seperti M044 |
| 2.4 | **Loss ordinal lain**: cumulative link model (cut-point), CORAL, penalti jarak (ordinal CE / fixed-penalty), soft label ordinal | M089 (cumulative link), M266 (20 kombinasi: 2 loss × 2 output × 5 soft label; terbaik CE + cumulative link + soft label beta), M039 & M095 (CORAL), M305 (penalti jarak), M028, M091, M150, M303 | mirip (B0 ≈ ambang kumulatif, B3/B4 CORN); **belum** untuk cumulative link, soft label, penalti jarak | Tinggi | Grid kecil seperti M266 pada kepala yang sama |
| 2.5 | **Kalibrasi ambang keputusan** pasca-latih untuk imbalance | M052 (inverse-frequency weight + decision threshold calibration) | belum | Tinggi, **tanpa training baru** (pakai logit tersimpan) | Ambang per kelas dipilih di val untuk BACC4 |
| 2.6 | **Retrieval-augmented / kNN classification** di atas embedding | M161 (konsisten naik pada 17 arsitektur) | mirip (B6 cRT juga memisahkan fitur dan classifier) | Tinggi, murah (embedding tersimpan) | kNN berbobot jarak vs linear head |
| 2.7 | **Mixup varian** (MixUp, DAB-mix) untuk kelas minoritas | M224, M269 | belum (Balanced-MixUp tercatat sebagai arm lanjutan E8) | Sedang | Balanced-MixUp, mixup hanya antargrade bertetangga (R) |
| 2.8 | **Augmentasi selektif kelas** (augmentasi lebih kuat untuk grade jarang) | M101, M275 | belum (E8 seragam untuk semua kelas) | Tinggi, murah | Gabungkan dengan resep E8 |
| 2.9 | **Data sintetis kelas minoritas** (CycleGAN, latent diffusion, StyleGAN3, GAN, sintesis lapisan sendi jari) | M041 (CycleGAN **divalidasi rheumatolog**), M068, M084, M165, M173, M191, M199 (sendi jari RA) | ditolak sebelumnya (grade hasil sintesis tidak bisa diverifikasi tanpa klinisi) | Rendah sampai ada validator klinis | Hanya bila ada rheumatolog yang memfilter, seperti M041 |
| 2.10 | **SMOTE di ruang fitur** | M162, M233 | belum | Rendah | — |
| 2.11 | **Uji shortcut lewat augmentasi adversarial** (hapus ROI, injeksi noise) | M249 (model tetap benar pada grade ekstrem tanpa sendi → bias) | belum | Tinggi sebagai **uji diagnostik** | (R) Masking lesi/ROI untuk melihat apakah model memakai konteks di luar erosi |
| 2.12 | **Ensemble** (weighted, majority voting) | M141, M148, M225 | mirip (seed ensemble pasca-hoc: gain kecil) | Rendah | — |

### RQ3: ketahanan lintas dataset (dievaluasi untuk semua varian)

Tidak ada kandidat metode yang berdiri sendiri. Dari studi yang masuk ada beberapa metode yang khusus menangani pergeseran domain, sebagai bahan pembanding:
- **Supervised contrastive domain adaptation** dengan seleksi sampel sumber (M219). Pada skema RQ3 ini butuh data target tanpa label dari RheumaVIT, jadi harus dibedakan dari transfer langsung.
- **Test-time batch-norm adaptation** (M095). Murah, bisa diuji pada RheumaVIT tanpa label.
- Studi dengan uji eksternal yang bisa jadi pembanding besar *domain drop*: M039, M044, M082, M083, M130, M155, M201.

### RQ4: kalibrasi, uncertainty, rujukan selektif (G3)

Yang sudah ada di tesis: temperature scaling, pemulihan log-prior, conformal ordinal APS (contiguous).

| # | Kandidat | Bukti | Status tesis | Cocok | Ide modifikasi |
|---|---|---|---|---|---|
| 4.1 | **Deep ensemble** (antar seed) sebagai uncertainty | M143/M159 hanya akurasi; tidak ada studi SLR-M yang menguji ensemble untuk UQ | belum untuk UQ | Tinggi, **tanpa training baru** (seed sudah ada) | Varians ensemble sebagai skor rujukan |
| 4.2 | **MC dropout** + kurva risk-coverage / selective prediction | M039 (gating MC dropout), M110 (perbandingan UQ), M280 (risk-coverage) | belum | Tinggi (perlu dropout di head) | Bandingkan dengan conformal untuk rujukan |
| 4.3 | **Evidential deep learning** | M270, M110 | belum | Sedang | — |
| 4.4 | **Varian conformal**: class-conditional/Mondrian (cakupan per grade), conformal setelah test-time adaptation | M095 (temperature scaling + test-time BN + conformal), M110 | mirip (APS ordinal) | Tinggi, murah | (R) Mondrian per grade dan per jenis sendi; cek cakupan grade ≥ 2 |
| 4.5 | **Kalibrasi lebih kaya**: Dirichlet / vector scaling | M095 (Dirichlet calibration term) | mirip (temperature scaling) | Tinggi, murah | Vector/Dirichlet vs temperature tunggal |
| 4.6 | **Uncertainty sebagai prediktor kesalahan**, divalidasi eksternal | M130 (hip OA, eksternal 52 pasien) | mirip | Tinggi | Laporkan AUROC uncertainty→salah pada RheumaVIT |
| 4.7 | **Bayesian decision tree** di atas embedding | M001 | belum | Rendah | — |

### RQ5 (baru): label noise, label correction, teacher-student (G4)

Belum ada di tesis kecuali E9 (penalti konsistensi pasangan kunjungan; hasilnya tidak ada gain).

| # | Kandidat | Bukti | Status tesis | Cocok | Ide modifikasi |
|---|---|---|---|---|---|
| 5.1 | **Confident learning / CleanLab relabeling** | M272 (relabel otomatis + buang outlier, KL lutut) | belum | Tinggi: pakai prediksi out-of-fold pada train | Relabel vs prune vs bobot turun; test label tetap tidak diubah |
| 5.2 | **Pemisahan sampel menurut reliabilitas + loss hibrida** | M200 (JBHI 2022: data dibagi dinamis menjadi yakin/tidak yakin) | belum | Tinggi | (R) Reliabilitas dari konsistensi antarkunjungan (temuan diagnosis 2026-11-09: grade 1→0 47%) |
| 5.3 | **Co-teaching** (dipandu uncertainty) | M095 | belum | Sedang-tinggi | Co-teaching ResNet34 × ViT-S |
| 5.4 | **Pembaruan reliabilitas label adaptif + metric learning ordinal** | M042 (ablasi kaskade) | belum | Sedang | — |
| 5.5 | **Label smoothing** dan **soft label ordinal** (beta/Gaussian) | M044, M238 (smoothing), M266 (5 pendekatan soft label) | belum | Tinggi, murah | (R) Lebar soft label disetel dari ketidaksepakatan antarpembaca (ICC pra-konsensus) |
| 5.6 | **Semi-supervised teacher-student / consistency regularization** memakai sendi tanpa label | M061 (active learning + consistency), M077, M185 (teacher-student hierarkis + pseudo label), M153 | belum (celah yang tercatat 2026-11-09) | **Tinggi pada 10% label**: pasien tanpa label tersedia | Mean teacher / FixMatch-style pada ROI sendi |
| 5.7 | **Pseudo-label + denoising** | M197 (pseudo label dibersihkan dengan CLIP) | belum | Sedang | — |
| 5.8 | **Knowledge distillation / self-distillation** | M155 (KD untuk generalisasi, uji eksternal), M095 (SWA-EMA self-distillation), M175 (kompresi) | belum | Tinggi | (R) Teacher = ensemble seed, student = model tunggal dengan soft target |
| 5.9 | **Evaluasi terhadap distribusi pembaca** | tidak ada di SLR-M | — | RheumaVIT punya 3 radiolog | (R) Evaluasi dengan soft label pembaca, karena label uji yang noisy bisa menyembunyikan gain RQ5 |

## 4. Metode yang lazim di literatur umum tetapi belum muncul di SLR-M

Belum ditemukan di 69 studi (masih berbasis abstrak; full text bisa mengubahnya): SimCLR, MoCo, BYOL, iBOT sebagai pembanding SSL langsung; LDAM-DRW dan class-balanced loss (effective number); DivideMix dan GCE/symmetric CE untuk label noise. Kandidat ini bisa ditambahkan dari paper aslinya, tetapi dasarnya bukan SLR-M. Sitasi aslinya belum diverifikasi.

## 5. Usulan prioritas (reader-inferred, untuk didiskusikan dengan dosen)

Disusun dari tiga hal: bukti di SLR-M, biaya (ada yang tidak perlu training baru), dan kecocokan dengan RAM-H1200.

- **Tanpa training baru, memakai logit/embedding tersimpan:** 2.5 kalibrasi ambang, 2.6 kNN, 4.1 deep ensemble, 4.4 Mondrian conformal, 4.5 vector/Dirichlet, 4.6 uncertainty→error.
- **Murah, satu sumbu per RQ:**
  - RQ1: 1.2 RAD-DINO/BiomedCLIP fine-tune, 1.3 SupCon, 1.8 strategi fine-tuning.
  - RQ2: 2.1 focal, 2.2 oversampling murni, 2.4 cumulative link + soft label.
  - RQ5: 5.1 CleanLab, 5.5 soft label, 5.8 self-distillation.
- **Lebih mahal, tetapi paling sesuai dengan celah yang sudah didiagnosis:** 5.6 semi-supervised teacher-student pada 10% label, 1.4 SSL normalitas pada sendi grade 0, 1.6 penyelarasan progresi temporal, 5.2 reliabilitas dari konsistensi kunjungan.
- Semua varian dilatih di RAM-H1200 dan diuji tanpa pelatihan ulang di RheumaVIT (RQ3).

## 6. Langkah berikutnya

1. Ambil full text: 38 open access (perlu izin unduh) dan 31 berbayar (perlu diunduh penulis; daftar di `screening/02_fulltext_candidates.csv`).
2. Penilaian full text, ekstraksi lengkap (§7 protokol), dan checklist kualitas 6 butir (§8).
3. Citation searching satu putaran dari studi yang punya ablasi.
4. Finalisasi daftar kandidat, lalu desain ablasi per RQ di tesis. Tidak ada run sebelum disetujui.
