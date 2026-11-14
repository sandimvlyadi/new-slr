# Protokol Systematic Review (PRISMA-P 2015): SLR-M

**Judul kerja:** Metode self-supervised learning, penanganan class imbalance, kalibrasi/uncertainty, dan penanganan label noise untuk grading ordinal radiografi muskuloskeletal (2021–2026): systematic review untuk menyusun kandidat ablasi

**Kode:** SLR-M (terpisah dari SLR erosi RA 22 studi di `report/`, yang tidak diubah)
**Versi:** v0.1 (draf, 2026-11-10). Menunggu konfirmasi penulis sebelum pencarian (checkpoint ARS fase 1).
**Registrasi:** belum terdaftar (OSF disarankan bila akan dipublikasikan).
**Asal:** bimbingan 2026-11-10 (`report/review/advisor_2026-11-10/`). Dosen meminta ablasi beberapa metode per RQ di proyek tesis. Kandidat metodenya diambil dari studi kasus yang mirip.
**Reviewer:** asisten AI (Claude) sebagai reviewer tunggal untuk screening, ekstraksi, dan penilaian. Semua keputusan **wajib diverifikasi** oleh reviewer manusia.

---

## 1. Rationale

Tesis menilai model untuk grading erosi tulang per sendi (SvdH 0–5) pada radiografi tangan RA (RAM-H1200; uji eksternal RheumaVIT). Tugas ini punya empat sifat: kelas ordinal, kelas 0 dominan (long-tailed), lesi kecil, dan label yang tidak stabil antarpembaca dan antarkunjungan. Literatur erosi RA sendiri hampir kosong untuk metode yang relevan: di SLR 22 studi hanya 2 studi memakai SSL dan tidak ada yang menangani label noise. Tugas lain yang memiliki keempat sifat yang sama, terutama grading Kellgren-Lawrence (KL) pada OA lutut, sudah lebih banyak diteliti. SLR-M mengumpulkan metode dari tugas-tugas tersebut supaya kandidat ablasi per RQ dipilih dengan dasar yang jelas, tidak berdasarkan selera.

## 2. Pertanyaan

**RQ utama:** Metode apa yang dipakai untuk meningkatkan grading ordinal radiografi muskuloskeletal dengan deep learning dalam empat kelompok berikut, dan seberapa besar perbedaannya terhadap baseline pada studi yang sama?

| Kelompok | Sesuai RQ tesis | Contoh lingkup |
|---|---|---|
| **G1** Pre-training dan SSL | RQ1 | contrastive (SimCLR, MoCo, BYOL), self-distillation (DINO, iBOT, DINOv2), masked image modeling (MAE, SimMIM), foundation model, transfer domain-medis vs ImageNet, modifikasi SSL (augmentasi khusus domain, objektif longitudinal/anatomis) |
| **G2** Class imbalance dan ordinalitas | RQ2 | re-sampling, re-weighting/cost-sensitive, focal/LDAM/class-balanced loss, logit adjustment, decoupling, loss ordinal (CORAL, CORN, ordinal CE, EMD), mixup/augmentasi kelas minoritas |
| **G3** Kalibrasi, uncertainty, selective prediction | RQ4 | temperature/vector scaling, MC dropout, deep ensemble, evidential, conformal prediction, rujukan/abstensi |
| **G4** Label noise, label correction, teacher-student | RQ5 (baru) | label smoothing, soft label dari banyak pembaca, co-teaching, DivideMix, confident learning, mean teacher, knowledge distillation, pseudo-label/semi-supervised, label correction |

RQ3 tesis (ketahanan RAM-H1200 → RheumaVIT) **tidak** punya kelompok sendiri. RQ3 menjadi cara mengevaluasi semua varian G1–G4. Dari setiap studi yang masuk tetap dicatat apakah ada uji lintas dataset atau senter.

**Sub-pertanyaan:**
1. Metode apa saja yang dipakai per kelompok, pada tugas, sendi, dan dataset apa?
2. Berapa selisih kinerjanya terhadap baseline pada split yang sama, dan apakah selisih itu berasal dari ablasi yang terkontrol?
3. Apakah selisih tersebut bertahan pada data eksternal atau lintas senter?
4. Metode mana yang bisa diterapkan pada ROI sendi RAM-H1200 (5 grade, data kecil, GPU RTX 3060), dan modifikasi apa yang diusulkan atau masuk akal?

## 3. Kriteria kelayakan

| Kriteria | Include | Exclude |
|---|---|---|
| Tahun | 2021–2026 | < 2021 (metode dasar tetap disitasi dari paper aslinya, di luar hitungan PRISMA) |
| Tipe | Artikel jurnal atau full paper conference yang sudah melalui peer review | Review, editorial, letter, abstrak konferensi, preprint yang belum terbit (arXiv saja), tesis |
| Bahasa | Inggris | Lainnya |
| Kuartil | **Tidak menjadi syarat.** Kuartil SJR (Q1–Q4, conference, tanpa kuartil) dicatat dan dipakai untuk **mengurutkan prioritas** bukti | — |
| Modalitas | Radiografi konvensional (X-ray) | MRI, CT, USG, DXA saja |
| Tugas | Grading/klasifikasi derajat keparahan ordinal per sendi atau per citra: KL atau OARSI pada OA lutut atau tangan, SvdH/mTSS erosi atau JSN pada RA, atau skala ordinal sejenis pada sendi muskuloskeletal | Hanya deteksi biner penyakit tanpa derajat (misalnya RA vs normal), hanya segmentasi, fraktur, bone age, skoliosis, prediksi progresi dari data non-citra |
| Metode | Model deep learning yang memakai ≥ 1 metode G1–G4 dalam pelatihan atau inferensi | Model tanpa satu pun metode G1–G4; inisialisasi ImageNet standar yang hanya disebut sebagai default dan tidak dianalisis |
| Outcome | ≥ 1 metrik kuantitatif (accuracy, BACC, QWK/kappa, F1, AUC, MAE, ECE/NLL, coverage, dsb.) | Tanpa evaluasi kuantitatif |

**Prioritas, bukan syarat:** (1) ada ablasi atau perbandingan terhadap baseline tanpa metode tersebut pada split yang sama; (2) jurnal Q1–Q3; (3) split per pasien; (4) uji eksternal. Studi tanpa ablasi tetap masuk, tetapi bobot buktinya lebih rendah.

## 4. Sumber informasi

| Sumber | Cara | Catatan |
|---|---|---|
| Scopus | Akses kampus ITB, ekspor CSV oleh penulis atau via browser | Sumber utama, termasuk LNCS/MICCAI |
| PubMed/MEDLINE | E-utilities API | Otomatis |
| IEEE Xplore | Antarmuka web, semua tipe konten | ISBI, EMBC, jurnal IEEE |
| Citation searching | OpenAlex API: backward + forward dari studi inklusi yang punya ablasi | Satu putaran |
| Web of Science | Tidak dipakai (kampus tidak berlangganan) | Sama dengan SLR sebelumnya |

## 5. Strategi pencarian

Struktur: **[Tugas] AND [Radiografi] AND [Deep learning] AND ([G1] OR [G2] OR [G3] OR [G4])**, 2021–2026, bahasa Inggris. Query lengkap ada di `slr-methods/search/`. Setiap rekaman diberi tag kelompok berdasarkan blok yang cocok dengan rekaman tersebut. Satu rekaman boleh punya beberapa tag.

- **Tugas:** "knee osteoarthritis", "hand osteoarthritis", Kellgren, "KL grade*", "rheumatoid arthritis", "Sharp score", "van der Heijde", "bone erosion*", "joint space narrowing", OARSI, mTSS, "total sharp score", plus MeSH Osteoarthritis, Knee dan Arthritis, Rheumatoid.
- **Radiografi:** radiograph*, "x-ray*", xray*, "plain film*", MeSH Radiography.
- **Deep learning:** "deep learning", "neural network*", convolutional, CNN, transformer*, "machine learning".
- **G1:** self-supervised, contrastive, "masked autoencoder*", "masked image model*", SimCLR, MoCo, BYOL, DINO/DINOv2, pretext, "foundation model*", pre-train*/pretrain*, "transfer learning".
- **G2:** imbalance*, long-tail*, "focal loss", over/undersampl*, "class weight*", "weighted loss", cost-sensitive, SMOTE, "ordinal regression/loss/classification", mixup, "data augmentation".
- **G3:** calibrat*, uncertaint*, conformal, "Monte Carlo dropout", "deep ensemble*", evidential, Bayesian, "selective prediction", out-of-distribution, abstention, "reject option", "confidence estimat*".
- **G4:** "label noise", "noisy label*", "label correction", "label smoothing", teacher-student, "mean teacher", "knowledge distillation", pseudo-label*, semi-supervised, co-teaching, inter-rater/inter-observer, "label uncertainty", "soft label*", weak supervision/weakly supervised.

**Uji coba PubMed (2026-11-10, hanya jumlah):** tugas + radiografi + DL = 377; G1 42, G2 26, G3 33, G4 23; gabungan G1–G4 = 110. Query final (dengan OARSI/mTSS dan G3 diperbaiki): lihat log pencarian. Pada uji coba ini blok G3 masih memuat `confidence[tiab]`, yang ikut menangkap "confidence interval". Di query final istilah itu diganti "confidence estimat*".

**Keterbatasan yang diterima:** metode yang hanya disebut di full text, misalnya class weighting yang tidak ditulis di abstrak, bisa terlewat. Mitigasinya adalah citation searching dan pencatatan semua metode G1–G4 yang ditemukan di full text studi inklusi, termasuk yang tidak muncul di abstrak.

## 6. Seleksi

1. Gabungkan semua sumber, deduplikasi berdasarkan DOI, lalu judul ternormalisasi.
2. Screening judul/abstrak terhadap §3, dengan kode alasan eksklusi EM1–EM7: EM1 bukan radiografi, EM2 bukan grading ordinal, EM3 tidak ada metode G1–G4, EM4 bukan riset primer, EM5 bukan DL, EM6 tanpa evaluasi kuantitatif, EM7 di luar tahun/bahasa/tipe.
3. Full text: unduh otomatis yang open access (dengan izin penulis). Yang berbayar diunduh penulis ke `slr-methods/papers/`. Yang tidak bisa diperoleh dicatat.
4. Penilaian full text dengan kode yang sama.
5. Satu putaran citation searching dari studi inklusi yang punya ablasi.
6. Seluruhnya dikerjakan reviewer AI tunggal. Untuk verifikasi manusia disiapkan sampel acak 20% screening judul/abstrak beserta semua eksklusi di tahap full text.

## 7. Data yang diekstraksi (berorientasi ablasi)

Per studi: sitasi, venue, kuartil SJR, tugas dan skala (KL 0–4, SvdH, dsb.), sendi, dataset (nama, n pasien/citra, senter, publik?), unit split, backbone, baseline.

Per metode (satu studi bisa punya beberapa baris):
- kelompok (G1–G4) dan keluarga metode (misalnya G2 → loss reweighting → focal)
- deskripsi singkat dan hyperparameter kunci
- **apakah ada ablasi** (metode vs tanpa metode pada split yang sama), dengan nilai baseline dan nilai dengan metode
- selisih pada metrik utama; jumlah seed atau run; ada CI atau tidak
- hasil pada data eksternal (bila ada)
- kode tersedia (link)
- biaya komputasi (GPU, waktu) bila dilaporkan
- **kesesuaian dengan RAM-H1200**: Tinggi, Sedang, atau Rendah, beserta alasannya (input ROI sendi 5 grade, ±15 ribu sendi latih, GPU 12 GB, dukungan untuk ResNet34/ViT-S)
- **status di tesis**: sudah ada (kode eksperimen), mirip, atau belum ada
- ide modifikasi dari penulis studi, ditambah usulan modifikasi dari reviewer yang ditandai *reader-inferred*

## 8. Penilaian kualitas bukti per klaim metode

Tidak memakai QUADAS-2, karena yang dinilai di sini adalah klaim bahwa metode tertentu memperbaiki kinerja, bukan akurasi diagnostik. Dipakai checklist 6 butir, masing-masing Ya/Tidak/Tidak jelas:
1. Ablasi pada split yang sama dengan resep yang sama.
2. Lebih dari satu seed atau run, atau ada CI.
3. Split per pasien.
4. Hiperparameter dipilih dengan data validasi, bukan data uji.
5. Ada uji eksternal atau lintas senter.
6. Kode tersedia.

Hasil checklist dipakai untuk mengurutkan kandidat, bukan untuk mengeluarkan studi.

## 9. Sintesis

Meta-analisis tidak direncanakan karena tugas, metrik, dan baseline terlalu heterogen. Sintesis naratif terstruktur mengikuti SWiM:
- Per kelompok: tabel keluarga metode × jumlah studi × arah efek terhadap baseline × kualitas bukti (§8).
- Tanpa vote counting sebagai kesimpulan. Arah efek dilaporkan bersama konteksnya.
- **Keluaran utama:** daftar kandidat ablasi per RQ tesis (RQ1 ← G1, RQ2 ← G2, RQ4 ← G3, RQ5 ← G4). Setiap kandidat memuat bukti, kesesuaian dengan RAM-H1200, status di tesis, dan ide modifikasi. Daftar disusun sepanjang mungkin, sesuai permintaan penulis, lalu diberi urutan prioritas.

## 10. Bias pelaporan dan keterbatasan yang diantisipasi

- Paper metode cenderung hanya melaporkan perbaikan (publication bias positif). Akibatnya selisih yang dilaporkan cenderung optimis. Inilah alasan kandidat tetap harus diuji ablasi di RAM-H1200.
- Literatur didominasi OA lutut (OAI/MOST). Keuntungan di KL lutut belum tentu berlaku untuk erosi kecil di sendi tangan.
- Reviewer AI tunggal (lihat §6.6).

## 11. Amandemen

| Versi | Tanggal | Perubahan |
|---|---|---|
| v0.1 | 2026-11-10 | Draf awal |
