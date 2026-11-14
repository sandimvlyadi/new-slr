# Verifikasi manusia SLR-M

Semua screening, ekstraksi, dan penilaian kualitas SLR-M dikerjakan satu reviewer AI. File ini untuk verifikasi oleh minimal satu reviewer manusia (protokol §6.6).

## 1. Screening (`01_screening_form.csv`)
- Isinya sampel acak 20% keputusan judul/abstrak dari database (68 rekaman, seed 20261110) ditambah 19 rekaman dari citation searching (4 inklusi + 15 eksklusi acak).
- Formulir ini **tidak** memuat keputusan AI. Isi kolom `reviewer_decision` dan `reviewer_reason` tanpa melihat `01_screening_key_AI.csv`.
- Kriteria: protokol §3. Kode eksklusi EM1–EM7 (§6). Inklusi diberi tag G1–G4.
- Setelah selesai, bandingkan dengan file kunci, lalu catat kesepakatan (kappa) dan ketidaksepakatan.

## 2. Ekstraksi (`02_extraction_check.csv`)
- 17 studi yang menopang kandidat prioritas A × 12 field.
- Cek nilai AI terhadap PDF di `papers/`. Kosongkan `reviewer_value` bila setuju. Bila tidak setuju, isi nilai yang benar beserta halaman atau tabelnya.

## 3. Yang paling perlu dicek
- Klaim yang dianggap tidak masuk akal: M095, M150, M175, M188.
- Label uji yang ikut diubah: M042, M272.
- 19 laporan yang tidak diperoleh (`papers/00_TODO_user_download.md`). Kalau ada yang bisa didapat, tambahkan ke `papers/`.

---

## 4. Ringkasan Hasil Verifikasi (Selesai — 13 November 2026)

Seluruh tahapan verifikasi protokol §6.6 telah diselesaikan dan didokumentasikan dalam berkas berikut di folder `verification/`:

1. **`01_screening_form.csv` & `01_screening_disagreements.csv` (Verifikasi Buta Screening Judul/Abstrak, $N = 87$):**
   - **Persentase Kesepakatan (*Overall Agreement*):** **96.55% (84/87)** — Basis data: `95.59% (65/68)`, *Citation searching*: `100.00% (19/19)`.
   - **Cohen's Kappa ($\kappa$):** **0.8880** (*almost perfect agreement*; Basis data $\kappa = 0.8532$, *Citation searching* $\kappa = 1.0000$).
   - **PABAK:** **0.9310**.
   - **Matriks Konfusi:** Both Include = `15`, Both Exclude = `69`, AI Include / Reviewer Exclude = `3` (`M173`, `M199`, `M222`), AI Exclude / Reviewer Include = `0` (**Sensitivitas AI = 100.0%**, nol *false negative*).
   - **Rekonsiliasi Full-Text atas 3 Ketidaksepakatan:** `M173` terkonfirmasi **Exclude (`EM2`)** (murni sintesis citra tanpa classifier grading); `M199` terkonfirmasi **Exclude (`EM7`)** (preprint arXiv anonim belum peer-review); `M222` terkonfirmasi **Include (`G2`)** (tujuan G2 berupa *downsampling*, `WeightedRandomSampler`, dan *class-weighted CE* tersembunyi di §2.5 & Tabel 2 teks penuh).

2. **`02_extraction_check.csv` (Verifikasi 17 Studi Prioritas A × 12 Field = 204 Sel terhadap PDF/XML Penuh):**
   - **Setuju Langsung (`reviewer_value` dikosongkan):** **181 / 204 sel (88.73%)**; akurasi faktual AI pada sel yang sudah diekstraksi (di luar 14 sel yang sengaja ditandai `U`/`NR in snippets`/`numbers not extracted`) mencapai **181 / 190 (95.26%)**.
   - **Pelengkapan Nilai `U` / `numbers not extracted` dari PDF Penuh:** **14 sel (6.86%)** (`M041 Q3_patient -> Y`, `M061 Q3_patient -> Y & angka BA Tabel 2`, `M071 Q3_patient -> N pada CV`, `M089 angka lengkap 21 jaringan Tabel 1–2`, `M185` [8 field], `C093 angka perbandingan sampling Tabel VI & Fig. 7–8`).
   - **Koreksi Faktual (`AI_value` keliru dibanding PDF penuh):** **9 sel (4.41%)**:
     - `M110 (dataset)`: ukuran *CP calibration split* adalah **1.000** citra (angka `827` di Tabel II adalah waktu inferensi dalam detik).
     - `M132 (key_comparison)`: pada ResNet-18-CE, angka `0.678` adalah **MSE**, sedangkan **Kappa adalah `0.774 ± 0.014`**.
     - `M041 (key_comparison)`: `FL-only` sebenarnya **kuat pada KL3** (recall KL3 tertinggi `0.510`, F1 `0.427` vs `0.280`), tetapi lemah pada KL1 dan KL4.
     - `M061 (runs_ci)`: melaporkan **mean ± SE** (*standard error*), bukan SD.
     - `M071 (key_comparison)`: selisih KL2 vs VGG19 adalah `+19.2` (bukan `+19.4`), dan penambahan *class weighting* di Tabel 4 justru menurunkan Macro-F1 sebesar `-5.0%` (`68.60% -> 63.6%`) sebelum dipulihkan oleh KD/TTA/ensemble.
     - `M130 (external & Q5_ext)`: dikoreksi dari `Y` ke **`N`** karena kohort uji 52 pasien berasal dari rumah sakit yang sama (Osaka University Hospital, §4.1 hlm. 8 & §6 hlm. 18).
     - `C093 (code & Q6_code)`: dikoreksi dari `NR`/`N` ke **`Y`** (tautan repositori Google Drive tersedia di hlm. 204).

3. **`03_fulltext_criteria_verification.csv` & `04_verification_report.md` (Audit Kasus Khusus & Verifikasi Kriteria Full-Text Seluruh 54 Paper di `papers/`):**
   - **Klaim Tidak Masuk Akal / Fabrikasi (`M095, M150, M175, M188`):** `M095` terbukti memuat placeholder template LLM (`[X]`, `[ F ]`, `[ TMI ]` di Tabel 11 hlm. 883) dan gambar mockup sintetis (§6.9 hlm. 882); `M175` tidak memiliki tabel/hasil eksperimen di §V (ditulis dalam kalimat masa depan) namun mengklaim akurasi `98.97%` di §VI; `M188` memiliki kontradiksi internal antara Tabel 5 (`AP 98.6%` dari kurva *training*) dan Fig. 10b matriks konfusi uji (`K2 = 48%, K3 = 65%, K4 = 75%`); `M150` (`98%` akurasi KL 5-kelas) tidak memiliki ablasi CORN vs CE (`Q1_abl = N`) dan mengindikasikan *leakage*.
   - **Modifikasi Label Uji (`M042, M272, M117`):** `M042` menggabungkan seluruh 8.260 citra (`train+val+test`) sebelum relabeling 286 citra oleh VGG19 Cascade-I; `M272` mengubah `22.0%` (`364/1.656`) label uji menggunakan Cleanlab (pada label uji asli yang tidak disentuh, peningkatan akurasi **tidak signifikan**, McNemar $p = 0.076$, §4.3 hlm. 15); `M117` membersihkan outlier dan merelabeli lintas fold CV.
   - **Status `papers/00_TODO_user_download.md`:** Dari 40 rekaman di daftar awal, **21 telah berhasil diunduh** ke `papers/` dan **19 masih belum diperoleh**.
   - **Rekomendasi Seleksi Full-Text Seluruh 54 Paper:** **44 studi unik dipertahankan** (`40` studi bersih + `4` studi dengan catatan peringatan metodologis `M042, M117, M150, M188`) dan **10 berkas dikeluarkan pada tahap full-text** (`M165, M173` [`EM2`]; `M051, M180, M249` [`EM3`]; `M095, M175` [`EM6/fabrikasi`]; `M191, M199` [`EM7 preprint`]; serta `M283` [`duplikat M141`]).
