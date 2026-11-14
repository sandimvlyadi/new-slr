# SLR-M: Systematic Literature Review Metodologis & Rencana Tesis Grading Ordinal Radiografi Muskuloskeletal (2021–2026)

Repositori ini berisi seluruh protokol, data pencarian, hasil screening, naskah teks penuh, ekstraksi terverifikasi, laporan audit forensik, serta rencana desain ablasi dan validasi eksternal untuk tesis:

> **Pembelajaran Representasi, Penanganan Ketidakseimbangan Ordinal, Kuantifikasi Ketidakpastian, dan Koreksi Derau Label untuk Grading Erosi Sendi Tangan pada Radiografi Artritis Reumatoid dengan Validasi Eksternal Lintas Populasi**  
> *(Dataset Internal: `RAM-H1200` [Sapporo, Jepang] $\to$ Validasi Eksternal: `RheumaVIT` [Moskow, Rusia])*

---

## Struktur Direktori & Dokumen Kunci

| Direktori / Berkas | Deskripsi |
|---|---|
| [`protocol/01_protocol_PRISMA-P.md`](protocol/01_protocol_PRISMA-P.md) | Protokol PRISMA-P 2015 untuk SLR-M (kriteria kelayakan `EM1–EM7`, kelompok metode `G1–G4`, dan checklist kualitas bukti 6 butir `Q1–Q6`). |
| [`search/`](search/) | Query dan hasil ekspor pencarian dari Scopus (`290`), PubMed (`96`), IEEE Xplore (`41` setelah filter), dan *citation searching* OpenAlex (`95` kandidat). |
| [`screening/`](screening/) | Rekaman gabungan pasca-deduplikasi (`341`), keputusan screening judul/abstrak (`69` lolos), kandidat teks penuh, dan hasil screening sitasi (`4` lolos). |
| [`papers/`](papers/) & [`fulltext/`](fulltext/) | 54 berkas PDF/XML teks penuh yang berhasil diambil beserta hasil ekstraksi teksnya (`19` laporan tidak terambil dicatat di [`papers/00_TODO_user_download.md`](papers/00_TODO_user_download.md)). |
| [`verification/`](verification/) | **Audit Verifikasi Independen (13 Nov 2026):**<br>- [`verification/README.md`](verification/README.md): Ringkasan eksekutif hasil verifikasi.<br>- [`verification/04_verification_report.md`](verification/04_verification_report.md): Laporan lengkap verifikasi buta judul/abstrak ($N=87$, $\kappa = 0,8880$, PABAK $= 0,9310$, sensitivitas AI $= 100\%$), audit 17 studi Prioritas A ($204$ sel), audit forensik paper fabrikasi (`M095, M175`) dan modifikasi label uji (`M042, M117, M272`), serta seleksi final **44 studi unik valid**.<br>- [`verification/02_extraction_check.csv`](verification/02_extraction_check.csv) & [`verification/03_fulltext_criteria_verification.csv`](verification/03_fulltext_criteria_verification.csv): Matriks verifikasi per sel dan per studi. |
| [`extraction/`](extraction/) | - [`extraction/01_fulltext_extraction.tsv`](extraction/01_fulltext_extraction.tsv): Tabel ekstraksi awal (54 baris).<br>- [`extraction/02_verified_fulltext_extraction.tsv`](extraction/02_verified_fulltext_extraction.tsv): **Tabel Master Ekstraksi Terverifikasi** yang telah menggabungkan seluruh koreksi faktual dan penilaian kualitas final. |
| [`report/`](report/) | **Laporan Sintesis & Rencana Eksperimen Tesis:**<br>- [`report/02_kandidat_ablasi_fulltext.md`](report/02_kandidat_ablasi_fulltext.md): Daftar kandidat ablasi per RQ pasca-verifikasi teks penuh.<br>- [`report/03_desain_ablasi.md`](report/03_desain_ablasi.md): Desain matriks eksperimen ablasi terkontrol (`RQ1–RQ5`).<br>- [`report/04_urutan_run_per_RQ.md`](report/04_urutan_run_per_RQ.md): Urutan eksekusi *run* dan estimasi jam GPU (`RTX 3060`).<br>- [`report/05_rencana_tesis_dan_validasi_eksternal.md`](report/05_rencana_tesis_dan_validasi_eksternal.md): **Dokumen Induk Rencana Tesis**, opsi judul, 5 protokol eksplorasi validasi eksternal, dan kerangka Bab I–VI. |
| [`scripts/`](scripts/) | Skrip Python untuk pengambilan data, deduplikasi, pemeriksaan Open Access, dan ekstraksi potongan teks. |

---

## Ringkasan Alur PRISMA & Kualitas Bukti (44 Studi Valid)

- **Rekaman di-screen (Judul/Abstrak):** `341` (basis data) + `95` (*citation searching*) = `436`
- **Laporan dicari teks penuhnya:** `73` (`19` tidak diperoleh, `54` dinilai pada teks penuh)
- **Dikeluarkan pada tahap teks penuh:** `10` (`M165, M173` [`EM2`]; `M051, M180, M249` [`EM3`]; `M095, M175` [`EM6/fabrikasi`]; `M191, M199` [`EM7 preprint`]; `M283` [`duplikat M141`])
- **Studi unik valid dalam sintesis:** **`44` studi** (`40` studi bersih + `4` studi dengan peringatan metodologis `M042, M117, M150, M188`)
  - **Q1 (Ablasi terkontrol):** `36 / 44 (81,8%)`
  - **Q2 (Multi-run / CI):** `19 / 44 (43,2%)`
  - **Q3 (Split tingkat pasien):** `13 / 44 (29,5%)`
  - **Q5 (Validasi eksternal lintas senter):** `11 / 44 (25,0%)` — **`0 / 5 (0%)` pada studi RA**
  - **Q6 (Kode/bobot publik):** `8 / 44 (18,2%)` penuh + `2 / 44 (4,5%)` sebagian
