# Dokumen Aturan Pra-Spesifikasi Eksperimen dan Analisis Statistik (*Pre-Specified Experimental & Statistical Analysis Protocol*)

**Kode Dokumen:** `RULES-SLR-THESIS-v1.0`  
**Tanggal Penguncian (*Lock Date*):** 14 November 2026 (Di-commit ke repositori Git sebelum eksekusi *run* GPU pertama `RQ0-01`)  
**Cakupan:** Seluruh eksperimen pelatihan dan evaluasi pada Bab III dan Bab IV (`RQ0`, `RQ1`, `RQ2`, `RQ3`, `RQ4`, dan `RQ5`) pada dataset internal **`RAM-H1200`** (Sapporo, Jepang) dan kohort validasi eksternal **`RheumaVIT`** (Moskow, Rusia).  
**Dasar Metodologis:** Menjawab enam kriteria kualitas bukti (`Q1_abl` s/d `Q6_code`) dan mencegah seluruh bias metodologis yang terungkap pada audit forensik `verification/04_verification_report.md` (kebocoran citra lintas pasien, *oversampling* pra-split, *cherry-picking* ambang di data uji, serta modifikasi label himpunan uji pada eksperimen derau label).

---

## Pasal 1. Prinsip Isolasi Satu Sumbu (*Single-Axis Controlled Ablation*, `Q1_abl = Y`)

1. **Konfigurasi Rujukan Bersama (`RQ0-01`):**
   - Seluruh eksperimen menggunakan satu titik jangkar (*common reference anchor*) yang identik, yaitu **`RQ0-01`**:
     - **Arsitektur Visual (*Backbone*):** `ResNet-34` dengan inisialisasi terawasi `ImageNet-1K` (`A1`).
     - **Kepala Klasifikasi & Fungsi Rugi:** Kepala linear 5-kelas ($y \in \{0, 1, 2, 3, 4+\}$) dengan **`Balanced Softmax / Logit Adjustment` (`B2`, $\tau_{\text{LA}} = 1,0$)**.
     - **Resep Augmentasi:** Resep augmentasi `base` dari eksperimen pendahuluan `E8` (transformasi afinitas/geometris ringan dan *intensity jitter* terukur) dengan pembangkit bilangan acak (*random number generator*) yang dikunci per *seed*.
2. **Aturan Penguncian Komponen per Pertanyaan Penelitian:**
   - **Keluarga `RQ1` (Sumbu Pre-training, SSL, & Strategi Transfer):** Hanya bobot inisialisasi *encoder*, objektif pra-pelatihan/kontras (`DINO`, `MAE`, `ChronoCon`, `RSNA Bone Age`, `SupCon-Ord`, `SS-FewSOME`, `SKELEX`, `TCN+MMD`), atau jadwal pembekuan lapisan (`Frozen Probe`, `LP-FT`, `LLRD`, `Gradual Unfreezing`) yang diubah. Kepala klasifikasi dan fungsi rugi hilir **dikunci pada `B2` (`Balanced Softmax`)**.
   - **Keluarga `RQ2` (Sumbu Ketidakseimbangan Kelas & Ordinalitas):** Hanya strategi *resampling* (`OS-only`, `Undersampling Grade 0`), bobot fungsi rugi (`CW-only`, `B2`, `Focal Loss`, `FOL`), atau struktur kepala ordinal (`Ordinal BCE B0`, `CORN B4`, `Decoupled cRT B6`, `Cumulative-Link ORM`, `Distance-Penalty / Ordinal Soft Labels`) yang diubah. *Backbone* visual **dikunci pada `ImageNet-1K ResNet-34`**.
   - **Eksperimen Interaksi `RQ1 × RQ2` (`RQ2-13`):** Hanya **2 (dua) konfigurasi kepala/loss terbaik** dari `RQ2` (dipilih murni berdasarkan metrik validasi $\mathcal{D}_{\text{val}}$, lihat Pasal 3) yang diuji ulang di atas **1 (satu) encoder terbaik** dari `RQ1`.
   - **Keluarga `RQ4` (Sumbu Kalibrasi, Conformal Prediction, & UQ):** Seluruh metode pasca-latih (`RQ4-P1` s/d `RQ4-P4`) dijalankan langsung di atas logit yang disimpan dari `RQ0-01` dan arm terbaik `RQ2` tanpa mengubah bobot jaringan. Arm pelatihan aktif (`RQ4-01 MC Dropout` dan `RQ4-02 EDL`) menggunakan backbone `ImageNet-1K ResNet-34` dengan hiperparameter optimisasi yang identik dengan `RQ0-01`.
   - **Keluarga `RQ5` (Sumbu Derau Label, Reliabilitas Longitudinal, & Semi-Supervised):** Seluruh intervensi (`CleanLab Relabel/Prune/Down-weight`, `Label Smoothing / Soft Labels`, `Longitudinal Visit-Reliability Loss`, dan `Mean-Teacher`) menggunakan arsitektur dan kepala yang **dikunci pada `RQ0-01`** (`ResNet-34 + B2`).

---

## Pasal 2. Partisi Data Tingkat Pasien (`Q3_patient = Y`) dan Protokol Multi-Run (`Q2_runs = Y`)

1. **Partisi *Strict Patient-Level* Bebas-Bocoran:**
   - Himpunan pasien pada `RAM-H1200` dipartisi secara saling lepas menjadi tiga himpunan pasien:
     $$\mathcal{P}_{\text{train}} \cap \mathcal{P}_{\text{val}} = \emptyset, \quad \mathcal{P}_{\text{train}} \cap \mathcal{P}_{\text{test}} = \emptyset, \quad \mathcal{P}_{\text{val}} \cap \mathcal{P}_{\text{test}} = \emptyset$$
   - Seluruh *Region of Interest* (ROI) sendi dari kedua tangan (kiri dan kanan) serta **seluruh kunjungan longitudinal** ($t = 1, \dots, T_p$) milik seorang pasien $p$ wajib berada di dalam partisi yang sama.
2. **Larangan Mutlak Operasi Pra-Split:**
   - Dilarang keras melakukan *oversampling*, SMOTE, rekonstruksi citra sintetis, normalisasi statistik berbasis seluruh dataset, ataupun penyaringan *outlier* sebelum pemisahan $\mathcal{D}_{\text{train}}, \mathcal{D}_{\text{val}}, \mathcal{D}_{\text{test}}$ (mencegah cacat metodologis yang ditemukan pada `M042, M117, M160, M162, M185, M272, M275, C037`).
   - Seluruh statistik normalisasi intensitas piksel ($\mu_{\text{img}}, \sigma_{\text{img}}$), frekuensi kelas prior $\pi_y = N_y / N_{\text{train}}$, indeks tetangga `FAISS`, dan matriks *confident joint* `CleanLab` dihitung **hanya dari $\mathcal{D}_{\text{train}}$**.
3. **Struktur Enam *Run* Wajib per Arm Eksperimen (`1 Arm = 6 Run`):**
   Setiap arm eksperimen pada `RQ0, RQ1, RQ2, RQ4, RQ5` wajib dieksekusi sebanyak **6 (enam) run independen** yang mencakup dua rezim ketersediaan label:
   - **Rezim Label Penuh (`f100`, 3 *Seed*):** Menggunakan 100% pasien latih ($\mathcal{P}_{\text{train}}$) dengan tiga *random seed* tetap:
     $$s \in \{42, 1337, 2026\}$$
     yang mengendalikan inisialisasi lapisan baru, urutan pengacakan *mini-batch*, dan aliran augmentasi daring (*online augmentation*).
   - **Rezim Efisiensi Label (`f010`, 3 *Stratified Patient Draws*):** Menggunakan tiga tarikan (*draws*) acak terstratifikasi sebesar **10% pasien latih**:
     $$\mathcal{P}_{\text{train}}^{(d)} \subset \mathcal{P}_{\text{train}}, \quad d \in \{0, 1, 2\}, \quad \text{pada } s = 42$$
     Stratifikasi dilakukan pada tingkat pasien berdasarkan skor erosi maksimum pasien agar proporsi kelas minoritas (`Grade 2–4+`) terjaga di setiap *draw*. Daftar ID pasien untuk $d \in \{0, 1, 2\}$ dikunci dalam berkas statis sebelum eksperimen dimulai dan dipakai secara identik di seluruh RQ.
   - **Aturan Data Tak Berlabel pada `RQ5-06` (*Semi-Supervised Mean-Teacher*):** Pada rezim `f010` (10% label) dan `f025` (25% label), himpunan data tak berlabel $\mathcal{D}_U$ diambil **hanya dari sisa 90% (atau 75%) pasien di dalam $\mathcal{P}_{\text{train}}$** ($\mathcal{P}_{\text{train}} \setminus \mathcal{P}_{\text{train}}^{(d)}$). Pasien di $\mathcal{P}_{\text{val}}$, $\mathcal{P}_{\text{test}}$, maupun `RheumaVIT` dilarang dimasukkan ke dalam $\mathcal{D}_U$.
4. **Konvensi Penamaan *Run* dan Artefak Keluaran:**
   - Format baku direktori *run*:
     ```text
     runs/rq<n>/rq<n>-<nomor>_<encoder>_<head>_f<100|010>_d<0|1|2>_s<42|1337|2026>/
     ```
   - Setiap *run* wajib menyimpan secara otomatis:
     1. `config.yaml` beserta *commit hash* Git dan `sha256` bobot inisialisasi;
     2. `metrics_val_history.tsv` (log metrik validasi setiap epoch);
     3. `logits_val.npz`, `logits_test_ramh1200.npz`, dan `logits_ext_rheumavit.npz` (logit mentah $\mathbf{z} \in \mathbb{R}^{N \times K}$, fitur embedding $\mathbf{h} \in \mathbb{R}^{N \times D}$, ID pasien, ID sendi, dan label asli);
     4. `summary_metrics.json`.

---

## Pasal 3. Aturan Seleksi Validasi Murni (`Q4_valsel = Y`) dan Partisi Kalibrasi Konformal

1. **Pemilihan *Checkpoint* Terbaik (*Early Stopping / Model Selection*):**
   - Selama pelatihan $E$ epoch, *checkpoint* model yang disimpan untuk evaluasi akhir dipilih **murni berdasarkan nilai `BACC4` tertinggi pada himpunan validasi $\mathcal{D}_{\text{val}}$** (dengan `QWK` validasi sebagai pemecah seri / *tie-breaker* apabila terdapat nilai `BACC4` yang identik hingga $10^{-4}$).
   - Metrik pada himpunan uji $\mathcal{D}_{\text{test}}$ (`RAM-H1200`) dan $\mathcal{D}_{\text{ext}}$ (`RheumaVIT`) tidak pernah dihitung selama iterasi pemilihan hiperparameter dan hanya dieksekusi satu kali setelah *checkpoint* terkunci.
2. **Penyetelan Parameter Pasca-Latih (*Post-Hoc Tuning*) di Validasi:**
   - **Ambang Keputusan Kelas (`RQ2-P`):** Vektor penggeser ambang/prior $\boldsymbol{\delta}^* = (\delta_1^*, \dots, \delta_K^*)$ dicari melalui optimisasi *coordinate ascent* / Nelder-Mead yang memaksimalkan `BACC4` **hanya pada $\mathcal{D}_{\text{val}}$**.
   - **Ukuran Tetangga `FAISS k-NN` (`RQ2-P`):** Nilai $k \in \{5, 10, 15, 25, 50\}$ dan bobot interpolasi $\lambda_{\text{knn}} \in \{0.1, 0.25, 0.5\}$ dipilih berdasarkan `BACC4` pada $\mathcal{D}_{\text{val}}$ (dengan indeks `FAISS` dibangun hanya dari embedding $\mathcal{D}_{\text{train}}$).
   - **Kalibrasi Probabilitas (`RQ4-P1`, `RQ4-P2`):** Suhu $T > 0$ pada *Temperature Scaling* serta matriks $\mathbf{W} \in \mathbb{R}^{K \times K}$ dan vektor bias $\mathbf{b} \in \mathbb{R}^K$ (dengan regularisasi off-diagonal L2 $\lambda_{\text{ODIR}}$) pada *Dirichlet Calibration* dioptimalkan dengan meminimalkan *Negative Log-Likelihood* (`NLL`) **hanya pada $\mathcal{D}_{\text{val-tune}}$**.
3. **Pemisahan Himpunan Validasi dan Himpunan Kalibrasi Konformal ($\mathcal{D}_{\text{val-tune}}$ vs. $\mathcal{D}_{\text{cal}}$):**
   - Untuk menjamin validitas teoretis cakupan *finite-sample* $P(Y_{\text{test}} \in \mathcal{C}_{1-\alpha}(X_{\text{test}})) \ge 1 - \alpha$ pada *Split Conformal Prediction* (`RQ4-P3`, mengacu pada koreksi `M110`), pasien di dalam $\mathcal{P}_{\text{val}}$ dibagi dua secara terstratifikasi tingkat pasien ($50\% : 50\%$):
     - $\mathcal{D}_{\text{val-tune}}$: digunakan untuk *early stopping* dan pengepasan parameter kalibrasi suhu $T$.
     - $\mathcal{D}_{\text{cal}}$: digunakan **khusus** untuk menghitung skor non-konformitas $s(x_i, y_i)$ dan kuantil empiris $\hat{q}_{1-\alpha}$ (baik marginal pada `APS`/`RAPS` maupun bersyarat per kelas $\hat{q}_{1-\alpha}^{(c)}$ pada `Mondrian Conformal`).

---

## Pasal 4. Hukum Besi Integritas Label pada Eksperimen Derau Label (`RQ5` *Golden Rule*)

Berdasarkan temuan audit forensik Bab II bahwa seluruh studi *CleanLab / relabeling* di literatur (`M042, M117, M272`) mencemari himpunan uji dengan merelabeli 3,5%–22,0% sampel uji, tesis ini menetapkan **empat aturan mengikat** untuk `RQ5`:

1. **Estimasi Probabilitas *Out-of-Fold* (`RQ5-00`) Murni di Dalam $\mathcal{D}_{\text{train}}$:**
   - Pasien di dalam $\mathcal{P}_{\text{train}}$ dibagi menjadi 5 lipatan tingkat pasien ($\mathcal{P}_{\text{train}}^{(1)}, \dots, \mathcal{P}_{\text{train}}^{(5)}$).
   - Lima model rujukan (`ResNet-34 + B2`) dilatih masing-masing pada 4 lipatan dan memprediksi probabilitas *out-of-fold* $\hat{\mathbf{P}}_{\text{oof}} \in [0, 1]^{N_{\text{train}} \times K}$ pada lipatan ke-5.
   - Tidak ada satu pun sampel dari $\mathcal{D}_{\text{val}}$, $\mathcal{D}_{\text{test}}$, atau `RheumaVIT` yang dimasukkan ke dalam proses estimasi *confident joint* $\mathbf{C}_{\tilde{y}, y^*}$.
2. **Imutabilitas Mutlak Label Validasi dan Uji:**
   - Intervensi `RQ5-01` (`CleanLab Relabel`), `RQ5-02` (`CleanLab Prune`), `RQ5-03` (`CleanLab Down-weight`), `RQ5-04` (`Label Smoothing / Reader-Agreement Soft Labels`), dan `RQ5-05` (`Longitudinal Visit-Reliability Reweighting`) **hanya boleh mengubah label atau bobot sampel di $\mathcal{D}_{\text{train}}$**.
   - Berkas label $\mathcal{D}_{\text{val}}$, $\mathcal{D}_{\text{test}}$ (`RAM-H1200`), dan $\mathcal{D}_{\text{ext}}$ (`RheumaVIT`) dikunci (*read-only*) dan diverifikasi dengan *checksum* `SHA-256` sebelum serta sesudah evaluasi.
3. **Evaluasi Ganda terhadap Label Konsensus dan Multi-Radiolog (`RQ5-E` & `RQ3-04`):**
   - Untuk menguji apakah mitigasi derau label benar-benar meningkatkan generalisasi klinis tanpa memanipulasi data uji, seluruh model `RQ5` dievaluasi pada:
     1. Label uji asli `RAM-H1200` ($\mathcal{D}_{\text{test}}$);
     2. Label konsensus pakar `RheumaVIT` ($\mathcal{D}_{\text{ext}}$);
     3. Masing-masing dari **3 radiolog independen** ($R_1, R_2, R_3$) pada `RheumaVIT`;
     4. Distribusi probabilitas empiris 3 radiolog $\mathbf{p}_{\text{human}}(x) \in \Delta^{K-1}$ pada `RheumaVIT` menggunakan *Kullback-Leibler (KL) Divergence*, *Total Variation Distance*, dan *Multi-Reader Soft Brier Score*.

---

## Pasal 5. Definisi dan Hirarki Metrik Evaluasi Pra-Spesifikasi

Mengingat sifat distribusi *long-tailed* ekstrem (di mana `Grade 0` mencakup $>75\%$ sendi) dan struktur keparahan berurut SvdH, *Overall Accuracy* **dilarang** digunakan sebagai metrik penentu keberhasilan. Hirarki metrik ditetapkan sebagai berikut:

| Kategori | Metrik | Simbol / Rentang | Arah Optimal | Peran dalam Pengambilan Keputusan |
|---|---|---|---|---|
| **Ko-Primer 1 (Keseimbangan Sensitivitas)** | *Macro-Averaged Balanced Recall* lintas strata keparahan erosi | $\text{BACC4} \in [0, 1]$ | $\uparrow$ Tinggi | Metrik utama pemilihan *checkpoint*, penyetelan ambang, dan uji hipotesis keseimbangan kelas (`RQ1, RQ2, RQ3, RQ5`) |
| **Ko-Primer 2 (Kesesuaian Urutan Ordinal)** | *Quadratic Weighted Cohen's Kappa* | $\text{QWK} \in [-1, 1]$ | $\uparrow$ Tinggi | Metrik utama ketepatan urutan keparahan klinis (`RQ1, RQ2, RQ3, RQ5`) dan perbandingan terhadap radiolog |
| **Sekunder Klasifikasi & Ordinal** | *Mean Absolute Error* antar-grade | $\text{MAE} \in [0, 4]$ | $\downarrow$ Rendah | Mengukur rata-rata penyimpangan absolut dalam satuan grade SvdH |
| **Sekunder Klasifikasi & Ordinal** | *Macro-F1* & *Recall / F1 per Grade* (`Grade 0, 1, 2, 3, 4+`) | $F_1^{\text{macro}}, \text{Rec}_c \in [0, 1]$ | $\uparrow$ Tinggi | Mendiagnosis sensitivitas khusus pada erosi dini (`Grade 1`) dan erosi lanjut (`Grade >= 2`) |
| **Sekunder Klasifikasi & Ordinal** | *Within-$\pm 1$ Grade Accuracy* | $\text{Acc}_{\pm 1} \in [0, 1]$ | $\uparrow$ Tinggi | Proporsi prediksi yang berada dalam toleransi $\le 1$ grade dari label acuan (`C007, C015`) |
| **Primer Kalibrasi (`RQ4`)** | *Expected Calibration Error* (15 *adaptive bins*) & *Classwise-ECE* | $\text{ECE}, \text{cwECE} \in [0, 1]$ | $\downarrow$ Rendah | Metrik utama kualitas kalibrasi probabilitas pasca-latih (`RQ4-P1, RQ4-P2`) |
| **Sekunder Kalibrasi (`RQ4`)** | *Negative Log-Likelihood* & *Multi-class Brier Score* | $\text{NLL}, \text{Brier}$ | $\downarrow$ Rendah | *Proper scoring rules* untuk mengevaluasi ketajaman dan kalibrasi distribusi probabilitas |
| **Primer Conformal (`RQ4`)** | *Empirical Marginal Coverage* & *Worst-Stratum (Grade $\ge 2$) Conditional Coverage* pada $\alpha = 0,10$ | $\text{Cov}_{\text{marg}}, \text{Cov}_{\ge 2} \in [0, 1]$ | Target $\ge 0,90$ | Menilai pemenuhan jaminan cakupan $90\%$ baik secara keseluruhan maupun pada kelas erosi minoritas |
| **Efisiensi Conformal (`RQ4`)** | Rata-rata Ukuran Himpunan Prediksi (*Average Prediction Set Size*) & Proporsi Himpunan *Singleton* | $\overline{|\mathcal{C}(X)|} \in [1, 5]$ | $\downarrow$ Kecil (asalkan $\text{Cov} \ge 0,90$) | Mengukur ketajaman klinis himpunan prediksi konformal (`APS`, `RAPS`, `Mondrian`) |
| **Primer UQ & Rujukan Selektif (`RQ4`)** | `AUROC (Uncertainty -> Error)` & *Area Under Risk-Coverage Curve* (`AURC`) | $\text{AUROC}_{\text{err}} \uparrow$, $\text{AURC} \downarrow$ | $\uparrow / \downarrow$ | Mengukur kemampuan skor ketidakpastian (`Deep Ensemble`, `MC Dropout`, `EDL`) memisahkan prediksi salah dan menekan risiko saat $10\%–30\%$ kasus dirujuk |
| **Primer Validasi Eksternal (`RQ3`)** | *External Performance* & *Cross-Population Domain Drop* | $\Delta_{\text{ext}} m = m_{\text{RheumaVIT}} - m_{\text{RAM-H1200}}$ | $|\Delta_{\text{ext}}|$ Kecil | Mengukur ketahanan lintas populasi Jepang $\to$ Rusia dan pemulihan pasca-adaptasi |

---

## Pasal 6. Protokol Pengujian Statistik, *Patient-Cluster Bootstrap*, dan Koreksi *Holm-Bonferroni*

1. **Agregasi Multi-Run dan Estimasi Varians Antar-Seed/Draw:**
   - Untuk setiap arm dan setiap rezim label (`f100` dengan 3 *seed*; `f010` dengan 3 *patient draw*), dilaporkan nilai **rata-rata $\pm$ simpangan baku ($\mu \pm \text{SD}$)** lintas 3 run.
2. **Interval Kepercayaan 95% via *Non-Parametric Patient-Level Cluster Bootstrap* ($B = 5.000$):**
   - Karena satu pasien memiliki hingga 32–42 sendi yang saling berkorelasi secara anatomis dan sistemik, *bootstrap* tingkat sendi biasa melanggar asumsi independensi dan menghasilkan interval kepercayaan yang terlalu sempit secara semu.
   - Oleh karena itu, seluruh Interval Kepercayaan 95% (95% CI) dihitung menggunakan **Patient-Level Cluster Bootstrap** dengan $B = 5.000$ replikasi (mengikuti standar tertinggi di SLR-M, yaitu `M015` dan `M157`):
     - Pada setiap replikasi $b \in \{1, \dots, 5.000\}$, diambil sampel sebanyak $|\mathcal{P}_{\text{test}}|$ pasien **dengan pengembalian (*with replacement*)** dari himpunan pasien uji $\mathcal{P}_{\text{test}}$ (dengan `rng_seed = 20261114`).
     - Seluruh ROI sendi milik pasien yang terpilih dimasukkan ke dalam sampel *bootstrap* ke-$b$, lalu metrik agregat $m^{(b)}$ (atau rata-rata metrik lintas 3 seed $\bar{m}^{(b)}$) dihitung.
     - Batas bawah dan batas atas 95% CI diambil dari persentil ke-$2,5$ dan ke-$97,5$ dari distribusi empiris $\{m^{(1)}, \dots, m^{(5.000)}\}$.
3. **Uji Hipotesis Berpasangan (*Paired Cluster Bootstrap Test*) terhadap Rujukan `RQ0-01`:**
   - Untuk membandingkan suatu arm perlakuan $A$ terhadap arm rujukan $R$ (`RQ0-01`) pada partisi dan *seed/draw* yang sama, selisih berpasangan pada replikasi *cluster bootstrap* ke-$b$ didefinisikan sebagai:
     $$\Delta^{(b)}(A, R) = \bar{m}_A^{(b)} - \bar{m}_R^{(b)}$$
   - Nilai-$p$ dua sisi (*two-sided paired cluster-bootstrap $p$-value*) dihitung dari proporsi replikasi *bootstrap* yang melintasi nol (dengan koreksi sampel hingga +1):
     $$p_{\text{raw}} = 2 \min\left(\frac{1 + \sum_{b=1}^{B} \mathbb{I}[\Delta^{(b)} \le 0]}{B + 1}, \; \frac{1 + \sum_{b=1}^{B} \mathbb{I}[\Delta^{(b)} \ge 0]}{B + 1}\right)$$
4. **Pengendalian *Family-Wise Error Rate* (FWER) dengan Prosedur *Step-Down Holm-Bonferroni*:**
   - Untuk mencegah inflasi positif palsu (*Type-I error*) akibat pengujian banyak arm terhadap rujukan `RQ0-01`, koreksi **Holm-Bonferroni** ($\alpha_{\text{FWER}} = 0,05$) diterapkan secara terpisah di dalam setiap keluarga pertanyaan penelitian (`RQ1`, `RQ2`, `RQ3`, `RQ4`, `RQ5`) dan setiap rezim label (`f100` dan `f010`):
     - Urutkan $M$ hipotesis dalam satu keluarga berdasarkan nilai-$p$ mentah dari terkecil ke terbesar: $p_{(1)} \le p_{(2)} \le \dots \le p_{(M)}$.
     - Nilai-$p$ terkoreksi Holm (*Holm-adjusted $p$-value*) untuk urutan ke-$j$ adalah:
       $$p_{(j)}^{\text{Holm}} = \min\left(1, \; \max_{1 \le i \le j} (M - i + 1) \, p_{(i)}\right)$$
     - Suatu arm dinyatakan **unggul secara signifikan secara statistik** hanya jika $\Delta > 0$ dan $p^{\text{Holm}} < 0,05$.
5. **Aturan Pra-Spesifikasi Pemilihan Komponen untuk Eksperimen Lanjutan (*Go / No-Go Selection Rules*):**
   - **Pemilihan Encoder Terbaik `RQ1` (untuk digunakan di `RQ1-10`, `RQ2-13`, dan `RQ3-02/03`):** Dipilih encoder dengan rata-rata skor gabungan validasi tertinggi pada $\mathcal{D}_{\text{val}}$:
     $$S_{\text{val}}(\text{arm}) = 0,5 \cdot \overline{\text{BACC4}}_{\text{val}}^{\text{f100}} + 0,5 \cdot \overline{\text{BACC4}}_{\text{val}}^{\text{f010}}$$
     (dengan $\overline{\text{QWK}}_{\text{val}}$ sebagai pemecah seri).
   - **Pemilihan 2 Kepala/Loss Terbaik `RQ2` (untuk diuji pada interaksi `RQ2-13`):** Dipilih 2 arm dari `RQ2-01..11` yang mencapai peningkatan $\Delta S_{\text{val}} > 0$ tertinggi di atas `RQ0-01` tanpa menurunkan $\overline{\text{QWK}}_{\text{val}}$ lebih dari $0,010$.
   - **Pelaporan Hasil Negatif (*Null/Negative Results Reporting Mandate*):** Apabila suatu kandidat metode populer dari SLR-M (misalnya `CleanLab Relabel` `RQ5-01`, `Class Weight` murni `RQ2-04`, atau *frozen linear probe* `RQ1-10a`) menghasilkan $\Delta \le 0$ atau $p^{\text{Holm}} \ge 0,05$, hasil tersebut **wajib dilaporkan secara utuh** di tabel utama Bab IV sebagai bukti empiris pembantah (*negative/refutational evidence*) dan tidak boleh disembunyikan dari naskah tesis.
