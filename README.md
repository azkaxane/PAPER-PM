# Dashboard Interaktif: Analisis Reliabilitas dan Penjadwalan Sustainable Preventive Maintenance

Dashboard ini mendukung paper: **"Pengembangan Dashboard Interaktif Berbasis Python untuk Analisis
Reliabilitas dan Penjadwalan Sustainable Preventive Maintenance Menggunakan Distribusi Weibull:
Studi Kasus Industri Manufaktur"**

## Cara Menjalankan Secara Lokal

```bash
pip install -r requirements.txt
streamlit run app.py
```

Dashboard akan terbuka otomatis di browser pada `http://localhost:8501`.

## Cara Deploy ke Streamlit Community Cloud (gratis)

1. Unggah `app.py` dan `requirements.txt` ke repository GitHub Anda.
2. Buka [share.streamlit.io](https://share.streamlit.io), login dengan akun GitHub.
3. Klik "New app", pilih repository dan file `app.py`.
4. Klik "Deploy" — dashboard akan online dengan URL publik dalam beberapa menit.

## Struktur Metodologi yang Diterapkan

| Komponen | Metode | Dasar Rumus |
|---|---|---|
| Estimasi keandalan mesin | Distribusi Weibull 2-parameter (MLE) | `scipy.stats.weibull_min.fit` |
| Validasi kesesuaian distribusi | Uji Kolmogorov-Smirnov | `scipy.stats.kstest` |
| Interval PM optimal (konvensional) | Age Replacement Policy (Barlow & Hunter) | Minimasi cost rate = [Cf·F(T)+Cp·R(T)] / ∫R(t)dt |
| Interval PM optimal (sustainable) | Weighted-sum multi-objective optimization | Normalisasi min-max biaya & emisi, digabung dengan bobot w |
| Model energi/emisi | Kerangka renewal-reward (konsisten dengan model cost) | [e_PM·R(T)+e_CM·F(T)] / ∫R(t)dt |

## Data yang Dibutuhkan

1. **Data historis Time Between Failures (TBF)** — minimal 20-30 observasi per mesin/komponen kritis.
2. **Biaya PM dan CM** — dari data internal perusahaan/UKM objek penelitian.
3. **Parameter energi** — dari nameplate mesin, tagihan listrik, atau estimasi (lihat catatan di sidebar dashboard).
4. **Faktor emisi grid listrik** — gunakan angka resmi Kementerian ESDM/PLN sesuai wilayah studi kasus.

## Catatan Penting untuk Penulisan Paper

- Model biaya emisi (`e_PM`, `e_CM`, weighted-sum normalization) adalah **adaptasi metodologis yang
  dikembangkan**, bukan rumus baku tunggal dari satu sumber — nyatakan ini eksplisit di bagian metodologi.
- Jika data energi bersifat estimasi (bukan pengukuran langsung), cantumkan sebagai **keterbatasan penelitian**.
- Gunakan fitur "Analisis Sensitivitas" di dashboard untuk menunjukkan kesadaran terhadap ketidakpastian
  parameter — ini memperkuat kredibilitas akademik paper Anda.

## File dalam Proyek Ini

- `app.py` — kode utama dashboard Streamlit
- `requirements.txt` — daftar dependency untuk instalasi/deployment
- `test_logic.py` — skrip pengujian logika perhitungan (opsional, untuk verifikasi ulang)
