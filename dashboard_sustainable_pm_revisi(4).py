"""
Dashboard Interaktif: Analisis Reliabilitas dan Penjadwalan
Sustainable Preventive Maintenance Berbasis Distribusi Weibull
Tema Warna: Cosmic Sunset

CATATAN REVISI (mengacu pada catatan reviewer):
1.  Klaim validasi KS-test diperlunak (tidak lagi "data valid dan cocok").
2.  Bug tanda delta biaya tahunan (panah hijau menyesatkan) diperbaiki.
3.  Label sumber data (contoh/sintetis vs data asli) dibuat eksplisit.
4.  Ditambahkan baseline pembanding: jadwal aktual perusahaan & run-to-failure.
5.  Ditambahkan metrik R(T), F(T), ekspektasi kegagalan/tahun, downtime,
    dan availability per kebijakan.
6.  Ditambahkan penjadwalan berbasis target keandalan (reliability-constrained).
7.  Grafik "Skor Gabungan" diganti kurva trade-off biaya vs emisi (Pareto).
8.  Ditambahkan tornado diagram sensitivitas terhadap Cf, Cp, beta, eta.
9.  Batas (boundary) perhitungan emisi dinyatakan eksplisit sebagai keterbatasan,
    dengan kolom sumber referensi faktor emisi & harga karbon.
10. Analogi "batang pohon" dihapus karena berpotensi menyesatkan pada skala kecil.
11. Ditambahkan tab "Kekokohan Akademik": uji tren Laplace, perbandingan
    distribusi (AIC/BIC), Anderson-Darling, peringatan n kecil / beta<=1,
    dan bootstrap CI untuk beta, eta, T_optimal (opsional, dipicu tombol).
12. Ekspor PDF dibuat lebih aman terhadap kegagalan Kaleido (pengecekan di awal).
13. Anotasi garis vertikal yang bertumpuk pada grafik biaya diperbaiki posisinya.

KETERBATASAN YANG SECARA SADAR TIDAK DISELESAIKAN DI SINI (bukan bug, tapi
batas lingkup model) dan dijelaskan secara eksplisit di dashboard:
- Batas emisi hanya mencakup konsumsi listrik saat PM/CM. Emisi dari scrap
  material, energi rework, idle saat downtime, dan jejak karbon suku cadang
  belum dimodelkan karena membutuhkan data tambahan yang belum tersedia.
- Model mengasumsikan PM bersifat "as good as new" (perfect PM / block
  replacement policy). Model PM tidak sempurna (mis. Kijima Type I/II)
  memerlukan data virtual age tambahan sehingga tidak diimplementasikan,
  namun keterbatasan ini dinyatakan eksplisit di tab Kekokohan Akademik.
"""

import io
import datetime
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from scipy.special import gamma as gamma_func
from scipy.optimize import minimize_scalar
from scipy.integrate import quad

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                 TableStyle, Image as RLImage, PageBreak)
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY

try:
    import kaleido  # noqa: F401
    KALEIDO_OK = True
except ImportError:
    KALEIDO_OK = False

# =====================================================================
# 1. KONFIGURASI HALAMAN & TAMPILAN KUSTOM (CSS COSMIC SUNSET)
# =====================================================================
st.set_page_config(
    page_title="Dashboard Sustainable PM",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp { background-color: #1A1C29; color: #9CA3AF; }
    .main-header { font-size: 2.2rem; font-weight: 700; color: #FFA3E5; margin-bottom: 0; }
    .sub-header { font-size: 1.1rem; color: #A6B1F7; margin-top: 5px; margin-bottom: 20px; }
    .section-title { font-size: 1.3rem; font-weight: 600; color: #F48A42; border-left: 4px solid #EC3D6D; padding-left: 10px; margin-top: 1.5rem; margin-bottom: 1rem; }
    .info-box { background-color: #292965; border-left: 4px solid #A6B1F7; padding: 15px; border-radius: 6px; font-size: 0.95rem; margin-bottom: 15px; color: #FFFFFF; }
    .warning-box { background-color: #4B5162; border-left: 4px solid #F48A42; padding: 15px; border-radius: 6px; font-size: 0.95rem; margin-bottom: 15px; color: #FFFFFF; }
    .danger-box { background-color: #5C2233; border-left: 4px solid #EC3D6D; padding: 15px; border-radius: 6px; font-size: 0.95rem; margin-bottom: 15px; color: #FFFFFF; }
    div[data-testid="stMetricValue"] { font-size: 1.5rem; color: #EC3D6D; font-weight: bold; }
    div[data-testid="stMetricLabel"] { color: #A6B1F7 !important; }
</style>
""", unsafe_allow_html=True)


# =====================================================================
# 2. FUNGSI-FUNGSI INTI MATEMATIKA & DISTRIBUSI WEIBULL
# =====================================================================
def reliability(t, beta, eta):
    t = np.asarray(t, dtype=float)
    return np.exp(-(t / eta) ** beta)

def hazard(t, beta, eta):
    t = np.asarray(t, dtype=float)
    t = np.where(t <= 0, 1e-6, t)
    return (beta / eta) * (t / eta) ** (beta - 1)

def pdf_weibull(t, beta, eta):
    t = np.asarray(t, dtype=float)
    return (beta / eta) * (t / eta) ** (beta - 1) * np.exp(-(t / eta) ** beta)

def mtbf_value(beta, eta):
    return eta * gamma_func(1 + 1 / beta)

def cycle_length(T, beta, eta):
    val, _ = quad(lambda t: reliability(t, beta, eta), 0, T)
    return val

def cost_conventional(T, beta, eta, Cp, Cf):
    if T <= 0:
        return np.inf
    R_T = float(reliability(T, beta, eta))
    F_T = 1 - R_T
    denom = cycle_length(T, beta, eta)
    if denom <= 1e-9:
        return np.inf
    return (Cf * F_T + Cp * R_T) / denom

def energi_dan_emisi_rate(T, beta, eta, e_PM, e_CM, faktor_emisi):
    R_T = float(reliability(T, beta, eta))
    F_T = 1 - R_T
    denom = cycle_length(T, beta, eta)
    if denom <= 1e-9:
        return np.inf, np.inf
    E_T = (e_PM * R_T + e_CM * F_T) / denom
    emisi_T = E_T * faktor_emisi
    return E_T, emisi_T

def biaya_emisi_rate(T, beta, eta, e_PM, e_CM, faktor_emisi, carbon_price):
    _, emisi_T = energi_dan_emisi_rate(T, beta, eta, e_PM, e_CM, faktor_emisi)
    return (emisi_T / 1000) * carbon_price

def build_normalization_bounds(beta, eta, Cp, Cf, e_PM, e_CM, faktor_emisi, carbon_price, T_max):
    T_range = np.linspace(1, T_max, 300)
    C_vals = np.array([cost_conventional(t, beta, eta, Cp, Cf) for t in T_range])
    E_vals = np.array([biaya_emisi_rate(t, beta, eta, e_PM, e_CM, faktor_emisi, carbon_price) for t in T_range])
    return C_vals.min(), C_vals.max(), E_vals.min(), E_vals.max(), T_range, C_vals, E_vals

def cost_sustainable_normalized(T, beta, eta, Cp, Cf, e_PM, e_CM, faktor_emisi, carbon_price, w, C_min, C_max, E_min, E_max):
    C_conv = cost_conventional(T, beta, eta, Cp, Cf)
    biaya_emisi = biaya_emisi_rate(T, beta, eta, e_PM, e_CM, faktor_emisi, carbon_price)
    C_norm = (C_conv - C_min) / (C_max - C_min + 1e-9)
    E_norm = (biaya_emisi - E_min) / (E_max - E_min + 1e-9)
    return (1 - w) * C_norm + w * E_norm

def compute_optimum_T(beta_, eta_, Cp_, Cf_, e_PM_, e_CM_, faktor_emisi_, carbon_price_, w_, T_max_):
    """Helper untuk analisis sensitivitas/tornado: mengembalikan T optimal sustainable."""
    Cmn, Cmx, Emn, Emx, _, _, _ = build_normalization_bounds(
        beta_, eta_, Cp_, Cf_, e_PM_, e_CM_, faktor_emisi_, carbon_price_, T_max_)
    r = minimize_scalar(cost_sustainable_normalized, bounds=(1, T_max_), method='bounded',
                         args=(beta_, eta_, Cp_, Cf_, e_PM_, e_CM_, faktor_emisi_, carbon_price_,
                               w_, Cmn, Cmx, Emn, Emx))
    return r.x

def T_for_target_reliability(R_target, beta, eta):
    """Interval PM (block replacement) agar keandalan saat servis >= R_target."""
    R_target = min(max(R_target, 1e-6), 0.999999)
    return eta * (-np.log(R_target)) ** (1.0 / beta)

def policy_metrics(beta, eta, Cp, Cf, e_PM, e_CM, faktor_emisi, MTTR, jam_operasi_tahun,
                    T=None, run_to_failure=False):
    """Menghitung metrik operasional & keandalan untuk satu kebijakan PM (T tertentu)
    atau kebijakan run-to-failure (tanpa PM terjadwal)."""
    if run_to_failure:
        mtbf = mtbf_value(beta, eta)
        cyc = mtbf
        R_T, F_T = 0.0, 1.0
        cost_rate = Cf / mtbf if mtbf > 0 else np.inf
        E_rate = e_CM / mtbf if mtbf > 0 else np.inf
        T_disp = np.nan
    else:
        T_disp = T
        R_T = float(reliability(T, beta, eta))
        F_T = 1 - R_T
        cyc = cycle_length(T, beta, eta)
        cost_rate = cost_conventional(T, beta, eta, Cp, Cf)
        E_rate, _ = energi_dan_emisi_rate(T, beta, eta, e_PM, e_CM, faktor_emisi)

    emisi_rate = E_rate * faktor_emisi
    n_cycles_year = jam_operasi_tahun / cyc if cyc and cyc > 1e-9 else np.nan
    expected_failures_year = n_cycles_year * F_T if not np.isnan(n_cycles_year) else np.nan
    downtime_year = expected_failures_year * MTTR if not np.isnan(expected_failures_year) else np.nan
    availability = (1 - downtime_year / jam_operasi_tahun) * 100 if not np.isnan(downtime_year) else np.nan
    annual_cost = cost_rate * jam_operasi_tahun
    annual_emisi = emisi_rate * jam_operasi_tahun

    return {
        "Interval PM (Jam)": "Run-to-Failure" if run_to_failure else f"{T_disp:.0f}",
        "R(T)": R_T,
        "F(T) - Peluang Gagal Sebelum PM": F_T,
        "Biaya/Jam (Rp)": cost_rate,
        "Biaya Tahunan (Rp)": annual_cost,
        "Emisi Tahunan (kg CO2)": annual_emisi,
        "Ekspektasi Kegagalan/Tahun": expected_failures_year,
        "Downtime/Tahun (Jam)": downtime_year,
        "Availability (%)": availability,
    }

def laplace_trend_test(tbf_data):
    """Uji tren Laplace untuk memeriksa asumsi renewal/i.i.d. pada data TBF.
    H0: tidak ada tren (asumsi renewal process terpenuhi).
    |U| > 1.96 -> tolak H0 pada alpha=5% (ada indikasi tren membaik/memburuk)."""
    t_cum = np.cumsum(tbf_data)
    n = len(tbf_data)
    T_obs = t_cum[-1]
    if T_obs <= 0 or n < 4:
        return None, None
    U = (np.sum(t_cum) / n - T_obs / 2) / (T_obs * np.sqrt(1.0 / (12 * n)))
    p_val = 2 * (1 - stats.norm.cdf(abs(U)))
    return U, p_val

def anderson_darling_stat(data, cdf_values_sorted):
    """Statistik Anderson-Darling generik dari nilai CDF (probability integral transform)."""
    n = len(cdf_values_sorted)
    F = np.clip(cdf_values_sorted, 1e-10, 1 - 1e-10)
    i = np.arange(1, n + 1)
    S = np.sum((2 * i - 1) * (np.log(F) + np.log(1 - F[::-1])))
    A2 = -n - S / n
    return A2

def fit_and_compare_distributions(data):
    """Membandingkan Weibull, Eksponensial, Lognormal, dan Gamma via AIC/BIC,
    serta statistik Anderson-Darling untuk masing-masing (loc dikunci = 0)."""
    n = len(data)
    rows = []

    fits = {
        "Weibull": (stats.weibull_min, 2),
        "Eksponensial": (stats.expon, 1),
        "Lognormal": (stats.lognorm, 2),
        "Gamma": (stats.gamma, 2),
    }
    for name, (dist, k) in fits.items():
        try:
            params = dist.fit(data, floc=0)
            loglik = np.sum(dist.logpdf(data, *params))
            aic = 2 * k - 2 * loglik
            bic = np.log(n) * k - 2 * loglik
            cdf_sorted = np.sort(dist.cdf(data, *params))
            ad = anderson_darling_stat(data, cdf_sorted)
            rows.append({"Distribusi": name, "Log-Likelihood": loglik,
                         "AIC": aic, "BIC": bic, "Anderson-Darling (A²)": ad})
        except Exception:
            continue
    df = pd.DataFrame(rows).sort_values("AIC").reset_index(drop=True)
    return df

def bootstrap_ci(tbf_data, Cp, Cf, e_PM, e_CM, faktor_emisi, carbon_price, w, T_max_search, n_boot=300, seed=123):
    """Bootstrap non-parametrik untuk interval kepercayaan beta, eta, dan T_optimal_sust."""
    rng = np.random.default_rng(seed)
    n = len(tbf_data)
    betas, etas, Ts = [], [], []
    for _ in range(n_boot):
        sample = rng.choice(tbf_data, size=n, replace=True)
        try:
            b, _, e = stats.weibull_min.fit(sample, floc=0)
            if b <= 0 or e <= 0:
                continue
            t_opt = compute_optimum_T(b, e, Cp, Cf, e_PM, e_CM, faktor_emisi, carbon_price, w, T_max_search)
            betas.append(b); etas.append(e); Ts.append(t_opt)
        except Exception:
            continue
    if len(betas) < 10:
        return None
    def ci(arr):
        return np.percentile(arr, [2.5, 50, 97.5])
    return {"beta": ci(betas), "eta": ci(etas), "T_optimal_sust": ci(Ts), "n_valid": len(betas)}


# =====================================================================
# 3. HEADER DASHBOARD & GLOSARIUM
# =====================================================================
st.markdown('<p class="main-header">🌌 Sistem Keputusan Perawatan Mesin (Sustainable PM)</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Menentukan jadwal servis optimal untuk meminimalkan biaya operasional dan menekan emisi karbon, berbasis Distribusi Weibull.</p>', unsafe_allow_html=True)

with st.expander("📖 PANDUAN PENGGUNAAN & GLOSARIUM ISTILAH (Klik untuk membuka)"):
    st.markdown("""
    <div class="info-box">
    <strong>Tujuan Dashboard:</strong> Menentukan interval servis rutin (PM) yang meminimalkan
    total biaya operasional sekaligus mempertimbangkan emisi karbon, dibandingkan terhadap
    praktik saat ini dan skenario run-to-failure.
    <hr style="margin: 10px 0; border-color: #A6B1F7;">
    <strong>Glosarium Istilah:</strong>
    <ul>
        <li><strong>TBF (Time Between Failures):</strong> Waktu (jam) antar kejadian kerusakan mesin.</li>
        <li><strong>Shape (β):</strong> β&gt;1 → laju kerusakan meningkat seiring usia (wear-out, PM efektif);
        β≈1 → kerusakan acak (PM rutin kurang berdampak); β&lt;1 → kegagalan dini (infant mortality).</li>
        <li><strong>Scale (η):</strong> Parameter skala umur (karakteristik waktu ke-63,2% populasi gagal).</li>
        <li><strong>MTBF:</strong> Rata-rata waktu antar kerusakan.</li>
        <li><strong>R(T):</strong> Peluang mesin masih bertahan (belum rusak) hingga waktu T.</li>
        <li><strong>PM (Preventive Maintenance):</strong> Servis rutin terjadwal sebelum mesin rusak.</li>
        <li><strong>CM (Corrective Maintenance):</strong> Perbaikan darurat karena mesin sudah rusak.</li>
        <li><strong>Run-to-Failure:</strong> Kebijakan tanpa PM terjadwal; mesin diganti/diperbaiki hanya
        setelah rusak.</li>
    </ul>
    <strong>Asumsi model:</strong> block replacement policy dengan PM bersifat "as good as new"
    (menyetel ulang usia mesin ke nol setiap PM/CM). Keterbatasan asumsi ini dijelaskan lebih
    lanjut di tab "Kekokohan Akademik".
    </div>
    """, unsafe_allow_html=True)


# =====================================================================
# 4. SIDEBAR — PANEL INPUT DATA
# =====================================================================
with st.sidebar:
    st.markdown("## 🎛️ Panel Input Data")
    st.markdown("---")

    st.markdown("#### 1. Data Historis Kerusakan (TBF)")
    input_mode = st.radio(
        "Sumber data:",
        ["Gunakan data contoh (SINTETIS)", "Upload CSV", "Input manual"]
    )

    data_is_synthetic = False
    if input_mode == "Upload CSV":
        uploaded = st.file_uploader("Upload file CSV (1 kolom: TBF dalam jam)", type=["csv"])
        if uploaded is not None:
            df_upload = pd.read_csv(uploaded)
            tbf_data = df_upload.iloc[:, 0].dropna().values.astype(float)
            data_is_synthetic = False
        else:
            tbf_data = np.array([
                111.7, 163.8, 169.7, 173.6, 217.7, 261.0, 265.7, 274.7, 294.1, 297.9,
                303.7, 316.0, 340.0, 342.9, 346.7, 372.3, 378.4, 385.3, 389.5, 395.4,
                399.5, 400.0, 409.9, 416.7, 421.2, 432.1, 445.0, 478.1, 483.3, 506.6,
                521.8, 532.6, 539.5, 554.9, 579.1, 644.0, 644.5, 671.8, 701.7, 770.0
            ])
            data_is_synthetic = True
            st.info("Belum ada file diunggah — menggunakan data contoh (sintetis) sementara.")
    elif input_mode == "Input manual":
        manual_text = st.text_area(
            "Masukkan data TBF (pisahkan koma)",
            value="111.7, 163.8, 169.7, 173.6, 217.7, 261.0, 265.7, 274.7, 294.1, 297.9, 303.7, 316.0, 340.0, 342.9, 346.7, 372.3, 378.4, 385.3, 389.5, 395.4, 399.5, 400.0, 409.9, 416.7, 421.2, 432.1, 445.0, 478.1, 483.3, 506.6, 521.8, 532.6, 539.5, 554.9, 579.1, 644.0, 644.5, 671.8, 701.7, 770.0"
        )
        try:
            tbf_data = np.array([float(x.strip()) for x in manual_text.split(",") if x.strip() != ""])
            data_is_synthetic = False
        except ValueError:
            st.error("Format data tidak valid.")
            tbf_data = np.array([111.7, 163.8, 169.7, 173.6, 217.7])
            data_is_synthetic = False
    else:
        tbf_data = np.array([
            111.7, 163.8, 169.7, 173.6, 217.7, 261.0, 265.7, 274.7, 294.1, 297.9,
            303.7, 316.0, 340.0, 342.9, 346.7, 372.3, 378.4, 385.3, 389.5, 395.4,
            399.5, 400.0, 409.9, 416.7, 421.2, 432.1, 445.0, 478.1, 483.3, 506.6,
            521.8, 532.6, 539.5, 554.9, 579.1, 644.0, 644.5, 671.8, 701.7, 770.0
        ])
        data_is_synthetic = True

    if data_is_synthetic:
        st.warning("⚠️ Sumber data saat ini: **data contoh (sintetis)**, dibangkitkan secara acak.")

    if len(tbf_data) < 20:
        st.warning(f"Jumlah data: {len(tbf_data)}. Idealnya minimal 20 observasi agar prediksi akurat "
                    "dan uji goodness-of-fit memiliki daya (power) yang memadai.")
    else:
        st.success(f"Jumlah data: {len(tbf_data)} observasi.")

    st.markdown("---")
    st.markdown("#### 2. Parameter Biaya (Finansial)")
    Cp = st.number_input("Biaya Servis Rutin / PM (Rp)", min_value=0, value=1500000, step=100000)
    Cf = st.number_input("Biaya Rusak Mendadak / CM (Rp)", min_value=0, value=8000000, step=100000)
    MTTR = st.number_input("Lama Waktu Perbaikan / MTTR (Jam)", min_value=0.1, value=8.0, step=0.5)

    st.markdown("---")
    st.markdown("#### 3. Parameter Lingkungan (Emisi)")
    st.caption("Batas perhitungan emisi HANYA mencakup konsumsi listrik saat PM/CM. "
               "Scrap material, energi rework, idle saat downtime, dan jejak karbon suku "
               "cadang belum dimodelkan.")
    e_PM = st.number_input("Listrik dipakai saat Servis PM (kWh)", min_value=0.0, value=3.5, step=0.5)
    e_CM = st.number_input("Listrik dipakai saat Rusak CM (kWh)", min_value=0.0, value=12.0, step=0.5)
    faktor_emisi = st.number_input("Faktor Emisi (kg CO2/kWh)", min_value=0.0, value=0.87, step=0.01)
    carbon_price = st.number_input("Harga Karbon (Rp/ton CO2)", min_value=0, value=75000, step=5000)
    sumber_faktor_emisi = st.text_input("Sumber & tahun faktor emisi (wajib dicantumkan di laporan)", value="")
    sumber_harga_karbon = st.text_input("Sumber & tahun harga karbon (wajib dicantumkan di laporan)", value="")

    st.markdown("---")
    st.markdown("#### 4. Kebijakan Manajemen")
    w = st.slider("Fokus Kepedulian Lingkungan (Bobot w)", 0.0, 1.0, 0.5, 0.05)
    R_target = st.slider("Target Keandalan Minimum saat PM — R(T)", 0.50, 0.99, 0.90, 0.01)

    st.markdown("---")
    st.markdown("#### 5. Baseline Pembanding")
    T_actual_company = st.number_input(
        "Jadwal PM Aktual Perusahaan Saat Ini (Jam) — isi 0 jika tidak ada data",
        min_value=0, value=0, step=10
    )

    st.markdown("---")
    st.markdown("#### 6. Proyeksi Bisnis Tahunan")
    jam_operasi_tahun = st.number_input("Total Jam Operasi Pabrik (1 Tahun)", min_value=1000, value=8000, step=500)
    T_max_search = st.number_input("Batas Atas Pencarian Jadwal (Jam)", min_value=100, value=2000, step=100)


# =====================================================================
# 5. PROSES KALKULASI UTAMA (WEIBULL FIT & OPTIMASI)
# =====================================================================
try:
    beta, loc_fit, eta = stats.weibull_min.fit(tbf_data, floc=0)
    D_stat, p_value = stats.kstest(tbf_data, 'weibull_min', args=(beta, loc_fit, eta))
    fit_success = True
except Exception as e:
    fit_success = False
    st.error(f"Gagal memproses data Weibull: {e}")

if fit_success:
    MTBF_val = mtbf_value(beta, eta)
    availability_mtbf = MTBF_val / (MTBF_val + MTTR)

    res_conv = minimize_scalar(cost_conventional, bounds=(1, T_max_search), method='bounded', args=(beta, eta, Cp, Cf))
    T_optimal_conv = res_conv.x
    cost_rate_conv = res_conv.fun

    C_min, C_max, E_min, E_max, T_range_plot, C_vals_plot, E_vals_plot = build_normalization_bounds(
        beta, eta, Cp, Cf, e_PM, e_CM, faktor_emisi, carbon_price, T_max_search)

    res_sust = minimize_scalar(cost_sustainable_normalized, bounds=(1, T_max_search), method='bounded',
                                args=(beta, eta, Cp, Cf, e_PM, e_CM, faktor_emisi, carbon_price, w, C_min, C_max, E_min, E_max))
    T_optimal_sust = res_sust.x

    T_reliability = T_for_target_reliability(R_target, beta, eta)
    T_reliability = min(T_reliability, T_max_search)

    cost_at_sust = cost_conventional(T_optimal_sust, beta, eta, Cp, Cf)

    if beta > 1.05:
        pola_kegagalan = "Keausan Seiring Waktu (Wear-out) — Perawatan rutin efektif."
    elif 0.95 <= beta <= 1.05:
        pola_kegagalan = "Kerusakan Acak (Random) — Perawatan rutin kurang berdampak signifikan."
    else:
        pola_kegagalan = "Kerusakan Dini (Infant Mortality) — Periksa instalasi awal / kualitas komponen."

    annual_cost_conv = cost_rate_conv * jam_operasi_tahun
    annual_cost_sust = cost_at_sust * jam_operasi_tahun
    # PERBAIKAN BUG TANDA: positif = biaya NAIK (buruk), negatif = biaya TURUN (baik)
    cost_change_rp = annual_cost_sust - annual_cost_conv
    cost_change_pct = (cost_change_rp / annual_cost_conv * 100) if annual_cost_conv != 0 else 0.0

    E_at_sust, emisi_at_sust = energi_dan_emisi_rate(T_optimal_sust, beta, eta, e_PM, e_CM, faktor_emisi)
    E_at_conv, emisi_at_conv = energi_dan_emisi_rate(T_optimal_conv, beta, eta, e_PM, e_CM, faktor_emisi)
    annual_emisi_conv = emisi_at_conv * jam_operasi_tahun
    annual_emisi_sust = emisi_at_sust * jam_operasi_tahun
    emisi_saved = annual_emisi_conv - annual_emisi_sust

    kebutuhan_sparepart = int(np.ceil(jam_operasi_tahun / T_optimal_sust)) if T_optimal_sust > 0 else 0

    # ------------------------------------------------------------
    # Tabel perbandingan kebijakan (baseline penting untuk validitas klaim)
    # ------------------------------------------------------------
    policy_rows = []
    policy_rows.append(("Run-to-Failure (tanpa PM)", policy_metrics(
        beta, eta, Cp, Cf, e_PM, e_CM, faktor_emisi, MTTR, jam_operasi_tahun, run_to_failure=True)))
    if T_actual_company > 0:
        policy_rows.append((f"Jadwal Aktual Perusahaan ({T_actual_company} jam)", policy_metrics(
            beta, eta, Cp, Cf, e_PM, e_CM, faktor_emisi, MTTR, jam_operasi_tahun, T=T_actual_company)))
    policy_rows.append(("Cost-Optimal / 'Jadwal Lama'", policy_metrics(
        beta, eta, Cp, Cf, e_PM, e_CM, faktor_emisi, MTTR, jam_operasi_tahun, T=T_optimal_conv)))
    policy_rows.append((f"Sustainable-Optimal / 'Jadwal Baru' (w={w:.2f})", policy_metrics(
        beta, eta, Cp, Cf, e_PM, e_CM, faktor_emisi, MTTR, jam_operasi_tahun, T=T_optimal_sust)))
    policy_rows.append((f"Reliability-Constrained (R≥{R_target:.2f})", policy_metrics(
        beta, eta, Cp, Cf, e_PM, e_CM, faktor_emisi, MTTR, jam_operasi_tahun, T=T_reliability)))

    df_policy = pd.DataFrame({name: metrics for name, metrics in policy_rows}).T
    df_policy_display = df_policy.copy()
    for col in ["R(T)", "F(T) - Peluang Gagal Sebelum PM"]:
        df_policy_display[col] = df_policy_display[col].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "-")
    for col in ["Biaya/Jam (Rp)", "Biaya Tahunan (Rp)", "Emisi Tahunan (kg CO2)"]:
        df_policy_display[col] = df_policy_display[col].apply(lambda x: f"{x:,.0f}" if pd.notna(x) else "-")
    for col in ["Ekspektasi Kegagalan/Tahun", "Downtime/Tahun (Jam)"]:
        df_policy_display[col] = df_policy_display[col].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "-")
    df_policy_display["Availability (%)"] = df_policy_display["Availability (%)"].apply(
        lambda x: f"{x:.2f}%" if pd.notna(x) else "-")

    # Referensi run-to-failure untuk narasi penghematan
    cost_rate_rtf = Cf / MTBF_val if MTBF_val > 0 else np.inf
    annual_cost_rtf = cost_rate_rtf * jam_operasi_tahun
    savings_vs_rtf_rp = annual_cost_rtf - annual_cost_sust
    savings_vs_rtf_pct = (savings_vs_rtf_rp / annual_cost_rtf * 100) if annual_cost_rtf != 0 else 0.0

    # =================================================================
    # 6. PEMBUATAN LAYOUT 5 TAB
    # =================================================================
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Profil Kesehatan Mesin",
        "💰 Optimasi & Perbandingan Kebijakan",
        "🌱 Trade-off & Sensitivitas",
        "🎓 Kekokohan Akademik",
        "📑 Ringkasan & Cetak Laporan"
    ])

    # -----------------------------------------------------------------
    # TAB 1: PROFIL & RELIABILITAS
    # -----------------------------------------------------------------
    with tab1:
        st.markdown('<p class="section-title">Hasil Prediksi Umur Mesin (Model Weibull)</p>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Pola Keausan (Shape/β)", f"{beta:.2f}")
        c2.metric("Skala Umur (Scale/η)", f"{eta:.0f} Jam")
        c3.metric("Rata-rata Umur (MTBF)", f"{MTBF_val:.0f} Jam")
        c4.metric("Availability (MTBF vs MTTR)", f"{availability_mtbf*100:.1f}%")

        if len(tbf_data) < 20 or beta <= 1.0:
            warn_msgs = []
            if len(tbf_data) < 20:
                warn_msgs.append(f"jumlah data historis hanya {len(tbf_data)} (idealnya ≥20), sehingga estimasi parameter dan uji goodness-of-fit memiliki ketidakpastian tinggi")
            if beta <= 1.0:
                warn_msgs.append("β ≤ 1 mengindikasikan laju kerusakan konstan/menurun, sehingga PM terjadwal secara teoritis TIDAK efektif menurunkan risiko kegagalan")
            st.markdown(f'<div class="danger-box">⚠️ <strong>Peringatan Interpretasi:</strong> {"; ".join(warn_msgs)}.</div>', unsafe_allow_html=True)

        ks_interpretation = ("Tidak terdapat cukup bukti statistik untuk menolak hipotesis bahwa data "
                              "mengikuti distribusi Weibull (gagal tolak H0)." if p_value > 0.05 else
                              "Terdapat indikasi ketidaksesuaian data historis terhadap model Weibull; "
                              "pertimbangkan menambah data atau membandingkan distribusi alternatif.")
        st.markdown(f"""
        <div class="warning-box">
        <strong>Diagnosis Pola Kerusakan:</strong> {pola_kegagalan}<br>
        <strong>Uji Kesesuaian Distribusi (Kolmogorov–Smirnov):</strong> D = {D_stat:.4f}, p-value = {p_value:.4f}.<br>
        {ks_interpretation}<br>
        <em>Catatan: karena parameter Weibull diestimasi dari data yang sama (bukan ditentukan a priori) dan
        ukuran sampel n={len(tbf_data)} relatif kecil, p-value uji KS ini cenderung optimistis (daya uji rendah)
        dan sebaiknya tidak dijadikan satu-satunya bukti kecocokan model. Lihat tab "Kekokohan Akademik" untuk
        perbandingan beberapa distribusi.</em>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("🔍 Lihat Plot Probabilitas Weibull (Uji Validitas Akademik)"):
            tbf_sorted = np.sort(tbf_data)
            n_data = len(tbf_sorted)
            F_emp = (np.arange(1, n_data + 1) - 0.3) / (n_data + 0.4)
            y_empirical = np.log(-np.log(1 - F_emp))
            x_empirical = np.log(tbf_sorted)

            x_line = np.linspace(min(x_empirical)*0.95, max(x_empirical)*1.05, 100)
            y_line = beta * x_line - beta * np.log(eta)

            fig_prob = go.Figure()
            fig_prob.add_trace(go.Scatter(x=x_empirical, y=y_empirical, mode='markers', name='Data Kerusakan Aktual', marker=dict(color='#EC3D6D', size=8)))
            fig_prob.add_trace(go.Scatter(x=x_line, y=y_line, mode='lines', name='Garis Teori Weibull', line=dict(color='#A6B1F7', dash='dash')))
            fig_prob.update_layout(template="plotly_dark", paper_bgcolor='#1A1C29', plot_bgcolor='#1A1C29', height=350, margin=dict(t=30, b=30), xaxis_title="ln(TBF) [Skala Waktu Logaritmik]", yaxis_title="ln(-ln(1 - F(t))) [Skala Probabilitas]")
            st.plotly_chart(fig_prob, use_container_width=True)

        st.markdown('<p class="section-title">Visualisasi Siklus Hidup Mesin</p>', unsafe_allow_html=True)

        t_plot = np.linspace(0.1, max(tbf_data.max() * 1.5, T_optimal_conv * 1.5), 300)
        fig1 = make_subplots(rows=1, cols=3, subplot_titles=("1. Peluang Bertahan Hidup R(t)", "2. Laju Kerusakan h(t)", "3. Kepadatan Peluang Gagal f(t)"))

        fig1.add_trace(go.Scatter(x=t_plot, y=reliability(t_plot, beta, eta), mode='lines', name='Peluang Bertahan', line=dict(color='#FFA3E5', width=3)), row=1, col=1)
        fig1.add_trace(go.Scatter(x=t_plot, y=hazard(t_plot, beta, eta), mode='lines', name='Laju Kerusakan', line=dict(color='#EC3D6D', width=3)), row=1, col=2)
        fig1.add_trace(go.Scatter(x=t_plot, y=pdf_weibull(t_plot, beta, eta), mode='lines', name='Kepadatan Rusak', line=dict(color='#F48A42', width=3), fill='tozeroy'), row=1, col=3)
        fig1.update_layout(template="plotly_dark", paper_bgcolor='#1A1C29', plot_bgcolor='#1A1C29', height=380, showlegend=False, margin=dict(t=50, b=30))
        st.plotly_chart(fig1, use_container_width=True)

        st.markdown('<p class="section-title">Data Waktu Historis Kerusakan</p>', unsafe_allow_html=True)
        col_a, col_b = st.columns([2, 1])
        with col_a:
            fig_hist = go.Figure(data=[go.Histogram(x=tbf_data, nbinsx=12, marker_color='#A6B1F7', opacity=0.75)])
            fig_hist.update_layout(template="plotly_dark", paper_bgcolor='#1A1C29', plot_bgcolor='#1A1C29', height=300, xaxis_title="Umur Mesin (Jam)", yaxis_title="Jumlah Kejadian", margin=dict(t=20, b=30))
            st.plotly_chart(fig_hist, use_container_width=True)
        with col_b:
            st.dataframe(pd.DataFrame({"Riwayat Kerusakan (Jam)": tbf_data}), height=300, use_container_width=True)

    # -----------------------------------------------------------------
    # TAB 2: OPTIMASI & PERBANDINGAN KEBIJAKAN
    # -----------------------------------------------------------------
    with tab2:
        st.markdown('<p class="section-title">Perbandingan Strategi Penjadwalan Servis</p>', unsafe_allow_html=True)
        colx, coly = st.columns(2)
        with colx:
            st.markdown("##### 🏢 Cost-Optimal ('Jadwal Lama' — fokus finansial murni)")
            st.metric("Interval PM", f"{T_optimal_conv:.0f} Jam")
            st.metric("Biaya Operasional", f"Rp {cost_rate_conv:,.0f} / Jam")
        with coly:
            st.markdown(f"##### 🌳 Sustainable-Optimal ('Jadwal Baru', Bobot w={w:.2f})")
            st.metric("Interval PM", f"{T_optimal_sust:.0f} Jam",
                       delta=f"{T_optimal_sust - T_optimal_conv:+.0f} Jam vs Cost-Optimal", delta_color="off")
            st.metric("Biaya Operasional", f"Rp {cost_at_sust:,.0f} / Jam")

        st.markdown(f"""
        <div class="info-box">
        🎯 <strong>Penjadwalan Berbasis Target Keandalan:</strong> Jika manajemen menetapkan syarat
        keandalan minimum R(T) ≥ <b>{R_target:.2f}</b> saat PM dilakukan, maka interval servis maksimum
        yang memenuhi syarat tersebut adalah <b>{T_reliability:.0f} jam</b> (dihitung dari
        T = η·(−ln R_target)^(1/β), independen dari biaya).
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<p class="section-title">Tabel Perbandingan Kebijakan (dengan Baseline)</p>', unsafe_allow_html=True)
        st.caption("Baseline pembanding mencakup jadwal aktual perusahaan (jika tersedia) dan "
                   "run-to-failure, selain kebijakan cost-optimal.")
        st.dataframe(df_policy_display, use_container_width=True)

        if cost_change_rp > 0:
            st.markdown(f"""
            <div class="danger-box">
            📌 Dibandingkan kebijakan cost-optimal murni, kebijakan sustainable-optimal menaikkan biaya
            tahunan sebesar <b>+Rp {cost_change_rp/1e6:,.1f} juta ({cost_change_pct:+.1f}%)</b>, sebagai
            konsekuensi dari pemberian bobot pada aspek lingkungan (w = {w:.2f}).
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="info-box">
            📌 Dibandingkan kebijakan cost-optimal murni, kebijakan sustainable-optimal menghemat
            <b>Rp {abs(cost_change_rp)/1e6:,.1f} juta ({abs(cost_change_pct):.1f}%)</b> per tahun.
            </div>
            """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="info-box">
        💡 <strong>Dibandingkan Run-to-Failure:</strong> Terhadap kebijakan tanpa PM terjadwal
        (≈Rp {annual_cost_rtf/1e6:,.1f} juta/tahun), kebijakan sustainable-optimal menghemat sekitar
        <b>Rp {savings_vs_rtf_rp/1e6:,.1f} juta/tahun ({savings_vs_rtf_pct:.1f}%)</b>.
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<p class="section-title">Proyeksi Tahunan & Kebutuhan Suku Cadang</p>', unsafe_allow_html=True)
        c1_b, c2_b, c3_b = st.columns(3)
        c1_b.metric("Total Biaya Tahunan (Jadwal Baru)", f"Rp {annual_cost_sust/1e6:,.1f} Juta",
                    delta=f"{cost_change_rp/1e6:+.1f} Juta ({cost_change_pct:+.1f}%) vs Cost-Optimal",
                    delta_color="inverse")
        c2_b.metric("Kebutuhan Suku Cadang (Forecast)", f"{kebutuhan_sparepart} Unit / Tahun")
        c3_b.metric("Emisi Tahunan (Jadwal Baru)", f"{annual_emisi_sust:,.0f} kg CO2",
                    delta=f"{-emisi_saved:+.0f} kg CO2 vs Cost-Optimal", delta_color="inverse")

        st.markdown(f"""
        <div class="info-box">
        📦 <strong>Saran Logistik (Inventory):</strong> Bagian gudang perlu menyiapkan sekitar
        <b>{kebutuhan_sparepart} paket suku cadang PM</b> per tahun pada interval {T_optimal_sust:.0f} jam
        agar operasional tidak terhenti akibat kehabisan stok.
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<p class="section-title">Kurva Biaya vs Interval Servis</p>', unsafe_allow_html=True)
        T_curve = np.linspace(10, T_max_search, 250)
        cost_curve = np.array([cost_conventional(t, beta, eta, Cp, Cf) for t in T_curve])
        emisi_cost_curve = np.array([biaya_emisi_rate(t, beta, eta, e_PM, e_CM, faktor_emisi, carbon_price) for t in T_curve])

        fig2 = make_subplots(specs=[[{"secondary_y": True}]])
        fig2.add_trace(go.Scatter(x=T_curve, y=cost_curve, name="Biaya Operasional (Rp/Jam)", line=dict(color='#A6B1F7', width=3)), secondary_y=False)
        fig2.add_trace(go.Scatter(x=T_curve, y=emisi_cost_curve, name="Setara Biaya Emisi (Rp/Jam)", line=dict(color='#F48A42', width=3, dash='dot')), secondary_y=True)
        fig2.add_vline(x=T_optimal_conv, line_dash="dash", line_color="#A6B1F7",
                        annotation_text=f"Cost-Optimal: {T_optimal_conv:.0f} Jam",
                        annotation_position="top left", annotation=dict(yshift=10))
        fig2.add_vline(x=T_optimal_sust, line_dash="dash", line_color="#F48A42",
                        annotation_text=f"Sustainable-Optimal: {T_optimal_sust:.0f} Jam",
                        annotation_position="top right", annotation=dict(yshift=35))
        fig2.update_layout(template="plotly_dark", paper_bgcolor='#1A1C29', plot_bgcolor='#1A1C29', height=440,
                            legend=dict(orientation="h", y=1.18), margin=dict(t=80, b=30),
                            yaxis_range=[min(cost_curve)*0.9, min(cost_curve)*3])
        st.plotly_chart(fig2, use_container_width=True)

    # -----------------------------------------------------------------
    # TAB 3: TRADE-OFF & SENSITIVITAS
    # -----------------------------------------------------------------
    with tab3:
        st.markdown('<p class="section-title">Kurva Trade-off Biaya vs Emisi (Pareto)</p>', unsafe_allow_html=True)
        st.caption("Setiap titik pada kurva merepresentasikan satu pilihan interval T, dengan sumbu X = "
                   "biaya operasional tahunan dan sumbu Y = emisi karbon tahunan, tanpa normalisasi.")

        annual_cost_curve_juta = np.array([cost_conventional(t, beta, eta, Cp, Cf) for t in T_curve]) * jam_operasi_tahun / 1e6
        annual_emisi_curve = np.array([
            energi_dan_emisi_rate(t, beta, eta, e_PM, e_CM, faktor_emisi)[1] for t in T_curve
        ]) * jam_operasi_tahun

        fig_pareto = go.Figure()
        fig_pareto.add_trace(go.Scatter(
            x=annual_cost_curve_juta, y=annual_emisi_curve, mode='lines+markers',
            marker=dict(size=4, color=T_curve, colorscale=[[0, '#A6B1F7'], [1, '#F48A42']],
                        showscale=True, colorbar=dict(title="Interval T (Jam)")),
            line=dict(color='#4B5162', width=1), name="Kurva Pilihan Interval T"
        ))
        highlight_points = [
            ("Cost-Optimal", T_optimal_conv, '#A6B1F7'),
            ("Sustainable-Optimal", T_optimal_sust, '#F48A42'),
            ("Reliability-Constrained", T_reliability, '#EC3D6D'),
        ]
        for label, T_h, color in highlight_points:
            c_h = cost_conventional(T_h, beta, eta, Cp, Cf) * jam_operasi_tahun / 1e6
            _, e_h_rate = energi_dan_emisi_rate(T_h, beta, eta, e_PM, e_CM, faktor_emisi)
            e_h = e_h_rate * jam_operasi_tahun
            fig_pareto.add_trace(go.Scatter(x=[c_h], y=[e_h], mode='markers+text', marker=dict(size=14, color=color, line=dict(width=2, color='white')),
                                             text=[label], textposition="top center", name=label))
        fig_pareto.update_layout(template="plotly_dark", paper_bgcolor='#1A1C29', plot_bgcolor='#1A1C29', height=460,
                                  xaxis_title="Biaya Operasional Tahunan (Rp Juta)", yaxis_title="Emisi Karbon Tahunan (kg CO2)",
                                  margin=dict(t=30, b=30), showlegend=False)
        st.plotly_chart(fig_pareto, use_container_width=True)

        # Insight: rentang T di mana biaya nyaris datar (mendekati optimum)
        min_cost_annual = annual_cost_curve_juta.min()
        band_mask = annual_cost_curve_juta <= min_cost_annual * 1.01
        if band_mask.sum() > 1:
            T_band_low, T_band_high = T_curve[band_mask].min(), T_curve[band_mask].max()
            emisi_band_low = annual_emisi_curve[band_mask].min()
            emisi_band_high = annual_emisi_curve[band_mask].max()
            st.markdown(f"""
            <div class="info-box">
            💡 <strong>Insight:</strong> Untuk interval servis antara <b>{T_band_low:.0f}–{T_band_high:.0f} jam</b>,
            biaya operasional tahunan berbeda kurang dari 1% dari titik minimumnya, sementara emisi tahunan
            pada rentang tersebut berkisar {emisi_band_low:,.0f}–{emisi_band_high:,.0f} kg CO2. Bergeser ke
            arah interval yang lebih rendah emisi dalam rentang ini <b>hampir tidak menambah biaya</b>.
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<p class="section-title">Sensitivitas Rekomendasi terhadap Bobot Kebijakan (w)</p>', unsafe_allow_html=True)
        w_range = np.linspace(0, 1, 11)
        T_vs_w = [compute_optimum_T(beta, eta, Cp, Cf, e_PM, e_CM, faktor_emisi, carbon_price, w_i, T_max_search) for w_i in w_range]
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(x=w_range, y=T_vs_w, mode='lines+markers', line=dict(color='#FFA3E5', width=3), marker=dict(size=8)))
        fig4.add_vline(x=w, line_dash="dot", line_color="#9CA3AF", annotation_text=f"Kebijakan Saat Ini ({w:.2f})")
        fig4.update_layout(template="plotly_dark", paper_bgcolor='#1A1C29', plot_bgcolor='#1A1C29', height=350, margin=dict(t=30, b=30), xaxis_title="Bobot Kepedulian Lingkungan (w)", yaxis_title="Rekomendasi Interval PM (Jam)")
        st.plotly_chart(fig4, use_container_width=True)

        st.markdown('<p class="section-title">Sensitivitas terhadap Ketidakpastian Konsumsi Listrik</p>', unsafe_allow_html=True)
        colp, colq = st.columns(2)
        with colp:
            variasi_pct = st.slider("Margin ketidakpastian estimasi listrik (±%)", 0, 50, 20, 5)

        skenario = [f"Lebih Hemat {variasi_pct}%", "Kondisi Saat Ini", f"Lebih Boros {variasi_pct}%"]
        faktor_var = [1 - variasi_pct/100, 1.0, 1 + variasi_pct/100]
        T_sens = []
        Cmn_base, Cmx_base, Emn_base, Emx_base, _, _, _ = build_normalization_bounds(beta, eta, Cp, Cf, e_PM, e_CM, faktor_emisi, carbon_price, T_max_search)
        for f in faktor_var:
            e_PM_s, e_CM_s = e_PM * f, e_CM * f
            r = minimize_scalar(cost_sustainable_normalized, bounds=(1, T_max_search), method='bounded',
                                 args=(beta, eta, Cp, Cf, e_PM_s, e_CM_s, faktor_emisi, carbon_price, w, Cmn_base, Cmx_base, Emn_base, Emx_base))
            T_sens.append(r.x)
        df_sens = pd.DataFrame({"Kondisi Kelistrikan": skenario, "Rekomendasi Interval PM (Jam)": [f"{t:.1f}" for t in T_sens]})
        st.table(df_sens)

        st.markdown('<p class="section-title">Diagram Tornado — Sensitivitas Parameter terhadap Rekomendasi Interval (T)</p>', unsafe_allow_html=True)
        st.caption("Setiap parameter diubah ±15% dari nilai dasar (parameter lain tetap), lalu dilihat "
                   "seberapa besar pergeseran interval PM sustainable-optimal yang dihasilkan.")

        pert = 0.15
        tornado_rows = []
        base_kwargs = dict(beta_=beta, eta_=eta, Cp_=Cp, Cf_=Cf, e_PM_=e_PM, e_CM_=e_CM,
                            faktor_emisi_=faktor_emisi, carbon_price_=carbon_price, w_=w, T_max_=T_max_search)
        param_specs = [("β (Shape)", "beta_", beta), ("η (Scale)", "eta_", eta),
                       ("Cf (Biaya CM)", "Cf_", Cf), ("Cp (Biaya PM)", "Cp_", Cp)]
        for label, key, base_val in param_specs:
            kwargs_low = dict(base_kwargs); kwargs_low[key] = base_val * (1 - pert)
            kwargs_high = dict(base_kwargs); kwargs_high[key] = base_val * (1 + pert)
            T_low = compute_optimum_T(**kwargs_low)
            T_high = compute_optimum_T(**kwargs_high)
            tornado_rows.append({"Parameter": label, "T_min": min(T_low, T_high), "T_max": max(T_low, T_high),
                                  "Rentang": abs(T_high - T_low)})
        df_tornado = pd.DataFrame(tornado_rows).sort_values("Rentang", ascending=True)

        fig_tornado = go.Figure()
        fig_tornado.add_trace(go.Bar(
            y=df_tornado["Parameter"], x=df_tornado["T_max"] - df_tornado["T_min"], base=df_tornado["T_min"],
            orientation='h', marker=dict(color='#EC3D6D'),
            text=[f"{lo:.0f}–{hi:.0f} jam" for lo, hi in zip(df_tornado["T_min"], df_tornado["T_max"])],
            textposition='outside'
        ))
        fig_tornado.add_vline(x=T_optimal_sust, line_dash="dot", line_color="#FFA3E5",
                               annotation_text=f"Baseline = {T_optimal_sust:.0f} jam")
        fig_tornado.update_layout(template="plotly_dark", paper_bgcolor='#1A1C29', plot_bgcolor='#1A1C29', height=350,
                                   margin=dict(t=30, b=30, l=10, r=80), xaxis_title="Rekomendasi Interval PM (Jam)")
        st.plotly_chart(fig_tornado, use_container_width=True)

    # -----------------------------------------------------------------
    # TAB 4: KEKOKOHAN AKADEMIK
    # -----------------------------------------------------------------
    with tab4:
        st.markdown('<p class="section-title">Uji Tren Laplace (Asumsi Renewal Process / i.i.d.)</p>', unsafe_allow_html=True)
        st.caption("Model Weibull renewal mengasumsikan TBF bersifat i.i.d. (tidak ada tren membaik/memburuk "
                   "antar kegagalan berurutan). Uji ini memeriksa asumsi tersebut secara kasar dari data TBF "
                   "yang tersedia (mengasumsikan urutan data mencerminkan urutan kronologis kegagalan).")
        U_stat, p_laplace = laplace_trend_test(tbf_data)
        if U_stat is not None:
            interpretasi_laplace = ("tidak ada bukti tren signifikan pada taraf 5% — asumsi renewal process "
                                     "masih dapat dipertahankan" if abs(U_stat) < 1.96 else
                                     "terdapat indikasi tren (membaik jika U<0 atau memburuk jika U>0) pada "
                                     "taraf 5% — asumsi renewal process i.i.d. perlu dipertimbangkan ulang, "
                                     "misalnya dengan model non-homogeneous Poisson process (NHPP)")
            st.markdown(f"""
            <div class="info-box">
            Statistik U = <b>{U_stat:.3f}</b>, p-value ≈ <b>{p_laplace:.4f}</b>.<br>
            Interpretasi: {interpretasi_laplace}.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Data tidak cukup untuk uji tren Laplace (minimal 4 observasi).")

        st.markdown('<p class="section-title">Perbandingan Distribusi Alternatif (AIC / BIC / Anderson-Darling)</p>', unsafe_allow_html=True)
        st.caption("Model dengan AIC/BIC terkecil dianggap paling didukung data, mempertimbangkan trade-off "
                   "antara kecocokan (log-likelihood) dan kompleksitas model (jumlah parameter).")
        df_dist = fit_and_compare_distributions(tbf_data)
        df_dist_display = df_dist.copy()
        for col in ["Log-Likelihood", "AIC", "BIC", "Anderson-Darling (A²)"]:
            df_dist_display[col] = df_dist_display[col].apply(lambda x: f"{x:,.2f}")
        st.dataframe(df_dist_display, use_container_width=True)
        best_dist = df_dist.iloc[0]["Distribusi"] if len(df_dist) > 0 else "Weibull"
        if best_dist != "Weibull":
            st.markdown(f"""
            <div class="danger-box">
            ⚠️ Berdasarkan AIC, distribusi <b>{best_dist}</b> memberikan kecocokan lebih baik daripada Weibull
            untuk data ini. Pertimbangkan untuk mengevaluasi ulang pemilihan model, karena seluruh perhitungan
            interval PM pada dashboard ini menggunakan asumsi distribusi Weibull.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.success("Weibull tetap menjadi distribusi dengan AIC terendah di antara kandidat yang diuji.")

        st.markdown('<p class="section-title">Interval Kepercayaan Bootstrap untuk β, η, dan T Optimal</p>', unsafe_allow_html=True)
        st.caption("Estimasi titik (β, η, T) tidak memberi gambaran ketidakpastian. Bootstrap non-parametrik "
                   "(resampling data TBF dengan pengembalian) memberikan interval kepercayaan 95% secara empiris. "
                   "Proses ini dijalankan hanya saat diminta (tombol di bawah) karena cukup intensif secara komputasi.")
        if st.button("🔄 Jalankan Bootstrap (n=300 iterasi)"):
            with st.spinner("Menjalankan resampling bootstrap..."):
                boot_result = bootstrap_ci(tbf_data, Cp, Cf, e_PM, e_CM, faktor_emisi, carbon_price, w, T_max_search, n_boot=300)
            if boot_result is None:
                st.error("Bootstrap gagal menghasilkan cukup sampel valid. Periksa kembali data TBF.")
            else:
                cib, cie, cit = boot_result["beta"], boot_result["eta"], boot_result["T_optimal_sust"]
                st.markdown(f"""
                <div class="info-box">
                Berdasarkan {boot_result['n_valid']} resample valid:<br>
                β: median = {cib[1]:.2f}, 95% CI = [{cib[0]:.2f}, {cib[2]:.2f}]<br>
                η: median = {cie[1]:.0f} jam, 95% CI = [{cie[0]:.0f}, {cie[2]:.0f}]<br>
                Interval PM sustainable-optimal: median = {cit[1]:.0f} jam,
                95% CI = [{cit[0]:.0f}, {cit[2]:.0f}] jam
                </div>
                """, unsafe_allow_html=True)

        st.markdown('<p class="section-title">Keterbatasan Model: Asumsi PM "As Good As New"</p>', unsafe_allow_html=True)
        st.markdown("""
        <div class="warning-box">
        Model pada dashboard ini menggunakan kebijakan <em>block replacement</em> dengan asumsi PM bersifat
        sempurna (mengembalikan mesin ke kondisi "sebaik baru"). Untuk mesin CNC, asumsi ini sering tidak
        realistis karena servis rutin umumnya tidak mengganti seluruh komponen kritis. Model perawatan tidak
        sempurna (mis. <em>Kijima Type I/II virtual age model</em>) dapat merepresentasikan kondisi ini secara
        lebih akurat, namun membutuhkan data tambahan (riwayat efektivitas tiap PM) yang belum tersedia pada
        studi ini.
        </div>
        """, unsafe_allow_html=True)

    # -----------------------------------------------------------------
    # TAB 5: RINGKASAN (PDF) & UNDUHAN EXCEL
    # -----------------------------------------------------------------
    with tab5:
        st.markdown('<p class="section-title">Ringkasan & Ekspor Dokumen</p>', unsafe_allow_html=True)

        summary_df = pd.DataFrame({
            "Indikator Kinerja": ["Sumber Data", "Pola Keausan (β)", "Rata-rata Umur Mesin (MTBF)",
                          "Kesesuaian Distribusi Weibull (KS-test)",
                          "Interval PM Cost-Optimal", "Interval PM Sustainable-Optimal",
                          "Interval PM Reliability-Constrained",
                          "Biaya Tahunan (Sustainable-Optimal)",
                          "Selisih Biaya vs Cost-Optimal",
                          "Penghematan vs Run-to-Failure",
                          "Kebutuhan Suku Cadang Tahunan"],
            "Hasil Kalkulasi": [
                "Data Contoh (SINTETIS)" if data_is_synthetic else "Data yang diinput pengguna",
                f"{beta:.2f} ({pola_kegagalan.split('—')[0].strip()})",
                f"{MTBF_val:.0f} Jam",
                "Tidak ada bukti untuk menolak Weibull" if p_value > 0.05 else "Indikasi ketidaksesuaian",
                f"{T_optimal_conv:.0f} Jam",
                f"{T_optimal_sust:.0f} Jam",
                f"{T_reliability:.0f} Jam (R≥{R_target:.2f})",
                f"Rp {annual_cost_sust:,.0f}",
                f"{'+' if cost_change_rp>=0 else ''}Rp {cost_change_rp:,.0f} ({cost_change_pct:+.1f}%)",
                f"Rp {savings_vs_rtf_rp:,.0f} ({savings_vs_rtf_pct:.1f}%)",
                f"{kebutuhan_sparepart} Unit"
            ]
        })
        st.table(summary_df)

        st.markdown(f"""
        <div class="warning-box">
        📌 <strong>Keterbatasan yang perlu dicantumkan dalam laporan/skripsi:</strong><br>
        1. Batas perhitungan emisi hanya mencakup listrik saat PM/CM (belum termasuk scrap, rework, idle,
        dan jejak karbon suku cadang).<br>
        2. Faktor emisi ({faktor_emisi} kg CO2/kWh) dan harga karbon (Rp {carbon_price:,.0f}/ton) bersumber
        dari: {sumber_faktor_emisi or '(belum dicantumkan — lengkapi di sidebar)'} /
        {sumber_harga_karbon or '(belum dicantumkan — lengkapi di sidebar)'}.<br>
        3. Model mengasumsikan PM "as good as new" (block replacement), bukan perawatan tidak sempurna.<br>
        4. Uji KS memiliki daya rendah pada n={len(tbf_data)}; gunakan bersama perbandingan AIC/BIC di tab
        Kekokohan Akademik.
        </div>
        """, unsafe_allow_html=True)

        if not KALEIDO_OK:
            st.error("Paket 'kaleido' belum terpasang atau versinya tidak kompatibel, sehingga ekspor PDF "
                     "(yang menyisipkan grafik sebagai gambar) tidak dapat dijalankan. Jalankan "
                     "`pip install -U \"kaleido>=1\"` dan tambahkan ke requirements.txt, lalu muat ulang aplikasi.")

        col_pdf, col_excel = st.columns(2)

        def generate_pdf():
            buffer = io.BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=1.5*cm, bottomMargin=1.5*cm, leftMargin=1.8*cm, rightMargin=1.8*cm)
            styles = getSampleStyleSheet()

            title_style = ParagraphStyle('TitleC', parent=styles['Title'], alignment=TA_CENTER, fontSize=18, spaceAfter=20, textColor=colors.HexColor('#292965'), fontName="Helvetica-Bold")
            subtitle_style = ParagraphStyle('SubC', parent=styles['Normal'], alignment=TA_CENTER, fontSize=11, spaceAfter=20, textColor=colors.HexColor('#4B5162'))
            heading_style = ParagraphStyle('Heading2Custom', parent=styles['Heading2'], fontSize=12, spaceBefore=15, spaceAfter=10, textColor=colors.white, backColor=colors.HexColor('#EC3D6D'), borderPadding=(6, 6, 6, 6), fontName="Helvetica-Bold")
            normal_style = ParagraphStyle('NormalCustom', parent=styles['Normal'], fontSize=10, spaceAfter=8, leading=15, alignment=TA_JUSTIFY)
            caption_style = ParagraphStyle('CaptionCustom', parent=styles['Normal'], fontSize=9, spaceBefore=5, spaceAfter=15, leading=13, textColor=colors.HexColor('#475569'), fontName="Helvetica-Oblique", alignment=TA_CENTER)

            story = []
            story.append(Paragraph("Laporan Eksekutif Penjadwalan Mesin", title_style))
            story.append(Paragraph(f"Model Sustainable Preventive Maintenance (Dicetak: {datetime.datetime.now().strftime('%d %B %Y')})", subtitle_style))

            if data_is_synthetic:
                story.append(Paragraph("<b>CATATAN:</b> Laporan ini dihasilkan dari data contoh (sintetis), bukan data historis kerusakan mesin yang sebenarnya.", normal_style))

            story.append(Paragraph("1. Ringkasan Kinerja Mesin & Prediksi", heading_style))
            story.append(Paragraph("Tabel berikut merangkum parameter reliabilitas mesin dan hasil optimasi jadwal, termasuk perbandingan terhadap baseline run-to-failure dan (jika tersedia) jadwal aktual perusahaan.", normal_style))

            data_param = [["Indikator Kunci", "Hasil Kalkulasi"]] + summary_df.values.tolist()
            t = Table(data_param, colWidths=[9*cm, 8*cm])
            t.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#292965')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.lightgrey),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8fafc')]),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('PADDING', (0, 0), (-1, -1), 8),
            ]))
            story.append(t)
            story.append(Spacer(1, 10))

            story.append(Paragraph("2. Tabel Perbandingan Kebijakan", heading_style))
            df_policy_pdf = df_policy_display.reset_index().rename(columns={"index": "Kebijakan"})
            data_policy = [df_policy_pdf.columns.tolist()] + df_policy_pdf.values.tolist()
            t2 = Table(data_policy, repeatRows=1)
            t2.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#292965')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('GRID', (0, 0), (-1, -1), 0.4, colors.lightgrey),
                ('FONTSIZE', (0, 0), (-1, -1), 6.5),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('PADDING', (0, 0), (-1, -1), 4),
            ]))
            story.append(t2)
            story.append(Spacer(1, 10))

            story.append(Paragraph("3. Analisis Keputusan & Dampak Bisnis", heading_style))
            arah_biaya = "menaikkan" if cost_change_rp > 0 else "menurunkan"
            rekomendasi_text = f"""
            Model matematis menunjukkan pola kegagalan mesin saat ini terdiagnosis sebagai <b>{pola_kegagalan}</b>.
            Dengan bobot lingkungan w = {w:.2f}, interval PM sustainable-optimal yang disarankan adalah
            <b>{T_optimal_sust:.0f} jam</b>, yang {arah_biaya} biaya tahunan sebesar
            <b>Rp {abs(cost_change_rp):,.0f} ({abs(cost_change_pct):.1f}%)</b> dibandingkan interval cost-optimal murni,
            namun tetap menghemat sekitar <b>Rp {savings_vs_rtf_rp:,.0f} ({savings_vs_rtf_pct:.1f}%)</b> per tahun
            dibandingkan kebijakan run-to-failure (tanpa PM terjadwal).
            <br/><br/>
            Dengan asumsi pabrik beroperasi {jam_operasi_tahun} jam/tahun, jadwal ini membutuhkan alokasi
            <b>{kebutuhan_sparepart} paket suku cadang</b> per tahun.
            """
            story.append(Paragraph(rekomendasi_text, normal_style))
            story.append(Spacer(1, 10))

            if KALEIDO_OK:
                try:
                    story.append(Paragraph("4. Visualisasi Siklus Hidup Mesin", heading_style))
                    fig1_png = fig1.to_image(format="png", width=950, height=350, scale=2)
                    story.append(RLImage(io.BytesIO(fig1_png), width=17*cm, height=6.2*cm))
                    story.append(Paragraph("Peluang bertahan (kiri), laju kerusakan (tengah), dan kepadatan peluang gagal (kanan) terhadap jam operasional.", caption_style))
                    story.append(Spacer(1, 10))
                    story.append(PageBreak())

                    story.append(Paragraph("5. Kurva Biaya vs Interval Servis", heading_style))
                    fig2_png = fig2.to_image(format="png", width=900, height=450, scale=2)
                    story.append(RLImage(io.BytesIO(fig2_png), width=16*cm, height=8*cm))
                    story.append(Paragraph(f"Garis vertikal menandai interval cost-optimal ({T_optimal_conv:.0f} jam) dan sustainable-optimal ({T_optimal_sust:.0f} jam).", caption_style))
                    story.append(Spacer(1, 10))

                    story.append(Paragraph("6. Kurva Trade-off Biaya vs Emisi (Pareto)", heading_style))
                    fig_pareto_png = fig_pareto.to_image(format="png", width=900, height=450, scale=2)
                    story.append(RLImage(io.BytesIO(fig_pareto_png), width=16*cm, height=8*cm))
                    story.append(Paragraph("Setiap titik merepresentasikan satu pilihan interval PM; titik berlabel menandai kebijakan-kebijakan yang dibandingkan.", caption_style))
                except Exception as img_err:
                    story.append(Paragraph(f"(Sebagian grafik gagal disisipkan karena Kaleido bermasalah: {img_err})", caption_style))

            story.append(Paragraph("7. Keterbatasan Studi", heading_style))
            keterbatasan_text = """
            Batas perhitungan emisi hanya mencakup konsumsi listrik saat PM/CM; belum mencakup scrap material,
            energi rework, energi idle saat downtime, dan jejak karbon suku cadang. Model mengasumsikan PM
            bersifat "as good as new" (block replacement), yang mungkin tidak sepenuhnya sesuai untuk mesin CNC.
            Uji Kolmogorov-Smirnov terhadap parameter hasil estimasi memiliki daya uji terbatas pada ukuran
            sampel kecil, sehingga diperkuat dengan perbandingan AIC/BIC terhadap distribusi alternatif.
            """
            story.append(Paragraph(keterbatasan_text, normal_style))

            doc.build(story)
            buffer.seek(0)
            return buffer

        def generate_excel():
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                summary_df.to_excel(writer, sheet_name='Ringkasan', index=False)
                df_policy_display.reset_index().rename(columns={"index": "Kebijakan"}).to_excel(
                    writer, sheet_name='Perbandingan_Kebijakan', index=False)
                df_sens.to_excel(writer, sheet_name='Sensitivitas_Listrik', index=False)
                df_tornado.to_excel(writer, sheet_name='Tornado_Parameter', index=False)
                df_dist.to_excel(writer, sheet_name='Perbandingan_Distribusi', index=False)
                pd.DataFrame({"Riwayat_Kerusakan_Jam": tbf_data}).to_excel(writer, sheet_name='Data_Historis', index=False)
            return output.getvalue()

        with col_pdf:
            if KALEIDO_OK:
                try:
                    pdf_buffer = generate_pdf()
                    st.download_button(
                        label="📄 Unduh Laporan (PDF)",
                        data=pdf_buffer,
                        file_name="Laporan_Sustainable_PM_Cosmic.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                        type="primary"
                    )
                except Exception as e:
                    st.error(f"Gagal membuat PDF: {e}")
            else:
                st.button("📄 Unduh Laporan (PDF) — dinonaktifkan (kaleido belum siap)", disabled=True, use_container_width=True)

        with col_excel:
            try:
                excel_data = generate_excel()
                st.download_button(
                    label="📊 Unduh Tabel Data (Excel)",
                    data=excel_data,
                    file_name="Data_Sustainable_PM_Cosmic.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                    type="secondary"
                )
            except Exception as e:
                st.error(f"Gagal membuat Excel. Pastikan 'xlsxwriter' terinstal. Error: {e}")

else:
    st.error("Silakan unggah data kerusakan (TBF) yang valid di panel sebelah kiri untuk memulai proses analisis.")
