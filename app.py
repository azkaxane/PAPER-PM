"""
Dashboard Interaktif: Analisis Reliabilitas dan Penjadwalan
Sustainable Preventive Maintenance Berbasis Distribusi Weibull
=================================================================
Studi Kasus Industri Manufaktur

Catatan metodologis:
- Model reliabilitas: distribusi Weibull 2-parameter (shape beta, scale eta)
- Model biaya konvensional: age replacement policy (Barlow & Hunter)
- Model sustainable: weighted-sum multi-objective optimization
  (normalisasi cost & emisi ke rentang [0,1] sebelum digabung)
"""

import io
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
from reportlab.lib.enums import TA_CENTER

# =====================================================================
# KONFIGURASI HALAMAN
# =====================================================================
st.set_page_config(
    page_title="Sustainable PM Dashboard | Weibull Reliability Analysis",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---- Styling custom ----
st.markdown("""
<style>
    .main-header {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1e3a5f;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1.0rem;
        color: #5a6b7d;
        margin-top: 0;
    }
    .metric-card {
        background-color: #f8f9fb;
        border: 1px solid #e3e7ec;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
    }
    .section-title {
        font-size: 1.3rem;
        font-weight: 600;
        color: #1e3a5f;
        border-left: 4px solid #2e86ab;
        padding-left: 10px;
        margin-top: 1.2rem;
        margin-bottom: 0.6rem;
    }
    .note-box {
        background-color: #fff8e6;
        border-left: 4px solid #e8a33d;
        padding: 10px 14px;
        border-radius: 6px;
        font-size: 0.88rem;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.4rem;
        color: #1e3a5f;
    }
</style>
""", unsafe_allow_html=True)


# =====================================================================
# FUNGSI-FUNGSI INTI PERHITUNGAN (sudah diuji terpisah di test_logic.py)
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
    return (emisi_T / 1000) * carbon_price  # kg -> ton CO2


def build_normalization_bounds(beta, eta, Cp, Cf, e_PM, e_CM, faktor_emisi, carbon_price, T_max):
    T_range = np.linspace(1, T_max, 300)
    C_vals = np.array([cost_conventional(t, beta, eta, Cp, Cf) for t in T_range])
    E_vals = np.array([biaya_emisi_rate(t, beta, eta, e_PM, e_CM, faktor_emisi, carbon_price) for t in T_range])
    return C_vals.min(), C_vals.max(), E_vals.min(), E_vals.max(), T_range, C_vals, E_vals


def cost_sustainable_normalized(T, beta, eta, Cp, Cf, e_PM, e_CM, faktor_emisi, carbon_price, w,
                                 C_min, C_max, E_min, E_max):
    C_conv = cost_conventional(T, beta, eta, Cp, Cf)
    biaya_emisi = biaya_emisi_rate(T, beta, eta, e_PM, e_CM, faktor_emisi, carbon_price)
    C_norm = (C_conv - C_min) / (C_max - C_min + 1e-9)
    E_norm = (biaya_emisi - E_min) / (E_max - E_min + 1e-9)
    return (1 - w) * C_norm + w * E_norm


# =====================================================================
# SIDEBAR — INPUT DATA & PARAMETER
# =====================================================================
with st.sidebar:
    st.markdown("## ⚙️ Panel Input")
    st.markdown("---")

    st.markdown("#### 1. Data Waktu Antar-Kegagalan (TBF)")
    input_mode = st.radio("Sumber data:", ["Gunakan data contoh", "Upload CSV", "Input manual"], index=0)

    if input_mode == "Upload CSV":
        uploaded = st.file_uploader("Upload file CSV (1 kolom: TBF dalam jam)", type=["csv"])
        if uploaded is not None:
            df_upload = pd.read_csv(uploaded)
            tbf_data = df_upload.iloc[:, 0].dropna().values.astype(float)
        else:
            st.info("Menunggu file diunggah. Menggunakan data contoh sementara.")
            np.random.seed(42)
            tbf_data = np.round(stats.weibull_min.rvs(2.3, loc=0, scale=480, size=28), 1)
    elif input_mode == "Input manual":
        manual_text = st.text_area("Masukkan data TBF (pisahkan dengan koma)",
                                    value="345.4, 775, 541, 461.4, 221.9, 221.9, 141.1, 650.4, 462.7, 525.4, "
                                          "89.1, 827.9, 617.7, 257.5, 238.8, 239.8, 308.9, 422.1, 374.6, 301.9")
        try:
            tbf_data = np.array([float(x.strip()) for x in manual_text.split(",") if x.strip() != ""])
        except ValueError:
            st.error("Format data tidak valid, gunakan angka dipisah koma.")
            tbf_data = np.array([345.4, 775, 541, 461.4, 221.9])
    else:
        np.random.seed(42)
        tbf_data = np.round(stats.weibull_min.rvs(2.3, loc=0, scale=480, size=28), 1)
        st.caption("Data contoh: 28 observasi TBF hasil simulasi (pola wear-out). "
                   "Ganti dengan data historis mesin riil Anda untuk hasil yang valid secara akademik.")

    st.caption(f"Jumlah observasi: **{len(tbf_data)}**" +
               ("" if len(tbf_data) >= 20 else "  ⚠️ Idealnya minimal 20-30 observasi agar estimasi parameter Weibull reliabel"))

    st.markdown("---")
    st.markdown("#### 2. Parameter Biaya Maintenance")
    Cp = st.number_input("Biaya Preventive Maintenance, Cp (Rp)", min_value=0, value=1_500_000, step=100_000, format="%d")
    Cf = st.number_input("Biaya Corrective Maintenance, Cf (Rp)", min_value=0, value=8_000_000, step=100_000, format="%d")
    MTTR = st.number_input("Rata-rata waktu perbaikan / MTTR (jam)", min_value=0.1, value=8.0, step=0.5)

    st.markdown("---")
    st.markdown("#### 3. Parameter Energi & Emisi")
    st.caption("Jika data presisi tidak tersedia, gunakan estimasi dari nameplate mesin / tagihan listrik (lihat catatan metodologi).")
    e_PM = st.number_input("Energi ekstra per event PM (kWh)", min_value=0.0, value=3.5, step=0.5)
    e_CM = st.number_input("Energi ekstra per event CM (kWh)", min_value=0.0, value=12.0, step=0.5)
    faktor_emisi = st.number_input("Faktor emisi grid listrik (kg CO2/kWh)", min_value=0.0, value=0.87, step=0.01,
                                    help="Gunakan angka resmi dari Kementerian ESDM/PLN sesuai wilayah studi kasus.")
    carbon_price = st.number_input("Nilai ekonomi karbon (Rp/ton CO2)", min_value=0, value=75_000, step=5_000,
                                    help="Asumsi penelitian — nyatakan sumber rujukan di paper (mis. NEK pemerintah).")

    st.markdown("---")
    st.markdown("#### 4. Preferensi Trade-off")
    w = st.slider("Bobot prioritas keberlanjutan (w)", 0.0, 1.0, 0.5, 0.05,
                   help="w=0 -> fokus penuh minimasi biaya. w=1 -> fokus penuh minimasi emisi.")
    T_max_search = st.number_input("Batas atas pencarian interval T (jam)", min_value=100, value=2000, step=100)

    st.markdown("---")
    hitung_btn = st.button("🔄 Hitung / Perbarui Analisis", use_container_width=True, type="primary")


# =====================================================================
# HEADER
# =====================================================================
st.markdown('<p class="main-header">⚙️ Dashboard Analisis Reliabilitas & Sustainable Preventive Maintenance</p>',
            unsafe_allow_html=True)
st.markdown('<p class="sub-header">Pendekatan Distribusi Weibull — Studi Kasus Industri Manufaktur</p>',
            unsafe_allow_html=True)
st.markdown("---")

# =====================================================================
# PERHITUNGAN UTAMA
# =====================================================================
try:
    beta, loc_fit, eta = stats.weibull_min.fit(tbf_data, floc=0)
    D_stat, p_value = stats.kstest(tbf_data, 'weibull_min', args=(beta, loc_fit, eta))
    fit_success = True
except Exception as e:
    fit_success = False
    st.error(f"Gagal melakukan fitting Weibull: {e}")

if fit_success:
    MTBF_val = mtbf_value(beta, eta)
    availability = MTBF_val / (MTBF_val + MTTR)

    res_conv = minimize_scalar(cost_conventional, bounds=(1, T_max_search), method='bounded',
                                args=(beta, eta, Cp, Cf))
    T_optimal_conv = res_conv.x
    cost_rate_conv = res_conv.fun

    C_min, C_max, E_min, E_max, T_range_plot, C_vals_plot, E_vals_plot = build_normalization_bounds(
        beta, eta, Cp, Cf, e_PM, e_CM, faktor_emisi, carbon_price, T_max_search)

    res_sust = minimize_scalar(cost_sustainable_normalized, bounds=(1, T_max_search), method='bounded',
                                args=(beta, eta, Cp, Cf, e_PM, e_CM, faktor_emisi, carbon_price, w,
                                      C_min, C_max, E_min, E_max))
    T_optimal_sust = res_sust.x

    E_at_conv, emisi_at_conv = energi_dan_emisi_rate(T_optimal_conv, beta, eta, e_PM, e_CM, faktor_emisi)
    E_at_sust, emisi_at_sust = energi_dan_emisi_rate(T_optimal_sust, beta, eta, e_PM, e_CM, faktor_emisi)
    cost_at_sust = cost_conventional(T_optimal_sust, beta, eta, Cp, Cf)

    pola_kegagalan = ("Wear-out (β > 1) — laju kegagalan meningkat seiring usia pakai; PM terjadwal efektif"
                       if beta > 1.05 else
                       "Random failure (β ≈ 1) — laju kegagalan relatif konstan; PM ketat mungkin kurang efektif"
                       if 0.95 <= beta <= 1.05 else
                       "Infant mortality (β < 1) — laju kegagalan menurun; fokus pada quality control produksi/instalasi")

    # =================================================================
    # TAB LAYOUT
    # =================================================================
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Ringkasan & Reliabilitas", "💰 Optimasi Interval PM",
        "🌱 Analisis Sensitivitas", "📄 Laporan & Unduh"
    ])

    # -----------------------------------------------------------------
    # TAB 1: RINGKASAN & RELIABILITAS
    # -----------------------------------------------------------------
    with tab1:
        st.markdown('<p class="section-title">Hasil Estimasi Parameter Weibull</p>', unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Shape (β)", f"{beta:.3f}")
        c2.metric("Scale (η)", f"{eta:.1f} jam")
        c3.metric("MTBF", f"{MTBF_val:.1f} jam")
        c4.metric("Availability", f"{availability*100:.2f}%")

        st.markdown(f"""
        <div class="note-box">
        <b>Interpretasi pola kegagalan:</b> {pola_kegagalan}<br>
        <b>Uji Goodness-of-Fit (Kolmogorov-Smirnov):</b> D = {D_stat:.4f}, p-value = {p_value:.4f} —
        {"data <b>sesuai</b> dengan distribusi Weibull (gagal tolak H0 pada α=0.05)" if p_value > 0.05 else
         "data <b>berpotensi tidak sesuai</b> dengan distribusi Weibull (tolak H0 pada α=0.05), pertimbangkan distribusi lain atau tambah data"}
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<p class="section-title">Kurva Reliabilitas, Hazard Rate, dan PDF</p>', unsafe_allow_html=True)

        t_plot = np.linspace(0.1, max(tbf_data.max() * 1.8, T_optimal_conv * 1.5), 300)
        R_plot = reliability(t_plot, beta, eta)
        h_plot = hazard(t_plot, beta, eta)
        f_plot = pdf_weibull(t_plot, beta, eta)

        fig1 = make_subplots(rows=1, cols=3, subplot_titles=(
            "Fungsi Reliabilitas R(t)", "Laju Kegagalan h(t)", "Distribusi Kepadatan f(t)"))

        fig1.add_trace(go.Scatter(x=t_plot, y=R_plot, mode='lines', name='R(t)',
                                   line=dict(color='#2e86ab', width=3)), row=1, col=1)
        fig1.add_trace(go.Scatter(x=t_plot, y=h_plot, mode='lines', name='h(t)',
                                   line=dict(color='#e8543c', width=3)), row=1, col=2)
        fig1.add_trace(go.Scatter(x=t_plot, y=f_plot, mode='lines', name='f(t)',
                                   line=dict(color='#2ca25f', width=3), fill='tozeroy'), row=1, col=3)

        fig1.update_layout(height=380, showlegend=False, margin=dict(t=50, b=30))
        fig1.update_xaxes(title_text="Waktu (jam)")
        st.plotly_chart(fig1, use_container_width=True)

        st.markdown('<p class="section-title">Data Waktu Antar-Kegagalan (TBF)</p>', unsafe_allow_html=True)
        col_a, col_b = st.columns([2, 1])
        with col_a:
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Histogram(x=tbf_data, nbinsx=12, marker_color='#2e86ab', opacity=0.75,
                                             name='Frekuensi data'))
            fig_hist.update_layout(height=300, xaxis_title="TBF (jam)", yaxis_title="Frekuensi",
                                    margin=dict(t=20, b=30))
            st.plotly_chart(fig_hist, use_container_width=True)
        with col_b:
            st.dataframe(pd.DataFrame({"TBF (jam)": tbf_data}), height=300, use_container_width=True)

    # -----------------------------------------------------------------
    # TAB 2: OPTIMASI INTERVAL PM
    # -----------------------------------------------------------------
    with tab2:
        st.markdown('<p class="section-title">Perbandingan Interval PM Optimal</p>', unsafe_allow_html=True)

        colx, coly = st.columns(2)
        with colx:
            st.markdown("##### 🔧 Model Konvensional (Cost-Based)")
            st.metric("Interval PM Optimal", f"{T_optimal_conv:.1f} jam")
            st.metric("Biaya Rata-rata", f"Rp {cost_rate_conv:,.0f} /jam")
        with coly:
            st.markdown("##### 🌱 Model Sustainable (Cost + Emisi, w={:.2f})".format(w))
            st.metric("Interval PM Optimal", f"{T_optimal_sust:.1f} jam",
                       delta=f"{T_optimal_sust - T_optimal_conv:+.1f} jam vs konvensional")
            st.metric("Biaya Rata-rata pada T ini", f"Rp {cost_at_sust:,.0f} /jam")

        st.markdown('<p class="section-title">Kurva Fungsi Biaya terhadap Interval T</p>', unsafe_allow_html=True)

        T_curve = np.linspace(5, T_max_search, 250)
        cost_curve = np.array([cost_conventional(t, beta, eta, Cp, Cf) for t in T_curve])
        emisi_cost_curve = np.array([biaya_emisi_rate(t, beta, eta, e_PM, e_CM, faktor_emisi, carbon_price) for t in T_curve])
        norm_curve = np.array([cost_sustainable_normalized(t, beta, eta, Cp, Cf, e_PM, e_CM, faktor_emisi,
                                                             carbon_price, w, C_min, C_max, E_min, E_max)
                                for t in T_curve])

        fig2 = make_subplots(specs=[[{"secondary_y": True}]])
        fig2.add_trace(go.Scatter(x=T_curve, y=cost_curve, name="Biaya Maintenance (Rp/jam)",
                                   line=dict(color='#2e86ab', width=2.5)), secondary_y=False)
        fig2.add_trace(go.Scatter(x=T_curve, y=emisi_cost_curve, name="Biaya Emisi (Rp/jam)",
                                   line=dict(color='#2ca25f', width=2.5, dash='dot')), secondary_y=True)
        fig2.add_vline(x=T_optimal_conv, line_dash="dash", line_color="#2e86ab",
                        annotation_text=f"T* konvensional = {T_optimal_conv:.0f} jam")
        fig2.add_vline(x=T_optimal_sust, line_dash="dash", line_color="#2ca25f",
                        annotation_text=f"T* sustainable = {T_optimal_sust:.0f} jam")
        fig2.update_layout(height=420, legend=dict(orientation="h", y=1.15), margin=dict(t=60, b=30))
        fig2.update_xaxes(title_text="Interval Maintenance T (jam)")
        fig2.update_yaxes(title_text="Biaya Maintenance (Rp/jam)", secondary_y=False)
        fig2.update_yaxes(title_text="Biaya Emisi (Rp/jam)", secondary_y=True)
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown('<p class="section-title">Skor Gabungan (Weighted-Sum, Ternormalisasi)</p>', unsafe_allow_html=True)
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=T_curve, y=norm_curve, name="Skor gabungan",
                                   line=dict(color='#7b3fa0', width=3), fill='tozeroy'))
        fig3.add_vline(x=T_optimal_sust, line_dash="dash", line_color="#7b3fa0",
                        annotation_text=f"T* optimal = {T_optimal_sust:.0f} jam")
        fig3.update_layout(height=350, margin=dict(t=30, b=30),
                            xaxis_title="Interval Maintenance T (jam)",
                            yaxis_title="Skor gabungan (0 = terbaik)")
        st.plotly_chart(fig3, use_container_width=True)

        st.markdown(f"""
        <div class="note-box">
        <b>Estimasi dampak keberlanjutan pada T* sustainable ({T_optimal_sust:.0f} jam):</b><br>
        - Konsumsi energi tambahan: {E_at_sust:.4f} kWh/jam operasi<br>
        - Estimasi emisi: {emisi_at_sust:.4f} kg CO2/jam operasi<br>
        <i>Model biaya emisi merupakan pendekatan/adaptasi yang dikembangkan berdasarkan kerangka renewal-reward,
        bukan rumus baku tunggal — nyatakan ini secara eksplisit di bagian metodologi paper.</i>
        </div>
        """, unsafe_allow_html=True)

    # -----------------------------------------------------------------
    # TAB 3: ANALISIS SENSITIVITAS
    # -----------------------------------------------------------------
    with tab3:
        st.markdown('<p class="section-title">Sensitivitas T Optimal terhadap Bobot Keberlanjutan (w)</p>',
                     unsafe_allow_html=True)

        w_range = np.linspace(0, 1, 11)
        T_vs_w = []
        for w_i in w_range:
            r = minimize_scalar(cost_sustainable_normalized, bounds=(1, T_max_search), method='bounded',
                                 args=(beta, eta, Cp, Cf, e_PM, e_CM, faktor_emisi, carbon_price, w_i,
                                       C_min, C_max, E_min, E_max))
            T_vs_w.append(r.x)

        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(x=w_range, y=T_vs_w, mode='lines+markers',
                                   line=dict(color='#e8543c', width=3), marker=dict(size=8)))
        fig4.add_vline(x=w, line_dash="dot", line_color="gray", annotation_text=f"w saat ini = {w:.2f}")
        fig4.update_layout(height=380, margin=dict(t=30, b=30),
                            xaxis_title="Bobot keberlanjutan (w)",
                            yaxis_title="Interval PM Optimal T* (jam)")
        st.plotly_chart(fig4, use_container_width=True)

        st.caption("Grafik ini menunjukkan bagaimana interval PM optimal bergeser ketika preferensi "
                   "pengambil keputusan bergeser dari fokus biaya murni (w=0) ke fokus keberlanjutan penuh (w=1). "
                   "Gunakan ini untuk pembahasan trade-off di paper Anda.")

        st.markdown('<p class="section-title">Sensitivitas terhadap Parameter Energi (e_PM, e_CM)</p>',
                     unsafe_allow_html=True)
        colp, colq = st.columns(2)
        with colp:
            variasi_pct = st.slider("Variasi parameter energi (±%)", 0, 50, 20, 5)
        with colq:
            st.write("")

        skenario = ["-{}%".format(variasi_pct), "Baseline", "+{}%".format(variasi_pct)]
        faktor = [1 - variasi_pct/100, 1.0, 1 + variasi_pct/100]
        T_sens = []
        for f in faktor:
            e_PM_s, e_CM_s = e_PM * f, e_CM * f
            Cmn, Cmx, Emn, Emx, _, _, _ = build_normalization_bounds(
                beta, eta, Cp, Cf, e_PM_s, e_CM_s, faktor_emisi, carbon_price, T_max_search)
            r = minimize_scalar(cost_sustainable_normalized, bounds=(1, T_max_search), method='bounded',
                                 args=(beta, eta, Cp, Cf, e_PM_s, e_CM_s, faktor_emisi, carbon_price, w,
                                       Cmn, Cmx, Emn, Emx))
            T_sens.append(r.x)

        df_sens = pd.DataFrame({"Skenario": skenario, "T Optimal (jam)": [f"{t:.1f}" for t in T_sens]})
        st.table(df_sens)
        st.caption("Uji sensitivitas ini penting dicantumkan di paper untuk menunjukkan kesadaran terhadap "
                   "ketidakpastian estimasi parameter energi (terutama jika data bukan hasil pengukuran langsung).")

    # -----------------------------------------------------------------
    # TAB 4: LAPORAN & UNDUH PDF
    # -----------------------------------------------------------------
    with tab4:
        st.markdown('<p class="section-title">Ringkasan Hasil Analisis</p>', unsafe_allow_html=True)

        summary_df = pd.DataFrame({
            "Parameter": ["Shape (β)", "Scale (η)", "MTBF", "Availability", "KS-test p-value",
                          "T Optimal Konvensional", "Biaya Rata-rata Konvensional",
                          "T Optimal Sustainable", "Bobot (w) digunakan",
                          "Estimasi Emisi pada T Sustainable"],
            "Nilai": [f"{beta:.4f}", f"{eta:.2f} jam", f"{MTBF_val:.2f} jam", f"{availability*100:.2f}%",
                      f"{p_value:.4f}", f"{T_optimal_conv:.1f} jam", f"Rp {cost_rate_conv:,.0f}/jam",
                      f"{T_optimal_sust:.1f} jam", f"{w:.2f}", f"{emisi_at_sust:.4f} kg CO2/jam"]
        })
        st.table(summary_df)

        st.markdown('<p class="section-title">Unduh Laporan</p>', unsafe_allow_html=True)
        st.caption("Laporan PDF berisi ringkasan parameter, hasil estimasi, grafik reliabilitas, dan "
                   "rekomendasi interval maintenance — siap dilampirkan ke laporan/paper Anda.")

        # ---------------- Generate PDF ----------------
        def generate_pdf():
            buffer = io.BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=A4,
                                     topMargin=1.5*cm, bottomMargin=1.5*cm,
                                     leftMargin=2*cm, rightMargin=2*cm)
            styles = getSampleStyleSheet()
            title_style = ParagraphStyle('TitleC', parent=styles['Title'], alignment=TA_CENTER, fontSize=15)
            sub_style = ParagraphStyle('SubC', parent=styles['Normal'], alignment=TA_CENTER,
                                        fontSize=10, textColor=colors.grey)
            heading_style = styles['Heading2']
            normal_style = styles['Normal']

            story = []
            story.append(Paragraph("Laporan Analisis Reliabilitas dan Penjadwalan", title_style))
            story.append(Paragraph("Sustainable Preventive Maintenance Berbasis Distribusi Weibull", title_style))
            story.append(Spacer(1, 6))
            story.append(Paragraph("Studi Kasus Industri Manufaktur", sub_style))
            story.append(Spacer(1, 16))

            story.append(Paragraph("1. Hasil Estimasi Parameter Weibull", heading_style))
            data_param = [["Parameter", "Nilai"]] + summary_df.values.tolist()
            t = Table(data_param, colWidths=[8*cm, 7*cm])
            t.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1e3a5f')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f2f5f8')]),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('TOPPADDING', (0, 0), (-1, -1), 5),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
            ]))
            story.append(t)
            story.append(Spacer(1, 12))

            story.append(Paragraph("2. Interpretasi Pola Kegagalan", heading_style))
            story.append(Paragraph(pola_kegagalan, normal_style))
            story.append(Spacer(1, 6))
            gof_text = ("Data historis TBF sesuai dengan distribusi Weibull berdasarkan uji Kolmogorov-Smirnov "
                        f"(D = {D_stat:.4f}, p-value = {p_value:.4f} &gt; 0.05)." if p_value > 0.05 else
                        "Data historis TBF berpotensi tidak sesuai distribusi Weibull berdasarkan uji "
                        f"Kolmogorov-Smirnov (D = {D_stat:.4f}, p-value = {p_value:.4f} &le; 0.05). "
                        "Disarankan menambah jumlah data atau mengeksplorasi distribusi lain.")
            story.append(Paragraph(gof_text, normal_style))
            story.append(Spacer(1, 12))

            story.append(Paragraph("3. Grafik Reliabilitas, Hazard Rate, dan Kurva Biaya", heading_style))

            fig1_png = fig1.to_image(format="png", width=1000, height=380, scale=2)
            story.append(RLImage(io.BytesIO(fig1_png), width=16*cm, height=6.1*cm))
            story.append(Spacer(1, 8))

            fig2_png = fig2.to_image(format="png", width=1000, height=420, scale=2)
            story.append(RLImage(io.BytesIO(fig2_png), width=16*cm, height=6.7*cm))
            story.append(PageBreak())

            story.append(Paragraph("4. Rekomendasi Interval Preventive Maintenance", heading_style))
            rekomendasi_text = f"""
            Berdasarkan model konvensional (minimasi biaya murni), interval preventive maintenance
            optimal adalah <b>{T_optimal_conv:.1f} jam</b> dengan estimasi biaya rata-rata
            Rp {cost_rate_conv:,.0f} per jam operasi.
            Setelah mempertimbangkan dimensi keberlanjutan (bobot w = {w:.2f}), interval optimal
            bergeser menjadi <b>{T_optimal_sust:.1f} jam</b>
            ({'lebih panjang' if T_optimal_sust > T_optimal_conv else 'lebih pendek' if T_optimal_sust < T_optimal_conv else 'relatif sama'}
            dibanding model konvensional), dengan estimasi emisi {emisi_at_sust:.4f} kg CO2 per jam operasi.
            """
            story.append(Paragraph(rekomendasi_text, normal_style))
            story.append(Spacer(1, 10))

            story.append(Paragraph("5. Catatan Metodologis", heading_style))
            catatan_text = """
            Model biaya konvensional menggunakan pendekatan age replacement policy (Barlow &amp; Hunter).
            Model sustainable menggabungkan biaya maintenance dan estimasi biaya emisi karbon melalui
            pendekatan weighted-sum multi-objective optimization dengan normalisasi min-max. Parameter
            energi (e_PM, e_CM) dan faktor emisi bersifat estimasi/adaptasi dan perlu diverifikasi dengan
            data primer (nameplate mesin, tagihan listrik, atau pengukuran langsung) untuk keperluan
            publikasi ilmiah.
            """
            story.append(Paragraph(catatan_text, normal_style))
            story.append(Spacer(1, 14))

            footer_text = "Laporan dihasilkan otomatis oleh Dashboard Sustainable Preventive Maintenance."
            story.append(Paragraph(footer_text, sub_style))

            doc.build(story)
            buffer.seek(0)
            return buffer

        try:
            pdf_buffer = generate_pdf()
            st.download_button(
                label="📥 Unduh Laporan PDF",
                data=pdf_buffer,
                file_name="Laporan_Sustainable_PM_Weibull.pdf",
                mime="application/pdf",
                use_container_width=True,
                type="primary"
            )
        except Exception as e:
            st.error(f"Gagal membuat PDF (pastikan 'kaleido' terinstal untuk ekspor grafik): {e}")

        st.markdown("---")
        csv_data = summary_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Unduh Ringkasan (CSV)", data=csv_data,
                            file_name="ringkasan_hasil_analisis.csv", mime="text/csv",
                            use_container_width=True)

else:
    st.warning("Periksa kembali data input TBF Anda.")

st.markdown("---")
st.caption("Dashboard ini merupakan alat bantu analisis akademik. Validitas hasil bergantung pada kualitas "
           "dan kuantitas data historis TBF serta parameter biaya/energi yang diinput. Pastikan asumsi yang "
           "digunakan dinyatakan secara eksplisit dalam laporan penelitian.")
