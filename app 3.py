"""
Dashboard Interaktif: Analisis Reliabilitas dan Penjadwalan
Sustainable Preventive Maintenance Berbasis Distribusi Weibull
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
# KONFIGURASI HALAMAN & TAMPILAN
# =====================================================================
st.set_page_config(
    page_title="Dashboard Sustainable PM",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #1e3a5f; margin-bottom: 0; }
    .sub-header { font-size: 1.1rem; color: #5a6b7d; margin-top: 5px; margin-bottom: 20px; }
    .section-title { font-size: 1.3rem; font-weight: 600; color: #1e3a5f; border-left: 4px solid #2e86ab; padding-left: 10px; margin-top: 1.5rem; margin-bottom: 1rem; }
    .info-box { background-color: #e0f2fe; border-left: 4px solid #0284c7; padding: 15px; border-radius: 6px; font-size: 0.95rem; margin-bottom: 15px; }
    .warning-box { background-color: #fff8e6; border-left: 4px solid #e8a33d; padding: 15px; border-radius: 6px; font-size: 0.95rem; margin-bottom: 15px; }
    div[data-testid="stMetricValue"] { font-size: 1.6rem; color: #1e3a5f; font-weight: bold; }
</style>
""", unsafe_allow_html=True)


# =====================================================================
# FUNGSI-FUNGSI INTI MATEMATIKA (TETAP LENGKAP)
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
    if T <= 0: return np.inf
    R_T = float(reliability(T, beta, eta))
    F_T = 1 - R_T
    denom = cycle_length(T, beta, eta)
    if denom <= 1e-9: return np.inf
    return (Cf * F_T + Cp * R_T) / denom

def energi_dan_emisi_rate(T, beta, eta, e_PM, e_CM, faktor_emisi):
    R_T = float(reliability(T, beta, eta))
    F_T = 1 - R_T
    denom = cycle_length(T, beta, eta)
    if denom <= 1e-9: return np.inf, np.inf
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


# =====================================================================
# HEADER & GLOSARIUM PANDUAN AWAM
# =====================================================================
st.markdown('<p class="main-header">⚙️ Sistem Keputusan Perawatan Mesin (Sustainable PM)</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Menentukan jadwal servis optimal untuk meminimalkan biaya operasional dan menekan emisi karbon, berbasis Distribusi Weibull.</p>', unsafe_allow_html=True)

with st.expander("📖 PANDUAN PENGGUNAAN & GLOSARIUM ISTILAH (Klik untuk membuka)"):
    st.markdown("""
    <div class="info-box">
    <strong>Tujuan Dashboard:</strong> Mencari tahu kapan persisnya mesin harus dimatikan untuk servis rutin, agar perusahaan tidak rugi karena mesin rusak mendadak, sekaligus peduli pada kelestarian lingkungan.
    <hr style="margin: 10px 0;">
    <strong>Glosarium Istilah:</strong>
    <ul>
        <li><strong>TBF (Time Between Failures):</strong> Waktu (dalam jam) dari mesin menyala hingga terjadi kerusakan.</li>
        <li><strong>Shape (β) / Pola Keausan:</strong> Indikator yang menunjukkan apakah mesin makin tua makin rentan rusak (>1) atau kerusakannya acak (≈1).</li>
        <li><strong>Scale (η) / Skala Umur:</strong> Umur garansi atau waktu di mana 63.2% mesin diprediksi akan mengalami kerusakan.</li>
        <li><strong>MTBF:</strong> Rata-rata umur mesin sebelum rusak.</li>
        <li><strong>PM (Preventive Maintenance):</strong> Servis rutin terjadwal sebelum mesin rusak.</li>
        <li><strong>CM (Corrective Maintenance):</strong> Perbaikan darurat karena mesin terlanjur rusak (biasanya lebih mahal & memakan lebih banyak energi).</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

# =====================================================================
# SIDEBAR — INPUT DATA LENGKAP DENGAN TOOLTIPS & PARAMETER TAHUNAN
# =====================================================================
with st.sidebar:
    st.markdown("## 🎛️ Panel Input Data")
    st.markdown("---")

    st.markdown("#### 1. Data Historis Kerusakan (TBF)")
    input_mode = st.radio("Sumber data:", ["Gunakan data contoh", "Upload CSV", "Input manual"], help="TBF adalah durasi mesin beroperasi sebelum rusak (dalam Jam).")

    if input_mode == "Upload CSV":
        uploaded = st.file_uploader("Upload file CSV (1 kolom: TBF dalam jam)", type=["csv"])
        if uploaded is not None:
            df_upload = pd.read_csv(uploaded)
            tbf_data = df_upload.iloc[:, 0].dropna().values.astype(float)
        else:
            np.random.seed(42)
            tbf_data = np.round(stats.weibull_min.rvs(2.3, loc=0, scale=480, size=28), 1)
    elif input_mode == "Input manual":
        manual_text = st.text_area("Masukkan data TBF (pisahkan koma)", value="345.4, 775, 541, 461.4, 221.9, 221.9, 141.1, 650.4, 462.7, 525.4, 89.1, 827.9, 617.7, 257.5, 238.8, 239.8, 308.9, 422.1, 374.6, 301.9")
        try:
            tbf_data = np.array([float(x.strip()) for x in manual_text.split(",") if x.strip() != ""])
        except ValueError:
            st.error("Format data tidak valid.")
            tbf_data = np.array([345.4, 775, 541, 461.4, 221.9])
    else:
        np.random.seed(42)
        tbf_data = np.round(stats.weibull_min.rvs(2.3, loc=0, scale=480, size=28), 1)

    if len(tbf_data) < 20:
        st.warning(f"Jumlah data: {len(tbf_data)}. Idealnya minimal 20 observasi agar prediksi akurat.")
    else:
        st.success(f"Jumlah data: {len(tbf_data)} observasi (Memadai).")

    st.markdown("---")
    st.markdown("#### 2. Parameter Biaya (Finansial)")
    Cp = st.number_input("Biaya Servis Rutin / PM (Rp)", min_value=0, value=1500000, step=100000, help="Biaya ganti oli, pembersihan, dll.")
    Cf = st.number_input("Biaya Rusak Mendadak / CM (Rp)", min_value=0, value=8000000, step=100000, help="Biaya kerugian produksi, ganti sparepart berat, dll.")
    MTTR = st.number_input("Lama Waktu Perbaikan (Jam)", min_value=0.1, value=8.0, step=0.5)

    st.markdown("---")
    st.markdown("#### 3. Parameter Lingkungan (Emisi)")
    e_PM = st.number_input("Listrik dipakai saat Servis PM (kWh)", min_value=0.0, value=3.5, step=0.5)
    e_CM = st.number_input("Listrik dipakai saat Rusak CM (kWh)", min_value=0.0, value=12.0, step=0.5, help="Rusak mendadak biasanya butuh energi perbaikan lebih besar.")
    faktor_emisi = st.number_input("Faktor Emisi (kg CO2/kWh)", min_value=0.0, value=0.87, step=0.01, help="Standar emisi dari pembangkit listrik yang digunakan pabrik.")
    carbon_price = st.number_input("Harga Karbon (Rp/ton CO2)", min_value=0, value=75000, step=5000, help="Denda atau nilai ekonomi atas polusi karbon.")

    st.markdown("---")
    st.markdown("#### 4. Kebijakan Manajemen")
    w = st.slider("Fokus Kepedulian Lingkungan (Bobot w)", 0.0, 1.0, 0.5, 0.05, help="0.0 = Hanya fokus hemat uang. 1.0 = Sangat fokus tekan emisi. 0.5 = Seimbang.")
    
    st.markdown("---")
    st.markdown("#### 5. Proyeksi Bisnis Tahunan")
    jam_operasi_tahun = st.number_input("Total Jam Operasi Pabrik (1 Tahun)", min_value=1000, value=8000, step=500, help="Asumsi pabrik beroperasi berapa jam dalam setahun (misal 24 jam x 330 hari = ~8000 jam).")
    T_max_search = st.number_input("Batas Atas Pencarian Jadwal (Jam)", min_value=100, value=2000, step=100, help="Batas maksimal sistem mencari angka ideal.")


# =====================================================================
# PROSES KALKULASI UTAMA
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
    availability = MTBF_val / (MTBF_val + MTTR)

    # Optimasi Konvensional
    res_conv = minimize_scalar(cost_conventional, bounds=(1, T_max_search), method='bounded', args=(beta, eta, Cp, Cf))
    T_optimal_conv = res_conv.x
    cost_rate_conv = res_conv.fun

    # Optimasi Sustainable (Ramah Lingkungan)
    C_min, C_max, E_min, E_max, T_range_plot, C_vals_plot, E_vals_plot = build_normalization_bounds(
        beta, eta, Cp, Cf, e_PM, e_CM, faktor_emisi, carbon_price, T_max_search)

    res_sust = minimize_scalar(cost_sustainable_normalized, bounds=(1, T_max_search), method='bounded',
                                args=(beta, eta, Cp, Cf, e_PM, e_CM, faktor_emisi, carbon_price, w, C_min, C_max, E_min, E_max))
    T_optimal_sust = res_sust.x

    # Ambil nilai emisi pada titik optimal
    E_at_conv, emisi_at_conv = energi_dan_emisi_rate(T_optimal_conv, beta, eta, e_PM, e_CM, faktor_emisi)
    E_at_sust, emisi_at_sust = energi_dan_emisi_rate(T_optimal_sust, beta, eta, e_PM, e_CM, faktor_emisi)
    cost_at_sust = cost_conventional(T_optimal_sust, beta, eta, Cp, Cf)

    # Bahasa Awam untuk Pola Kegagalan
    if beta > 1.05:
        pola_kegagalan = "Keausan Seiring Waktu (Wear-out) — Semakin lama dipakai, risiko rusak membesar. Perawatan terjadwal sangat efektif."
    elif 0.95 <= beta <= 1.05:
        pola_kegagalan = "Kerusakan Acak (Random) — Mesin bisa rusak kapan saja tanpa gejala. Servis terjadwal mungkin kurang berdampak."
    else:
        pola_kegagalan = "Kerusakan Dini (Infant Mortality) — Sering rusak di awal masa pakai. Periksa instalasi atau cacat pabrik."

    # =================================================================
    # LAYOUT 4 TAB (LENGKAP DENGAN FITUR BARU)
    # =================================================================
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Profil Kesehatan Mesin", 
        "💰 Optimasi & Dampak Bisnis",
        "🌱 Analisis Sensitivitas (Skenario)", 
        "📑 Ringkasan & Cetak PDF"
    ])

    # -----------------------------------------------------------------
    # TAB 1: PROFIL & RELIABILITAS
    # -----------------------------------------------------------------
    with tab1:
        st.markdown('<p class="section-title">Hasil Prediksi Umur Mesin (Model Weibull)</p>', unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Pola Keausan (Shape/β)", f"{beta:.2f}", help="> 1 berarti mesin aus seiring berjalannya waktu")
        c2.metric("Skala Umur (Scale/η)", f"{eta:.0f} Jam", help="Waktu di mana peluang kerusakan mencapai 63.2%")
        c3.metric("Rata-rata Umur (MTBF)", f"{MTBF_val:.0f} Jam", help="Waktu rata-rata mesin beroperasi sebelum mati")
        c4.metric("Tingkat Kesiapan", f"{availability*100:.1f}%", help="Persentase waktu mesin siap digunakan")

        st.markdown(f"""
        <div class="warning-box">
        <strong>Diagnosis Pola Kerusakan:</strong> {pola_kegagalan}<br>
        <strong>Validitas Data (Uji KS):</strong> p-value = {p_value:.4f} — 
        {"Data historis Anda <b>valid dan cocok</b> diprediksi dengan model ini." if p_value > 0.05 else "Data historis <b>berpotensi kurang akurat</b>, pertimbangkan menambah jumlah data kerusakan."}
        </div>
        """, unsafe_allow_html=True)
        
        # FITUR BARU: Weibull Probability Plot (Log-Log)
        with st.expander("🔍 Lihat Plot Probabilitas Weibull (Uji Validitas Akademik)"):
            st.write("Plot *log-log* ini memvisualisasikan seberapa presisi data historis Anda (titik merah) mengikuti garis lurus Distribusi Weibull (garis putus-putus biru). Semakin menempel titik pada garis, semakin valid dan dapat dipertanggungjawabkan analisis ini secara akademis.")
            
            tbf_sorted = np.sort(tbf_data)
            n_data = len(tbf_sorted)
            # Bernard's approximation untuk Median Ranks
            F_emp = (np.arange(1, n_data + 1) - 0.3) / (n_data + 0.4)
            y_empirical = np.log(-np.log(1 - F_emp))
            x_empirical = np.log(tbf_sorted)

            # Regresi Garis Weibull
            x_line = np.linspace(min(x_empirical)*0.95, max(x_empirical)*1.05, 100)
            y_line = beta * x_line - beta * np.log(eta)

            fig_prob = go.Figure()
            fig_prob.add_trace(go.Scatter(x=x_empirical, y=y_empirical, mode='markers', name='Data Kerusakan Aktual', marker=dict(color='#e8543c', size=8)))
            fig_prob.add_trace(go.Scatter(x=x_line, y=y_line, mode='lines', name='Garis Teori Weibull', line=dict(color='#2e86ab', dash='dash')))
            fig_prob.update_layout(height=350, margin=dict(t=30, b=30), xaxis_title="ln(TBF) [Skala Waktu Logaritmik]", yaxis_title="ln(-ln(1 - F(t))) [Skala Probabilitas]")
            st.plotly_chart(fig_prob, use_container_width=True)

        st.markdown('<p class="section-title">Visualisasi Siklus Hidup Mesin</p>', unsafe_allow_html=True)
        st.caption("Tiga grafik di bawah memvisualisasikan bagaimana peluang mesin bertahan, laju kerusakan, dan kepadatan peluang rusaknya seiring waktu.")

        t_plot = np.linspace(0.1, max(tbf_data.max() * 1.5, T_optimal_conv * 1.5), 300)
        
        fig1 = make_subplots(rows=1, cols=3, subplot_titles=(
            "1. Peluang Bertahan Hidup", "2. Kecepatan Laju Kerusakan", "3. Titik Rawan Kerusakan"))

        fig1.add_trace(go.Scatter(x=t_plot, y=reliability(t_plot, beta, eta), mode='lines', name='Peluang Bertahan', line=dict(color='#2e86ab', width=3)), row=1, col=1)
        fig1.add_trace(go.Scatter(x=t_plot, y=hazard(t_plot, beta, eta), mode='lines', name='Laju Kerusakan', line=dict(color='#e8543c', width=3)), row=1, col=2)
        fig1.add_trace(go.Scatter(x=t_plot, y=pdf_weibull(t_plot, beta, eta), mode='lines', name='Kepadatan Rusak', line=dict(color='#2ca25f', width=3), fill='tozeroy'), row=1, col=3)

        fig1.update_layout(height=380, showlegend=False, margin=dict(t=50, b=30))
        fig1.update_xaxes(title_text="Waktu Menyala (Jam)")
        st.plotly_chart(fig1, use_container_width=True)

        st.markdown('<p class="section-title">Data Waktu Historis Kerusakan</p>', unsafe_allow_html=True)
        col_a, col_b = st.columns([2, 1])
        with col_a:
            fig_hist = go.Figure(data=[go.Histogram(x=tbf_data, nbinsx=12, marker_color='#2e86ab', opacity=0.75)])
            fig_hist.update_layout(height=300, xaxis_title="Umur Mesin (Jam)", yaxis_title="Jumlah Kejadian", margin=dict(t=20, b=30))
            st.plotly_chart(fig_hist, use_container_width=True)
        with col_b:
            st.dataframe(pd.DataFrame({"Riwayat Kerusakan (Jam)": tbf_data}), height=300, use_container_width=True)

    # -----------------------------------------------------------------
    # TAB 2: OPTIMASI & DAMPAK BISNIS
    # -----------------------------------------------------------------
    with tab2:
        st.markdown('<p class="section-title">Perbandingan Strategi Penjadwalan Servis</p>', unsafe_allow_html=True)

        colx, coly = st.columns(2)
        with colx:
            st.markdown("##### 🏢 Strategi Lama (Hanya Fokus Finansial)")
            st.metric("Matikan Mesin Setiap", f"{T_optimal_conv:.0f} Jam")
            st.metric("Estimasi Biaya Operasional", f"Rp {cost_rate_conv:,.0f} / Jam")
        with coly:
            st.markdown(f"##### 🌳 Strategi Baru (Seimbang Uang & Lingkungan, Bobot {w:.2f})")
            st.metric("Matikan Mesin Setiap", f"{T_optimal_sust:.0f} Jam", delta=f"{T_optimal_sust - T_optimal_conv:+.0f} Jam (Perubahan Jadwal)", delta_color="off")
            st.metric("Estimasi Biaya Operasional", f"Rp {cost_at_sust:,.0f} / Jam")

        # FITUR BARU: Dampak Bisnis Tahunan & Analogi Lingkungan
        st.markdown('<p class="section-title">Proyeksi Tahunan & Dampak Bisnis</p>', unsafe_allow_html=True)
        
        annual_cost_conv = cost_rate_conv * jam_operasi_tahun
        annual_cost_sust = cost_at_sust * jam_operasi_tahun
        savings_rp = annual_cost_conv - annual_cost_sust
        
        annual_emisi_conv = emisi_at_conv * jam_operasi_tahun
        annual_emisi_sust = emisi_at_sust * jam_operasi_tahun
        emisi_saved = annual_emisi_conv - annual_emisi_sust
        pohon_setara = max(0, emisi_saved / 22.0) # 1 Pohon setara 22kg CO2/Tahun
        
        kebutuhan_sparepart = int(np.ceil(jam_operasi_tahun / T_optimal_sust))

        c1_b, c2_b, c3_b = st.columns(3)
        c1_b.metric("Total Biaya Tahunan (Baru)", f"Rp {annual_cost_sust/1e6:,.1f} Juta", delta=f"{savings_rp/1e6:+.1f} Juta vs Jadwal Lama", delta_color="inverse")
        c2_b.metric("Kebutuhan Suku Cadang (Forecast)", f"{kebutuhan_sparepart} Unit / Tahun", help=f"Dihitung dari {jam_operasi_tahun} jam beroperasi dibagi siklus servis {T_optimal_sust:.0f} jam.")
        
        if emisi_saved > 0:
            c3_b.metric("Reduksi Emisi Karbon Tahunan", f"{emisi_saved:,.0f} kg CO2", help=f"Telah menghemat emisi dibandingkan metode lama.")
        else:
            c3_b.metric("Total Emisi Tahunan", f"{annual_emisi_sust:,.0f} kg CO2", help="Jadwal baru berpotensi menghasilkan emisi sedikit lebih tinggi untuk mengompensasi efisiensi biaya yang masif.")
            
        st.markdown(f"""
        <div class="info-box">
        <strong>🌳 Analogi Jejak Karbon:</strong> Dengan menerapkan jadwal baru ini (setiap <b>{T_optimal_sust:.0f} jam</b>), total jejak emisi karbon pabrik dalam setahun diproyeksikan sebesar <b>{annual_emisi_sust:,.0f} kg CO2</b>. 
        {"Pengurangan/penghematan emisi dibandingkan jadwal lama ini setara dengan daya serap <b>" + str(int(pohon_setara)) + " batang pohon dewasa</b> dalam setahun penuh!" if emisi_saved > 0 else ""}
        <br><br>
        <strong>📦 Saran Logistik (Inventory):</strong> Pastikan bagian Gudang menyiapkan setidaknya <b>{kebutuhan_sparepart} paket suku cadang</b> (spare part PM) per tahun agar operasional tidak terhenti akibat <i>out-of-stock</i>.
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<p class="section-title">Kurva Pencarian Jadwal Termurah (Biaya vs Emisi)</p>', unsafe_allow_html=True)
        st.caption("Grafik ini menunjukkan pergerakan biaya servis dan biaya emisi. Garis putus-putus vertikal adalah titik rekomendasi terbaik.")

        T_curve = np.linspace(10, T_max_search, 250)
        cost_curve = np.array([cost_conventional(t, beta, eta, Cp, Cf) for t in T_curve])
        emisi_cost_curve = np.array([biaya_emisi_rate(t, beta, eta, e_PM, e_CM, faktor_emisi, carbon_price) for t in T_curve])
        norm_curve = np.array([cost_sustainable_normalized(t, beta, eta, Cp, Cf, e_PM, e_CM, faktor_emisi, carbon_price, w, C_min, C_max, E_min, E_max) for t in T_curve])

        fig2 = make_subplots(specs=[[{"secondary_y": True}]])
        fig2.add_trace(go.Scatter(x=T_curve, y=cost_curve, name="Biaya Operasional (Rp)", line=dict(color='#2e86ab', width=3)), secondary_y=False)
        fig2.add_trace(go.Scatter(x=T_curve, y=emisi_cost_curve, name="Biaya Polusi Emisi (Rp)", line=dict(color='#2ca25f', width=3, dash='dot')), secondary_y=True)
        fig2.add_vline(x=T_optimal_conv, line_dash="dash", line_color="#2e86ab", annotation_text=f"Jadwal Lama: {T_optimal_conv:.0f} Jam")
        fig2.add_vline(x=T_optimal_sust, line_dash="dash", line_color="#2ca25f", annotation_text=f"Jadwal Baru: {T_optimal_sust:.0f} Jam")
        
        fig2.update_layout(height=420, legend=dict(orientation="h", y=1.15), margin=dict(t=60, b=30), yaxis_range=[min(cost_curve)*0.9, min(cost_curve)*3])
        fig2.update_xaxes(title_text="Pilihan Jadwal Servis (Jam)")
        fig2.update_yaxes(title_text="Biaya Operasional (Rp/jam)", secondary_y=False)
        fig2.update_yaxes(title_text="Biaya Emisi Karbon (Rp/jam)", secondary_y=True)
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown('<p class="section-title">Skor Gabungan (0 Berarti Paling Sempurna)</p>', unsafe_allow_html=True)
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=T_curve, y=norm_curve, name="Skor Penilaian Gabungan", line=dict(color='#7b3fa0', width=3), fill='tozeroy'))
        fig3.add_vline(x=T_optimal_sust, line_dash="dash", line_color="#7b3fa0", annotation_text=f"Jadwal Paling Sempurna = {T_optimal_sust:.0f} Jam")
        fig3.update_layout(height=350, margin=dict(t=30, b=30), xaxis_title="Pilihan Jadwal Servis (Jam)", yaxis_title="Skor Penilaian (Makin kecil makin baik)")
        st.plotly_chart(fig3, use_container_width=True)

    # -----------------------------------------------------------------
    # TAB 3: ANALISIS SENSITIVITAS (SKENARIO)
    # -----------------------------------------------------------------
    with tab3:
        st.markdown('<p class="section-title">Bagaimana Jika Kebijakan Manajemen Berubah?</p>', unsafe_allow_html=True)
        st.caption("Uji sensitivitas ini menunjukkan pergeseran jadwal servis jika pimpinan mengubah bobot fokus dari 0 (hanya peduli uang) hingga 1 (hanya peduli lingkungan).")

        w_range = np.linspace(0, 1, 11)
        T_vs_w = []
        for w_i in w_range:
            r = minimize_scalar(cost_sustainable_normalized, bounds=(1, T_max_search), method='bounded',
                                 args=(beta, eta, Cp, Cf, e_PM, e_CM, faktor_emisi, carbon_price, w_i, C_min, C_max, E_min, E_max))
            T_vs_w.append(r.x)

        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(x=w_range, y=T_vs_w, mode='lines+markers', line=dict(color='#e8543c', width=3), marker=dict(size=8)))
        fig4.add_vline(x=w, line_dash="dot", line_color="gray", annotation_text=f"Pilihan Kebijakan Saat Ini (Bobot {w:.2f})")
        fig4.update_layout(height=380, margin=dict(t=30, b=30), xaxis_title="Bobot Kepedulian Lingkungan (0 sampai 1)", yaxis_title="Rekomendasi Jadwal (Jam)")
        st.plotly_chart(fig4, use_container_width=True)

        st.markdown('<p class="section-title">Bagaimana Jika Penggunaan Listrik Mesin Berubah?</p>', unsafe_allow_html=True)
        st.caption("Karena data energi kadang berupa estimasi, tabel ini mensimulasikan apa yang terjadi jika asumsi konsumsi listrik ternyata lebih tinggi atau lebih rendah dari dugaan kita.")
        
        colp, colq = st.columns(2)
        with colp:
            variasi_pct = st.slider("Coba geser margin kesalahan estimasi listrik (±%)", 0, 50, 20, 5)

        skenario = [f"Lebih Hemat {variasi_pct}%", "Kondisi Saat Ini", f"Lebih Boros {variasi_pct}%"]
        faktor = [1 - variasi_pct/100, 1.0, 1 + variasi_pct/100]
        T_sens = []
        for f in faktor:
            e_PM_s, e_CM_s = e_PM * f, e_CM * f
            Cmn, Cmx, Emn, Emx, _, _, _ = build_normalization_bounds(beta, eta, Cp, Cf, e_PM_s, e_CM_s, faktor_emisi, carbon_price, T_max_search)
            r = minimize_scalar(cost_sustainable_normalized, bounds=(1, T_max_search), method='bounded',
                                 args=(beta, eta, Cp, Cf, e_PM_s, e_CM_s, faktor_emisi, carbon_price, w, Cmn, Cmx, Emn, Emx))
            T_sens.append(r.x)

        df_sens = pd.DataFrame({"Kondisi Kelistrikan": skenario, "Rekomendasi Jadwal Baru (Jam)": [f"{t:.1f}" for t in T_sens]})
        st.table(df_sens)

    # -----------------------------------------------------------------
    # TAB 4: LAPORAN & PDF
    # -----------------------------------------------------------------
    with tab4:
        st.markdown('<p class="section-title">Ringkasan Eksekutif</p>', unsafe_allow_html=True)

        summary_df = pd.DataFrame({
            "Indikator Kinerja": ["Pola Keausan (β)", "Rata-rata Umur Mesin", "Tingkat Kesiapan Mesin", "Status Validitas Data", 
                          "Rekomendasi Jadwal Lama", "Estimasi Biaya Jadwal Lama", 
                          "Rekomendasi Jadwal Baru", "Estimasi Emisi Jadwal Baru",
                          "Proyeksi Total Biaya per Tahun", "Kebutuhan Suku Cadang Tahunan"],
            "Hasil Kalkulasi": [f"{beta:.2f} ({pola_kegagalan.split('—')[0]})", f"{MTBF_val:.0f} Jam", f"{availability*100:.1f}%",
                      "Sangat Valid" if p_value > 0.05 else "Kurang Valid", f"{T_optimal_conv:.0f} Jam", f"Rp {cost_rate_conv:,.0f} / Jam",
                      f"{T_optimal_sust:.0f} Jam", f"{emisi_at_sust:.4f} kg CO2 / Jam",
                      f"Rp {annual_cost_sust:,.0f}", f"{kebutuhan_sparepart} Unit"]
        })
        st.table(summary_df)

        st.markdown('<p class="section-title">Cetak Dokumen Resmi</p>', unsafe_allow_html=True)
        st.caption("Cetak laporan PDF lengkap berisi parameter teknis, kurva, proyeksi bisnis, dan rekomendasi keputusan untuk diserahkan ke manajemen atau lampiran akademik.")

        # ---------------- Generate PDF (Diperbarui dengan Dampak Bisnis) ----------------
        def generate_pdf():
            buffer = io.BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=1.5*cm, bottomMargin=1.5*cm, leftMargin=2*cm, rightMargin=2*cm)
            styles = getSampleStyleSheet()
            title_style = ParagraphStyle('TitleC', parent=styles['Title'], alignment=TA_CENTER, fontSize=15)
            sub_style = ParagraphStyle('SubC', parent=styles['Normal'], alignment=TA_CENTER, fontSize=10, textColor=colors.grey)
            heading_style = styles['Heading2']
            normal_style = styles['Normal']

            story = []
            story.append(Paragraph("Laporan Analisis Keputusan Penjadwalan Mesin", title_style))
            story.append(Paragraph("Sustainable Preventive Maintenance Berbasis Weibull", title_style))
            story.append(Spacer(1, 16))

            story.append(Paragraph("1. Ringkasan Kinerja Mesin & Prediksi", heading_style))
            data_param = [["Indikator", "Nilai"]] + summary_df.values.tolist()
            t = Table(data_param, colWidths=[8*cm, 8*cm])
            t.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1e3a5f')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f2f5f8')]),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
            ]))
            story.append(t)
            story.append(Spacer(1, 12))

            story.append(Paragraph("2. Kesimpulan Jadwal dan Dampak Bisnis", heading_style))
            rekomendasi_text = f"""
            Jika pabrik hanya berfokus pada penghematan finansial (pendekatan konvensional), mesin direkomendasikan untuk dihentikan dan diservis setiap <b>{T_optimal_conv:.0f} jam</b>. 
            Namun, setelah menyeimbangkan faktor finansial dan komitmen lingkungan (dengan bobot {w:.2f}), jadwal paling optimal berubah menjadi <b>{T_optimal_sust:.0f} jam</b>. 
            <br/><br/>
            Berdasarkan proyeksi {jam_operasi_tahun} jam operasi per tahun, jadwal baru ini diperkirakan menelan biaya operasional sebesar Rp {annual_cost_sust:,.0f} per tahun, dengan total estimasi jejak karbon mencapai {annual_emisi_sust:,.0f} kg CO2. 
            Dari sisi logistik, pihak gudang direkomendasikan untuk menyetok sekurang-kurangnya {kebutuhan_sparepart} unit suku cadang perawatan dalam setahun.
            """
            story.append(Paragraph(rekomendasi_text, normal_style))
            story.append(Spacer(1, 12))

            story.append(Paragraph("3. Visualisasi Kurva Probabilitas", heading_style))
            fig1_png = fig1.to_image(format="png", width=1000, height=380, scale=2)
            story.append(RLImage(io.BytesIO(fig1_png), width=16*cm, height=6.1*cm))
            story.append(Spacer(1, 8))

            fig2_png = fig2.to_image(format="png", width=1000, height=420, scale=2)
            story.append(RLImage(io.BytesIO(fig2_png), width=16*cm, height=6.7*cm))
            story.append(PageBreak())

            story.append(Paragraph("Dokumen dicetak secara otomatis dari Dashboard Sustainable PM.", sub_style))
            doc.build(story)
            buffer.seek(0)
            return buffer

        try:
            pdf_buffer = generate_pdf()
            st.download_button(
                label="📥 Unduh Laporan Resmi (PDF)",
                data=pdf_buffer,
                file_name="Laporan_Keputusan_Sustainable_PM.pdf",
                mime="application/pdf",
                use_container_width=True,
                type="primary"
            )
        except Exception as e:
            st.error(f"Gagal membuat PDF. Pastikan library 'kaleido' telah terinstal di komputer Anda: {e}")

        st.markdown("---")
        csv_data = summary_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Ekspor Tabel ke Excel (CSV)", data=csv_data, file_name="ringkasan_keputusan.csv", mime="text/csv", use_container_width=True)

else:
    st.error("Silakan unggah data kerusakan (TBF) yang valid di panel sebelah kiri.")