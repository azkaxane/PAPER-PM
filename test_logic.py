"""
Uji logika perhitungan sebelum diintegrasikan ke dashboard Streamlit.
Memastikan rumus Weibull, reliability, cost model konvensional, dan sustainable model benar.
"""
import numpy as np
from scipy import stats
from scipy.special import gamma as gamma_func
from scipy.optimize import minimize_scalar
from scipy.integrate import quad

# ------------------------------------------------------------------
# 1. Data dummy realistis (waktu antar kegagalan / TBF dalam jam)
# ------------------------------------------------------------------
np.random.seed(42)
true_beta, true_eta = 2.3, 480  # pola wear-out (beta > 1)
tbf_data = stats.weibull_min.rvs(true_beta, loc=0, scale=true_eta, size=28)
tbf_data = np.round(tbf_data, 1)
print("Data TBF (jam):", tbf_data)

# ------------------------------------------------------------------
# 2. Estimasi parameter Weibull via MLE
# ------------------------------------------------------------------
beta, loc, eta = stats.weibull_min.fit(tbf_data, floc=0)
print(f"\nEstimasi parameter -> beta (shape) = {beta:.4f}, eta (scale) = {eta:.4f}")

# ------------------------------------------------------------------
# 3. Uji goodness-of-fit (Kolmogorov-Smirnov)
# ------------------------------------------------------------------
D, p_value = stats.kstest(tbf_data, 'weibull_min', args=(beta, loc, eta))
print(f"KS test -> D = {D:.4f}, p-value = {p_value:.4f}")
print("Kesimpulan:", "Data sesuai distribusi Weibull (gagal tolak H0)" if p_value > 0.05
      else "Data TIDAK sesuai distribusi Weibull (tolak H0)")

# ------------------------------------------------------------------
# 4. Fungsi reliability, hazard, MTBF, availability
# ------------------------------------------------------------------
def reliability(t, beta, eta):
    return np.exp(-(t / eta) ** beta)

def hazard(t, beta, eta):
    return (beta / eta) * (t / eta) ** (beta - 1)

def mtbf(beta, eta):
    return eta * gamma_func(1 + 1 / beta)

MTTR = 8.0  # asumsi rata-rata waktu perbaikan (jam), idealnya dari data riil
MTBF_val = mtbf(beta, eta)
availability = MTBF_val / (MTBF_val + MTTR)
print(f"\nMTBF = {MTBF_val:.2f} jam")
print(f"Availability = {availability:.4f} ({availability*100:.2f}%)")

# sanity check reliability harus turun monoton dan berada di [0,1]
t_test = np.linspace(0, 1500, 10)
R_test = reliability(t_test, beta, eta)
assert np.all(np.diff(R_test) <= 0), "ERROR: reliability tidak monoton turun!"
assert np.all((R_test >= 0) & (R_test <= 1)), "ERROR: reliability keluar rentang [0,1]!"
print("Sanity check reliability: OK (monoton turun, dalam rentang [0,1])")

# ------------------------------------------------------------------
# 5. Model biaya konvensional (age replacement policy - Barlow & Hunter)
#    C(T) = [Cf*F(T) + Cp*R(T)] / integral_0^T R(t) dt
# ------------------------------------------------------------------
Cp = 1_500_000    # biaya preventive maintenance (Rp)
Cf = 8_000_000    # biaya corrective maintenance / kegagalan (Rp)

def cycle_length(T, beta, eta):
    val, _ = quad(reliability, 0, T, args=(beta, eta))
    return val

def cost_conventional(T, beta, eta, Cp, Cf):
    if T <= 0:
        return np.inf
    R_T = reliability(T, beta, eta)
    F_T = 1 - R_T
    denom = cycle_length(T, beta, eta)
    if denom <= 1e-9:
        return np.inf
    return (Cf * F_T + Cp * R_T) / denom

res_conv = minimize_scalar(cost_conventional, bounds=(1, 2000), method='bounded',
                            args=(beta, eta, Cp, Cf))
T_optimal_conv = res_conv.x
print(f"\n[Model Konvensional] T optimal = {T_optimal_conv:.1f} jam, "
      f"Cost rate = Rp {res_conv.fun:,.0f}/jam")

# sanity check: cost rate harus positif dan T optimal dalam rentang wajar
assert res_conv.fun > 0, "ERROR: cost rate negatif, ada kesalahan logika!"
assert 0 < T_optimal_conv < 2000, "ERROR: T optimal di luar rentang wajar!"
print("Sanity check cost model: OK")

# ------------------------------------------------------------------
# 6. Model sustainable (tambahan energi & emisi)
# ------------------------------------------------------------------
e_PM = 3.5     # kWh energi ekstra per event PM
e_CM = 12.0    # kWh energi ekstra per event CM (lebih besar krn mendadak)
faktor_emisi = 0.87   # kg CO2 / kWh (contoh, harus diganti data resmi)
carbon_price = 75_000  # Rp per ton CO2 (asumsi, harus dijustifikasi di paper)

def energi_dan_emisi(T, beta, eta, e_PM, e_CM, faktor_emisi):
    """
    Laju konsumsi energi ekstra (kWh/jam) akibat aktivitas maintenance,
    dihitung dengan kerangka renewal-reward yang SAMA dengan model biaya
    konvensional: [energi per event x probabilitas event] / expected cycle length.
    Ini memastikan E(T) dan C(T) konsisten secara struktur matematis
    (sama-sama berbentuk rate-per-unit-time dari satu siklus renewal),
    sehingga tidak muncul solusi degenerate (T -> tak hingga).
    """
    R_T = reliability(T, beta, eta)
    F_T = 1 - R_T
    denom = cycle_length(T, beta, eta)
    if denom <= 1e-9:
        return np.inf, np.inf
    E_T = (e_PM * R_T + e_CM * F_T) / denom   # kWh per jam (rate)
    emisi_T = E_T * faktor_emisi               # kg CO2 per jam (rate)
    return E_T, emisi_T

def biaya_emisi_rate(T, beta, eta, e_PM, e_CM, faktor_emisi, carbon_price):
    _, emisi_T = energi_dan_emisi(T, beta, eta, e_PM, e_CM, faktor_emisi)
    return (emisi_T / 1000) * carbon_price  # kg -> ton CO2, dikali harga karbon

# --- Normalisasi (weighted-sum method untuk multi-objective optimization) ---
# Karena skala biaya maintenance (Rp jutaan) dan biaya emisi (Rp puluhan) sangat
# berbeda, kedua objective dinormalisasi ke rentang [0,1] terlebih dahulu sebelum
# digabung. Ini adalah praktik standar dalam optimasi multi-objektif (weighted-sum
# method) agar bobot w benar-benar merepresentasikan preferensi, bukan didominasi
# oleh objective dengan skala numerik lebih besar.
T_range = np.linspace(1, 2000, 500)
C_vals = np.array([cost_conventional(t, beta, eta, Cp, Cf) for t in T_range])
E_vals = np.array([biaya_emisi_rate(t, beta, eta, e_PM, e_CM, faktor_emisi, carbon_price) for t in T_range])

C_min, C_max = C_vals.min(), C_vals.max()
E_min, E_max = E_vals.min(), E_vals.max()

def cost_sustainable_normalized(T, beta, eta, Cp, Cf, e_PM, e_CM, faktor_emisi, carbon_price, w,
                                 C_min, C_max, E_min, E_max):
    C_conv = cost_conventional(T, beta, eta, Cp, Cf)
    biaya_emisi = biaya_emisi_rate(T, beta, eta, e_PM, e_CM, faktor_emisi, carbon_price)
    C_norm = (C_conv - C_min) / (C_max - C_min + 1e-9)
    E_norm = (biaya_emisi - E_min) / (E_max - E_min + 1e-9)
    return (1 - w) * C_norm + w * E_norm

w_test = 0.8
res_sust = minimize_scalar(cost_sustainable_normalized, bounds=(1, 2000), method='bounded',
                            args=(beta, eta, Cp, Cf, e_PM, e_CM, faktor_emisi, carbon_price, w_test,
                                  C_min, C_max, E_min, E_max))
T_optimal_sust = res_sust.x
print(f"\n[Model Sustainable, w={w_test}] T optimal = {T_optimal_sust:.1f} jam (skor normalisasi = {res_sust.fun:.4f})")

# Bandingkan beberapa nilai w untuk memastikan T bergeser secara logis
print("\nUji sensitivitas terhadap bobot w:")
for w_try in [0.0, 0.25, 0.5, 0.75, 1.0]:
    r = minimize_scalar(cost_sustainable_normalized, bounds=(1, 2000), method='bounded',
                         args=(beta, eta, Cp, Cf, e_PM, e_CM, faktor_emisi, carbon_price, w_try,
                               C_min, C_max, E_min, E_max))
    print(f"  w={w_try:.2f} -> T optimal = {r.x:.1f} jam")

print("\n=== SEMUA UJI LOGIKA BERHASIL, SIAP DIINTEGRASIKAN KE DASHBOARD ===")
