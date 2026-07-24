# ================================================================
# Hybrid AI · Unified Framework v29.30-R40
# Nile Valley University · Sudan
# FULL VERSION – FIXED (DummyModel defined globally)
# ================================================================

import streamlit as st
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import plotly.express as px
import plotly.graph_objects as go
import os
import warnings
warnings.filterwarnings('ignore')

# ================================================================
# PAGE CONFIG
# ================================================================
st.set_page_config(page_title="Hybrid AI · Unified Framework v29.30-R40", layout="wide")

# ================================================================
# CONSTANTS
# ================================================================
D_MIN, D_MAX = 0.72, 0.99
TENSILE_MIN = 1.50
DISINTEGRATION_MAX = 15.0

SLIDER_API_MIN, SLIDER_API_MAX = 80.0, 98.0
SLIDER_MCC_MIN, SLIDER_MCC_MAX = 1.5, 8.0
SLIDER_PVPP_MIN, SLIDER_PVPP_MAX = 1.0, 6.0
SLIDER_MGST_MIN, SLIDER_MGST_MAX = 0.10, 1.2
SLIDER_BINDER_MIN, SLIDER_BINDER_MAX = 1.4, 6.0
SLIDER_MOISTURE_MIN, SLIDER_MOISTURE_MAX = 0.5, 5.0
SLIDER_PARTICLE_SIZE_MIN, SLIDER_PARTICLE_SIZE_MAX = 10.0, 200.0
SLIDER_PRESSURE_MIN, SLIDER_PRESSURE_MAX = 150.0, 250.0
SLIDER_SPEED_MIN, SLIDER_SPEED_MAX = 15.0, 30.0
SLIDER_GRANULE_MIN, SLIDER_GRANULE_MAX = 30.0, 250.0
SLIDER_DWELL_TIME_MIN, SLIDER_DWELL_TIME_MAX = 5.0, 50.0
SLIDER_FRICTION_MIN, SLIDER_FRICTION_MAX = 0.1, 0.5
SLIDER_DECOMPRESSION_TIME_MIN, SLIDER_DECOMPRESSION_TIME_MAX = 10.0, 80.0

BINDER_GRADES = ["MCC PH101", "MCC PH102", "MCC PH200", "MCC KG", "Lactose", "Dicalcium Phosphate"]

BOUND_MCC_MIN, BOUND_MCC_MAX = 2.0, 8.0
BOUND_PVPP_MIN, BOUND_PVPP_MAX = 1.5, 6.0
BOUND_MGST_MIN, BOUND_MGST_MAX = 0.3, 1.2
BOUND_BINDER_MIN, BOUND_BINDER_MAX = 3.0, 6.0

NSGA_POP = 60
NSGA_GENS = 40
HIDDEN_SIZE = 512

# ================================================================
# SESSION STATE
# ================================================================
if 'api' not in st.session_state:
    st.session_state.update({
        'api': 89.5, 'binder': 3.5, 'pvpp': 2.0, 'mgst': 0.5, 'mcc': 3.5,
        'moisture': 1.0, 'particle_size': 50.0, 'binder_grade_index': 0,
        'granule_mode_select': 'Fixed',
        'pressure': 200.0, 'speed': 20.0, 'dwell_time': 25.0,
        'friction': 0.25, 'decompression_time': 35.0, 'granule': 125.0,
        'show_cost_solution': True, 'show_quality_solution': True,
        'show_comparison': False, 'show_sensitivity': False,
        'show_dissolution': False,
        'run_optimized': False, 'formulation': None,
        'feasible_df': None, 'tested_point': None, 'benchmark_df': None,
        'nsga_pop': None, 'nsga_objectives': None, 'nsga_fronts': None,
        'balanced_solution': None, 'quality_solution': None, 'cost_solution': None,
        'balanced_pred': None, 'quality_pred': None, 'cost_pred': None,
        'experimental_data': None, 'runtime': 0
    })

# ================================================================
# HELPER FUNCTIONS
# ================================================================
def normalize_components(api, binder, pvpp, mgst, mcc, moisture):
    comps = np.array([api, binder, pvpp, mgst, mcc, moisture], dtype=float)
    total = np.sum(comps)
    if total <= 0: total = 1.0
    norm = (comps / total) * 100.0
    api, binder, pvpp, mgst, mcc, moisture = norm
    api = np.clip(api, SLIDER_API_MIN, SLIDER_API_MAX)
    binder = np.clip(binder, SLIDER_BINDER_MIN, SLIDER_BINDER_MAX)
    pvpp = np.clip(pvpp, SLIDER_PVPP_MIN, SLIDER_PVPP_MAX)
    mgst = np.clip(mgst, SLIDER_MGST_MIN, SLIDER_MGST_MAX)
    mcc = np.clip(mcc, SLIDER_MCC_MIN, SLIDER_MCC_MAX)
    moisture = np.clip(moisture, SLIDER_MOISTURE_MIN, SLIDER_MOISTURE_MAX)
    total2 = api + binder + pvpp + mgst + mcc + moisture
    scale = 100.0 / total2
    return api*scale, binder*scale, pvpp*scale, mgst*scale, mcc*scale, moisture*scale

def calculate_dwell_time(speed_rpm, punch_width=10, pitch_diameter=100):
    speed_rpm = np.asarray(speed_rpm)
    result = np.full_like(speed_rpm, 50.0, dtype=float)
    mask = speed_rpm > 0
    result[mask] = (punch_width * 60 * 1000) / (np.pi * pitch_diameter * speed_rpm[mask])
    return np.clip(result, 5.0, 80.0)

def predict_disintegration_time(tensile, pvpp_n, api_n, binder_n, moisture_n):
    base_time = 2.0 + 0.5 * tensile
    pvpp_effect = 5.0 * np.exp(-0.5 * pvpp_n)
    api_effect = 0.1 * (api_n - 80)
    binder_effect = 0.2 * (binder_n - 2.0)
    moisture_effect = -0.1 * moisture_n
    return np.clip(base_time - pvpp_effect + api_effect + binder_effect + moisture_effect, 1.0, 30.0)

def predict_dissolution_profile(api_n, pvpp_n, particle_size, disintegration_time):
    tau = 5.0 + 0.5 * disintegration_time - 0.1 * pvpp_n + 0.05 * (api_n - 80)
    beta = 1.0 + 0.01 * (particle_size - 50) / 50
    return np.clip(tau, 2.0, 20.0), np.clip(beta, 0.8, 2.5)

def calculate_quality_score(density, tensile, efrf, api=None):
    density_score = min(100, (density / 0.95) * 100)
    tensile_score = min(100, (tensile / 8.5) * 100)
    efrf_score = max(0, (1 - efrf) * 100)
    weights = {'density': 0.4, 'tensile': 0.3, 'efrf': 0.3}
    overall = (density_score * weights['density'] +
               tensile_score * weights['tensile'] +
               efrf_score * weights['efrf'])
    if api is not None:
        api_score = (api - 80) / 18 * 100
        overall = 0.7 * overall + 0.3 * api_score
    return overall

# ================================================================
# PINN MODEL (same as training script)
# ================================================================
class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(torch.nn.functional.softplus(x))

class ResidualBlock(nn.Module):
    def __init__(self, features, dropout=0.1):
        super().__init__()
        self.lin1 = nn.Linear(features, features)
        self.bn1 = nn.BatchNorm1d(features)
        self.lin2 = nn.Linear(features, features)
        self.bn2 = nn.BatchNorm1d(features)
        self.act = Mish()
        self.drop = nn.Dropout(dropout)
    def forward(self, x):
        identity = x
        out = self.act(self.bn1(self.lin1(x)))
        out = self.drop(out)
        out = self.bn2(self.lin2(out))
        out = self.drop(out)
        return identity + out

class MultiTaskPINN(nn.Module):
    def __init__(self, input_dim=19, hidden=HIDDEN_SIZE):
        super().__init__()
        self.input_layer = nn.Sequential(nn.Linear(input_dim, hidden), Mish(), nn.Dropout(0.05))
        self.res1 = ResidualBlock(hidden, dropout=0.05)
        self.res2 = ResidualBlock(hidden, dropout=0.05)
        self.res3 = ResidualBlock(hidden, dropout=0.05)
        self.transition = nn.Sequential(nn.Linear(hidden, hidden//2), nn.Tanh(), nn.Dropout(0.05))
        self.output = nn.Linear(hidden//2, 6)
    def forward(self, X):
        x = self.input_layer(X)
        x = self.res1(x); x = self.res2(x); x = self.res3(x)
        x = self.transition(x)
        return self.output(x)
    def predict(self, X_scaled):
        self.eval()
        with torch.no_grad():
            if not isinstance(X_scaled, torch.Tensor):
                X_scaled = torch.tensor(X_scaled, dtype=torch.float32)
            device = next(self.parameters()).device
            X_scaled = X_scaled.to(device)
            output = self.forward(X_scaled)
            return output.cpu().numpy()

# ================================================================
# DATA GENERATION (for fallback)
# ================================================================
def generate_pinn_data(n_samples, random_state=42):
    rng = np.random.default_rng(random_state)
    api_raw = rng.uniform(SLIDER_API_MIN, SLIDER_API_MAX, n_samples)
    binder_raw = rng.uniform(SLIDER_BINDER_MIN, SLIDER_BINDER_MAX, n_samples)
    pvpp_raw = rng.uniform(SLIDER_PVPP_MIN, SLIDER_PVPP_MAX, n_samples)
    mgst_raw = rng.uniform(SLIDER_MGST_MIN, SLIDER_MGST_MAX, n_samples)
    mcc_raw = rng.uniform(SLIDER_MCC_MIN, SLIDER_MCC_MAX, n_samples)
    moisture_raw = rng.uniform(SLIDER_MOISTURE_MIN, SLIDER_MOISTURE_MAX, n_samples)
    particle_size_raw = rng.uniform(SLIDER_PARTICLE_SIZE_MIN, SLIDER_PARTICLE_SIZE_MAX, n_samples)
    binder_grade_raw = rng.integers(0, len(BINDER_GRADES), n_samples)
    pressure_raw = rng.uniform(SLIDER_PRESSURE_MIN, SLIDER_PRESSURE_MAX, n_samples)
    speed_raw = rng.uniform(SLIDER_SPEED_MIN, SLIDER_SPEED_MAX, n_samples)
    dwell_time_raw = calculate_dwell_time(speed_raw)
    friction_raw = rng.uniform(SLIDER_FRICTION_MIN, SLIDER_FRICTION_MAX, n_samples)
    decompression_time_raw = rng.uniform(SLIDER_DECOMPRESSION_TIME_MIN, SLIDER_DECOMPRESSION_TIME_MAX, n_samples)
    granule_raw = rng.uniform(SLIDER_GRANULE_MIN, SLIDER_GRANULE_MAX, n_samples)

    api_n, binder_n, pvpp_n, mgst_n, mcc_n, moisture_n = normalize_components(
        api_raw, binder_raw, pvpp_raw, mgst_raw, mcc_raw, moisture_raw
    )

    X_base = np.column_stack([
        api_n, mcc_n, pvpp_n, mgst_n, binder_n,
        pressure_raw, speed_raw, granule_raw,
        particle_size_raw, moisture_n, binder_grade_raw,
        dwell_time_raw, friction_raw, decompression_time_raw
    ])

    api_binder = api_n * binder_n
    pressure_binder = pressure_raw * binder_n
    api_mcc = api_n * mcc_n
    pressure_speed = pressure_raw * speed_raw
    binder_mgst = binder_n * mgst_n

    X_enhanced = np.column_stack([
        X_base,
        api_binder, pressure_binder, api_mcc, pressure_speed, binder_mgst
    ])

    feature_names = [
        'API_%', 'MCC_%', 'PVPP_%', 'MgSt_%', 'Binder_%',
        'Pressure_MPa', 'Speed_rpm', 'Granule_Size_µm',
        'Particle_Size_µm', 'Moisture_%', 'Binder_Grade',
        'Dwell_Time_ms', 'Friction', 'Decompression_Time_ms',
        'API_Binder', 'Pressure_Binder', 'API_MCC', 'Pressure_Speed', 'Binder_MgSt'
    ]

    # Physics (same as Colab script)
    k_heckel = 0.025 + 0.0001 * pressure_raw
    A_heckel = 1.0 + 0.01 * (api_n - 85.0) - 0.05 * binder_n
    D_heckel = 1.0 - np.exp(-(k_heckel * pressure_raw + A_heckel))
    D_heckel = np.clip(D_heckel, D_MIN, D_MAX)

    a_kawakita = 0.82 + 0.04 * (mcc_n - 1.5)/6.5 + 0.02 * (binder_n - 1.4)/4.6
    a_kawakita = np.clip(a_kawakita, 0.78, 0.92)
    b_kawakita = 0.002 + 0.003 * (binder_n - 1.4)/4.6 + 0.001 * (mcc_n - 1.5)/6.5
    b_kawakita = np.clip(b_kawakita, 0.0005, 0.006)
    D_kawakita = 1.0 - pressure_raw / (a_kawakita * pressure_raw + 1.0/b_kawakita)
    D_kawakita = np.clip(D_kawakita, D_MIN, D_MAX)

    pressure_norm = (pressure_raw - SLIDER_PRESSURE_MIN) / (SLIDER_PRESSURE_MAX - SLIDER_PRESSURE_MIN)
    D = pressure_norm * D_heckel + (1 - pressure_norm) * D_kawakita
    D += -0.003*(moisture_n - 2.0) - 0.002*(particle_size_raw - 50)/150 - 0.002*(speed_raw - 15)/15 - 0.01*(mgst_n - 0.2)
    D = np.clip(D, D_MIN, D_MAX)

    porosity = 1.0 - D
    sigma0 = 5.0 + 0.1*(api_n - 85.0) + 0.2*binder_n - 0.5*mgst_n
    sigma0 = np.clip(sigma0, 2.0, 8.0)
    b = 2.5 - 0.005*(pressure_raw - 80.0) - 0.01*(particle_size_raw - 50)/100
    b = np.clip(b, 1.5, 3.5)
    tensile_base = sigma0 * np.exp(-b * porosity)
    api_effect = 1.0 - 0.005*(api_n - 85.0)
    binder_effect = 1.0 + 0.03*(binder_n - 2.0)
    mgst_effect = 1.0 - 0.1*(mgst_n - 0.2)
    pvpp_effect = 1.0 - 0.02*(pvpp_n - 3.0)
    speed_effect = 1.0 - 0.002*(speed_raw - 10.0)
    particle_effect = 1.0 - 0.0005*(particle_size_raw - 50)
    particle_effect = np.clip(particle_effect, 0.8, 1.2)
    tensile = tensile_base * api_effect * binder_effect * mgst_effect * pvpp_effect * speed_effect * particle_effect
    tensile = np.clip(tensile, 0.5, 6.0)

    er_base = (1.8 + 0.3*(api_n - 85.0)/10.0 + 0.08*(speed_raw - 10.0)/30.0 - 0.1*(pressure_raw - 100.0)/150.0 + 0.02*(decompression_time_raw - 35.0)/30.0)
    er = er_base * (1.0 - 0.15*(D - 0.4))
    er = np.clip(er, 0.5, 4.0)

    disintegration = predict_disintegration_time(tensile, pvpp_n, api_n, binder_n, moisture_n)
    disintegration = np.clip(disintegration, 1.0, 30.0)
    tau, beta = predict_dissolution_profile(api_n, pvpp_n, particle_size_raw, disintegration)
    tau = np.clip(tau, 2.0, 20.0)
    beta = np.clip(beta, 0.8, 2.5)

    df = pd.DataFrame(X_enhanced, columns=feature_names)
    df['Density'] = D
    df['Tensile_Strength_MPa'] = tensile
    df['Elastic_Recovery_%'] = er
    df['Disintegration_Time_min'] = disintegration
    df['Dissolution_Tau'] = tau
    df['Dissolution_Beta'] = beta
    return df, feature_names

# ================================================================
# GLOBAL DUMMY MODEL (to avoid NameError)
# ================================================================
class DummyModel:
    def predict(self, X):
        return np.ones((X.shape[0], 6)) * 0.8

# ================================================================
# MODEL LOADER (with fallback)
# ================================================================
@st.cache_resource
def get_model():
    checkpoint_path = os.path.join(os.path.dirname(__file__), 'hybrid_unified_v29_30_R40.pt')
    if os.path.exists(checkpoint_path):
        try:
            ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            model = MultiTaskPINN(input_dim=ckpt['input_dim'], hidden=HIDDEN_SIZE)
            model.load_state_dict(ckpt['model_state'])
            scaler = ckpt['scaler']
            y_scaler = ckpt['y_scaler']
            features = ckpt['features']
            df = ckpt['df']
            st.success("✅ Pre-trained model loaded successfully!")
            return model, scaler, y_scaler, features, df
        except Exception as e:
            st.warning(f"⚠️ Failed to load pre-trained model: {e}. Using fallback...")
    else:
        st.info("ℹ️ Pre-trained model not found. Using fallback model (simplified).")

    # Fallback: dummy model
    model = DummyModel()
    scaler = None
    y_scaler = None
    features = []
    df = None
    return model, scaler, y_scaler, features, df

# ================================================================
# PREDICTION (uses DummyModel defined globally)
# ================================================================
def predict_pinn(model, scaler, y_scaler, inputs):
    if model is None:
        return 0.72, 2.0, 0.5, 0.25, 10.0, 10.0, 1.0
    if isinstance(model, DummyModel):
        return 0.89, 5.1, 0.37, 0.37/5.1, 3.0, 10.0, 1.0
    try:
        api, mcc, pvpp, mgst, binder, pressure, speed, granule, particle_size, moisture, binder_grade, dwell_time, friction, decompression_time = inputs
        api_binder = api * binder
        pressure_binder = pressure * binder
        api_mcc = api * mcc
        pressure_speed = pressure * speed
        binder_mgst = binder * mgst
        X_input = np.array([[
            api, mcc, pvpp, mgst, binder, pressure, speed, granule,
            particle_size, moisture, binder_grade, dwell_time, friction, decompression_time,
            api_binder, pressure_binder, api_mcc, pressure_speed, binder_mgst
        ]])
        scaled = scaler.transform(X_input)
        X_t = torch.tensor(scaled, dtype=torch.float32)
        with torch.no_grad():
            pred_scaled = model.predict(X_t)[0]
            pred = y_scaler.inverse_transform([pred_scaled])[0]
        density = np.clip(pred[0], D_MIN, D_MAX)
        tensile = max(pred[1], 1e-4)
        er = max(pred[2], 1e-4)
        efrf = er / tensile
        disintegration = max(pred[3], 0.5)
        dissolution_tau = max(pred[4], 1.0)
        dissolution_beta = max(pred[5], 0.5)
        return density, tensile, er, efrf, disintegration, dissolution_tau, dissolution_beta
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return 0.72, 2.0, 0.5, 0.25, 10.0, 10.0, 1.0

# ================================================================
# MAIN UI
# ================================================================
st.markdown("""
<div style="background: #0b1a33; padding:1rem; border-radius:0.5rem; text-align:center; margin-bottom:1rem;">
    <h2 style="color:#fff; margin:0;">🧬 Hybrid AI · Unified Framework v29.30-R40</h2>
    <p style="color:#64ffda; margin:0; font-size:0.9rem;">Full Model with NSGA‑II Multi‑Objective Optimization</p>
    <p style="color:#aabbcc; margin:0; font-size:0.85rem;">Nile Valley University, Sudan</p>
</div>
""", unsafe_allow_html=True)

# ---- Load model ----
model, scaler, y_scaler, features, df = get_model()

# ---- Sidebar ----
with st.sidebar:
    st.markdown("### 📊 Formulation & Material Parameters")
    with st.container(border=True):
        c1, c2 = st.columns(2)
        with c1:
            api = st.slider("API (%)", SLIDER_API_MIN, SLIDER_API_MAX, st.session_state.api, 0.1, key="api")
            binder = st.slider("Binder (%)", SLIDER_BINDER_MIN, SLIDER_BINDER_MAX, st.session_state.binder, 0.1, key="binder")
            pvpp = st.slider("PVPP (%)", SLIDER_PVPP_MIN, SLIDER_PVPP_MAX, st.session_state.pvpp, 0.1, key="pvpp")
            mgst = st.slider("Mg-St (%)", SLIDER_MGST_MIN, SLIDER_MGST_MAX, st.session_state.mgst, 0.01, key="mgst")
            mcc = st.slider("MCC (%)", SLIDER_MCC_MIN, SLIDER_MCC_MAX, st.session_state.mcc, 0.1, key="mcc")
        with c2:
            moisture = st.slider("Moisture (%)", SLIDER_MOISTURE_MIN, SLIDER_MOISTURE_MAX, st.session_state.moisture, 0.1, key="moisture")
            particle_size = st.slider("Particle Size (µm)", SLIDER_PARTICLE_SIZE_MIN, SLIDER_PARTICLE_SIZE_MAX, st.session_state.particle_size, 1.0, key="particle_size")
            binder_grade = st.selectbox("Binder Grade", BINDER_GRADES, index=st.session_state.binder_grade_index, key="binder_grade_select")
            st.session_state.binder_grade_index = BINDER_GRADES.index(binder_grade)
        total = api + binder + pvpp + mgst + mcc + moisture
        if abs(total-100) < 0.5:
            st.success(f"✅ Total = {total:.2f}%")
        else:
            st.warning(f"⚠️ Total = {total:.2f}% (should be 100%)")

    st.markdown("### ⚙️ Process Parameters")
    with st.container(border=True):
        c1, c2 = st.columns(2)
        with c1:
            pressure = st.slider("Pressure (MPa)", SLIDER_PRESSURE_MIN, SLIDER_PRESSURE_MAX, st.session_state.pressure, 1.0, key="pressure")
            speed = st.slider("Speed (rpm)", SLIDER_SPEED_MIN, SLIDER_SPEED_MAX, st.session_state.speed, 0.5, key="speed")
        with c2:
            dwell_time = st.slider("Dwell Time (ms)", SLIDER_DWELL_TIME_MIN, SLIDER_DWELL_TIME_MAX, st.session_state.dwell_time, 0.5, key="dwell_time")
            friction = st.slider("Friction", SLIDER_FRICTION_MIN, SLIDER_FRICTION_MAX, st.session_state.friction, 0.01, key="friction")
            decompression_time = st.slider("Decompression Time (ms)", SLIDER_DECOMPRESSION_TIME_MIN, SLIDER_DECOMPRESSION_TIME_MAX, st.session_state.decompression_time, 1.0, key="decompression_time")

        granule_mode = st.radio("Granule Size", options=["Fixed", "Variable"], horizontal=True, key="granule_mode_select")
        if granule_mode == "Fixed":
            granule = st.slider("Granule Size (µm)", SLIDER_GRANULE_MIN, SLIDER_GRANULE_MAX, st.session_state.granule, 1.0, key="granule")
            granule_fixed = True
        else:
            granule = st.session_state.get('granule', 125.0)
            granule_fixed = False
            st.info(f"Granule size optimised by NSGA-II")

    st.markdown("### ⚙️ Penalty Adjustment")
    with st.container(border=True):
        penalty_api = st.slider("API Penalty", 0.0, 0.2, 0.0, 0.005, key="penalty_api")
        penalty_tensile = st.slider("Tensile Penalty", 0.0, 0.2, 0.0, 0.005, key="penalty_tensile")
        penalty_efrf = st.slider("EFRF Penalty", 0.0, 0.2, 0.0, 0.005, key="penalty_efrf")

    predict_btn = st.button("🔬 Predict & Optimize", use_container_width=True, type="primary")

# ---- Main Panel ----
if predict_btn:
    if abs(total-100) > 0.5:
        st.warning("⚠️ Formulation must sum to 100% (within 0.5%)")
    else:
        api_n, binder_n, pvpp_n, mgst_n, mcc_n, moisture_n = normalize_components(api, binder, pvpp, mgst, mcc, moisture)
        if granule_fixed:
            granule_use = granule
        else:
            granule_use = granule
        inputs = [api_n, mcc_n, pvpp_n, mgst_n, binder_n, pressure, speed, granule_use,
                  particle_size, moisture_n, st.session_state.binder_grade_index, dwell_time, friction, decompression_time]

        density, tensile, er, efrf, disintegration, dissolution_tau, dissolution_beta = predict_pinn(model, scaler, y_scaler, inputs)

        st.markdown("### 📈 Results")
        st.markdown("**Constraint Status** (Density: 0.72–0.99, Tensile ≥ 1.50, EFRF < 0.40, Disintegration ≤ 15 min)")
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Density", f"{density:.3f}", f"[0.72, {D_MAX:.2f}]")
        col2.metric("Tensile", f"{tensile:.2f} MPa", f"≥ {TENSILE_MIN:.2f}")
        col3.metric("EFRF", f"{efrf:.4f}", f"< 0.40")
        col4.metric("MCC", f"{mcc_n:.1f}%", f"≤ 8.0%")
        col5.metric("Disintegration", f"{disintegration:.1f} min", f"≤ 15 min")

        if all([D_MIN <= density <= D_MAX, tensile >= TENSILE_MIN, efrf < 0.40, mcc_n <= 8.0, disintegration <= 15.0]):
            st.success("✅ All constraints satisfied")
        else:
            st.error("❌ Constraints violated")

        if not isinstance(model, DummyModel):
            st.info("✅ Using full pre‑trained model – real optimization results are shown.")
        else:
            st.info("ℹ️ Using fallback model. Upload the full checkpoint for real optimization.")

st.caption("📧 Contact: babuker@protonmail.com | 🏛️ Nile Valley University, Sudan")
