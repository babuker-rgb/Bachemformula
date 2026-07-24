import streamlit as st
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Hybrid AI · Tablet Optimization", layout="wide")

# ================================================================
# CONSTANTS
# ================================================================
D_MIN, D_MAX = 0.72, 0.99
TENSILE_MIN = 1.50

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

# ================================================================
# SESSION STATE
# ================================================================
if 'api' not in st.session_state:
    st.session_state.update({
        'api': 89.5, 'binder': 3.5, 'pvpp': 2.0, 'mgst': 0.5, 'mcc': 3.5,
        'moisture': 1.0, 'particle_size': 50.0, 'binder_grade_index': 0,
        'pressure': 200.0, 'speed': 20.0, 'dwell_time': 25.0,
        'friction': 0.25, 'decompression_time': 35.0, 'granule': 125.0,
        'run_optimized': False
    })

# ================================================================
# HELPER FUNCTIONS
# ================================================================
def normalize_components(api, binder, pvpp, mgst, mcc, moisture):
    comps = np.array([api, binder, pvpp, mgst, mcc, moisture], dtype=float)
    total = np.sum(comps)
    if total <= 0:
        total = 1.0
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

# ================================================================
# PINN MODEL (very simple dummy model for demonstration)
# ================================================================
class DummyModel:
    def predict(self, X):
        # Return constant predictions
        return np.ones((X.shape[0], 3)) * 0.8

model = DummyModel()
scaler = None
y_scaler = None

# ================================================================
# UI
# ================================================================
st.markdown("""
<div style="background: #0b1a33; padding:1rem; border-radius:0.5rem; text-align:center; margin-bottom:1rem;">
    <h2 style="color:#fff; margin:0;">🧬 Hybrid AI · Simplified Tablet Optimization</h2>
    <p style="color:#aabbcc; margin:0; font-size:0.85rem;">Nile Valley University, Sudan</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 📊 Formulation Parameters")
    api = st.slider("API (%)", SLIDER_API_MIN, SLIDER_API_MAX, st.session_state.api, 0.1, key="api")
    binder = st.slider("Binder (%)", SLIDER_BINDER_MIN, SLIDER_BINDER_MAX, st.session_state.binder, 0.1, key="binder")
    pvpp = st.slider("PVPP (%)", SLIDER_PVPP_MIN, SLIDER_PVPP_MAX, st.session_state.pvpp, 0.1, key="pvpp")
    mgst = st.slider("Mg-St (%)", SLIDER_MGST_MIN, SLIDER_MGST_MAX, st.session_state.mgst, 0.01, key="mgst")
    mcc = st.slider("MCC (%)", SLIDER_MCC_MIN, SLIDER_MCC_MAX, st.session_state.mcc, 0.1, key="mcc")
    moisture = st.slider("Moisture (%)", SLIDER_MOISTURE_MIN, SLIDER_MOISTURE_MAX, st.session_state.moisture, 0.1, key="moisture")
    binder_grade = st.selectbox("Binder Grade", BINDER_GRADES, index=st.session_state.binder_grade_index, key="binder_grade_select")
    st.session_state.binder_grade_index = BINDER_GRADES.index(binder_grade)
    particle_size = st.slider("Particle Size (µm)", SLIDER_PARTICLE_SIZE_MIN, SLIDER_PARTICLE_SIZE_MAX, st.session_state.particle_size, 1.0, key="particle_size")

    st.markdown("### ⚙️ Process Parameters")
    pressure = st.slider("Pressure (MPa)", SLIDER_PRESSURE_MIN, SLIDER_PRESSURE_MAX, st.session_state.pressure, 1.0, key="pressure")
    speed = st.slider("Speed (rpm)", SLIDER_SPEED_MIN, SLIDER_SPEED_MAX, st.session_state.speed, 0.5, key="speed")
    dwell_time = st.slider("Dwell Time (ms)", SLIDER_DWELL_TIME_MIN, SLIDER_DWELL_TIME_MAX, st.session_state.dwell_time, 0.5, key="dwell_time")
    friction = st.slider("Friction", SLIDER_FRICTION_MIN, SLIDER_FRICTION_MAX, st.session_state.friction, 0.01, key="friction")
    decompression_time = st.slider("Decompression Time (ms)", SLIDER_DECOMPRESSION_TIME_MIN, SLIDER_DECOMPRESSION_TIME_MAX, st.session_state.decompression_time, 1.0, key="decompression_time")
    granule = st.slider("Granule Size (µm)", SLIDER_GRANULE_MIN, SLIDER_GRANULE_MAX, st.session_state.granule, 1.0, key="granule")

    predict_btn = st.button("🔬 Predict & Optimize", use_container_width=True, type="primary")

if predict_btn:
    total = api + binder + pvpp + mgst + mcc + moisture
    if abs(total-100) > 0.5:
        st.warning(f"⚠️ Total = {total:.2f}% (should be 100%)")
    else:
        # Dummy predictions
        density = 0.89
        tensile = 5.1
        efrf = 0.37
        disintegration = 3.1

        st.markdown("### 📈 Results")
        st.markdown("**Constraint Status** (Density: 0.72–0.99, Tensile ≥ 1.50, EFRF < 0.40)")
        col1, col2, col3 = st.columns(3)
        col1.metric("Density", f"{density:.3f}", "[0.72, 0.99]")
        col2.metric("Tensile", f"{tensile:.2f} MPa", "≥ 1.50")
        col3.metric("EFRF", f"{efrf:.4f}", "< 0.40")

        if all([D_MIN <= density <= D_MAX, tensile >= TENSILE_MIN, efrf < 0.40]):
            st.success("✅ All constraints satisfied")
        else:
            st.error("❌ Constraints violated")

        st.info("ℹ️ This is a simplified demo. For full multi‑objective optimization, use the complete framework with the pre‑trained model.")

else:
    st.info("👆 Adjust parameters and click 'Predict & Optimize' to see results.")
