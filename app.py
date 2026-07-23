# ================================================================
# Hybrid AI · Unified Framework v29.30-R40
# Nile Valley University · Sudan
# ================================================================

import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import plotly.graph_objects as go
import pandas as pd

st.set_page_config(page_title="Hybrid AI · Unified Optimization", layout="wide")

# -----------------------------
# إعداد العقوبات الديناميكية (من Co‑HYBAI)
# -----------------------------
st.sidebar.markdown("## 🧬 Hybrid AI Framework – Unified")
penalty_api = st.sidebar.slider("قوة عقوبة API", 0.0, 0.2, 0.08, 0.01)
penalty_tensile = st.sidebar.slider("قوة عقوبة Tensile", 0.0, 0.2, 0.05, 0.01)

# -----------------------------
# إدخال المعاملات (من PLEX‑HYBAI)
# -----------------------------
st.markdown("## 🧪 Formulation Parameters")
col1, col2 = st.columns(2)
with col1:
    api = st.slider("API Content (%)", 80.0, 98.0, 90.5)
    binder = st.slider("Binder (%)", 1.4, 6.0, 3.5)
    pvpp = st.slider("PVPP (%)", 1.0, 6.0, 2.0)
    mgst = st.slider("MgSt (%)", 0.1, 1.2, 0.5)
with col2:
    mcc = st.slider("MCC (%)", 1.5, 8.0, 3.5)
    moisture = st.slider("Moisture (%)", 0.5, 5.0, 1.0)
    particle_size = st.slider("Particle Size (µm)", 10.0, 200.0, 50.0)
    binder_grade = st.selectbox("Binder Grade", ["MCC PH101", "MCC PH102", "MCC PH200", "MCC KG", "Lactose", "Dicalcium Phosphate"])

# -----------------------------
# Mass Balance
# -----------------------------
total = api + binder + pvpp + mgst + mcc + moisture
st.metric("Total", f"{total:.1f}%", "✅ Mass Balance" if abs(total-100)<0.5 else "⚠️ Check")

# -----------------------------
# نموذج PINN (مبسط للعرض – في الإنتاج يستخدم النموذج المدرب)
# -----------------------------
# هنا يمكنك تحميل النموذج المدرب من ملف checkpoint
# نستخدم قيماً افتراضية للعرض
density = 0.85 + 0.05 * np.random.random()
tensile = 2.0 + 0.8 * np.random.random()
efrf = 0.25 + 0.15 * np.random.random()
disintegration = 8.0 + 4.0 * np.random.random()
quality_score = (density/0.95*40 + tensile/8.5*30 + (1-efrf)*30)

# -----------------------------
# نتائج التحسين
# -----------------------------
st.markdown("## 📊 Optimization Results")
c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Density", f"{density:.3f}", "✅ ≥0.80")
    st.metric("API%", f"{api:.1f}%", "🎯 Maximize")
with c2:
    st.metric("Tensile", f"{tensile:.2f} MPa", "✅ ≥1.5")
    st.metric("EFRF", f"{efrf:.3f}", "✅ <0.40")
with c3:
    st.metric("Disintegration", f"{disintegration:.1f} min", "≤ 15 min")
    st.metric("Quality Score", f"{quality_score:.1f}%", "Good" if quality_score>60 else "Needs improvement")

# -----------------------------
# Pareto Front Explorer (بيانات محاكاة)
# -----------------------------
st.markdown("## 🌐 Pareto Front Explorer")
gen_slider = st.slider("Select Generation", 1, 80, 80)
# بيانات محاكاة لـ Pareto Front
np.random.seed(gen_slider)
api_vals = np.random.uniform(80, 98, 50)
density_vals = np.random.uniform(0.75, 0.95, 50)
tensile_vals = np.random.uniform(1.0, 2.0, 50)
efrf_vals = np.random.uniform(0.15, 0.35, 50)

fig = go.Figure(data=[go.Scatter3d(
    x=density_vals, y=tensile_vals, z=efrf_vals,
    mode='markers',
    marker=dict(size=6, color=api_vals, colorscale='Viridis', showscale=True,
                colorbar=dict(title="API%")),
    text=[f"API: {a:.1f}%" for a in api_vals],
    hovertemplate='Density: %{x:.3f}<br>Tensile: %{y:.2f}<br>EFRF: %{z:.3f}<br>%{text}<extra></extra>'
)])
fig.update_layout(scene=dict(
    xaxis_title='Density', yaxis_title='Tensile', zaxis_title='EFRF',
    camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
), height=500)
st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Golden Solution
# -----------------------------
st.markdown("## 🏆 Golden Solution (Balanced Trade‑off)")
st.success("""
**API:** 84.5% | **Binder:** 5.5% | **PVPP:** 4.1% | **MgSt:** 0.25% | **MCC:** 4.4% | **Moisture:** 1.3%  
**Density:** 0.946 ✅ | **Tensile:** 1.38 MPa ⚠️ | **EFRF:** 0.241 ✅ | **Quality Score:** 67.5% 🏆
""")

# -----------------------------
# Side-by-Side Comparison
# -----------------------------
st.markdown("## 📊 Side‑by‑Side Comparison")
solutions = pd.DataFrame([
    {"Solution":"S2 (Golden)","API":84.5,"Binder":5.5,"PVPP":4.1,"MgSt":0.25,"MCC":4.4,"Moisture":1.3,"Density":0.946,"Tensile":1.38,"EFRF":0.241,"Quality":67.5},
    {"Solution":"S1","API":85.4,"Binder":5.0,"PVPP":2.1,"MgSt":0.79,"MCC":3.0,"Moisture":3.7,"Density":0.945,"Tensile":1.33,"EFRF":0.279,"Quality":66.1},
    {"Solution":"S5","API":87.7,"Binder":3.0,"PVPP":2.5,"MgSt":0.59,"MCC":4.8,"Moisture":1.5,"Density":0.940,"Tensile":1.20,"EFRF":0.262,"Quality":65.9}
])
st.dataframe(solutions, use_container_width=True)

st.markdown("### 🎯 Performance Radar")
categories = ["Density", "Tensile (MPa)", "EFRF (inverted)", "Quality Score"]
fig_radar = go.Figure()
for _, row in solutions.iterrows():
    fig_radar.add_trace(go.Scatterpolar(
        r=[row["Density"], row["Tensile"]/8.5, 1-row["EFRF"], row["Quality"]/100],
        theta=categories,
        fill='toself',
        name=row["Solution"]
    ))
fig_radar.update_layout(
    polar=dict(radialaxis=dict(visible=True, range=[0,1])),
    showlegend=True,
    height=400,
    margin=dict(l=40, r=40, t=40, b=40)
)
st.plotly_chart(fig_radar, use_container_width=True)
