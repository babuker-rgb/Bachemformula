# ================================================================
# Hybrid AI v31.1-FastPhysics-2D · Fixed UI Load Order
# ================================================================

import streamlit as st
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import plotly.graph_objects as go
import time
import warnings
import json
import os
import tempfile
from datetime import datetime

warnings.filterwarnings('ignore')

# ================================================================
# CONFIG
# ================================================================
st.set_page_config(page_title="Hybrid AI v31.1-FastPhysics", page_icon="🧬", layout="wide")

API_MIN, API_MAX = 80.0, 98.0
BINDER_MIN, BINDER_MAX = 1.4, 6.0
PVPP_MIN, PVPP_MAX = 1.0, 6.0
MGST_MIN, MGST_MAX = 0.10, 1.2
MCC_MIN, MCC_MAX = 1.5, 8.0
MOISTURE_MIN, MOISTURE_MAX = 0.5, 5.0
PRESSURE_MIN, PRESSURE_MAX = 150.0, 250.0
SPEED_MIN, SPEED_MAX = 15.0, 30.0
EFRF_THRESHOLD = 0.40

SAMPLES = 20000
EPOCHS = 1200
GENERATIONS = 60
POP_SIZE = 120

# ================================================================
# DATA GENERATION
# ================================================================
def generate_synthetic_data(n_samples=SAMPLES, seed=42):
    rng = np.random.default_rng(seed)
    api = rng.uniform(API_MIN, API_MAX, n_samples)
    binder = rng.uniform(BINDER_MIN, BINDER_MAX, n_samples)
    pvpp = rng.uniform(PVPP_MIN, PVPP_MAX, n_samples)
    mgst = rng.uniform(MGST_MIN, MGST_MAX, n_samples)
    mcc = rng.uniform(MCC_MIN, MCC_MAX, n_samples)
    moisture = rng.uniform(MOISTURE_MIN, MOISTURE_MAX, n_samples)
    pressure = rng.uniform(PRESSURE_MIN, PRESSURE_MAX, n_samples)
    speed = rng.uniform(SPEED_MIN, SPEED_MAX, n_samples)

    X = np.column_stack([api, binder, pvpp, mgst, mcc, moisture, pressure, speed]).astype(np.float32)
    density = np.clip(0.55 + 0.3 * (pressure-150)/100 - 0.01*(binder-3.0) + rng.normal(0,0.01,n_samples), 0.55, 0.95)
    tensile = np.clip(1.0 + 6.0*(density-0.55) + 0.2*(api-80)/18 - 0.5*(mgst-0.1) + rng.normal(0,0.2,n_samples), 0.5, 8.5)
    efrf = np.clip(0.6 - 0.5*(density-0.55) + 0.2*(mgst-0.1) + rng.normal(0,0.05,n_samples), 0.02, 0.98)
    disintegration = np.clip(10.0 - 2.0*(pvpp-1.0)/5.0 + 3.0*(binder-1.4)/4.6 + rng.normal(0,1.0,n_samples), 2.0, 45.0)
    dissolution = np.clip(2.0*disintegration + 10.0 + rng.normal(0,2.0,n_samples), 10.0, 90.0)
    y = np.column_stack([density, tensile, efrf, disintegration, dissolution]).astype(np.float32)
    return X, y

# ================================================================
# SCALER & PINN MODEL
# ================================================================
class InputScaler:
    def fit(self, X): 
        self.mean_ = X.mean(0)
        self.std_ = X.std(0)
        self.std_[self.std_ < 1e-8] = 1
        return self
    def transform(self, X): 
        return (X - self.mean_) / self.std_

class HybridTabletModel(nn.Module):
    def __init__(self, input_dim=8, hidden_dim=512):
        super().__init__()
        self.fc1, self.bn1 = nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim)
        self.fc2, self.bn2 = nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim)
        self.fc3, self.bn3 = nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim)
        self.fc4, self.bn4 = nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim)
        self.fc5, self.dropout = nn.Linear(hidden_dim, 5), nn.Dropout(0.1)
        for m in self.modules():
            if isinstance(m, nn.Linear): nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x):
        h1 = torch.relu(self.bn1(self.fc1(x))); h1 = self.dropout(h1)
        h2 = torch.relu(self.bn2(self.fc2(h1)))+h1; h2 = self.dropout(h2)
        h3 = torch.relu(self.bn3(self.fc3(h2)))+h2; h3 = self.dropout(h3)
        out = self.fc5(h3)
        density = torch.sigmoid(out[:,0])*0.4+0.55
        tensile = torch.sigmoid(out[:,1])*8.0+0.5
        efrf = torch.sigmoid(out[:,2])
        disintegration = torch.sigmoid(out[:,3])*45.0+2.0
        dissolution = torch.sigmoid(out[:,4])*80.0+10.0
        return torch.stack([density, tensile, efrf, disintegration, dissolution], 1)

    def predict_with_uncertainty(self, x, n_samples=20):
        self.train()
        with torch.no_grad():
            if not torch.is_tensor(x): x = torch.tensor(x, dtype=torch.float32)
            x_repeat = x.repeat(n_samples, 1)
            preds = self.forward(x_repeat).numpy().reshape(n_samples, -1, 5)
        self.eval()
        return np.mean(preds, 0), np.std(preds, 0)

# ================================================================
# PHYSICS CONSTRAINT
# ================================================================
def calculate_heckel_density(pressure, binder):
    return 0.55 + 0.3 * (pressure - 150) / 100 - 0.01 * (binder - 3.0)

# ================================================================
# HEAVY TRAINING LOOP (CACHED)
# ================================================================
CHECKPOINT_PATH = os.path.join(tempfile.gettempdir(), 'hybrid_ai_fastphysics_2d.pt')

@st.cache_resource(show_spinner=False)
def train_model():
    # Check for cached model
    if os.path.exists(CHECKPOINT_PATH):
        try:
            ckpt = torch.load(CHECKPOINT_PATH, map_location='cpu', weights_only=False)
            model = HybridTabletModel(input_dim=8, hidden_dim=512)
            model.load_state_dict(ckpt['model_state'])
            model.eval()
            return model, ckpt['scaler']
        except:
            pass

    # Train from scratch if no cache
    X, y = generate_synthetic_data()
    scaler = InputScaler().fit(X)
    X_scaled = scaler.transform(X)
    X_t, y_t = torch.tensor(X_scaled, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
    
    model = HybridTabletModel(input_dim=8, hidden_dim=512)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    target_var = torch.clamp(y_t.var(0, unbiased=False), min=1e-6)
    def mse(pred, true): return (((pred - true) ** 2) / target_var).mean()
    
    pressure_input, binder_input = X[:, 6], X[:, 1]
    best_loss = np.inf
    patience = 0
    
    for epoch in range(EPOCHS):
        model.train()
        opt.zero_grad()
        pred = model(X_t)
        loss = mse(pred, y_t)
        physical = torch.tensor(calculate_heckel_density(pressure_input, binder_input), dtype=torch.float32)
        physics_loss = torch.mean((pred[:, 0] - physical) ** 2) * 0.1
        loss += physics_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        
        if epoch % 100 == 0:
            val = mse(model(X_t), y_t).item()
            if val < best_loss:
                best_loss = val
                patience = 0
            else:
                patience += 1
            if patience >= 120: break
            
    model.eval()
    torch.save({'model_state': model.state_dict(), 'scaler': scaler}, CHECKPOINT_PATH)
    return model, scaler

# ================================================================
# OPTIMIZER
# ================================================================
class FastOptimizer:
    def __init__(self, model, scaler, pop_size=POP_SIZE, generations=GENERATIONS):
        self.model, self.scaler = model, scaler
        self.pop_size, self.generations = pop_size, generations
        self.mutation_rate = 0.1
        self.GENE_BOUNDS = [
            (API_MIN, API_MAX), (BINDER_MIN, BINDER_MAX), (PVPP_MIN, PVPP_MAX),
            (MGST_MIN, MGST_MAX), (MCC_MIN, MCC_MAX), (MOISTURE_MIN, MOISTURE_MAX),
            (PRESSURE_MIN, PRESSURE_MAX), (SPEED_MIN, SPEED_MAX)
        ]

    def enforce_mass_balance(self, pop):
        balanced = pop.copy()
        lo = np.array([b[0] for b in self.GENE_BOUNDS[:6]])
        hi = np.array([b[1] for b in self.GENE_BOUNDS[:6]])
        comps = np.clip(pop[:, :6], lo, hi)
        total = comps.sum(axis=1, keepdims=True)
        balanced[:, :6] = np.clip(comps / (total if (total > 0).all() else 1.0) * 100.0, lo, hi)
        return balanced

    def evaluate(self, pop):
        pop_scaled = self.scaler.transform(pop)
        if isinstance(pop_scaled, pd.DataFrame): pop_scaled = pop_scaled.values
        self.model.eval()
        with torch.no_grad():
            pred = self.model(torch.tensor(pop_scaled, dtype=torch.float32)).numpy()
        density, tensile, efrf = pred[:, 0], pred[:, 1], pred[:, 2]
        penalty = 1.0 / (1.0 + np.clip(np.abs(pop_scaled) - 2.5, 0, None).sum(axis=1))
        fitness = np.column_stack([-density*penalty, -tensile*penalty, -pop[:,0]*penalty, efrf*penalty])
        fitness[:, 3] += np.maximum(0, efrf - EFRF_THRESHOLD) * 20.0
        return fitness

    def optimize(self):
        pop = np.random.rand(self.pop_size, 8)
        for i, (lo, hi) in enumerate(self.GENE_BOUNDS): pop[:, i] = pop[:, i] * (hi - lo) + lo
        pop = self.enforce_mass_balance(pop)
        obj = self.evaluate(pop)
        for gen in range(self.generations):
            diversity = np.std(pop, axis=0).mean()
            self.mutation_rate = min(0.2, self.mutation_rate + 0.02) if diversity < 0.05 else max(0.02, self.mutation_rate - 0.01)
            
            selected = []
            for _ in range(self.pop_size):
                idx = np.random.choice(self.pop_size, 2, replace=False)
                selected.append(idx[np.argmin(obj[idx].sum(axis=1))])
            sel_pop = pop[selected]
            
            offspring = []
            for i in range(0, self.pop_size, 2):
                p1, p2 = sel_pop[i], sel_pop[(i+1)%self.pop_size]
                c1, c2 = p1.copy(), p2.copy()
                for j in range(8):
                    if np.random.rand() < 0.8:
                        beta = 1.0 + 2.0 * np.random.rand()
                        c1[j] = 0.5 * ((1+beta)*p1[j] + (1-beta)*p2[j])
                        c2[j] = 0.5 * ((1-beta)*p1[j] + (1+beta)*p2[j])
                    if np.random.rand() < self.mutation_rate:
                        lo, hi = self.GENE_BOUNDS[j]
                        c1[j] = np.clip(c1[j] + np.random.normal(0, 0.1) * (hi-lo), lo, hi)
                offspring.extend([c1, c2])
            offspring = np.array(offspring[:self.pop_size])
            offspring = self.enforce_mass_balance(offspring)
            
            combined = np.vstack([pop, offspring])
            combined_obj = np.vstack([obj, self.evaluate(offspring)])
            pareto_idx = np.argsort(combined_obj.sum(axis=1))[:self.pop_size]
            pop, obj = combined[pareto_idx], combined_obj[pareto_idx]
            yield pop, obj, gen

# ================================================================
# 2D PLOTTING FUNCTION
# ================================================================
def render_2d_pareto(pop, obj, golden_idx, tested_data=None):
    api_vals = pop[:, 0]
    efrf_vals = obj[:, 3]
    
    valid_mask = (api_vals >= API_MIN) & (api_vals <= API_MAX) & (efrf_vals <= EFRF_THRESHOLD)
    api_vals, efrf_vals = api_vals[valid_mask], efrf_vals[valid_mask]
    
    if len(api_vals) == 0:
        st.warning("No solutions strictly meet the hard constraints.")
        return

    sort_idx = np.argsort(api_vals)
    api_sorted, efrf_sorted = api_vals[sort_idx], efrf_vals[sort_idx]
    
    pareto_api, pareto_efrf = [], []
    min_efrf_so_far = np.inf
    for a, e in zip(api_sorted, efrf_sorted):
        if e < min_efrf_so_far:
            min_efrf_so_far = e
            pareto_api.append(a)
            pareto_efrf.append(e)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=pareto_api, y=pareto_efrf, mode='lines+markers', name='Pareto Front',
        line=dict(color='red', width=2), marker=dict(size=6, color='#a3c4f3', line=dict(width=1, color='black'))
    ))

    if golden_idx is not None:
        g_api = pop[golden_idx, 0]
        g_efrf = obj[golden_idx, 3]
        fig.add_trace(go.Scatter(
            x=[g_api], y=[g_efrf], mode='markers', name='🏆 Golden Solution',
            marker=dict(size=16, color='gold', symbol='star', line=dict(width=1.5, color='black'))
        ))

    if tested_data is not None:
        fig.add_trace(go.Scatter(
            x=[tested_data['api']], y=[tested_data['efrf']], mode='markers', name='🔵 Tested Formulation',
            marker=dict(size=12, color='blue', symbol='circle', line=dict(width=1, color='white'))
        ))

    fig.add_vline(x=API_MIN, line_dash='dash', line_color='gray', annotation_text=f'API min ({API_MIN}%)')
    fig.add_vline(x=API_MAX, line_dash='dash', line_color='gray', annotation_text=f'API max ({API_MAX}%)')
    fig.add_hline(y=EFRF_THRESHOLD, line_dash='dash', line_color='gray', annotation_text=f'EFRF limit ({EFRF_THRESHOLD})')

    fig.update_layout(
        title='2D Pareto Front: API% vs EFRF',
        xaxis_title='API (%)', yaxis_title='EFRF',
        height=500, template='plotly_white',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)

# ================================================================
# MAIN APPLICATION (REORDERED FOR INSTANT UI)
# ================================================================
def main():
    st.title("🧬 Hybrid AI v31.1-FastPhysics-2D (Instant UI)")
    
    # ===== 1. Draw Sidebar and Main Inputs IMMEDIATELY =====
    st.sidebar.header("⚖️ Custom Recommender")
    w_api = st.sidebar.slider("Weight for API", 0.0, 1.0, 0.4)
    w_quality = st.sidebar.slider("Weight for Quality", 0.0, 1.0, 0.6)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("⚙️ Formulation & Process")
        api = st.slider("API (%)", API_MIN, API_MAX, 85.0)
        binder = st.slider("Binder (%)", BINDER_MIN, BINDER_MAX, 5.0)
        pvpp = st.slider("PVPP (%)", PVPP_MIN, PVPP_MAX, 2.0)
        mgst = st.slider("MgSt (%)", MGST_MIN, MGST_MAX, 0.5)
    with col2:
        mcc = st.slider("MCC (%)", MCC_MIN, MCC_MAX, 4.0)
        moisture = st.slider("Moisture (%)", MOISTURE_MIN, MOISTURE_MAX, 1.5)
        pressure = st.slider("Pressure (MPa)", PRESSURE_MIN, PRESSURE_MAX, 200.0)
        speed = st.slider("Speed (rpm)", SPEED_MIN, SPEED_MAX, 20.0)

    # ===== 2. Heavy Logic runs ONLY when button is clicked =====
    if st.button("🚀 Run Optimization & Show 2D Pareto"):
        
        # Train / Load model inside the button action
        with st.spinner("Loading/Training FastPhysics Model (1st time takes 15-30s)..."):
            model, scaler = train_model()
        
        start_time = time.time()
        with st.status("Running NSGA-II Optimization...", expanded=True) as status:
            progress_bar = st.progress(0)
            optimizer = FastOptimizer(model, scaler)
            final_pop, final_obj = None, None
            for i, (pop, obj, gen) in enumerate(optimizer.optimize()):
                final_pop, final_obj = pop, obj
                progress_bar.progress((gen+1)/GENERATIONS)
                if gen % 5 == 0:
                    status.update(label=f"Generation {gen+1}/{GENERATIONS}")
            status.update(label="Optimization Complete ✅", state="complete")
        
        # Recommender Scoring
        weights = np.array([w_api, w_quality])
        results = []
        for i in range(len(final_pop)):
            score = (final_pop[i,0]/100 * weights[0]) + ((1 - final_obj[i].sum()/4) * weights[1])
            results.append(score)
        golden_idx = np.argmax(results)
        best_sol = final_pop[golden_idx]
        
        # Predict Uncertainty
        pop_scaled = scaler.transform([best_sol])
        preds, uncertainty = model.predict_with_uncertainty(torch.tensor(pop_scaled, dtype=torch.float32))
        preds, uncertainty = preds[0], uncertainty[0]
        
        st.success(f"🏆 Golden Solution Found!\nAPI: {best_sol[0]:.2f}% | EFRF: {preds[2]:.3f} ± {uncertainty[2]:.3f}")
        st.caption(f"Optimization took {time.time() - start_time:.2f} seconds.")

        # Evaluate current slider formulation (Tested Data)
        slider_form = np.array([[api, binder, pvpp, mgst, mcc, moisture, pressure, speed]], dtype=np.float32)
        slider_preds, _ = model.predict_with_uncertainty(torch.tensor(scaler.transform(slider_form), dtype=torch.float32))
        tested_data = {'api': float(api), 'efrf': float(slider_preds[0][2])}
        
        # Render 2D Chart
        render_2d_pareto(final_pop, final_obj, golden_idx, tested_data=tested_data)

        # Export Report
        report = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'golden_api': float(best_sol[0]), 'golden_efrf': float(preds[2]),
            'tested_formulation': tested_data
        }
        st.download_button("📥 Download Report (JSON)", data=json.dumps(report, indent=2, default=str), file_name="2d_pareto_report.json")

if __name__ == "__main__":
    main()
