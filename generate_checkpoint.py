# Generate checkpoint for Hybrid AI Framework
!pip install streamlit numpy pandas torch plotly scikit-learn scipy xgboost fpdf2 -q

import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# ---------------------------------------------------------------------
# Copy the necessary definitions from app.py (they are reproduced below)
# ---------------------------------------------------------------------
class Mish(torch.nn.Module):
    def forward(self, x):
        return x * torch.tanh(torch.nn.functional.softplus(x))

class ResidualBlock(torch.nn.Module):
    def __init__(self, features, dropout=0.1):
        super().__init__()
        self.lin1 = torch.nn.Linear(features, features)
        self.bn1 = torch.nn.BatchNorm1d(features)
        self.lin2 = torch.nn.Linear(features, features)
        self.bn2 = torch.nn.BatchNorm1d(features)
        self.act = Mish()
        self.drop = torch.nn.Dropout(dropout)
    def forward(self, x):
        identity = x
        out = self.act(self.bn1(self.lin1(x)))
        out = self.drop(out)
        out = self.bn2(self.lin2(out))
        out = self.drop(out)
        return identity + out

class MultiTaskPINN(torch.nn.Module):
    def __init__(self, input_dim=19, hidden=512):
        super().__init__()
        self.input_layer = torch.nn.Sequential(torch.nn.Linear(input_dim, hidden), Mish(), torch.nn.Dropout(0.05))
        self.res1 = ResidualBlock(hidden, dropout=0.05)
        self.res2 = ResidualBlock(hidden, dropout=0.05)
        self.res3 = ResidualBlock(hidden, dropout=0.05)
        self.transition = torch.nn.Sequential(torch.nn.Linear(hidden, hidden//2), torch.nn.Tanh(), torch.nn.Dropout(0.05))
        self.output = torch.nn.Linear(hidden//2, 6)
    def forward(self, X):
        x = self.input_layer(X)
        x = self.res1(x); x = self.res2(x); x = self.res3(x)
        x = self.transition(x)
        return self.output(x)

# ---------------------------------------------------------------------
# Data generation – copy the helper functions from app.py
# (or we can generate synthetic data directly, but for consistency we copy them)
# ---------------------------------------------------------------------
# For brevity, we define the same constants as in app.py (copy from your own file)
# or we can generate random data with known physics. Let's use a simplified generator:
def generate_synthetic_data(n=25000):
    np.random.seed(42)
    api = np.random.uniform(80, 98, n)
    binder = np.random.uniform(1.4, 6, n)
    pvpp = np.random.uniform(1, 6, n)
    mgst = np.random.uniform(0.1, 1.2, n)
    mcc = np.random.uniform(1.5, 8, n)
    moisture = np.random.uniform(0.5, 5, n)
    particle_size = np.random.uniform(10, 200, n)
    pressure = np.random.uniform(150, 250, n)
    speed = np.random.uniform(15, 30, n)
    granule = np.random.uniform(30, 250, n)
    # ... more features (simplified) – for a real checkpoint we need full 19 features,
    # but we can generate them similarly.
    # To keep this short, I'll assume you have the full generate_pinn_data function in app.py.
    # We'll import from app.py if it's in the same folder, but in Colab we'll just reproduce.
    # The easiest: copy the entire generate_pinn_data from your app.py into this notebook.
    # For now, we'll use a placeholder – but you should copy the full function from your app.py.
    pass

# Instead of rewriting all, I recommend you to upload your app.py to Colab first,
# then run the following to import the functions:
from app import generate_pinn_data, MultiTaskPINN, HIDDEN_SIZE
# But we are in Colab, so we need to have app.py available. Alternatively, you can paste the whole generate_pinn_data function here.

# ---------------------------------------------------------------------
# Train and save the checkpoint
# ---------------------------------------------------------------------
N_SAMPLES = 25000
df, features = generate_pinn_data(N_SAMPLES)
X_raw = df[features].values
y = df[['Density','Tensile_Strength_MPa','Elastic_Recovery_%',
        'Disintegration_Time_min','Dissolution_Tau','Dissolution_Beta']].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)
y_scaler = StandardScaler()
y_scaled = y_scaler.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_scaled, test_size=0.2, random_state=42
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MultiTaskPINN(input_dim=X_raw.shape[1], hidden=512).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=30, factor=0.5)

X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_t = torch.tensor(y_train, dtype=torch.float32).to(device)
X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test_t = torch.tensor(y_test, dtype=torch.float32).to(device)

best_r2 = -np.inf
for epoch in range(800):
    model.train()
    optimizer.zero_grad()
    y_pred = model(X_train_t)
    loss = torch.nn.MSELoss()(y_pred, y_train_t)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step(loss.item())
    
    if epoch % 50 == 0:
        model.eval()
        with torch.no_grad():
            val_pred = model(X_test_t).cpu().numpy()
            val_true = y_test_t.cpu().numpy()
            val_pred_actual = y_scaler.inverse_transform(val_pred)
            val_true_actual = y_scaler.inverse_transform(val_true)
            r2_t = r2_score(val_true_actual[:, 1], val_pred_actual[:, 1])
        print(f"Epoch {epoch}: R² = {r2_t:.4f}")
        if r2_t > best_r2:
            best_r2 = r2_t
            checkpoint = {
                'model_state': model.cpu().state_dict(),
                'scaler': scaler,
                'y_scaler': y_scaler,
                'features': features,
                'df': df,
                'input_dim': X_raw.shape[1]
            }
            torch.save(checkpoint, 'hybrid_unified_v29_30_R40.pt')
            print("Checkpoint saved!")

print("Training complete. Download the file 'hybrid_unified_v29_30_R40.pt' from Colab.")
