# generate_checkpoint.py
# This script runs locally (not on Streamlit Cloud) to produce the pre‑trained model.

import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Import from your app (this is safe, as generate_checkpoint.py is not imported by app.py)
from app import generate_pinn_data, MultiTaskPINN, HIDDEN_SIZE

# --- Generate data ---
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
model = MultiTaskPINN(input_dim=X_raw.shape[1], hidden=HIDDEN_SIZE).to(device)
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

print("Training complete.")
