# ================================================================
# generate_checkpoint.py
# Run this ONCE locally (or on Colab) to produce the pre‑trained model file.
# It uses the definitions from your app.py – no duplication needed.
# ================================================================

import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import os
import sys

# ---------------------------------------------------------------------
# Import from your app.py (make sure it's in the same folder)
# ---------------------------------------------------------------------
try:
    from app import generate_pinn_data, MultiTaskPINN, HIDDEN_SIZE
except ImportError:
    print("❌ Could not import from app.py. Make sure this file is in the same folder as app.py.")
    sys.exit(1)

# ---------------------------------------------------------------------
# Constants – must match those in app.py
# ---------------------------------------------------------------------
N_SAMPLES = 25000          # 25k samples for high accuracy
ADAM_EPOCHS = 800          # enough to converge
HIDDEN_SIZE = 512          # as in the unified framework
CHECKPOINT_NAME = "hybrid_unified_v29_30_R40.pt"

def main():
    print("🚀 Generating pre‑trained model for Hybrid AI Framework...")
    print(f"   Samples: {N_SAMPLES}")
    print(f"   Epochs:  {ADAM_EPOCHS}")
    print(f"   Hidden:  {HIDDEN_SIZE}")
    print("")

    # ---- 1. Generate data ----
    print("📊 Generating synthetic training data...")
    df, features = generate_pinn_data(N_SAMPLES)
    X_raw = df[features].values
    y = df[['Density','Tensile_Strength_MPa','Elastic_Recovery_%',
            'Disintegration_Time_min','Dissolution_Tau','Dissolution_Beta']].values
    print(f"   Generated {len(df)} samples with {X_raw.shape[1]} features.")
    print("")

    # ---- 2. Scale data ----
    print("📈 Scaling features and targets...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    y_scaler = StandardScaler()
    y_scaled = y_scaler.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_scaled, test_size=0.2, random_state=42
    )
    print(f"   Training samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")
    print("")

    # ---- 3. Build model ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Using device: {device}")
    model = MultiTaskPINN(input_dim=X_raw.shape[1], hidden=HIDDEN_SIZE).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=30, factor=0.5)

    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).to(device)

    # ---- 4. Train ----
    print("🏋️  Training started...")
    best_r2 = -np.inf
    for epoch in range(ADAM_EPOCHS):
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
                r2_t = r2_score(val_true_actual[:, 1], val_pred_actual[:, 1])  # tensile R²
            print(f"   Epoch {epoch:4d}/{ADAM_EPOCHS} – Tensile R² = {r2_t:.4f}")

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
                torch.save(checkpoint, CHECKPOINT_NAME)
                print(f"   ✅ Checkpoint saved (R² = {r2_t:.4f})")

    # ---- 5. Final checkpoint ----
    # Ensure the best model is saved at the end
    print("")
    print(f"🏁 Training complete. Best Tensile R² = {best_r2:.4f}")
    if os.path.exists(CHECKPOINT_NAME):
        print(f"✅ Checkpoint file saved as '{CHECKPOINT_NAME}'")
        print("📤 Upload this file to your repository's root folder (same as app.py).")
        print("   Then commit, push, and redeploy your Streamlit app.")
    else:
        print("❌ Checkpoint file not found – something went wrong.")

if __name__ == "__main__":
    main()
