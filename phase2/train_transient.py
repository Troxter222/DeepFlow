import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.model import DeepFlowNet

# === CONFIG ===
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_PATH = "data/taylor_green.npy"
BATCH_SIZE = 10000 
EPOCHS = 50
LR = 1e-4

# === SCALING CONSTANTS ===
SPATIAL_SCALE = np.pi  
TIME_SCALE = 2.0       

def compute_transient_physics(model, coords_norm, nu_est):
    """
    Physic Loss with CHAIN RULE adjustment for normalized inputs.
    """
    coords_norm.requires_grad = True
    output = model(coords_norm)
    psi, p = output[:, 0:1], output[:, 1:2]
    
    # 1. Gradients
    grads_psi = torch.autograd.grad(psi, coords_norm, grad_outputs=torch.ones_like(psi), create_graph=True)[0]
    
    # Chain rule factors
    inv_s = 1.0 / SPATIAL_SCALE
    inv_t = 1.0 / TIME_SCALE
    
    u = grads_psi[:, 1:2] * inv_s
    v = -grads_psi[:, 0:1] * inv_s
    
    # 2. First Derivatives
    grads_u = torch.autograd.grad(u, coords_norm, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_x = grads_u[:, 0:1] * inv_s
    u_y = grads_u[:, 1:2] * inv_s
    u_t = grads_u[:, 2:3] * inv_t   
    
    grads_v = torch.autograd.grad(v, coords_norm, grad_outputs=torch.ones_like(v), create_graph=True)[0]
    v_x = grads_v[:, 0:1] * inv_s
    v_y = grads_v[:, 1:2] * inv_s
    v_t = grads_v[:, 2:3] * inv_t
    
    grads_p = torch.autograd.grad(p, coords_norm, grad_outputs=torch.ones_like(p), create_graph=True)[0]
    p_x = grads_p[:, 0:1] * inv_s
    p_y = grads_p[:, 1:2] * inv_s
    
    # 3. Second Derivatives (Laplacian)
    u_xx = torch.autograd.grad(u_x, coords_norm, grad_outputs=torch.ones_like(u_x), create_graph=True)[0][:, 0:1] * inv_s
    u_yy = torch.autograd.grad(u_y, coords_norm, grad_outputs=torch.ones_like(u_y), create_graph=True)[0][:, 1:2] * inv_s
    
    v_xx = torch.autograd.grad(v_x, coords_norm, grad_outputs=torch.ones_like(v_x), create_graph=True)[0][:, 0:1] * inv_s
    v_yy = torch.autograd.grad(v_y, coords_norm, grad_outputs=torch.ones_like(v_y), create_graph=True)[0][:, 1:2] * inv_s
    
    # 4. Residuals
    f_u = u_t + (u*u_x + v*u_y) + p_x - nu_est * (u_xx + u_yy)
    f_v = v_t + (u*v_x + v*v_y) + p_y - nu_est * (v_xx + v_yy)
    
    return torch.mean(f_u**2 + f_v**2)

def train_phase2():
    print(f"PHASE 2: Transient Dynamics (Stabilized) on {DEVICE}")
    
    try:
        raw_data = np.load(DATA_PATH)
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    print(f"Loaded {raw_data.shape[0]} points.")
    
    # Input: [x, y, t] -> Indices 1, 2, 0
    X_raw = raw_data[:, [1, 2, 0]]
    U_raw = raw_data[:, [3, 4]]
    
    # Normalize inputs manually
    X_norm = np.copy(X_raw)
    X_norm[:, 0] /= SPATIAL_SCALE
    X_norm[:, 1] /= SPATIAL_SCALE
    X_norm[:, 2] /= TIME_SCALE
    
    X = torch.tensor(X_norm, dtype=torch.float32)
    U = torch.tensor(U_raw, dtype=torch.float32)
    
    dataset = TensorDataset(X, U)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    
    model = DeepFlowNet().to(DEVICE)
    # Start closer to expected value to avoid huge gradients initially
    # Target ln(0.1) ~ -2.3. Start at -2.0
    model.lambda_2.data = torch.tensor([-2.0], device=DEVICE)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    print("Training started with WARM-UP Strategy...")
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        phys_loss_accum = 0
        
        # === STRATEGY: WARM-UP ===
        if epoch < 5:
            phys_weight = 0.0
            status = "WARMUP (Data only)"
        else:
            phys_weight = 1e-3
            status = "PHYSICS ON"

        for batch_in, batch_out in loader:
            batch_in, batch_out = batch_in.to(DEVICE), batch_out.to(DEVICE)
            optimizer.zero_grad()
            
            # 1. Forward (Data Loss)
            batch_in.requires_grad = True
            pred = model(batch_in)
            psi = pred[:, 0:1]
            
            grads = torch.autograd.grad(psi, batch_in, grad_outputs=torch.ones_like(psi), create_graph=True)[0]
            
            u_pred = grads[:, 1:2] * (1.0 / SPATIAL_SCALE)
            v_pred = -grads[:, 0:1] * (1.0 / SPATIAL_SCALE)
            
            loss_d = torch.mean((u_pred - batch_out[:, 0:1])**2 + (v_pred - batch_out[:, 1:2])**2)
            
            # 2. Physics Loss
            loss_p = torch.tensor(0.0, device=DEVICE)
            if phys_weight > 0:
                est_nu = torch.exp(model.lambda_2)
                loss_p = compute_transient_physics(model, batch_in, est_nu)
            
            # Total Loss
            loss = loss_d + phys_weight * loss_p
            
            loss.backward()
            
            # === SAFETY: TIGHT CLIPPING ===
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            
            optimizer.step()
            
            total_loss += loss.item()
            phys_loss_accum += loss_p.item()
            
        avg_loss = total_loss / len(loader)
        est_nu_val = torch.exp(model.lambda_2).item()
        
        print(f"Epoch {epoch+1}/{EPOCHS} [{status}] | Loss: {avg_loss:.5f} | Phys: {phys_loss_accum/len(loader):.5f} | 🌊 nu: {est_nu_val:.4f}")
        
    os.makedirs("phase2", exist_ok=True)
    torch.save(model.state_dict(), "phase2/model_transient.pth")
    print("Phase 2 Model Saved.")

if __name__ == "__main__":
    train_phase2()