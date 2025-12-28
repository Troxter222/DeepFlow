import torch
import numpy as np
import matplotlib.pyplot as plt
from src.model import DeepFlowNet
from src.physics import NavierStokes2D
import os

N_SAMPLES = 6000
LBFGS_HISTORY = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = "deepflow_model.pth"
DATA_PATH = "data/flow_data.npy"
FINAL_MODEL_PATH = "deepflow_final.pth"

def finalize_project():
    print("🚀 ACTIVATING SCIENTIFIC MODE [L-BFGS OPTIMIZER]")
    print("🖥️ Hardware: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print("⚙️ Tuning Strategy: Low-VRAM High-Precision convergence")
    
    model = DeepFlowNet().to(DEVICE)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH))
        print("✅ Pre-trained model loaded.")
    else:
        print("❌ Error: Train the model first using train.py!")
        return

    print(f"📦 Loading {N_SAMPLES} sparse sensor points...")
    raw_data = np.load(DATA_PATH)
    indices = np.random.choice(raw_data.shape[0], N_SAMPLES, replace=False)
    batch_data = torch.tensor(raw_data[indices], dtype=torch.float32).to(DEVICE)
    
    X_batch = batch_data[:, [1, 2, 0]].requires_grad_(True) # x, y, t
    U_batch = batch_data[:, [3, 4]]    # u, v

    # 3. L-BFGS
    optimizer = torch.optim.LBFGS(
        model.parameters(), 
        lr=1.0, 
        max_iter=1000,
        max_eval=1000, 
        history_size=LBFGS_HISTORY,
        tolerance_grad=1e-5, 
        tolerance_change=1e-5,
        line_search_fn="strong_wolfe"
    )

    print("🔥 Starting High-Precision Fine-Tuning...")
    
    global step_count
    step_count = 0

    def closure():
        global step_count
        optimizer.zero_grad()
        
        # Forward pass
        pred_out = model(X_batch)
        psi = pred_out[:, 0:1]
        
        # Gradients (Physics)
        grads = torch.autograd.grad(psi, X_batch, grad_outputs=torch.ones_like(psi), create_graph=True)[0]
        u_pred = grads[:, 1:2]
        
        # Loss
        loss_data = torch.mean((u_pred - U_batch)**2)
        loss_phys = NavierStokes2D.compute_residuals(model, X_batch)
        
        loss = loss_data + 1.5 * loss_phys 
        
        loss.backward()
        
        step_count += 1
        if step_count % 50 == 0:
            est_re = 1.0 / torch.exp(model.lambda_2).item()
            if step_count % 200 == 0:
                 torch.cuda.empty_cache()
            print(f"\r⚡ Step {step_count} | Loss: {loss.item():.6f} | 🎯 Re Estimate: {est_re:.4f} (True: 20.0)", end="")
        return loss

    try:
        optimizer.step(closure)
    except RuntimeError as e:
        if "out of memory" in str(e):
            print("\n\n💥 OOM Error again! Try reducing N_SAMPLES in code to 2000.")
            torch.cuda.empty_cache()
            return
        else:
            raise e

    print("\n✅ Optimization Complete.")

    torch.save(model.state_dict(), FINAL_MODEL_PATH)
    
    # === 4. Visulaization ===
    render_pro_plot(model)

def render_pro_plot(model):
    print("\n🎨 Rendering Production-Grade Visualization...")
    torch.cuda.empty_cache()
    
    res = 200
    x = np.linspace(-0.5, 1.0, res)
    y = np.linspace(-0.5, 1.5, res)
    X, Y = np.meshgrid(x, y)
    
    flat_X = torch.tensor(X.flatten(), dtype=torch.float32).to(DEVICE)
    flat_Y = torch.tensor(Y.flatten(), dtype=torch.float32).to(DEVICE)
    flat_T = torch.zeros_like(flat_X)
    
    inputs = torch.stack([flat_X, flat_Y, flat_T], dim=1)
    
    with torch.no_grad():
        out = model(inputs)
        p_pred = out[:, 1].cpu().numpy().reshape(res, res)
        
    # Analytical Solution (Ground Truth)
    Re_true = 20.0
    lam = Re_true / 2 - np.sqrt(Re_true**2 / 4 + 4 * np.pi**2)
    p_true = 0.5 * (1 - np.exp(2 * lam * X))
    
    p_pred_norm = p_pred - np.mean(p_pred)
    p_true_norm = p_true - np.mean(p_true)
    
    fig, ax = plt.subplots(1, 2, figsize=(16, 7))
    
    # Ground Truth
    im1 = ax[0].contourf(X, Y, p_true_norm, levels=120, cmap='turbo')
    ax[0].set_title("Ground Truth Pressure (Normalized)", fontsize=14, fontweight='bold')

    plt.colorbar(im1, ax=ax[0])

    ax[0].axis('off')
    
    # Prediction
    est_re = 1.0 / torch.exp(model.lambda_2).item()
    im2 = ax[1].contourf(X, Y, p_pred_norm, levels=120, cmap='turbo')
    ax[1].set_title(f"DeepFlow Prediction (Re: {est_re:.2f})", fontsize=14, fontweight='bold')

    plt.colorbar(im2, ax=ax[1])

    ax[1].axis('off')
    
    plt.suptitle(f"DeepFlow: Inverse Navier-Stokes Solver (Re_error: {abs(est_re-20.0)/20.0*100:.2f}%)", fontsize=18)
    plt.tight_layout()
    plt.savefig("deepflow_final_result.png", dpi=300, bbox_inches='tight')

    print("🖼️ Saved chart to deepflow_final_result.png")

    plt.show()

if __name__ == "__main__":
    finalize_project()