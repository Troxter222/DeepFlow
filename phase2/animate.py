import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.model import DeepFlowNet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "phase2/model_transient.pth"

def create_animation():
    print("🎥 Rendering Animation...")
    model = DeepFlowNet().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH))
    
    # Grid
    res = 100
    x = np.linspace(-np.pi, np.pi, res)
    y = np.linspace(-np.pi, np.pi, res)
    X, Y = np.meshgrid(x, y)
    flat_X = torch.tensor(X.flatten(), dtype=torch.float32).to(DEVICE)
    flat_Y = torch.tensor(Y.flatten(), dtype=torch.float32).to(DEVICE)
    
    fig, ax = plt.subplots(figsize=(6, 5))
    
    # Initial plot
    t_start = 0.0
    flat_T = torch.full_like(flat_X, t_start)
    inputs = torch.stack([flat_X, flat_Y, flat_T], dim=1)
    
    with torch.no_grad():
        out = model(inputs)
        # Visualize Vorticity (Curl) or Speed
        # Let's simplify: Visualize Stream Function (Psi)
        psi = out[:, 0].cpu().numpy().reshape(res, res)
        
    contour = ax.contourf(X, Y, psi, levels=50, cmap='inferno')
    title = ax.set_title(f"Time: {t_start:.2f}")
    plt.colorbar(contour, ax=ax)

    def update(frame):
        ax.clear()
        time_val = frame * 0.05 # 2 seconds total roughly
        flat_T = torch.full_like(flat_X, time_val)
        inputs = torch.stack([flat_X, flat_Y, flat_T], dim=1)
        
        with torch.no_grad():
            out = model(inputs)
            psi = out[:, 0].cpu().numpy().reshape(res, res)
            
        # Re-plot
        ax.contourf(X, Y, psi, levels=50, cmap='inferno')
        ax.set_title(f"DeepFlow Phase 2 | Time: {time_val:.2f}s | Decay")
        return ax,

    print("⏳ Generatng frames...")
    # 40 frames
    ani = animation.FuncAnimation(fig, update, frames=40, interval=100)
    
    save_path = "phase2/vortex_decay.gif"
    ani.save(save_path, writer='pillow', fps=10)
    print(f"🎬 Animation saved to {save_path}")

if __name__ == "__main__":
    create_animation()