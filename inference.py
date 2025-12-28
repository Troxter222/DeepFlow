import torch
import numpy as np
import matplotlib.pyplot as plt
from src.model import DeepFlowNet

# Настройки
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "deepflow_model.pth" # Этот файл появится после train.py

def plot_results():
    print("🎨 Rendering results...")
    
    # 1. Загружаем модель
    model = DeepFlowNet().to(DEVICE)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print("✅ Model loaded successfully.")
    except FileNotFoundError:
        print("❌ Error: Run train.py first to save the model!")
        return

    # 2. Узнаем, какое число Рейнольдса нашла сеть
    estimated_Re = 1.0 / torch.exp(model.lambda_2).item()
    print(f"🧐 Network estimates Reynolds Number Re ≈ {estimated_Re:.2f} (True: 20.0)")

    # 3. Создаем сетку для рисования (плотная сетка 200x200)
    x = np.linspace(-0.5, 1.0, 200)
    y = np.linspace(-0.5, 1.5, 200)
    X, Y = np.meshgrid(x, y)
    
    # Превращаем в тензор для нейросети
    # Вход: (x, y, t=0)
    flat_X = X.flatten()
    flat_Y = Y.flatten()
    flat_T = np.zeros_like(flat_X)
    
    inputs = np.stack([flat_X, flat_Y, flat_T], axis=1)
    inputs_tensor = torch.tensor(inputs, dtype=torch.float32).to(DEVICE)

    # 4. Прогоняем через сеть
    with torch.no_grad():
        output = model(inputs_tensor)
        # Выход: [psi, p]
        psi_pred = output[:, 0].cpu().numpy().reshape(200, 200)
        p_pred = output[:, 1].cpu().numpy().reshape(200, 200)

    # 5. Рисуем аналитическое решение (Истину) для сравнения
    Re_true = 20.0
    lam = Re_true / 2 - np.sqrt(Re_true**2 / 4 + 4 * np.pi**2)
    p_true = 0.5 * (1 - np.exp(2 * lam * X)) # Формула давления Kovasznay

    # === ГРАФИКИ ===
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    
    # График 1: Функция тока (Линии тока)
    c1 = ax[0].contourf(X, Y, psi_pred, levels=50, cmap='viridis')
    ax[0].set_title("Predicted Stream Function (Ψ)")
    plt.colorbar(c1, ax=ax[0])
    
    # График 2: Предсказанное Давление (Скрытая переменная!)
    c2 = ax[1].contourf(X, Y, p_pred, levels=50, cmap='magma')
    ax[1].set_title(f"Predicted Pressure (Re_est={estimated_Re:.1f})")
    plt.colorbar(c2, ax=ax[1])
    
    # График 3: Истинное Давление (Ground Truth)
    c3 = ax[2].contourf(X, Y, p_true, levels=50, cmap='magma')
    ax[2].set_title("Ground Truth Pressure (Analytical)")
    plt.colorbar(c3, ax=ax[2])

    plt.suptitle("DeepFlow Results: Inverse Navier-Stokes Solving", fontsize=16)
    plt.savefig("result_plot.png", dpi=300)
    print("✅ Saved visualization to result_plot.png")
    plt.show()

if __name__ == "__main__":
    plot_results()