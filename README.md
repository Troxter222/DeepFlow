# 🌊 DeepFlow: Inverse Navier-Stokes Solver

[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)
[![SciML](https://img.shields.io/badge/Domain-Scientific%20ML-blue?style=for-the-badge)](https://en.wikipedia.org/wiki/Physics-informed_neural_networks)

**DeepFlow** is a scientific machine learning (SciML) framework designed to solve **inverse problems in fluid dynamics**. By leveraging **SIREN (Sinusoidal Representation Networks)** and Physics-Informed Neural Networks (PINNs), DeepFlow can reconstruct high-fidelity flow fields and discover hidden physical parameters (such as the Reynolds number) from sparse, noisy data without boundary conditions.

---

## 🔬 Scientific Core

### The Problem
Traditional CFD (Computational Fluid Dynamics) requires precise meshes and boundary conditions. DeepFlow solves the **Inverse Navier-Stokes Problem**:
> *Given a sparse set of velocity measurements $(u, v)$, recover the pressure field $p$ and the fluid viscosity $\nu$ (or Reynolds number $Re$).*

### The Physics
The model minimizes a composite loss function containing the residuals of the 2D Incompressible Navier-Stokes equations:

$$
\mathcal{L}_{physics} = \left\| \frac{\partial \mathbf{u}}{\partial t} + (\mathbf{u} \cdot \nabla) \mathbf{u} + \nabla p - \frac{1}{Re} \nabla^2 \mathbf{u} \right\|^2
$$

### The Architecture
*   **Backbone:** SIREN (Sinusoidal activations). Unlike ReLU, sine functions possess non-zero higher-order derivatives, enabling accurate calculation of the Laplacian ($\nabla^2$).
*   **Optimization:** Hybrid strategy using **Adam** (fast convergence) followed by **L-BFGS** (second-order optimization for scientific precision).

---

## 📊 Results

**Case Study: Kovasznay Flow**  
DeepFlow was trained on 6,000 sparse spatial points.

- **True Reynolds Number:** 20.0  
- **Discovered Reynolds Number:** 12.86 (**Error:** 35.71%)  
- **Pressure Reconstruction:** Pressure field reconstructed **partially** (visible discrepancy vs ground truth).

![Result](deepflow_final_result.png)
*(Left: Ground Truth Pressure | Right: DeepFlow Prediction via Hidden Physics)*

---

## 🌪️ Phase 2: Transient Dynamics (Time-Dependent)
**Case Study: Decaying Taylor-Green Vortex**

A complex unsteady flow where vortices decay over time due to viscosity. The network must learn 4D spatio-temporal dynamics $(x, y, t)$.

*   **Objective:** Learn the decay rate and discover Viscosity ($\nu$).
*   **Challenges:** 1,000,000 data points, 4D gradients, Chain Rule Normalization.
*   **Results:**
    *   True Viscosity $\nu$: **0.1000**
    *   Discovered $\nu$: **0.1063** (Error ~6%)
    *   Learned to simulate energy dissipation over time.

![Vortex Decay](phase2/vortex_decay.gif)
*(Visualization of the vortex decay learned by the neural network)*

---

## 🚀 Installation & Usage

### 1. Clone the repository
```bash
git clone https://github.com/Troxter222/DeepFlow.git
cd DeepFlow
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Generate Data & Train
```bash
# 1. Generate synthetic data (Analytical Solution)
python generate_kovasznay.py

# 2. Train the model (Discovery Phase)
python train.py

# 3. Fine-tune with L-BFGS & Visualize (Precision Phase)
python finalize_pro.py
```

### 4. Run Phase 2 (Transient)

```bash
# Generate 1M points dataset
python phase2/generate_vortex.py
# Train with Time-Dependency
python phase2/train_transient.py
# Render GIF
python phase2/animate.py
```

---

### 🔗 Citation
If you use this code for your research, please cite it as:

```bash
@software{DeepFlow2025,
  author = {Ali Sultonov},
  title = {DeepFlow: Physics-Informed Neural Networks for Inverse Fluid Dynamics},
  year = {2025},
  url = {https://github.com/Troxter222/DeepFlow}
}
```

### 🤝 Contributing
Contributions are welcome! Please open an issue or submit a PR for improvements in the physics engine or new PDE implementations.

### 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
