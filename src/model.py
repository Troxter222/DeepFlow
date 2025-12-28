import torch
import torch.nn as nn
import numpy as np

class SineLayer(nn.Module):
    """
    Fundamental building block for SIREN (Sinusoidal Representation Networks).
    
    Paper: Implicit Neural Representations with Periodic Activation Functions
    (Sitzmann et al., NeurIPS 2020)
    """
    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                # First layer initialization: uniform distribution
                self.linear.weight.uniform_(-1 / self.linear.in_features, 
                                             1 / self.linear.in_features)
            else:
                # Subsequent layers: dependent on omega_0 for stability
                limit = np.sqrt(6 / self.linear.in_features) / self.omega_0
                self.linear.weight.uniform_(-limit, limit)
        
    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))

class DeepFlowNet(nn.Module):
    """
    The main PINN architecture.
    Inputs:  (x, y, t)
    Outputs: (psi, p) -> Stream function and Pressure
    """
    def __init__(self, in_features=3, hidden_dim=256, layers=5, out_features=2):
        super().__init__()
        
        layers_list = [SineLayer(in_features, hidden_dim, is_first=True, omega_0=30.0)]
        
        for _ in range(layers):
            layers_list.append(SineLayer(hidden_dim, hidden_dim, omega_0=30.0))
        
        self.net = nn.Sequential(*layers_list)
        
        # Final linear layer to map to physical quantities (unbounded)
        self.final_linear = nn.Linear(hidden_dim, out_features)
        self.init_final_layer(hidden_dim)

        # === PHYSICAL PARAMETERS (INVERSE PROBLEM) ===
        # The network will attempt to learn the Reynolds numbers (Re) from data.
        # We model lambda_1 = 1 (convection) and lambda_2 = 1/Re (viscosity)
        self.lambda_1 = nn.Parameter(torch.tensor([0.0], dtype=torch.float32)) # Learnable
        self.lambda_2 = nn.Parameter(torch.tensor([-2.0], dtype=torch.float32)) # Learnable (log scale)

    def init_final_layer(self, hidden_dim):
        with torch.no_grad():
            limit = np.sqrt(6 / hidden_dim) / 30.0
            self.final_linear.weight.uniform_(-limit, limit)

    def forward(self, x):
        feat = self.net(x)
        return self.final_linear(feat)