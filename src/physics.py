import torch

class NavierStokes2D:
    """
    Computes the residuals for the 2D Incompressible Navier-Stokes equations.
    """
    @staticmethod
    def compute_residuals(model, coords):
        """
        Calculates the physics loss (residual) for a batch of coordinates.
        
        Args:
            model: The neural network
            coords: Input tensor (batch_size, 3) -> (x, y, t)
        
        Returns:
            mse_residual: Mean Squared Error of the physical equations.
        """
        coords.requires_grad = True
        outputs = model(coords)
        
        psi = outputs[:, 0:1] # Stream function
        p = outputs[:, 1:2]   # Pressure
        
        # --- Automatic Differentiation ---
        
        # 1. Velocity components from Stream Function (Ensures mass conservation automatically)
        # u = d(psi)/dy, v = -d(psi)/dx
        grads_psi = torch.autograd.grad(psi, coords, grad_outputs=torch.ones_like(psi), create_graph=True)[0]
        u = grads_psi[:, 1:2]
        v = -grads_psi[:, 0:1]
        
        # 2. First derivatives of velocity and pressure
        # grads_u shape: [batch, 3] -> [du/dx, du/dy, du/dt]
        grads_u = torch.autograd.grad(u, coords, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_x, u_y, u_t = grads_u[:, 0:1], grads_u[:, 1:2], grads_u[:, 2:3]
        
        grads_v = torch.autograd.grad(v, coords, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        v_x, v_y, v_t = grads_v[:, 0:1], grads_v[:, 1:2], grads_v[:, 2:3]
        
        grads_p = torch.autograd.grad(p, coords, grad_outputs=torch.ones_like(p), create_graph=True)[0]
        p_x, p_y = grads_p[:, 0:1], grads_p[:, 1:2]
        
        # 3. Second derivatives (Laplacian components)
        u_xx = torch.autograd.grad(u_x, coords, grad_outputs=torch.ones_like(u_x), create_graph=True)[0][:, 0:1]
        u_yy = torch.autograd.grad(u_y, coords, grad_outputs=torch.ones_like(u_y), create_graph=True)[0][:, 1:2]
        
        v_xx = torch.autograd.grad(v_x, coords, grad_outputs=torch.ones_like(v_x), create_graph=True)[0][:, 0:1]
        v_yy = torch.autograd.grad(v_y, coords, grad_outputs=torch.ones_like(v_y), create_graph=True)[0][:, 1:2]
        
        # --- PDE Formulation ---
        # Momentum Equations:
        # u_t + (u*u_x + v*u_y) = -p_x + (1/Re)*(u_xx + u_yy)
        
        l1 = model.lambda_1 # Convection coefficient (usually 1.0)
        l2 = torch.exp(model.lambda_2) # Viscosity coefficient (1/Re)
        
        momentum_u = u_t + l1 * (u * u_x + v * u_y) + p_x - l2 * (u_xx + u_yy)
        momentum_v = v_t + l1 * (u * v_x + v * v_y) + p_y - l2 * (v_xx + v_yy)
        
        return torch.mean(momentum_u**2 + momentum_v**2)