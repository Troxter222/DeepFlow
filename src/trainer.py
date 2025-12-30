import torch
from tqdm import tqdm
from .physics import NavierStokes2D

class PINNTrainer:
    def __init__(self, model, train_loader, optimizer, device):
        self.model = model
        self.train_loader = train_loader
        self.optimizer = optimizer
        self.device = device
        
    def train_epoch(self, epoch_idx, physics_weight=0.1):
        self.model.train()
        total_loss = 0
        data_loss_accum = 0
        phys_loss_accum = 0
        
        loop = tqdm(self.train_loader, desc=f"Epoch {epoch_idx}")
        
        for batch_in, batch_out in loop:
            batch_in = batch_in.to(self.device)
            batch_out = batch_out.to(self.device)
            
            # Reset gradients
            self.optimizer.zero_grad()
            
            # 1. Data Loss
            batch_in.requires_grad = True
            
            pred_out = self.model(batch_in)
            psi = pred_out[:, 0:1]
            
            # (dPsi/dy, -dPsi/dx)
            grads = torch.autograd.grad(psi, batch_in, grad_outputs=torch.ones_like(psi), create_graph=True)[0]
            u_pred = grads[:, 1:2]
            v_pred = -grads[:, 0:1]
            
            loss_d = torch.mean((u_pred - batch_out[:, 0:1])**2 + (v_pred - batch_out[:, 1:2])**2)
            
            # 2. Physics Loss
            loss_p = NavierStokes2D.compute_residuals(self.model, batch_in)
            
            if torch.isnan(loss_p) or torch.isinf(loss_p):
                loss = loss_d 
                loop.set_postfix(status="PHYS_INF!", d_loss=loss_d.item())
            else:
                loss = loss_d + physics_weight * loss_p
                loop.set_postfix(d_loss=loss_d.item(), p_loss=loss_p.item())

            # Backward pass
            loss.backward()
            
            # Clip Gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Logging
            total_loss += loss.item()
            data_loss_accum += loss_d.item()
            if not (torch.isnan(loss_p) or torch.isinf(loss_p)):
                phys_loss_accum += loss_p.item()
            
        return total_loss / len(self.train_loader)