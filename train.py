import torch
from torch.utils.data import DataLoader
from src.model import DeepFlowNet
from src.dataset import FlowDataset
from src.trainer import PINNTrainer

# Configuration
BATCH_SIZE = 5000
EPOCHS = 50
LR = 1e-4
DATA_PATH = "data/flow_data.npy"

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Starting DeepFlow on {device}")
    
    # 1. Dataset
    dataset = FlowDataset(DATA_PATH)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    
    # 2. Model
    model = DeepFlowNet().to(device)
    print(f"🧠 Model initialized. Parameters: {sum(p.numel() for p in model.parameters())}")
    
    # 3. Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    # 4. Trainer
    trainer = PINNTrainer(model, loader, optimizer, device)
    
    # 5. Training Loop
    print("Training started... (Watch 'Discovered Re' converge to 20.0)")
    for epoch in range(EPOCHS):
        # Physics weight можно постепенно повышать, но начнем с малого
        avg_loss = trainer.train_epoch(epoch, physics_weight=0.01)
        
        # Check physical parameter discovery
        est_inv_re = torch.exp(model.lambda_2).item()
        est_re = 1.0 / est_inv_re if est_inv_re > 0 else 0
        
        print(f"Epoch {epoch} Summary | Avg Loss: {avg_loss:.6f} | 🎯 Discovered Re: {est_re:.2f} (Target: 20.0)")

    # 6. Save
    torch.save(model.state_dict(), "deepflow_model.pth")
    print("Model saved successfully.")

if __name__ == "__main__":
    main()