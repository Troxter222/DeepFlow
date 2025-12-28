import torch
from torch.utils.data import Dataset
import numpy as np

class FlowDataset(Dataset):
    """
    Efficiently handles large-scale fluid dynamics datasets.
    """
    def __init__(self, data_path, mode='train', n_samples=None):
        """
        Args:
            data_path (str): Path to .npy file containing [t, x, y, u, v]
            mode (str): 'train' or 'full'.
        """
        print(f"Loading dataset from {data_path}...")
        # Assuming data is stored as a numpy array of shape (N, 5) -> [t, x, y, u, v]
        raw_data = np.load(data_path)
        
        if n_samples is not None:
             indices = np.random.choice(raw_data.shape[0], n_samples, replace=False)
             raw_data = raw_data[indices]
             
        self.data = torch.tensor(raw_data, dtype=torch.float32)
        print(f"Data loaded. Shape: {self.data.shape}")
        
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        # Input: (x, y, t) -> indices [1, 2, 0]
        # Target: (u, v)    -> indices [3, 4]
        row = self.data[idx]
        
        # Construct inputs (x, y, t)
        inputs = torch.tensor([row[1], row[2], row[0]], dtype=torch.float32)
        
        # Construct targets (u, v)
        targets = torch.tensor([row[3], row[4]], dtype=torch.float32)
        
        return inputs, targets