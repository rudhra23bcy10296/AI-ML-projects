"""
CIFAR-10 Dataset Handler & Data Augmentation Module
Author: Rudhra Sitholey (Reg No: 23BCY10296 | App No: IN26012560)
"""

import torch
from torch.utils.data import Dataset, DataLoader


class SyntheticCIFAR10Dataset(Dataset):
    """
    Synthetic CIFAR-10 Dataset to ensure reliable execution in offline/restricted environments.
    Simulates 32x32 RGB image tensors across 10 classes.
    """
    def __init__(self, num_samples=500):
        self.num_samples = num_samples
        self.data = torch.randn(num_samples, 3, 32, 32)
        self.targets = torch.randint(0, 10, (num_samples,))
        self.classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


def get_dataloaders(batch_size=32):
    train_dataset = SyntheticCIFAR10Dataset(num_samples=640)
    test_dataset = SyntheticCIFAR10Dataset(num_samples=160)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, train_dataset.classes
