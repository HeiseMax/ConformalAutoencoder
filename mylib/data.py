import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def make_half_sphere(n_samples=1000, noise=0.1, random_state=None):
    """
    Generate a half-sphere dataset.
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # Generate random points on a sphere
    phi = np.random.uniform(0, np.pi, n_samples)
    theta = np.random.uniform(0, 2 * np.pi, n_samples)
    
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    
    # Keep only the upper hemisphere
    mask = z >= 0
    x = x[mask]
    y = y[mask]
    z = z[mask]
    
    # Add noise
    noise_x = np.random.normal(0, noise, x.shape)
    noise_y = np.random.normal(0, noise, y.shape)
    noise_z = np.random.normal(0, noise, z.shape)
    
    x += noise_x
    y += noise_y
    z += noise_z

    labels = z
    
    return np.column_stack((x, y, z)), labels

def load_CelebA(transform=[], split=0.8, seed=36):
    """
    Load the CelebA dataset.
    """
    data_path = 'data/CelebA/'
    train_dataset = datasets.ImageFolder(
        root=data_path,
        transform = transforms.Compose([
            transforms.ToTensor(),
            *transform
        ])
    )

    # create train test split
    train_size = int(split * len(train_dataset))
    test_size = len(train_dataset) - train_size
    generator = torch.Generator().manual_seed(seed)
    train_dataset, test_dataset = torch.utils.data.random_split(train_dataset, [train_size, test_size], generator=generator)

    return train_dataset, test_dataset
