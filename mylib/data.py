from pathlib import Path
from PIL import Image
import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from sklearn.model_selection import train_test_split

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

# CelebA
def parse_celeba_attr_file(path, to_zero_one=True, max_lines=None):
    """
    Parse CelebA `list_attr_celeba.txt`.

    Returns:
      filenames: list[str]                # filenames (e.g. '000001.jpg')
      classnames: list[str]               # attribute names (e.g. 'Smiling', ...)
      labels: np.ndarray (N, M) uint8/int8 # binary matrix (or -1/1 if to_zero_one=False)
    """
    path = Path(path)
    with path.open('r', encoding='utf-8') as f:
        # first line: number of images (we can ignore or use for validation)
        first = f.readline()
        try:
            total = int(first.strip())
        except ValueError:
            total = None

        # second line: attribute names
        header = f.readline().strip()
        classnames = header.split()
        M = len(classnames)

        filenames = []
        labels = {}
        for i, line in enumerate(f):
            if max_lines is not None and i >= max_lines:
                break
            parts = line.strip().split()
            if len(parts) == 0:
                continue
            fname = parts[0]
            vals = parts[1:]
            if len(vals) != M:
                # optionally skip, or raise. Here we raise to surface format problems.
                raise ValueError(f'Line {i+3} ({fname}) has {len(vals)} values, expected {M}')
            # convert to ints: -1/1 -> (0/1) if requested
            arr = np.array(vals, dtype=np.int8)
            if to_zero_one:
                arr = (arr == 1).astype(np.uint8)
            filenames.append(fname)
            labels[fname] = arr

    # labels = np.vstack(rows).astype(np.uint8 if to_zero_one else np.int8)
    return filenames, classnames, labels

def get_celeba_label_names(root_dir=""):
    """
    Get the attribute names for CelebA dataset.
    """
    path = Path(f"{root_dir}data/CelebA/list_attr_celeba.txt")
    _, classnames, _ = parse_celeba_attr_file(path, to_zero_one=True)
    return classnames


class CelebA(Dataset):
    def __init__(self, root_dir="", transform=None, filter_categories = [], extensions=(".jpg",), test_size=0.2, split='train', device='cpu'):
        self.root_dir = Path(f"{root_dir}data/CelebA/img_align_celeba")
        self.device = device

        if transform is None:
            transform = [
                transforms.Pad((0, 3), padding_mode='symmetric'), # pad and crop from 218x178 to 224x176 (divisible by 16)
                transforms.CenterCrop((224, 176)),
            ]

        self.transform = transforms.Compose(transform + [transforms.ToTensor()])

        labels = parse_celeba_attr_file(Path(f"{root_dir}data/CelebA/list_attr_celeba.txt"))[2]
        self.samples = []

        folder = self.root_dir
        for p in folder.rglob("*"):
            if p.is_file() and p.suffix.lower() in extensions:
                if not all(labels[p.name][cat] == is_true for cat, is_true in filter_categories):
                    continue
                label = labels[p.name]
                self.samples.append((p, label))

        self.train_samples, self.test_samples = train_test_split(self.samples, test_size=test_size, random_state=42)

        if split == 'train':
            self.samples = self.train_samples
        else:
            self.samples = self.test_samples


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        with Image.open(path) as im:
            # handle grayscale/alpha robustly
            if im.mode in ("RGBA", "LA"):
                im = im.convert("RGBA").convert("RGB")
            else:
                im = im.convert("RGB")
        if self.transform:
            im = self.transform(im)
        return im.to(self.device), label
    
    def get_sample(self, num_samples=16, seed=42):
        """
        Get a batch of samples from the dataset.
        """
        np.random.seed(seed)
        indices = np.random.choice(len(self), num_samples, replace=False)
        images = []
        labels = []
        for idx in indices:
            img, label = self[idx]
            images.append(img)
            labels.append(torch.tensor(label))
        return torch.stack(images), torch.stack(labels)


class CelebA_small(Dataset):
    def __init__(self, root_dir="", transform=None, filter_categories = [], extensions=(".jpg",), test_size=0.2, split='train', device='cpu'):
        self.root_dir = Path(f"{root_dir}data/CelebA_small/images")
        self.device = device

        if transform is None:
            transform = []

        self.transform = transforms.Compose(transform + [transforms.ToTensor()])

        labels = parse_celeba_attr_file(Path(f"{root_dir}data/CelebA/list_attr_celeba.txt"))[2]
        self.samples = []

        folder = self.root_dir
        for p in folder.rglob("*"):
            if p.is_file() and p.suffix.lower() in extensions:
                if not all(labels[p.name][cat] == is_true for cat, is_true in filter_categories):
                    continue
                label = labels[p.name]
                self.samples.append((p, label))

        self.train_samples, self.test_samples = train_test_split(self.samples, test_size=test_size, random_state=42)

        if split == 'train':
            self.samples = self.train_samples
        else:
            self.samples = self.test_samples


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        with Image.open(path) as im:
            # handle grayscale/alpha robustly
            if im.mode in ("RGBA", "LA"):
                im = im.convert("RGBA").convert("RGB")
            else:
                im = im.convert("RGB")
        if self.transform:
            im = self.transform(im)
        return im.to(self.device), label
    
    def get_sample(self, num_samples=16, seed=42):
        """
        Get a batch of samples from the dataset.
        """
        np.random.seed(seed)
        indices = np.random.choice(len(self), num_samples, replace=False)
        images = []
        labels = []
        for idx in indices:
            img, label = self[idx]
            images.append(img)
            labels.append(torch.tensor(label))
        return torch.stack(images), torch.stack(labels)
