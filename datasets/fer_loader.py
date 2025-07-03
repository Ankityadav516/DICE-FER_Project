from torch.utils.data import Dataset
from PIL import Image
import torch

class FERDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.transform = transform

        # Step 1: Get unique string labels and map to integer IDs
        unique_labels = sorted(set(labels))
        self.label2idx = {label: idx for idx, label in enumerate(unique_labels)}

        # Step 2: Convert all string labels to integer indices
        self.labels = [self.label2idx[label] for label in labels]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(self.labels[idx], dtype=torch.long)
