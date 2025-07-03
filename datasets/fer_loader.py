from torch.utils.data import Dataset
from PIL import Image
import torch

class FERDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.transform = transform

        # 🔢 Map string labels to indices
        unique_labels = sorted(set(labels))
        self.label2idx = {label: idx for idx, label in enumerate(unique_labels)}
        self.idx2label = {idx: label for label, idx in self.label2idx.items()}

        # 🧾 Convert string labels to integers
        self.labels = [self.label2idx[label] for label in labels]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.image_paths[idx]).convert('RGB')
        except Exception as e:
            print(f"❌ Error loading image: {self.image_paths[idx]} — {e}")
            return self.__getitem__((idx + 1) % len(self))

        if self.transform:
            img = self.transform(img)

        label = torch.tensor(self.labels[idx]).item()  # returns int
        return img, label
