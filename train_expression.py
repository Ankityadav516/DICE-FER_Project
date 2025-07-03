import torch
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import os
import pandas as pd

from models.encoder import ExpressionEncoder
from models.mine import MINE
from datasets.fer_loader import FERDataset
from utils.losses import l1_loss

# Device config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 📁 Path to CSV
csv_path = "/content/datasets/rafdb/train/labels.csv"
base_path = "/content/datasets/rafdb/train"

# ✅ Filter for only existing image paths
df = pd.read_csv(csv_path)
df = df[df["filename"].apply(lambda x: os.path.exists(os.path.join(base_path, x)))]

image_paths = [os.path.join(base_path, fname) for fname in df["filename"]]
labels = df["expression"].tolist()

# 🧼 Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# 📦 Dataset and Loader
dataset = FERDataset(image_paths, labels, transform=transform)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# 🧠 Models
expr_enc = ExpressionEncoder().to(device)
mine = MINE(input_dim=64).to(device)

# ⚙️ Optimizer
optimizer = optim.Adam(list(expr_enc.parameters()) + list(mine.parameters()), lr=1e-4)

# 🔁 Training loop
epochs = 10
for epoch in range(epochs):
    print(f"\n📘 Epoch {epoch+1}/{epochs}")
    for step, (img, _) in enumerate(dataloader):
        img = img.to(device)
        z = expr_enc(img)
        z1, z2 = torch.chunk(z, 2, dim=0)

        # 🔄 Shuffle for negative samples
        idx = torch.randperm(z1.size(0))
        z2_shuffled = z2[idx]

        # 💡 MINE loss
        mi_pos = mine(z1, z2)
        mi_neg = mine(z1, z2_shuffled)
        mine_loss = -torch.mean(mi_pos) + torch.log(torch.mean(torch.exp(mi_neg)))

        # 🧮 L1 reconstruction loss
        recon_loss = l1_loss(z1, z2)

        # 🎯 Total loss
        total_loss = mine_loss + 0.1 * recon_loss

        optimizer.zero_grad()
        total_loss.backward()

        # ✅ Gradient clipping
        torch.nn.utils.clip_grad_norm_(expr_enc.parameters(), max_norm=5.0)

        optimizer.step()

        if step % 10 == 0:
            print(f"  Step {step}: Loss = {total_loss.item():.4f}")

# ✅ 6. Save model
torch.save(expr_enc.state_dict(), "expression_model.pth")
print("✅ Expression encoder saved.")
