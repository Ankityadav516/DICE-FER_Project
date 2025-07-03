import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F
import os
import pandas as pd
from PIL import Image

from models.encoder import ExpressionEncoder
from models.mine import MINE
from utils.losses import l1_loss
from datasets.fer_loader import FERDataset

# ✅ 1. Load dataset safely
csv_path = "/content/datasets/rafdb/train/labels.csv"
df = pd.read_csv(csv_path)

# Only keep images that actually exist
base_path = "/content/datasets/rafdb/train"
df = df[df["filename"].apply(lambda x: os.path.exists(os.path.join(base_path, x)))]

image_paths = [os.path.join(base_path, fname) for fname in df["filename"]]
labels = df["expression"].tolist()

# ✅ 2. Apply transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

dataset = FERDataset(image_paths, labels, transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# ✅ 3. Initialize models
expr_enc = ExpressionEncoder().cuda()
mine = MINE(128).cuda()

# ✅ 4. Optimizer
opt = optim.Adam(list(expr_enc.parameters()) + list(mine.parameters()), lr=1e-4)

# ✅ 5. Training Loop
for epoch in range(10):
    print(f"\n📘 Epoch {epoch+1}/10")
    for step, (img, _) in enumerate(dataloader):
        img = img.cuda()

        # Get embeddings
        e1 = expr_enc(img)
        e2 = expr_enc(img[torch.randperm(img.size(0))])

        # 🔒 Normalize to prevent MI from exploding
        e1 = F.normalize(e1, dim=1)
        e2 = F.normalize(e2, dim=1)

        # Compute joint and marginal
        joint = mine(e1, e1).mean()
        log_marg = torch.log(torch.exp(mine(e1, e2)).clamp(min=1e-6)).mean()

        # Mutual Information loss
        mi_loss = -(joint - log_marg)

        # Add L1 loss
        l1 = l1_loss(e1, e2)

        # Total loss
        loss = mi_loss + 0.1 * l1

        opt.zero_grad()
        loss.backward()
        opt.step()

        # 📉 Print every 10 steps
        if step % 10 == 0:
            print(f"  Step {step}: Loss = {loss.item():.4f}")

# ✅ 6. Save model
torch.save(expr_enc.state_dict(), "expression_model.pth")
print("✅ Expression encoder saved.")
