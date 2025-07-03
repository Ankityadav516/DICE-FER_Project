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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

csv_path = "/content/datasets/rafdb/train/labels.csv"
base_path = "/content/datasets/rafdb/train"

df = pd.read_csv(csv_path)
df = df[df["filename"].apply(lambda x: os.path.exists(os.path.join(base_path, x)))]

image_paths = [os.path.join(base_path, fname) for fname in df["filename"]]
labels = df["expression"].tolist()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

dataset = FERDataset(image_paths, labels, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, drop_last=True)

expr_enc = ExpressionEncoder().to(device)
mine = MINE(input_dim=256).to(device)  # 128 + 128 = 256

# 🔽 Smaller learning rate for expr_enc, weight decay added for mine
expr_opt = optim.Adam(expr_enc.parameters(), lr=5e-5)
mine_opt = optim.Adam(mine.parameters(), lr=1e-5, weight_decay=1e-4)

epochs = 10
for epoch in range(epochs):
    expr_enc.train()
    mine.train()
    print(f"\n📘 Epoch {epoch+1}/{epochs}")

    for step, (img, _) in enumerate(dataloader):
        img = img.to(device)

        z = expr_enc(img)
        z1, z2 = torch.chunk(z, 2, dim=0)

        if z1.size(0) != z2.size(0):
            continue

        idx = torch.randperm(z2.size(0))
        z2_shuffled = z2[idx]

        mi_pos = mine(z1, z2)
        mi_neg = mine(z1, z2_shuffled)

        mine_loss = -torch.mean(mi_pos) + torch.logsumexp(mi_neg, dim=0).mean() - torch.log(torch.tensor(z1.size(0), dtype=torch.float).to(device))
        recon_loss = l1_loss(z1, z2)
        total_loss = mine_loss + 0.1 * recon_loss

        expr_opt.zero_grad()
        mine_opt.zero_grad()
        total_loss.backward()

        # 🛡️ Gradient clipping
        torch.nn.utils.clip_grad_norm_(expr_enc.parameters(), max_norm=5.0)
        torch.nn.utils.clip_grad_norm_(mine.parameters(), max_norm=5.0)

        # 🧪 Check for gradient issues (debug)
        for name, param in expr_enc.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any():
                    print(f"❌ NaN in gradient: {name}")
                if torch.max(param.grad) > 1e3:
                    print(f"⚠️ Large gradient in: {name}, max = {torch.max(param.grad)}")

        expr_opt.step()
        mine_opt.step()

        if step % 10 == 0:
            print(f"  Step {step}: Loss = {total_loss.item():.6f}")

torch.save(expr_enc.state_dict(), "expression_model.pth")
print("✅ Expression encoder saved.")
