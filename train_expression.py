import torch
from torch import optim
from models.encoder import ExpressionEncoder
from models.mine import MINE
from utils.losses import l1_loss
from datasets.fer_loader import FERDataset
from torch.utils.data import DataLoader
import os
import pandas as pd
from torchvision import transforms
from PIL import Image

# Prepare dataset
csv_path = "/content/datasets/rafdb/train/labels.csv"
df = pd.read_csv(csv_path)
image_paths = [os.path.join("datasets", "rafdb", "train", fname) for fname in df['filename']]
labels = df['expression'].tolist()
transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
dataset = FERDataset(image_paths, labels, transform)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

expr_enc = ExpressionEncoder().cuda()
mine = MINE(128).cuda()
opt = optim.Adam(list(expr_enc.parameters()) + list(mine.parameters()), lr=1e-4)

for epoch in range(10):
    for img, _ in dataloader:
        img = img.cuda()
        e1 = expr_enc(img)
        e2 = expr_enc(img[torch.randperm(img.size(0))])
        
        joint = mine(e1, e1).mean()
        marg = torch.exp(mine(e1, e2)).mean()
        mi_loss = -(joint - torch.log(marg))
        l1 = l1_loss(e1, e2)

        loss = mi_loss + 0.1 * l1
        opt.zero_grad()
        loss.backward()
        opt.step()

torch.save(expr_enc.state_dict(), "expression_model.pth")
