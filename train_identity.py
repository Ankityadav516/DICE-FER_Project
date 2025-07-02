import torch
from models.encoder import IdentityEncoder, ExpressionEncoder
from models.discriminator import Discriminator
from models.mine import MINE
from torch import optim
from datasets.fer_loader import FERDataset
from torch.utils.data import DataLoader
import os
import pandas as pd
from torchvision import transforms

csv_path = os.path.join("datasets", "rafdb", "train", "labels.csv")
df = pd.read_csv(csv_path)
image_paths = [os.path.join("datasets", "rafdb", "train", fname) for fname in df['filename']]
labels = df['expression'].tolist()
transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
dataset = FERDataset(image_paths, labels, transform)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

expr_enc = ExpressionEncoder().cuda()
expr_enc.load_state_dict(torch.load("expression_model.pth"))
expr_enc.eval()
id_enc = IdentityEncoder().cuda()
dis = Discriminator(128).cuda()
mine = MINE(256).cuda()
opt = optim.Adam(list(id_enc.parameters()) + list(dis.parameters()) + list(mine.parameters()), lr=1e-4)

for epoch in range(10):
    for img, _ in dataloader:
        img = img.cuda()
        with torch.no_grad():
            e = expr_enc(img)
        i = id_enc(img)

        joint = mine(e, i).mean()
        marg = torch.exp(mine(e, i[torch.randperm(i.size(0))])).mean()
        mi_loss = -(joint - torch.log(marg))

        real = dis(e, i)
        fake = dis(e, i[torch.randperm(i.size(0))])
        adv = -torch.mean(torch.log(real + 1e-8) + torch.log(1 - fake + 1e-8))

        loss = mi_loss + 0.1 * adv
        opt.zero_grad()
        loss.backward()
        opt.step()

torch.save(id_enc.state_dict(), "identity_model.pth")

