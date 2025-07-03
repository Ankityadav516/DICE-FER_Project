import torch
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd
import os

from models.encoder import IdentityEncoder, ExpressionEncoder
from models.discriminator import Discriminator
from models.mine import MINE
from datasets.fer_loader import FERDataset

# 📍 Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 📁 Load data
csv_path = "/content/datasets/rafdb/train/labels.csv"
df = pd.read_csv(csv_path)
image_paths = [os.path.join("/content/datasets/rafdb/train", fname) for fname in df["filename"]]
labels = df["expression"].tolist()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

dataset = FERDataset(image_paths, labels, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, drop_last=True)  # ✅ Larger batch

# 🧠 Models
expr_enc = ExpressionEncoder().to(device)
expr_enc.load_state_dict(torch.load("expression_model.pth", map_location=device))
expr_enc.eval()

id_enc = IdentityEncoder().to(device)
dis = Discriminator(input_dim=128).to(device)
mine = MINE(input_dim=256).to(device)  # e and i are 256-dim

# 🧮 Optimizer
opt = optim.Adam(
    list(id_enc.parameters()) + list(dis.parameters()) + list(mine.parameters()),
    lr=1e-4,
    weight_decay=1e-4
)

# 🔁 Training loop
for epoch in range(100):
    print(f"\n📘 Epoch {epoch+1}/10")
    id_enc.train()
    dis.train()
    mine.train()

    for step, (img, _) in enumerate(dataloader):
        img = img.to(device)

        with torch.no_grad():
            e = expr_enc(img)  # Expression embeddings

        i = id_enc(img)       # Identity embeddings

        # 🔀 Shuffle for negative MI samples
        i_shuffled = i[torch.randperm(i.size(0))]

        # 📊 Mutual Information loss
        mi_pos = mine(e, i)
        mi_neg = mine(e, i_shuffled)
        mi_loss = -torch.mean(mi_pos) + torch.logsumexp(mi_neg, dim=0).mean() - torch.log(torch.tensor(i.size(0), dtype=torch.float).to(device))

        # 🧨 Adversarial Loss
        real_logits = dis(e, i)
        fake_logits = dis(e, i_shuffled)
        adv_loss = -torch.mean(torch.log(real_logits + 1e-6) + torch.log(1 - fake_logits + 1e-6))

        # 🎯 Total loss
        total_loss = 1.0 * mi_loss + 0.1 * adv_loss

        # ✅ Backprop
        opt.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(id_enc.parameters(), max_norm=5.0)
        torch.nn.utils.clip_grad_norm_(mine.parameters(), max_norm=5.0)
        torch.nn.utils.clip_grad_norm_(dis.parameters(), max_norm=5.0)
        opt.step()

        if step % 10 == 0:
            print(f"  Step {step}: Loss = {total_loss.item():.6f}")
    if (epoch + 1) % 10 == 0:
        torch.save(id_enc.state_dict(), f"identity_model_epoch{epoch+1}.pth")
        print(f"✅ Saved checkpoint: identity_model_epoch{epoch+1}.pth")
# 💾 Save identity encoder
torch.save(id_enc.state_dict(), "identity_model.pth")
print("✅ Identity encoder saved.")
