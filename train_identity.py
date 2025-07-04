import torch
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd
import os
import matplotlib.pyplot as plt

from models.encoder import IdentityEncoder, ExpressionEncoder
from models.discriminator import Discriminator
from models.mine import MINE
from datasets.fer_loader import FERDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

csv_path = "/content/datasets/rafdb/train/labels.csv"
df = pd.read_csv(csv_path)
image_paths = [os.path.join("/content/datasets/rafdb/train", fname) for fname in df["filename"]]
labels = df["expression"].tolist()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

dataset = FERDataset(image_paths, labels, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, drop_last=True)

expr_enc = ExpressionEncoder().to(device)
expr_enc.load_state_dict(torch.load("/content/drive/MyDrive/expression_model_final.pth", map_location=device))
expr_enc.eval()  

id_enc = IdentityEncoder().to(device)
dis = Discriminator(input_dim=128).to(device)
mine = MINE(input_dim=256).to(device)

opt = optim.Adam(
    list(id_enc.parameters()) + list(dis.parameters()) + list(mine.parameters()),
    lr=1e-4,
    weight_decay=1e-4
)

epoch_losses = []
epochs = 60

for epoch in range(epochs):
    print(f"\n Epoch {epoch+1}/{epochs}")
    id_enc.train()
    dis.train()
    mine.train()

    running_loss = 0
    for step, (img, _) in enumerate(dataloader):
        img = img.to(device)

        with torch.no_grad():
            e = expr_enc(img)

        i = id_enc(img)
        i_shuffled = i[torch.randperm(i.size(0))]

        #  MINE Loss
        mi_pos = mine(e, i)
        mi_neg = mine(e, i_shuffled)
        log_batch_size = torch.log(torch.tensor(i.size(0), dtype=torch.float, device=device))
        mi_loss = -torch.mean(mi_pos) + torch.logsumexp(mi_neg, dim=0).mean() - log_batch_size

        #  Adversarial Loss
        real_logits = dis(e, i)
        fake_logits = dis(e, i_shuffled)
        adv_loss = torch.mean((real_logits - 1)**2 + fake_logits**2)

        total_loss = 1.0 * mi_loss + 0.1 * adv_loss
        running_loss += total_loss.item()

        #  Backprop
        opt.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(id_enc.parameters(), max_norm=5.0)
        torch.nn.utils.clip_grad_norm_(mine.parameters(), max_norm=5.0)
        torch.nn.utils.clip_grad_norm_(dis.parameters(), max_norm=5.0)
        opt.step()

        if step % 10 == 0:
            print(f"  Step {step}: Loss = {total_loss.item():.6f}")

    avg_loss = running_loss / len(dataloader)
    epoch_losses.append(avg_loss)
    print(f" Epoch {epoch+1} Avg Loss: {avg_loss:.4f}")

    #  Save every 10 epochs to Drive
    if (epoch + 1) % 10 == 0:
        os.makedirs("/content/drive/MyDrive/DICE-FER-Checkpoints", exist_ok=True)
        torch.save(id_enc.state_dict(), f"/content/drive/MyDrive/DICE-FER-Checkpoints/identity_model_epoch{epoch+1}.pth")
        torch.save(dis.state_dict(), f"/content/drive/MyDrive/DICE-FER-Checkpoints/discriminator_epoch{epoch+1}.pth")
        torch.save(mine.state_dict(), f"/content/drive/MyDrive/DICE-FER-Checkpoints/mine_epoch{epoch+1}.pth")
        print(f" Saved checkpoint at epoch {epoch+1}")

#  Final Save
torch.save(id_enc.state_dict(), "/content/drive/MyDrive/identity_model_final.pth")
print(" Identity encoder saved to Drive")

#  Plot Loss Curve
plt.plot(epoch_losses, label="Total Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Identity Training Loss Curve")
plt.grid(True)
plt.savefig("/content/drive/MyDrive/identity_loss_plot.png")
print(" Loss curve saved to Drive")