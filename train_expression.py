import torch
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import os
import pandas as pd
import matplotlib.pyplot as plt

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
expr_cls_head = torch.nn.Linear(128, 7).to(device)  
mine = MINE(input_dim=256).to(device)  

expr_opt = optim.Adam(expr_enc.parameters(), lr=5e-5)
cls_opt = optim.Adam(expr_cls_head.parameters(), lr=5e-5)
mine_opt = optim.Adam(mine.parameters(), lr=1e-5, weight_decay=1e-4)

accuracies = []

epochs = 60  
for epoch in range(epochs):
    expr_enc.train()
    expr_cls_head.train()
    mine.train()

    print(f"\n Epoch {epoch+1}/{epochs}")
    total_correct = 0
    total_samples = 0

    for step, (img, lbl) in enumerate(dataloader):
        img = img.to(device)
        lbl = lbl.to(device)

        z = expr_enc(img)              
        z1, z2 = torch.chunk(z, 2, dim=0)
        lbl1, lbl2 = torch.chunk(lbl, 2, dim=0)

        if z1.size(0) != z2.size(0): continue

        idx = torch.randperm(z2.size(0))
        z2_shuffled = z2[idx]

        mi_pos = mine(z1, z2)
        mi_neg = mine(z1, z2_shuffled)

        mine_loss = -torch.mean(mi_pos) + torch.logsumexp(mi_neg, dim=0).mean() - torch.log(torch.tensor(z1.size(0), dtype=torch.float).to(device))
        recon_loss = l1_loss(z1, z2)
        logits = expr_cls_head(z1)
        cls_loss = torch.nn.functional.cross_entropy(logits, lbl1)

        total_loss = mine_loss + 0.1 * recon_loss + 0.5 * cls_loss

        expr_opt.zero_grad()
        mine_opt.zero_grad()
        cls_opt.zero_grad()
        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(expr_enc.parameters(), max_norm=5.0)
        torch.nn.utils.clip_grad_norm_(mine.parameters(), max_norm=5.0)

        expr_opt.step()
        mine_opt.step()
        cls_opt.step()

        _, preds = torch.max(logits, dim=1)
        total_correct += (preds == lbl1).sum().item()
        total_samples += lbl1.size(0)

        if step % 10 == 0:
            print(f"  Step {step}: Loss = {total_loss.item():.6f}")

    acc_epoch = total_correct / total_samples
    accuracies.append(acc_epoch)
    print(f" Epoch {epoch+1} Accuracy (on z1): {acc_epoch:.4f}")

    if (epoch + 1) % 10 == 0:
        os.makedirs("/content/drive/MyDrive/DICE-FER-Checkpoints", exist_ok=True)
        torch.save(expr_enc.state_dict(), f"/content/drive/MyDrive/DICE-FER-Checkpoints/expression_encoder_epoch{epoch+1}.pth")
        torch.save(expr_cls_head.state_dict(), f"/content/drive/MyDrive/DICE-FER-Checkpoints/expression_classifier_epoch{epoch+1}.pth")
        print(f" Saved checkpoint at epoch {epoch+1}")

torch.save(expr_enc.state_dict(), "/content/drive/MyDrive/expression_model_final.pth")
torch.save(expr_cls_head.state_dict(), "/content/drive/MyDrive/expression_classifier_final.pth")
print(" Expression encoder and classifier saved to Drive.")

plt.plot(range(1, len(accuracies) + 1), accuracies, marker='o')
plt.xlabel("Epoch")
plt.ylabel("Accuracy on z1")
plt.title("Expression Encoder Accuracy Over Epochs")
plt.grid(True)
plt.savefig("/content/drive/MyDrive/expression_accuracy_plot.png")
plt.show()