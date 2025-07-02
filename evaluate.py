import torch
from models.encoder import ExpressionEncoder
from datasets.fer_loader import FERDataset
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import os
import pandas as pd
from torchvision import transforms

csv_path = os.path.join("datasets", "rafdb", "train", "labels.csv")
df = pd.read_csv(csv_path)
image_paths = [os.path.join("datasets", "rafdb", "train", fname) for fname in df['filename']]
labels = df['expression'].tolist()
transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
dataset = FERDataset(image_paths, labels, transform)
dataloader = DataLoader(dataset, batch_size=8, shuffle=False)

expr_enc = ExpressionEncoder().cuda()
expr_enc.load_state_dict(torch.load("expression_model.pth"))
expr_enc.eval()

X, y = [], []
for img, label in dataloader:
    with torch.no_grad():
        feat = expr_enc(img.cuda()).cpu()
    X.append(feat)
    y.extend(label)

X = torch.cat(X).numpy()
clf = LogisticRegression(max_iter=1000)
clf.fit(X, y)
y_pred = clf.predict(X)
print("Accuracy:", accuracy_score(y, y_pred))
