import torch
from models.encoder import ExpressionEncoder
from datasets.fer_loader import FERDataset
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import os
import pandas as pd
from torchvision import transforms
import numpy as np

# 📁 Load dataset
csv_path = "/content/datasets/rafdb/train/labels.csv"
df = pd.read_csv(csv_path)
image_paths = [os.path.join("/content/datasets/rafdb/train", fname) for fname in df['filename']]
labels = df["expression"].tolist()

# 🧼 Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

dataset = FERDataset(image_paths, labels, transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

# 🧠 Load pretrained Expression Encoder
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
expr_enc = ExpressionEncoder().to(device)
expr_enc.load_state_dict(torch.load("expression_model.pth", map_location=device))
expr_enc.eval()

# 🧪 Extract features
X, y = [], []
with torch.no_grad():
    for img, label in dataloader:
        img = img.to(device)
        features = expr_enc(img).cpu().numpy()
        X.append(features)
        y.extend(label.tolist())

X = np.concatenate(X, axis=0)
y = np.array(y)

# 📈 Logistic Regression
clf = LogisticRegression(max_iter=1000)
clf.fit(X, y)
y_pred = clf.predict(X)
acc = accuracy_score(y, y_pred)
print(f"📊 Expression Classification Accuracy (linear probe): {acc:.4f}")
