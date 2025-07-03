import torch
from models.encoder import ExpressionEncoder
from datasets.fer_loader import FERDataset
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import os
import pandas as pd
from torchvision import transforms
import numpy as np

# 📁 Load dataset
csv_path = "/content/datasets/rafdb/train/labels.csv"
df = pd.read_csv(csv_path)
image_paths = [os.path.join("/content/datasets/rafdb/train", fname) for fname in df['filename']]
labels = df["expression"].tolist()

# 🧼 Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

dataset = FERDataset(image_paths, labels, transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

# 🧠 Load pretrained encoder
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
expr_enc = ExpressionEncoder().to(device)
expr_enc.load_state_dict(torch.load("expression_model.pth", map_location=device))
expr_enc.eval()

# 🧪 Feature extraction
X, y = [], []
with torch.no_grad():
    for img, label in dataloader:
        img = img.to(device)
        z = expr_enc(img)
        z1, _ = torch.chunk(z, 2, dim=0)
        label = torch.chunk(label, 2, dim=0)[0]

        X.append(z1.cpu().numpy())
        y.extend(label.cpu().numpy())

X = np.concatenate(X, axis=0)
y = np.array(y)

# ✂️ Train-test split (stratified)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# 📈 Train probe
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

# 🔍 Evaluate
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"📊 Linear Probe Accuracy: {acc:.4f}")
print("\n🔎 Classification Report:\n")
print(classification_report(y_test, y_pred))
