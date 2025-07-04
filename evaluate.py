import torch
from models.encoder import ExpressionEncoder
from datasets.fer_loader import FERDataset
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os
import pandas as pd
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#  Paths
test_csv = "/content/datasets/rafdb/test/labels.csv"
test_base = "/content/datasets/rafdb/test"

#  Load test metadata
df = pd.read_csv(test_csv)
image_paths = [os.path.join(test_base, fname) for fname in df['filename']]
labels = df['expression'].tolist()

#  Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

#  Dataset
test_dataset = FERDataset(image_paths, labels, transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

#  Load encoder
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
expr_enc = ExpressionEncoder().to(device)
expr_enc.load_state_dict(torch.load("/content/drive/MyDrive/expression_model_final.pth", map_location=device))
expr_enc.eval()

#  Extract test features
X_test, y_test = [], []
with torch.no_grad():
    for img, label in test_loader:
        img = img.to(device)
        z = expr_enc(img)
        z1, _ = torch.chunk(z, 2, dim=0)
        label = torch.chunk(label, 2, dim=0)[0]
        X_test.append(z1.cpu().numpy())
        y_test.extend(label.cpu().numpy())

X_test = np.concatenate(X_test, axis=0)
y_test = np.array(y_test)

#  Load train features (from saved file or re-extract)
# If you don’t have saved train features, extract from train again here
train_csv = "/content/datasets/rafdb/train/labels.csv"
df_train = pd.read_csv(train_csv)
train_paths = [os.path.join("/content/datasets/rafdb/train", fname) for fname in df_train['filename']]
train_labels = df_train['expression'].tolist()

train_dataset = FERDataset(train_paths, train_labels, transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)

X_train, y_train = [], []
with torch.no_grad():
    for img, label in train_loader:
        img = img.to(device)
        z = expr_enc(img)
        z1, _ = torch.chunk(z, 2, dim=0)
        label = torch.chunk(label, 2, dim=0)[0]
        X_train.append(z1.cpu().numpy())
        y_train.extend(label.cpu().numpy())

X_train = np.concatenate(X_train, axis=0)
y_train = np.array(y_train)

#  Train classifier
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

#  Evaluate
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n Linear Probe Accuracy on Test Set: {acc:.4f}")
print("\n Classification Report:\n")
print(classification_report(y_test, y_pred))

#  Plot confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix (Test Set)")
plt.xlabel("Predicted")
plt.ylabel("True")
os.makedirs("/content/drive/MyDrive/DICE-FER-Results", exist_ok=True)
plt.savefig("/content/drive/MyDrive/DICE-FER-Results/expression_confusion_matrix_testset.png")
print(" Test Confusion matrix saved to Drive.")
