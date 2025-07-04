!git clone https://github.com/Ankityadav516/DICE-FER_Project.git
%cd DICE-FER_Project

!pip install -r requirements.txt
from google.colab import drive
drive.mount('/content/drive')
!cp "/content/drive/MyDrive/raf-db-dataset.zip" /content/
import zipfile, os, shutil, pandas as pd

# Extract
with zipfile.ZipFile("/content/raf-db-dataset.zip", "r") as zip_ref:
    zip_ref.extractall("/content/rafdb_raw")

# Organize images + labels
src_root = "/content/rafdb_raw/DATASET/train"
dst_root = "/content/datasets/rafdb/train"
os.makedirs(dst_root, exist_ok=True)

label_map = {
    "1": "surprise", "2": "fear", "3": "disgust",
    "4": "happy", "5": "sad", "6": "angry", "7": "neutral"
}

records = []
img_counter = 0
for class_id in sorted(os.listdir(src_root)):
    class_path = os.path.join(src_root, class_id)
    if not os.path.isdir(class_path): continue
    for fname in os.listdir(class_path):
        try:
            old_path = os.path.join(class_path, fname)
            new_fname = f"img_{img_counter:05d}.jpg"
            new_path = os.path.join(dst_root, new_fname)
            shutil.copy(old_path, new_path)
            records.append([new_fname, label_map[class_id]])
            img_counter += 1
        except:
            continue

df = pd.DataFrame(records, columns=["filename", "expression"])
df.to_csv(os.path.join(dst_root, "labels.csv"), index=False)

print(f"✅ Copied {img_counter} images and created synced labels.csv")
%cd /content/DICE-FER_Project
!git pull
!python3 /content/DICE-FER_Project/train_expression.py

!python3 /content/DICE-FER_Project/train_identity.py
import os
import shutil
import pandas as pd

# Paths
src_root = "/content/rafdb_raw/DATASET/test"
dst_root = "/content/datasets/rafdb/test"
os.makedirs(dst_root, exist_ok=True)

label_map = {
    "1": "surprise", "2": "fear", "3": "disgust",
    "4": "happy", "5": "sad", "6": "angry", "7": "neutral"
}

records = []
img_counter = 0

# Loop through class folders
for class_id in sorted(os.listdir(src_root)):
    class_path = os.path.join(src_root, class_id)
    if not os.path.isdir(class_path): continue

    for fname in os.listdir(class_path):
        old_path = os.path.join(class_path, fname)
        new_fname = f"img_{img_counter:05d}.jpg"
        new_path = os.path.join(dst_root, new_fname)

        try:
            shutil.copy(old_path, new_path)
            records.append([new_fname, label_map[class_id]])
            img_counter += 1
        except:
            continue  # skip broken images

# Save test labels.csv
df = pd.DataFrame(records, columns=["filename", "expression"])
df.to_csv(os.path.join(dst_root, "labels.csv"), index=False)

print(f"✅ Copied {img_counter} test images and created labels.csv")


!python3 /content/DICE-FER_Project/evaluate.py


!python3 /content/DICE-FER_Project/inference.py /content/image4.jpeg


