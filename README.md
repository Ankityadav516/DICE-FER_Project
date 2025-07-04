# DICE-FER Project: Disentangled Identity and Expression Representations for Facial Expression Recognition

This repository contains our course project implementation of the DICE-FER architecture for facial expression recognition (FER), developed as part of **EE656: Artificial Intelligence, Machine Learning, Deep Learning and Its Applications** at **IIT Kanpur**.

Our goal was to disentangle expression and identity features from facial images using mutual information estimation and adversarial training.

---

## Team Members

* Ankit
* Akash Guru
* Kushal Agrawal
* Sanjay

---

## 📁 Repository Structure

```
DICE-FER_Project/
├── train_expression.py           # Train expression encoder + classifier
├── train_identity.py            # Train identity encoder + MINE + discriminator
├── evaluate.py                  # Linear probing evaluation on test set
├── inference.py                 # Inference on a single image
├── models/
│   ├── encoder.py               # Expression & Identity Encoders (ResNet18)
│   ├── discriminator.py         # Adversarial Discriminator
│   └── mine.py                  # Mutual Information Estimator
├── datasets/
│   └── fer_loader.py           # Custom PyTorch Dataset for RAF-DB
├── utils/
│   └── losses.py               # Custom loss functions (MI, L1, Adv)
├── config.yaml                  # Config file (hyperparameters, paths)
├── requirements.txt             # Dependencies
├── google_collab_code.py        # Full Colab workflow for training & evaluation
└── README.md                    # This file
```

---

## Dataset: RAF-DB

* The **Real-world Affective Faces Database (RAF-DB)** contains \~30,000 images labeled with 7 basic emotions:

  * Happy, Sad, Angry, Disgust, Fear, Surprise, Neutral
* Pre-aligned, RGB face images
* We generated custom `labels.csv` files for PyTorch loading

---

## Model Overview

We implemented a simplified DICE-FER pipeline consisting of:

* **Expression Encoder**: ResNet-18 → 128D features
* **Identity Encoder**: ResNet-18 → 128D features
* **Classifier Head**: Linear(128 → 7)
* **MINE Module**: Mutual information minimization between features
* **Discriminator**: Adversarial loss to enforce expression–identity disentanglement

---

## Training Setup

* **Stage 1**: Train expression encoder + classifier using:

  * MI loss, L1 loss, and cross-entropy loss
* **Stage 2**: Freeze expression encoder, train identity encoder + MINE + discriminator
* All models trained from scratch on RAF-DB (Colab)

| Parameter     | Value        |
| ------------- | ------------ |
| Epochs        | 60 per stage |
| Batch size    | 32           |
| Input size    | 224x224      |
| Optimizer     | Adam         |
| LR (Expr)     | 5e-5         |
| LR (ID/MINE)  | 1e-4 / 1e-5  |
| Grad Clipping | 5.0          |

---

## Evaluation & Results

We performed **linear probing** on expression features using logistic regression:

| Metric         | Value  |
| -------------- | ------ |
| Test Accuracy  | 82.27% |
| Macro F1 Score | 72.3%  |

Visual outputs:

* `expression_accuracy_plot.png`: accuracy over epochs
* `identity_loss_plot.png`: MI + adversarial loss trend
* `expression_confusion_matrix_testset.png`: test confusion matrix

---

## Pretrained Models

Download the pretrained models and place them in the root folder (or update paths in the code):

| Model                      | Link                                                                                         |
| -------------------------- | -------------------------------------------------------------------------------------------- |
| Expression Encoder         | [Download](https://drive.google.com/uc?export=download&id=1wF9gVtHABeM6O2ozJrTe7urDqXJXvuXO) |
| Expression Classifier Head | [Download](https://drive.google.com/uc?export=download&id=1AAJt_b5MrVHNcFyy0cLWUNK9Ufn4DZeL) |
| Identity Encoder           | [Download](https://drive.google.com/uc?export=download&id=14Zv940i78ViRDf88C9C1867E6El2pRu0) |

---

##  How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train expression encoder

```bash
python train_expression.py
```

### 3. Train identity encoder

```bash
python train_identity.py
```

### 4. Evaluate with logistic regression

```bash
python evaluate.py
```

### 5. Run inference on a new image

```bash
python inference.py path/to/image.jpg
```

---

## ⚠Limitations and Future Work

While the original DICE-FER paper included more components, we focused on the core disentanglement pipeline. Components we did not implement:

* Ablation studies on loss terms
* Image retrieval from embeddings
* Cross-dataset evaluation (e.g., CK+, AffectNet)
* Use of multiple MINE modules (local/global)

These remain directions for future exploration.

---

## References

* DICE-FER: Disentangled Identity and Expression Representations for FER (CVPR 2020 Workshop)
* RAF-DB: Real-world Affective Faces Database

---

## License

This repository is part of a course project and shared for educational purposes only.
