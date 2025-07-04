import torch
from torchvision import transforms
from PIL import Image
import os

from models.encoder import ExpressionEncoder
import torch.nn.functional as F

# ✅ Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ Expression classes (as used during training)
class_names = ['surprise', 'fear', 'disgust', 'happy', 'sad', 'angry', 'neutral']

# ✅ Load models from Drive
expr_enc_path = "/content/drive/MyDrive/expression_model_final.pth"
cls_head_path = "/content/drive/MyDrive/expression_classifier_final.pth"

expr_enc = ExpressionEncoder().to(device)
expr_enc.load_state_dict(torch.load(expr_enc_path, map_location=device))
expr_enc.eval()

classifier_head = torch.nn.Linear(128, 7).to(device)
classifier_head.load_state_dict(torch.load(cls_head_path, map_location=device))
classifier_head.eval()

# ✅ Define preprocessing (same as training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ✅ Function to predict expression from image path
def predict_expression(image_path):
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        z = expr_enc(img_tensor)
        z1, _ = torch.chunk(z, 2, dim=0)  # use z1 as in training
        logits = classifier_head(z1)
        probs = F.softmax(logits, dim=1)
        top_prob, top_class = torch.max(probs, dim=1)

    predicted = class_names[top_class.item()]
    confidence = top_prob.item()
    print(f"🧠 Predicted Expression: {predicted} ({confidence * 100:.2f}% confidence)")

    return predicted, confidence



for fname in uploaded.keys():
    print(f"\n📷 Running inference on: {fname}")
    predict_expression(fname)
