import torch
from torchvision import transforms
from PIL import Image
import os
import torch.nn.functional as F
import sys

from models.encoder import ExpressionEncoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_names = ['surprise', 'fear', 'disgust', 'happy', 'sad', 'angry', 'neutral']

expr_enc_path = "/content/drive/MyDrive/expression_model_final.pth"
cls_head_path = "/content/drive/MyDrive/expression_classifier_final.pth"

expr_enc = ExpressionEncoder().to(device)
expr_enc.load_state_dict(torch.load(expr_enc_path, map_location=device))
expr_enc.eval()

classifier_head = torch.nn.Linear(128, 7).to(device)
classifier_head.load_state_dict(torch.load(cls_head_path, map_location=device))
classifier_head.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def predict_expression(image_path):
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)  # batch of 1

    with torch.no_grad():
        z = expr_enc(img_tensor)          # No need to chunk
        logits = classifier_head(z)
        probs = F.softmax(logits, dim=1)
        top_prob, top_class = torch.max(probs, dim=1)

    predicted = class_names[top_class.item()]
    confidence = top_prob.item()
    print(f"🧠 Predicted Expression: {predicted} ({confidence * 100:.2f}% confidence)")
    return predicted, confidence

# 👇 Manual path mode
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("⚠️  Usage: python inference.py <image_path>")
    else:
        predict_expression(sys.argv[1])
