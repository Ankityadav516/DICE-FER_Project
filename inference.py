import torch
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
from facenet_pytorch import MTCNN
import os

from models.encoder import ExpressionEncoder

# ✅ Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ Expression classes
class_names = ['surprise', 'fear', 'disgust', 'happy', 'sad', 'angry', 'neutral']

# ✅ Load models
expr_enc = ExpressionEncoder().to(device)
expr_enc.load_state_dict(torch.load("/content/drive/MyDrive/expression_model_final.pth", map_location=device))
expr_enc.eval()

classifier_head = torch.nn.Linear(128, 7).to(device)
classifier_head.load_state_dict(torch.load("/content/drive/MyDrive/expression_classifier_final.pth", map_location=device))
classifier_head.eval()

# ✅ MTCNN face detector
mtcnn = MTCNN(image_size=224, margin=0, post_process=False, device=device)

# ✅ Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ✅ Expression prediction function
def predict_expression(image_path):
    img = Image.open(image_path).convert("RGB")

    # 🔍 Face detection
    face = mtcnn(img)
    if face is None:
        print("❌ No face detected in image.")
        return None, 0.0

    face = face.unsqueeze(0).to(device)

    with torch.no_grad():
        z = expr_enc(face)
        z1, _ = torch.chunk(z, 2, dim=0)
        logits = classifier_head(z1)
        probs = F.softmax(logits, dim=1)
        top_prob, top_class = torch.max(probs, dim=1)

    predicted = class_names[top_class.item()]
    confidence = top_prob.item()
    print(f"🧠 Predicted Expression: {predicted} ({confidence * 100:.2f}% confidence)")
    return predicted, confidence


# ✅ Manual input image path
image_path = input("📂 Enter the full path of the image you uploaded (e.g., /content/image1.jpeg): ").strip()

if image_path and os.path.exists(image_path):
    predict_expression(image_path)
else:
    print("❌ Provided path is invalid or image not found.")
