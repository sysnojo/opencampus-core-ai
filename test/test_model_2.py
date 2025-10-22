import torch
from torchvision import models, transforms
from PIL import Image
import os
import torch.nn as nn
import torch.nn.functional as F

# === CONFIG ===
BASE_DIR = r"C:\Users\Alpabit\Documents\[!] TUGAS AKHIR\Project\OpenCampus"
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODEL_PATH = os.path.join(BASE_DIR, "exported", "openibl_lite_vgg16.pt")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Class names ===
classes = sorted([d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))])

# === Load model ===
num_classes = len(classes)
model = models.vgg16()
model.classifier[6] = nn.Linear(4096, num_classes)
state_dict = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state_dict)
model = model.float().to(device)
model.eval()

# === Transform ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# === Path gambar test ===
img_path = r"C:\Users\Alpabit\Documents\[!] TUGAS AKHIR\Project\OpenCampus\dataset\ruang_dosen_1_35m\frame_002.jpg"

img = Image.open(img_path).convert("RGB")
img_t = transform(img).unsqueeze(0).to(device)

# === Predict ===
with torch.no_grad():
    output = model(img_t)
    probs = F.softmax(output, dim=1)
    conf, pred_idx = torch.max(probs, 1)

pred_class = classes[pred_idx.item()]
print(f"🔍 Predicted Class: {pred_class} (Confidence: {conf.item() * 100:.2f}%)")