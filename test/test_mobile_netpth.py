import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import json

# === 1. SETUP DEVICE ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === 2. NAMA FILE MODEL & KELAS ===
model_path = r"C:\Users\Alpabit\Documents\[!] TUGAS AKHIR\Project\OpenCampus\exported\best_model.pth"

# urutan kelas sesuai urutan folder dataset kamu
with open(r"C:\Users\Alpabit\Documents\[!] TUGAS AKHIR\Project\OpenCampus\exported\class_names.json", "r") as f:
    class_names = json.load(f)


# === 3. LOAD MODEL ===
num_classes = len(class_names)
model = models.mobilenet_v2(weights=None)
model.classifier = nn.Sequential(
    nn.Dropout(0.2),
    nn.Linear(1280, 512),
    nn.ReLU(),
    nn.BatchNorm1d(512),
    nn.Linear(512, num_classes)
)

model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()


# === 4. TRANSFORMASI GAMBAR (SAMA DENGAN TRAINING) ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# === 5. FUNGSI UNTUK PREDIKSI ===
def predict_image(image_path):
    # Pastikan file ada
    if not os.path.exists(image_path):
        print(f"❌ File tidak ditemukan: {image_path}")
        return

    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        _, preds = torch.max(outputs, 1)
        predicted_class = class_names[preds.item()]

    print(f"✅ Prediksi: {predicted_class}")

# === 6. TEST DENGAN GAMBAR ===
# Ganti path di bawah dengan lokasi gambar kamu
test_image_path = r"C:\Users\Alpabit\Documents\[!] TUGAS AKHIR\Project\OpenCampus\dataset_val\crophalf.jpg"
predict_image(test_image_path)
