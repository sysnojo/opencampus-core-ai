import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import os

# === CONFIG ===
BASE_DIR = r"C:\Users\Alpabit\Documents\[!] TUGAS AKHIR\Project\OpenCampus"
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
CACHE_PATH = os.path.join(BASE_DIR, "exported", "descriptor_cache.npz")
MODEL_EXPORT_PATH = os.path.join(BASE_DIR, "exported", "openibl_vgg16_netvlad_lite.pt")

device = torch.device("cpu")
os.makedirs(os.path.dirname(MODEL_EXPORT_PATH), exist_ok=True)

# === 1. Load model ===
print("🔄 Loading OpenIBL model...")
model = torch.hub.load('yxgeee/OpenIBL', 'vgg16_netvlad', pretrained=True)
model.eval().to(device)

# === 2. Transform ===
transformer = transforms.Compose([
    transforms.Resize((480, 640)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.48501960784313836, 0.4579568627450961, 0.4076039215686255],
        std=[0.00392156862745098, 0.00392156862745098, 0.00392156862745098]
    )
])

# === 3. Ekstraksi descriptor ===
def extract_descriptor(image_path):
    img = Image.open(image_path).convert("RGB")
    img = transformer(img).unsqueeze(0).to(device)
    with torch.no_grad():
        des = model(img)[0]
    return des.cpu().numpy()

# === 4. Bangun cache descriptor ===
if os.path.exists(CACHE_PATH):
    print(f"⚡ Loading cached descriptors from {CACHE_PATH}...")
    cache = np.load(CACHE_PATH, allow_pickle=True)
    descriptors = cache["descriptors"]
    image_paths = cache["image_paths"].tolist()
else:
    print("🧠 Building descriptor database (first time only)...")
    descriptors, image_paths = [], []
    for root, _, files in os.walk(DATASET_DIR):
        for file in files:
            if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                path = os.path.join(root, file)
                des = extract_descriptor(path)
                descriptors.append(des)
                image_paths.append(path)
    descriptors = np.vstack(descriptors)
    np.savez(CACHE_PATH, descriptors=descriptors, image_paths=image_paths)
    print(f"✅ Cached descriptors saved to {CACHE_PATH}")

print(f"📸 Total indexed images: {len(image_paths)}")

# === 5. Export model ke TorchScript Lite ===
print("🚀 Exporting model to TorchScript Lite...")
dummy_input = torch.randn(1, 3, 480, 640).to(device)
traced_model = torch.jit.trace(model, dummy_input)
traced_model._save_for_lite_interpreter(MODEL_EXPORT_PATH)
print(f"✅ Model TorchScript Lite berhasil disimpan ke: {MODEL_EXPORT_PATH}")

# === 6. Konfirmasi file ===
print("\n📦 Export completed!")
print(f" - Model: {MODEL_EXPORT_PATH}")
print(f" - Descriptor Cache: {CACHE_PATH}")
