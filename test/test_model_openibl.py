import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import math

# === CONFIG ===
BASE_DIR = r"C:\Users\Alpabit\Documents\[!] TUGAS AKHIR\Project\OpenCampus"
MODEL_PATH = os.path.join(BASE_DIR, "exported", "openibl_new_lite.pt")
CACHE_PATH = os.path.join(BASE_DIR, "exported", "descriptor_cache.npz")
TEST_IMAGE = os.path.join(BASE_DIR, "dataset", "ruang_dosen_1_5m", "frame_003.jpg")  # ganti sesuai path test

device = torch.device("cpu")

# === 1. Load Model ===
print("🚀 Loading model...")
model = torch.jit.load(MODEL_PATH, map_location=device)
model.eval()

# === 2. Load Descriptor Cache ===
print("📦 Loading descriptor cache...")
cache = np.load(CACHE_PATH, allow_pickle=True)
descriptors = cache["descriptors"]
image_paths = cache["image_paths"].tolist()

# === 3. Build Label List ===
labels = [os.path.basename(os.path.dirname(p)) for p in image_paths]
class_names = sorted(list(set(labels)))
print(f"🏷️ Classes: {class_names}")

# === 4. Transform ===
transformer = transforms.Compose([
    transforms.Resize((480, 640)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.48501960784313836, 0.4579568627450961, 0.4076039215686255],
        std=[0.00392156862745098, 0.00392156862745098, 0.00392156862745098]
    )
])

# === 5. Cosine Similarity ===
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# === 6. Predict One Image ===
def predict_image(img_path):
    img = Image.open(img_path).convert("RGB")
    img_tensor = transformer(img).unsqueeze(0).to(device)

    with torch.no_grad():
        feat = model(img_tensor)[0].cpu().numpy()

    # Hitung similarity ke semua descriptor
    sims = [cosine_similarity(feat, d) for d in descriptors]
    best_idx = int(np.argmax(sims))
    pred_label = labels[best_idx]
    score = sims[best_idx]

    print(f"\n🖼️ Image: {os.path.basename(img_path)}")
    print(f"🔍 Predicted class: {pred_label}")
    print(f"📊 Cosine similarity: {score:.4f}")
    return pred_label, score

# === 7. Akurasi Batch Test ===
def test_dataset(base_dir):
    total, correct, losses = 0, 0, []
    for root, _, files in os.walk(base_dir):
        label = os.path.basename(root)
        if label not in class_names:
            continue
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                total += 1
                path = os.path.join(root, f)
                pred, score = predict_image(path)
                losses.append(1 - score)
                if pred == label:
                    correct += 1
    acc = correct / total * 100 if total > 0 else 0
    avg_loss = np.mean(losses) if losses else 0
    print(f"\n✅ Accuracy: {acc:.2f}%")
    print(f"❌ Avg Loss (1 - cosine): {avg_loss:.4f}")

# === 8. Run test ===
# Tes satu gambar dulu
predict_image(TEST_IMAGE)

# Kalau mau test seluruh dataset:
# test_dataset(os.path.join(BASE_DIR, "dataset"))
