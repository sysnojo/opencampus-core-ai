import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity

# === 1. Load model ===
print("Loading model...")
model = torch.hub.load('yxgeee/OpenIBL', 'vgg16_netvlad', pretrained=True).eval()

device = torch.device('cpu')
model = model.to(device)

# === 2. Preprocessing ===
transformer = transforms.Compose([
    transforms.Resize((480, 640)),  # (height, width)
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.48501960784313836, 0.4579568627450961, 0.4076039215686255],
        std=[0.00392156862745098, 0.00392156862745098, 0.00392156862745098]
    )
])

# === 3. Fungsi ekstraksi descriptor ===
def extract_descriptor(image_path):
    img = Image.open(image_path).convert('RGB')
    img = transformer(img).unsqueeze(0).to(device)
    with torch.no_grad():
        des = model(img)[0]
    return des.cpu().numpy()

# === 4. Bangun database deskriptor dari dataset ===
dataset_dir = r"C:\Users\Alpabit\Documents\[!] TUGAS AKHIR\Project\OpenCampus\dataset"
print("Building descriptor database...")

descriptors = []
image_paths = []

for root, dirs, files in os.walk(dataset_dir):
    for file in files:
        if file.lower().endswith(('.jpg', '.png', '.jpeg')):
            img_path = os.path.join(root, file)
            des = extract_descriptor(img_path)
            descriptors.append(des)
            image_paths.append(img_path)

descriptors = np.vstack(descriptors)
print(f"Total images indexed: {len(image_paths)}")

# === 5. Input query image ===
query_path = r"C:\Users\Alpabit\Documents\[!] TUGAS AKHIR\Project\OpenCampus\dataset\ruang_dosen_1_5m\frame_000.jpg"
query_des = extract_descriptor(query_path)

# === 6. Hitung kesamaan cosine ===
sims = cosine_similarity(query_des.reshape(1, -1), descriptors)[0]
best_idx = np.argmax(sims)

# === 7. Tampilkan hasil ===
print("\n📍 Query image:", query_path)
print(f"🔍 Most similar image: {image_paths[best_idx]}")
print(f"🔢 Similarity score: {sims[best_idx]:.4f}")
