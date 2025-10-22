import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

# === CONFIG ===
dataset_dir = r"C:\Users\Alpabit\Documents\[!] TUGAS AKHIR\Project\OpenCampus\dataset"
cache_file = r"C:\Users\Alpabit\Documents\[!] TUGAS AKHIR\Project\OpenCampus\trainer\descriptor_cache.npz"
device = torch.device("cpu")

# === 1. Load model (sekali saja) ===
print("🔄 Loading model...")
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

# === 3. Fungsi ekstraksi descriptor ===
def extract_descriptor(image_path):
    img = Image.open(image_path).convert("RGB")
    img = transformer(img).unsqueeze(0).to(device)
    with torch.no_grad():
        des = model(img)[0]
    return des.cpu().numpy()

# === 4. Load cache jika ada ===
if os.path.exists(cache_file):
    print(f"⚡ Loading cached descriptors from {cache_file}...")
    cache = np.load(cache_file, allow_pickle=True)
    descriptors = cache["descriptors"]
    image_paths = cache["image_paths"].tolist()
else:
    print("🧠 Building descriptor database (first time only)...")
    descriptors, image_paths = [], []
    for root, _, files in os.walk(dataset_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                path = os.path.join(root, file)
                des = extract_descriptor(path)
                descriptors.append(des)
                image_paths.append(path)
    descriptors = np.vstack(descriptors)
    np.savez(cache_file, descriptors=descriptors, image_paths=image_paths)
    print(f"✅ Cached descriptors saved to {cache_file}")

print(f"📸 Total indexed images: {len(image_paths)}")

# === 5. Fungsi query ===
def query_image(query_path):
    query_des = extract_descriptor(query_path)
    sims = cosine_similarity(query_des.reshape(1, -1), descriptors)[0]
    best_idx = np.argmax(sims)

    best_match = image_paths[best_idx]
    score = sims[best_idx]

    # Tampilkan hasil
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(Image.open(query_path))
    axs[0].set_title("Query Image")
    axs[0].axis("off")

    axs[1].imshow(Image.open(best_match))
    axs[1].set_title(f"Best Match\nScore: {score:.4f}")
    axs[1].axis("off")

    plt.show()

    print("\n📍 Query:", query_path)
    print(f"🔍 Match: {best_match}")
    print(f"🔢 Similarity: {score:.4f}")

# === 6. Contoh query ===
query_path = r"C:\Users\Alpabit\Documents\[!] TUGAS AKHIR\Project\OpenCampus\dataset\ruang_dosen_1_5m\frame_009.jpg"
query_image(query_path)
