import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

# === CONFIG ===
BASE_DIR = r"C:\Users\Alpabit\Documents\[!] TUGAS AKHIR\Project\OpenCampus"
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
EXPORT_DIR = os.path.join(BASE_DIR, "exported")
VGG_PATH = os.path.join(EXPORT_DIR, "vgg.pth")
MODEL_EXPORT_PATH = os.path.join(EXPORT_DIR, "openibl_lite_vgg16.pt")

BATCH_SIZE = 8
EPOCHS = 10
LR = 1e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(EXPORT_DIR, exist_ok=True)

# === 1. Load pretrained VGG16 (skip download if exists) ===
if os.path.exists(VGG_PATH):
    print("⚡ Loading local VGG16 weights...")
    model = models.vgg16()
    model.load_state_dict(torch.load(VGG_PATH, map_location=device))
else:
    print("⬇️ Downloading pretrained VGG16...")
    model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    torch.save(model.state_dict(), VGG_PATH)
    print("✅ Saved pretrained weights to:", VGG_PATH)

# === 2. Freeze backbone, replace classifier ===
for param in model.features.parameters():
    param.requires_grad = False

# Detect number of classes from folder names
classes = sorted([d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))])
num_classes = len(classes)
print(f"🏷️ Detected {num_classes} classes: {classes}")

model.classifier[6] = nn.Linear(4096, num_classes)
model = model.to(device)

# === 3. Data transforms ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

dataset = datasets.ImageFolder(DATASET_DIR, transform=transform)

# Split dataset (80% train, 20% test)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# === 4. Training setup ===
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=LR)

# === 5. Training loop ===
print("\n🚀 Starting fine-tuning for 10 epochs...")
for epoch in range(EPOCHS):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    acc = 100 * correct / total
    avg_loss = running_loss / len(train_loader)
    print(f"📈 Epoch [{epoch+1}/{EPOCHS}] Loss: {avg_loss:.4f} | Acc: {acc:.2f}%")

# === 6. Evaluate ===
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

test_acc = 100 * correct / total
print(f"✅ Test Accuracy: {test_acc:.2f}%")

# === 7. Save lightweight FP16 model ===
model = model.half()  # convert weights to float16
torch.save(model.state_dict(), MODEL_EXPORT_PATH)
print(f"💾 Model saved to {MODEL_EXPORT_PATH}")
print("💡 Expected size < 100MB after FP16 compression.")
