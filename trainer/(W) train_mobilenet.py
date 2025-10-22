import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import os, json

# ==== CONFIGURATION ====
DATA_DIR = r"C:\Users\Alpabit\Documents\[!] TUGAS AKHIR\Project\OpenCampus\dataset"
EXPORT_DIR = r"C:\Users\Alpabit\Documents\[!] TUGAS AKHIR\Project\OpenCampus\exported"
SAVE_PATH = os.path.join(EXPORT_DIR, "best_model.pth")
CLASS_JSON = os.path.join(EXPORT_DIR, "class_names.json")

BATCH_SIZE = 4
EPOCHS = 25
LR = 5e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==== TRANSFORMS ====
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ==== LOAD DATA ====
full_dataset = datasets.ImageFolder(DATA_DIR, transform=train_transform)
class_names = full_dataset.classes
num_classes = len(class_names)

# 🔥 Simpan urutan kelas sebenarnya yang dipakai model
os.makedirs(EXPORT_DIR, exist_ok=True)
with open(CLASS_JSON, "w") as f:
    json.dump(class_names, f, indent=4)
print(f"💾 Saved class names order to: {CLASS_JSON}")
print(f"📂 Class order used for training:\n{class_names}\n")

# ==== SPLIT DATA ====
train_size = int(0.9 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
test_dataset.dataset.transform = test_transform

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ==== MODEL ====
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)

# Fine-tune layer akhir
for name, param in model.features.named_parameters():
    param.requires_grad = True if "17" in name or "18" in name or "conv" in name else False

model.classifier = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(model.last_channel, 512),
    nn.ReLU(),
    nn.BatchNorm1d(512),
    nn.Linear(512, num_classes)
)
model = model.to(DEVICE)

# ==== LOSS & OPTIMIZER ====
criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LR, steps_per_epoch=len(train_loader), epochs=EPOCHS)

# ==== TRAIN LOOP ====
best_acc = 0.0
for epoch in range(EPOCHS):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels).item()
        total += labels.size(0)

    train_acc = 100 * correct / total
    train_loss = running_loss / total

    # ==== VALIDATION ====
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels).item()
            total += labels.size(0)

    test_acc = 100 * correct / total

    print(f"📈 Epoch {epoch+1}/{EPOCHS} | Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")

    if test_acc > best_acc:
        best_acc = test_acc
        torch.save(model.state_dict(), SAVE_PATH)
        print(f"💾 Saved new best model ({best_acc:.2f}%)")

print(f"\n✅ Training complete! Best Test Accuracy: {best_acc:.2f}%")
