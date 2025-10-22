"""
train_localization_model.py
---------------------------
Script untuk melatih model klasifikasi posisi berbasis gambar
dari dataset per-meter dan per-ruang, dengan data augmentation.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import os

# ==========================
# USER CONFIG
# ==========================
CONFIG = {
    "dataset_dir": "C:/Users/Alpabit/Documents/[!] TUGAS AKHIR/Project/OpenCampus/dataset",
    "batch_size": 8,
    "num_epochs": 60,  # disarankan sedikit lebih lama karena dataset kecil
    "learning_rate": 1e-4,
    "model_save_path": "C:/Users/Alpabit/Documents/[!] TUGAS AKHIR/Project/OpenCampus/models/opencampus-v0.pt",
    "image_size": (224, 224),
    "use_augmentation": True
}
# ==========================


def train():
    # Data augmentation untuk memperbanyak variasi data secara virtual
    if CONFIG["use_augmentation"]:
        transform = transforms.Compose([
            transforms.Resize(CONFIG["image_size"]),
            transforms.RandomHorizontalFlip(p=0.5),          # flip kiri-kanan
            transforms.RandomRotation(degrees=15),           # rotasi ringan
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
            transforms.RandomResizedCrop(CONFIG["image_size"], scale=(0.8, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(CONFIG["image_size"]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    # Dataset dan DataLoader
    dataset = datasets.ImageFolder(CONFIG["dataset_dir"], transform=transform)
    dataloader = DataLoader(dataset, batch_size=CONFIG["batch_size"], shuffle=True)

    # Model MobileNetV2 pretrained
    model = models.mobilenet_v2(pretrained=True)
    num_features = model.classifier[1].in_features
    num_classes = len(dataset.classes)
    model.classifier[1] = nn.Linear(num_features, num_classes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Loss dan optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])

    # Training loop
    for epoch in range(CONFIG["num_epochs"]):
        model.train()
        running_loss = 0.0

        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{CONFIG['num_epochs']}], Loss: {running_loss/len(dataloader):.4f}")

    # Simpan model
    torch.save(model.state_dict(), CONFIG["model_save_path"])
    print(f"Model disimpan ke: {CONFIG['model_save_path']}")
    print(f"Total class: {num_classes}")
    print("Daftar class:", dataset.classes)


if __name__ == "__main__":
    train()
