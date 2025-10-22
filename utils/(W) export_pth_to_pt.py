import torch
import torch.nn as nn
from torchvision import models
import json

# === PATHS ===
pth_path = r"C:\Users\Alpabit\Documents\[!] TUGAS AKHIR\Project\OpenCampus\exported\best_model.pth"
ptl_path = r"C:\Users\Alpabit\Documents\[!] TUGAS AKHIR\Project\OpenCampus\trained_exported\oc-mnet-v0.ptl"
json_path = r"C:\Users\Alpabit\Documents\[!] TUGAS AKHIR\Project\OpenCampus\trained_exported\class_names.json"

# === LOAD CLASS NAMES ===
with open(json_path, "r") as f:
    class_names = json.load(f)
num_classes = len(class_names)

# === LOAD MODEL ===
model = models.mobilenet_v2(weights=None)
model.classifier = nn.Sequential(
    nn.Dropout(0.2),
    nn.Linear(1280, 512),
    nn.ReLU(),
    nn.BatchNorm1d(512),
    nn.Linear(512, num_classes)
)

model.load_state_dict(torch.load(pth_path, map_location="cpu"))
model.eval()

# === DUMMY INPUT ===
example_input = torch.randn(1, 3, 224, 224)

# === CONVERT TO TORCHSCRIPT ===
traced_model = torch.jit.trace(model, example_input)
traced_model._save_for_lite_interpreter(ptl_path)
print(f"✅ Model .ptl berhasil dibuat: {ptl_path}")
