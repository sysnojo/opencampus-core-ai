import torch
import torch.nn as nn
from torchvision import models
from torch.utils.mobile_optimizer import optimize_for_mobile

# 1️⃣ Buat ulang model arsitektur yang sama
model = models.vgg16(pretrained=False)
num_features = model.classifier[6].in_features
model.classifier[6] = nn.Linear(num_features, 17)  # ubah sesuai jumlah kelas kamu

# 2️⃣ Load state_dict dari file kamu
state_dict = torch.load(
    "C:/Users/Alpabit/Documents/[!] TUGAS AKHIR/Project/OpenCampus/exported/openibl_lite_vgg16.pt",
    map_location="cpu"
)
model.load_state_dict(state_dict)
model.eval()

# 3️⃣ Convert ke TorchScript
example = torch.rand(1, 3, 224, 224)
traced = torch.jit.trace(model, example)

# 4️⃣ Optimize for mobile
optimized_traced = optimize_for_mobile(traced)

# 5️⃣ Save versi Lite-nya
optimized_traced._save_for_lite_interpreter(
    "C:/Users/Alpabit/Documents/[!] TUGAS AKHIR/Project/OpenCampus/exported/openibl_lite_vgg16-exported.pt"
)

print("✅ Model Lite saved successfully!")
