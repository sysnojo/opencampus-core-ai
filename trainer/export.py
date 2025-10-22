import torch
from torchvision import models

STATE_DICT_PATH = r"C:\Users\Alpabit\Documents\[!] TUGAS AKHIR\Project\OpenCampus\trained_exported\oc-v1-model.pth"
EXPORT_PATH = r"C:\Users\Alpabit\Documents\[!] TUGAS AKHIR\Project\OpenCampus\trained_exported\oc-mnet-v1.pt"

# 1️⃣ Buat ulang arsitektur model
model = models.mobilenet_v2(pretrained=False)
num_classes = 17  # ganti sesuai jumlah kelas
model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)

# 2️⃣ Load bobot hasil training
model.load_state_dict(torch.load(STATE_DICT_PATH, map_location="cpu"))
model.eval()

# 3️⃣ Trace model
example_input = torch.rand(1, 3, 224, 224)
traced_model = torch.jit.trace(model, example_input)

# 4️⃣ Simpan versi TorchScript Lite
traced_model._save_for_lite_interpreter(EXPORT_PATH)
print(f"✅ Model TorchScript Lite berhasil disimpan ke: {EXPORT_PATH}")
