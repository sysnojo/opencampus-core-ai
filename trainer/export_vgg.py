import torch
import os

# === 1. Load model ===
print("Loading model...")
model = torch.hub.load('yxgeee/OpenIBL', 'vgg16_netvlad', pretrained=True).eval()

# === 2. Gunakan CPU ===
device = torch.device('cpu')
model = model.to(device)

# === 3. Buat folder penyimpanan ===
export_dir = r"C:\Users\Alpabit\Documents\[!] TUGAS AKHIR\Project\OpenCampus\exported"
os.makedirs(export_dir, exist_ok=True)

# === 4. Dummy input ===
dummy_input = torch.randn(1, 3, 480, 640)

# === 5. Export ke TorchScript ===
print("Tracing model to TorchScript...")
traced_model = torch.jit.trace(model, dummy_input)
torchscript_path = os.path.join(export_dir, "openibl_vgg16_netvlad.pt")
traced_model.save(torchscript_path)
print(f"✅ TorchScript model saved to: {torchscript_path}")

# === 6. Export ke Lite Interpreter ===
print("Exporting to Lite Interpreter (.ptl)...")
lite_path = os.path.join(export_dir, "openibl_vgg16_netvlad.ptl")

# Langsung simpan tanpa optimize_for_mobile
traced_model._save_for_lite_interpreter(lite_path)
print(f"📱 Lite model saved to: {lite_path}")

print("\n🎉 Export complete! TorchScript (.pt) and Lite (.ptl) are ready.")
