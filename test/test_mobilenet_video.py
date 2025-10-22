import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import json
import os

# === 1. SETUP DEVICE ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === 2. MODEL DAN KELAS ===
model_path = r"C:\Users\Alpabit\Documents\[!] TUGAS AKHIR\Project\OpenCampus\exported\best_model.pth"
with open(r"C:\Users\Alpabit\Documents\[!] TUGAS AKHIR\Project\OpenCampus\exported\class_names.json", "r") as f:
    class_names = json.load(f)
num_classes = len(class_names)

# === 3. LOAD MODEL ===
model = models.mobilenet_v2(weights=None)
model.classifier = nn.Sequential(
    nn.Dropout(0.2),
    nn.Linear(1280, 512),
    nn.ReLU(),
    nn.BatchNorm1d(512),
    nn.Linear(512, num_classes)
)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# === 4. TRANSFORMASI GAMBAR ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# === 5. BACA VIDEO ===
video_path = r"C:\Users\Alpabit\Documents\[!] TUGAS AKHIR\Project\OpenCampus\raw\ruang_dosen_1.mp4"
log_path = r"C:\Users\Alpabit\Documents\[!] TUGAS AKHIR\Project\OpenCampus\logs\video_predictions.txt"

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# === 6. PROSES FRAME PER FRAME ===
frame_count = 0

with open(log_path, "w", encoding="utf-8") as log_file:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        # Kalau mau lebih cepat: ambil 1 dari setiap 5 frame
        # if frame_count % 5 != 0:
        #     continue

        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        input_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(input_tensor)
            _, preds = torch.max(outputs, 1)
            predicted_class = class_names[preds.item()]

        current_time = frame_count / fps
        log_file.write(f"{current_time:.2f} detik -> {predicted_class}\n")

        cv2.putText(frame, f"{predicted_class} ({current_time:.1f}s)",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Video Classification", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

print(f"\n✅ Selesai! Total frame: {total_frames}")
print(f"📄 Log hasil tersimpan di:\n{log_path}")
