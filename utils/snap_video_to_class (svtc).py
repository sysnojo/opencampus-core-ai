"""
snap_video_to_class.py
---------------------------------
Program ini mengambil beberapa screenshot (5–7) dari video untuk setiap jarak tertentu (misalnya setiap 5 meter).
Setiap jarak akan memiliki folder sendiri (misalnya ruang_dosen_1_5m, ruang_dosen_1_10m, dst),
yang berisi beberapa gambar dari rentang waktu tertentu.

Cocok untuk membuat dataset Image-Based Localization (IBL)
yang lebih representatif dengan variasi posisi dalam satu jarak.

Author: John
Date: 2025
"""

import cv2
import os
import random

# ==============================
# USER CONFIGURATION
# ==============================
CONFIG = {
    # Path ke video input
    "video_path": "C:/Users/Alpabit/Documents/[!] TUGAS AKHIR/Project/OpenCampus/raw/ruang_dosen_2.mp4",

    # Path folder output dataset
    "output_dir": "C:/Users/Alpabit/Documents/[!] TUGAS AKHIR/Project/OpenCampus/dataset",

    # Interval waktu antar kelas (dalam detik)
    # Misal setiap 10 detik dianggap berpindah 5 meter
    "interval_per_class_seconds": 10,

    # Jarak antar kelas dalam meter
    "distance_step_m": 5,

    # Jumlah gambar yang diambil per kelas (acak antara min-max)
    "min_images_per_class": 10,
    "max_images_per_class": 10,

    # Format gambar output
    "image_format": "jpg"
}
# ==============================


def create_folder(path):
    """Membuat folder jika belum ada."""
    if not os.path.exists(path):
        os.makedirs(path)


def snap_video_to_class():
    """Mengambil beberapa frame per kelas (per meter) dari video."""
    video_path = CONFIG["video_path"]
    base_output_dir = CONFIG["output_dir"]
    interval_class = CONFIG["interval_per_class_seconds"]
    distance_step = CONFIG["distance_step_m"]
    min_imgs = CONFIG["min_images_per_class"]
    max_imgs = CONFIG["max_images_per_class"]
    img_format = CONFIG["image_format"]

    # Pastikan video ada
    if not os.path.exists(video_path):
        print(f"Error: File video '{video_path}' tidak ditemukan.")
        return

    # Ambil nama dasar video (tanpa ekstensi)
    base_name = os.path.splitext(os.path.basename(video_path))[0]

    # Buka video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Gagal membuka video.")
        return

    # Dapatkan informasi video
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    print(f"Video: {video_path}")
    print(f"Durasi: {duration:.2f} detik")
    print(f"Interval per kelas: {interval_class} detik")

    distance = distance_step  # Mulai dari 5m, bukan 0m
    class_index = 0
    total_saved = 0

    # Loop per kelas (setiap segmen waktu)
    while class_index * interval_class < duration:
        start_time = class_index * interval_class
        end_time = min(start_time + interval_class, duration)

        # Buat nama folder berdasarkan nama video + jarak, misal ruang_dosen_1_5m
        folder_name = f"{base_name}_{int(distance)}m"
        folder_path = os.path.join(base_output_dir, folder_name)
        create_folder(folder_path)

        # Tentukan jumlah gambar acak untuk kelas ini
        num_images = random.randint(min_imgs, max_imgs)
        times_to_capture = sorted(random.uniform(start_time, end_time) for _ in range(num_images))

        print(f"\n[{folder_name}] Mengambil {num_images} frame dari {start_time:.1f}s - {end_time:.1f}s")

        for i, t in enumerate(times_to_capture):
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            success, frame = cap.read()
            if not success:
                print(f"  Gagal membaca frame di {t:.2f} detik.")
                continue

            img_name = f"frame_{i:03d}.{img_format}"
            img_path = os.path.join(folder_path, img_name)
            cv2.imwrite(img_path, frame)
            total_saved += 1
            print(f"  Menyimpan {img_path}")

        distance += distance_step
        class_index += 1

    cap.release()
    print("\nProses selesai.")
    print(f"Total gambar disimpan: {total_saved}")


if __name__ == "__main__":
    snap_video_to_class()
