"""
video_label_gui_dynamic_hotkey.py
---------------------------------
UI sederhana untuk meninjau frame video dan menyimpan frame tertentu
ke folder kelas (dataset) secara manual.

Kontrol:
  - Tombol "<" atau panah kiri  : Mundur beberapa detik
  - Tombol ">" atau panah kanan : Maju beberapa detik
  - Tombol "s"                  : Simpan frame saat ini ke kelas yang dipilih
  - Tombol "n"                  : Tambah kelas baru lewat input prompt
  - Mouse UI tombol & radio tetap bisa digunakan

Author: John
Date: 2025
"""

import cv2
import os
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RadioButtons, TextBox
import numpy as np

CONFIG = {
    "video_path": "C:/Users/Alpabit/Documents/[!] TUGAS AKHIR/Project/OpenCampus/raw/new/Half_4.mp4",
    "output_dir": "C:/Users/Alpabit/Documents/[!] TUGAS AKHIR/Project/OpenCampus/dataset_new",
    "step_seconds": 0.5,
    "image_format": "jpg"
}

class VideoLabeler:
    def __init__(self, config):
        self.config = config
        self.cap = cv2.VideoCapture(config["video_path"])
        if not self.cap.isOpened():
            raise FileNotFoundError(f"Gagal membuka video: {config['video_path']}")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.total_frames / self.fps
        self.current_time = 0.0
        self.image_format = config["image_format"]
        self.output_dir = config["output_dir"]

        print("\n=== Masukkan daftar kelas (pisahkan dengan koma) ===")
        raw_input = input("Kelas: ").strip()
        if not raw_input:
            self.classes = ["default_class"]
        else:
            self.classes = [c.strip() for c in raw_input.split(",")]
        self.current_class = self.classes[0]

        self.fig, self.ax = plt.subplots()
        plt.subplots_adjust(bottom=0.35)
        self.ax.set_title("Video Frame Labeler")

        axprev = plt.axes([0.05, 0.05, 0.1, 0.075])
        axnext = plt.axes([0.17, 0.05, 0.1, 0.075])
        axsave = plt.axes([0.8, 0.05, 0.15, 0.075])
        axradio = plt.axes([0.35, 0.05, 0.25, 0.2])
        axaddbox = plt.axes([0.05, 0.15, 0.22, 0.05])
        axaddbtn = plt.axes([0.28, 0.15, 0.07, 0.05])

        self.btn_prev = Button(axprev, "< Prev")
        self.btn_next = Button(axnext, "Next >")
        self.btn_save = Button(axsave, "Save Frame")
        self.radio = RadioButtons(axradio, self.classes)
        self.txt_newclass = TextBox(axaddbox, "New Class:")
        self.btn_addclass = Button(axaddbtn, "+ Add")

        self.btn_prev.on_clicked(self.prev_frame)
        self.btn_next.on_clicked(self.next_frame)
        self.btn_save.on_clicked(self.save_frame)
        self.radio.on_clicked(self.select_class)
        self.btn_addclass.on_clicked(self.add_class)

        # === HOTKEY SUPPORT ===
        self.fig.canvas.mpl_connect("key_press_event", self.on_key_press)

        self.im = self.ax.imshow(np.zeros((100, 100, 3), dtype=np.uint8))
        self.update_frame()

        plt.show()

    # ==============================
    # EVENT HANDLERS
    # ==============================
    def on_key_press(self, event):
        key = event.key.lower()
        if key in ["left", "<"]:
            self.prev_frame(None)
        elif key in ["right", ">"]:
            self.next_frame(None)
        elif key == "m":
            self.save_frame(None)
        elif key == "n":
            new_class = input("Masukkan nama kelas baru: ").strip()
            if new_class:
                self.add_class_manual(new_class)
        else:
            print(f"(i) Tombol '{key}' tidak dikenali.")

    def update_frame(self):
        self.cap.set(cv2.CAP_PROP_POS_MSEC, self.current_time * 1000)
        ret, frame = self.cap.read()
        if not ret:
            print("⚠️ Gagal membaca frame. Kembali ke awal video.")
            self.current_time = 0
            self.cap.set(cv2.CAP_PROP_POS_MSEC, 0)
            ret, frame = self.cap.read()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.im.set_data(frame_rgb)
        self.ax.set_title(f"Time: {self.current_time:.2f}s | Class: {self.current_class}")
        self.fig.canvas.draw_idle()

    def prev_frame(self, event):
        self.current_time = max(0, self.current_time - self.config["step_seconds"])
        self.update_frame()

    def next_frame(self, event):
        self.current_time = min(self.duration, self.current_time + self.config["step_seconds"])
        self.update_frame()

    def select_class(self, label):
        self.current_class = label
        print(f"✅ Kelas dipilih: {self.current_class}")

    def add_class_manual(self, new_class):
        """Tambahkan kelas baru via hotkey prompt."""
        if new_class in self.classes:
            print(f"⚠️ Kelas '{new_class}' sudah ada.")
            return
        self.classes.append(new_class)
        self._refresh_radio()
        self.current_class = new_class
        print(f"✅ Kelas baru ditambahkan: {new_class}")

    def add_class(self, event):
        new_class = self.txt_newclass.text.strip()
        if not new_class:
            print("⚠️ Nama kelas kosong, abaikan.")
            return
        self.add_class_manual(new_class)

    def _refresh_radio(self):
        self.radio.ax.clear()
        self.radio = RadioButtons(self.radio.ax, self.classes)
        self.radio.on_clicked(self.select_class)
        self.fig.canvas.draw_idle()

    def save_frame(self, event):
        self.cap.set(cv2.CAP_PROP_POS_MSEC, self.current_time * 1000)
        ret, frame = self.cap.read()
        if not ret:
            print("❌ Gagal menyimpan frame (tidak terbaca).")
            return

        class_dir = os.path.join(self.output_dir, self.current_class)
        os.makedirs(class_dir, exist_ok=True)

        frame_name = f"frame_Half_4_{int(self.current_time * 1000)}.{self.image_format}"
        path = os.path.join(class_dir, frame_name)
        cv2.imwrite(path, frame)
        print(f"💾 Disimpan: {path}")

if __name__ == "__main__":
    VideoLabeler(CONFIG)
