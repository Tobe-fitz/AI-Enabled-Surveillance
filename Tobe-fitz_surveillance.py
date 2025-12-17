#!/usr/bin/env python3
"""
TOBE-FITZ - Surveillance System (Hugging Face + YOLOv8 + Tkinter)


Features
--------
1) Multi-source camera input (Webcam index, IP camera (RTSP/HTTP), Video file)
2) Object detection + simple tracking IDs (centroid tracker)
3) Auto-generate image description via Hugging Face (BLIP) + threat assessment
4) Thick detection boxes; "THREAT" in red, bold labels
5) Voice report (gTTS MP3) + playback + email sending (report or audio)
6) GUI: Left (live feed top + controls bottom), Right (2 thumbnails + analysis + editable report + send)
7) Playback option for saved videos
8) Blue-themed, colorful buttons, fitted layout
9) Footer: “Designed by NAVY AI COURSE 5 SYN 2”

Setup
-----
- Python 3.9+ recommended
- pip install -r requirements.txt
- Set your Hugging Face token (one-time):
    export HUGGINGFACE_API_TOKEN="hf_..."
  Or paste it into the GUI when prompted (Settings -> HF Token).

- Email:
    Uses SMTP (e.g., Gmail). For Gmail, create an App Password and use:
    SMTP server: smtp.gmail.com, Port: 587, TLS.

Notes
-----
- Threat assessment is rule-based from detections and caption keywords.
- For IP camera, use a full URL (e.g., rtsp://user:pass@ip:554/stream or http://ip:port/video).
"""

import os
import cv2
import sys
import time
import json
import base64
import queue
import smtplib
import threading
import tempfile
import requests
import numpy as np

from email.message import EmailMessage
from PIL import Image, ImageTk, ImageDraw, ImageFont
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional

# GUI
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# AI / Models
from ultralytics import YOLO

# Voice
from gtts import gTTS
from playsound import playsound

# -----------------------------------------------------------------------------
# Utility: Simple Centroid Tracker for basic "tracking" IDs
# -----------------------------------------------------------------------------
SAVE_DIR = "surv_outputs"

class CentroidTracker:
    def __init__(self, max_disappeared=20, max_distance=80):
        self.nextObjectID = 1
        self.objects = {}            # objectID -> centroid
        self.disappeared = {}        # objectID -> frames disappeared
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def register(self, centroid):
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        if objectID in self.objects:
            del self.objects[objectID]
        if objectID in self.disappeared:
            del self.disappeared[objectID]

    def update(self, rects: List[Tuple[int,int,int,int]]):
        # rects: list of (x1,y1,x2,y2)
        if len(rects) == 0:
            # mark disappeared
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.max_disappeared:
                    self.deregister(objectID)
            return self.objects

        # calculate centroids
        inputCentroids = []
        for (x1, y1, x2, y2) in rects:
            cX = int((x1 + x2) / 2.0)
            cY = int((y1 + y2) / 2.0)
            inputCentroids.append((cX, cY))

        if len(self.objects) == 0:
            for c in inputCentroids:
                self.register(c)
            return self.objects

        # match existing to new centroids by nearest distance
        objectIDs = list(self.objects.keys())
        objectCentroids = list(self.objects.values())

        D = np.zeros((len(objectCentroids), len(inputCentroids)), dtype=float)
        for i, oc in enumerate(objectCentroids):
            for j, ic in enumerate(inputCentroids):
                D[i, j] = np.linalg.norm(np.array(oc) - np.array(ic))

        # greedy matching
        usedRows = set()
        usedCols = set()

        while True:
            minVal = None
            minPos = None
            for i in range(D.shape[0]):
                if i in usedRows: continue
                for j in range(D.shape[1]):
                    if j in usedCols: continue
                    if minVal is None or D[i, j] < minVal:
                        minVal = D[i, j]
                        minPos = (i, j)
            if minPos is None or minVal is None:
                break
            i, j = minPos
            if D[i, j] > self.max_distance:
                break
            objectID = objectIDs[i]
            self.objects[objectID] = inputCentroids[j]
            self.disappeared[objectID] = 0
            usedRows.add(i)
            usedCols.add(j)

        # any unmatched existing: mark disappeared
        for i in range(D.shape[0]):
            if i not in usedRows:
                objectID = objectIDs[i]
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.max_disappeared:
                    self.deregister(objectID)

        # unmatched new centroids -> register
        for j in range(D.shape[1]):
            if j not in usedCols:
                self.register(inputCentroids[j])

        return self.objects
    
def assess_threat(detections: List[Dict], caption: str) -> Tuple[str, List[str]]:
    """
    Basic military-context threat heuristic.
    Returns (level, reasons)
      level in {'LOW', 'MEDIUM', 'HIGH'}
    """
    reasons = []
    person_count = sum(1 for d in detections if d['cls_name'] == 'person')
    weapon_like = any(k in caption.lower() for k in ['gun', 'rifle', 'pistol', 'knife', 'grenade'])
    camo_like = any(k in caption.lower() for k in ['camouflage', 'military uniform', 'uniform', 'fatigues'])
    crowding = person_count >= 3

    if weapon_like:
        reasons.append("Possible weapon reference in caption.")
    if camo_like:
        reasons.append("Caption suggests uniforms/camouflage.")
    if crowding:
        reasons.append(f"{person_count} persons detected.")

    # Box labels that suggest risk
    risk_labels = {'knife', 'gun', 'backpack', 'cell phone'}
    if any(d['cls_name'] in risk_labels for d in detections):
        reasons.append("Potentially risky object detected.")

    # Simple scoring
    score = 0
    if weapon_like: score += 2
    if camo_like: score += 1
    if crowding: score += 1
    if score >= 3:
        return "HIGH", reasons
    if score == 2:
        return "MEDIUM", reasons
    return "LOW", reasons

# threat configuration
THREAT_KEYWORDS = {
    "gun","rifle","weapon","pistol","camouflage","soldier","grenade","tank",
    "knife","hostile","firearm","shotgun","assault","explosive"
}
THREAT_OBJECTS = {
    "knife","truck","car","bus","boat","train","motorcycle","airplane","backpack"
}


# -----------------------------------------------------------------------------
# Email Utility
# -----------------------------------------------------------------------------

def send_email(smtp_server, smtp_port, use_tls, sender_email, sender_password,
               to_email, subject, body, attachments: List[Tuple[str, bytes]] = None):
    msg = EmailMessage()
    msg['From'] = sender_email
    msg['To'] = to_email
    msg['Subject'] = subject
    msg.set_content(body)

    attachments = attachments or []
    for filename, filebytes in attachments:
        msg.add_attachment(filebytes, maintype='application', subtype='octet-stream', filename=filename)

    with smtplib.SMTP(smtp_server, smtp_port, timeout=60) as server:
        if use_tls:
            server.starttls()
        server.login(sender_email, sender_password)
        server.send_message(msg)

# -----------------------------------------------------------------------------
# Main App Class
# -----------------------------------------------------------------------------

@dataclass
class AppConfig:
    hf_token: str = ""
    smtp_server: str = "smtp.gmail.com"
    smtp_port: int = 587
    use_tls: bool = True
    sender_email: str = " "
    sender_password: str = " "


class SurveillanceApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("TOBE-FITZ - AI SURVEILLANCE SYSTEM")
        self.root.geometry("1350x800")
        self.root.configure(bg='#0a2a66')
        #self.root.configure(bg='#0b3d91')  # navy blue-ish

        self.config = AppConfig(hf_token=os.environ.get("HUGGINGFACE_API_TOKEN", "").strip())

        # Video / capture state
        self.cap = None
        self.current_source = None
        self.running = False
        self.recording = False
        self.writer = None
        self.last_frame = None
        self.last_analysis = ""
        self.last_audio_path = None
        self.snapshot_paths = []  # Recent snapshot paths
        self.video_source_cap = None  # VideoCapture for video file source
        self.zoom_level = 1.0  # Initial zoom level
        self.zoom_step = 0.2   # Zoom increment/decrement
        self.playback_cap = None  # VideoCapture for playback
        self.playback_running = False
        self.playback_canvas = None  # Canvas for playback
        self.playback_thread = None
        
        # Detection
        self.model = YOLO('yolov8n.pt')
        self.tracker = CentroidTracker()
        self.class_names = self.model.names  # dict id->name
        self.model_loaded = True
        self.detector_conf = 0.25  # Confidence threshold

        # Threading
        self.frame_queue = queue.Queue(maxsize=2)
        self.capture_thread = None

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        
    

    # --------------------------- UI Layout -----------------------------------


    def _build_ui(self):
        # Layout: Left (live + controls), Right (thumbs + analysis + email + voice)
        header = tk.Frame(self.root, bg='#0a2a66')
        header.pack(side="top", fill="x")

        # Existing logo (upper left)
        #logo = self._ensure_logo_image().resize((200, 40))
        #self.logo_tk = ImageTk.PhotoImage(logo)
        #tk.Label(header, image=self.logo_tk, bg='#0a2a66').pack(side="left", padx=10, pady=6)

        # Center frame for new logos and heading
        center_frame = tk.Frame(header, bg='#0a2a66')
        center_frame.pack(side="left", expand=True, padx=(0, 10), pady=6)  # Centered with padding

        # New logo before heading
        new_logo_path = r"C:\Users\OWNER\Desktop\AI AERIAL SURVEILLANCE SYSTEM CODE\python logo.png"  # Replace with your new logo file path
        try:
            new_logo_img = Image.open(new_logo_path)
            new_logo_img = new_logo_img.resize((50, 50), Image.Resampling.LANCZOS)  # Resize to 50x50
            self.new_logo_tk = ImageTk.PhotoImage(new_logo_img)
            tk.Label(center_frame, image=self.new_logo_tk, bg='#0a2a66').pack(side="left", padx=(0, 10))
        except Exception as e:
            print(f"Failed to load new logo: {e}")
            tk.Label(center_frame, text="New Logo", bg='#0a2a66', fg="white").pack(side="left", padx=(0, 10))

        # Heading
        tk.Label(center_frame, text="AI SURVEILLANCE SYSTEM", fg="white", bg='#0a2a66', font=("Arial", 25, "bold")).pack(side="left", padx=8)

        # New logo after heading
        new_logo_path = r"C:\Users\OWNER\Desktop\AI AERIAL SURVEILLANCE SYSTEM CODE\bulletcam.png"
        try:
            new_logo_img = Image.open(new_logo_path)
            new_logo_img = new_logo_img.resize((50, 50), Image.Resampling.LANCZOS)  # Resize to 50x50
            self.new_logo_tk_after = ImageTk.PhotoImage(new_logo_img)
            tk.Label(center_frame, image=self.new_logo_tk_after, bg='#0a2a66').pack(side="left", padx=(10, 0))
        except Exception as e:
            print(f"Failed to load new logo: {e}")
            tk.Label(center_frame, text="New Logo", bg='#0a2a66', fg="white").pack(side="left", padx=(10, 0))

        outer = ttk.Frame(self.root)
        outer.pack(fill='both', expand=True)
        outer.configure(style="TFrame")
       
        # Use PanedWindow for resizable split
        paned = ttk.PanedWindow(outer, orient=tk.HORIZONTAL)
        paned.pack(fill='both', expand=True, padx=8, pady=8)

        

        self._init_styles()

        left = ttk.Frame(paned)
        right = ttk.Frame(paned)
        paned.add(left, weight=3)
        paned.add(right, weight=2)
        #Left_upper: live video canvas
        live_frame = ttk.LabelFrame(left, text="Live Camera Feed")
        live_frame.pack(fill='both', expand=True, padx=5, pady=5)
        self.canvas = tk.Canvas(live_frame, bg='black', highlightthickness=0)
        self.canvas.pack(fill='both', expand=True)
        
        

        # Left lower: controls
        controls = ttk.LabelFrame(left, text="Controls")
        controls.pack(fill='x', padx=5, pady=5)

        self.source_var = tk.StringVar(value="Webcam 0")
        source_label = ttk.Label(controls, text="Source:")
        source_label.grid(row=0, column=0, padx=4, pady=4, sticky='w')

        self.source_chooser = ttk.Combobox(controls, textvariable=self.source_var, state="readonly",
                                           values=["Webcam 0", "Webcam 1", "IP Camera", "Video File"])
        self.source_chooser.grid(row=0, column=1, padx=4, pady=4, sticky='we')
        self.source_chooser.bind("<<ComboboxSelected>>", self._on_source_change)

        self.ip_var = tk.StringVar()
        self.ip_entry = ttk.Entry(controls, textvariable=self.ip_var, width=35)
        self.ip_entry.grid(row=0, column=2, padx=8, pady=8, sticky='we')
        self.ip_entry.insert(0, "http://ip:port/video")

        

        self.btn_choose = ttk.Button(controls, text="Choose Video File", command=self.choose_file)
        self.btn_choose.grid(row=0, column=3, padx=0, pady=0, sticky='we')  # Extend to 2 columns

        self.btn_start = ttk.Button(controls, text="Start", command=self.start_stream)
        self.btn_start.grid(row=1, column=0, padx=4, pady=6, sticky='we')

        self.btn_stop = ttk.Button(controls, text="Stop", command=self.stop_stream)
        self.btn_stop.grid(row=1, column=1, padx=4, pady=6, sticky='we')

        self.btn_capture = ttk.Button(controls, text="Capture Photo", command=self.capture_photo)
        self.btn_capture.grid(row=1, column=2, padx=4, pady=6, sticky='we')

        self.btn_record = ttk.Button(controls, text="Start Record", command=self.toggle_recording)
        self.btn_record.grid(row=1, column=3, padx=0, pady=0, sticky='we')

        self.btn_playback = ttk.Button(controls, text="Playback", command=self.playback_video)
        self.btn_playback.grid(row=1, column=6, padx=4, pady=6, sticky='we')  # Extend to 2 columns

        self.btn_settings = ttk.Button(controls, text="Settings", command=self.open_settings)
        self.btn_settings.grid(row=1, column=5, padx=4, pady=6, sticky='we')

        self.btn_zoom_in = ttk.Button(controls, text="Zoom In", command=self.zoom_in)
        self.btn_zoom_in.grid(row=0, column=5, padx=8, pady=6, sticky='we')

        self.btn_zoom_out = ttk.Button(controls, text="Zoom Out", command=self.zoom_out)
        self.btn_zoom_out.grid(row=0, column=6, padx=2, pady=6, sticky='we')


        for i in range(8):
            controls.columnconfigure(i, weight=1)

        # Right: thumbnails + analysis + email + voice
        thumbs = ttk.LabelFrame(right, text="Screenshots")
        thumbs.pack(fill='x', padx=5, pady=5)

        self.thumb_labels = [ttk.Label(thumbs), ttk.Label(thumbs)]
        self.thumb_labels[0].grid(row=0, column=0, padx=6, pady=6)
        self.thumb_labels[1].grid(row=0, column=1, padx=6, pady=6)

        report_frame = ttk.LabelFrame(right, text="Detection & Analysis Report")
        report_frame.pack(fill='both', expand=True, padx=5, pady=5)

        self.auto_report = tk.Text(report_frame, height=12, wrap='word')
        self.auto_report.pack(fill='both', expand=True, padx=4, pady=4)

        btns = ttk.Frame(report_frame)
        btns.pack(fill='x', pady=4)
        self.btn_analyze = ttk.Button(btns, text="Generate Analysis From Current Frame", command=self.generate_analysis_from_frame)
        self.btn_analyze.pack(side='left', padx=4)

        email_frame = ttk.LabelFrame(right, text="Email Report")
        email_frame.pack(fill='x', padx=5, pady=5)

        self.to_email_var = tk.StringVar()
        ttk.Label(email_frame, text="To:").grid(row=0, column=0, sticky='e', padx=4, pady=4)
        ttk.Entry(email_frame, textvariable=self.to_email_var, width=28).grid(row=0, column=1, sticky='we', padx=4, pady=4)

        self.btn_send_report = ttk.Button(email_frame, text="Send Report Email", command=self.send_report_email)
        self.btn_send_report.grid(row=0, column=2, padx=4, pady=4)

        # Voice
        voice_frame = ttk.LabelFrame(right, text="Voice Report")
        voice_frame.pack(fill='x', padx=5, pady=5)

        self.btn_generate_voice = ttk.Button(voice_frame, text="Generate Voice from Report", command=self.generate_voice_from_report)
        self.btn_generate_voice.grid(row=0, column=0, padx=4, pady=4)

        self.btn_play_voice = ttk.Button(voice_frame, text="Play Voice", command=self.play_voice)
        self.btn_play_voice.grid(row=0, column=1, padx=4, pady=4)

        self.btn_send_voice = ttk.Button(voice_frame, text="Email Voice MP3", command=self.send_voice_email)
        self.btn_send_voice.grid(row=0, column=2, padx=4, pady=4)

        # Footer
        footer = ttk.Label(self.root, text="DESIGNED BY NAVY AI COURSE 5 SYN 2 (2025)", anchor='center', style='Footer.TLabel')
        footer.pack(side='bottom', fill='x', pady=6)
        
    def _ensure_logo_image(self):
        p = os.path.join(SAVE_DIR, "logo_fixed.png")
        if os.path.exists(p):
            return Image.open(p)
        img = Image.new("RGBA", (380, 80), (0,0,0,0))
        d = ImageDraw.Draw(img)
        d.ellipse((10,10,70,70), outline=(30,200,80,255), width=4)
        d.line((40,40,70,25), fill=(30,200,80,255), width=4)
        d.rectangle((90,15,150,65), outline=(200,200,200,255), width=3)
        try:
            font = ImageFont.truetype("arial.ttf", 30)
        except Exception:
            font = ImageFont.load_default()
        d.text((170, 28), "AI SURVEILLANCE", fill=(255,255,255,255), font=font)
        img.save(p)
        return img

    def _init_styles(self):
        style = ttk.Style()
        # Try a colorful theme
        style.theme_use('clam')

        style.configure("TFrame", background='#0a2a66')
        style.configure("TLabelframe", background='#0a2a66', foreground='white')
        style.configure("TLabelframe.Label", background='#0a2a66', foreground='white', font=('Arial', 11, 'bold'))
        style.configure("TLabel", background='#0a2a66', foreground='white')
        style.configure("TButton", font=('Arial', 10, 'bold'))
        style.map("TButton",
                  foreground=[('active', '#0b3d91'), ('!active', '#0b3d91')],
                  background=[('active', '#ffd166'), ('!active', '#ffd166')])
        style.configure("Footer.TLabel", background='#072a63', foreground='white', font=('Arial', 10, 'bold'))

    # --------------------------- Control actions ------------------------------

    def _on_source_change(self, event=None):
        choice = self.source_var.get()
        self.root.after(0, lambda: (
            self.ip_entry.configure(state='normal' if choice == "IP Camera" else 'disabled'),
            self.btn_choose.configure(state='normal' if choice == "Video File" else 'disabled')
        ))

    def choose_file(self):
        path = filedialog.askopenfilename(title="Choose video file",
                                          initialdir=".",
                                          filetypes=[("Video files", "*.mp4 *.avi *.mkv *.mov"), ("All files", "*.*")])
        if path:
            self.root.after(0, lambda: self._set_video_source(path))

    def _set_video_source(self, path):
        self.stop_stream()
        self.config.camera_source = path
        self.current_source = path  # Ensure current_source is set
        self.video_source_cap = cv2.VideoCapture(path)
        if not self.video_source_cap.isOpened():
            self.root.after(0, lambda: messagebox.showerror("Error", "Cannot open selected video."))
            self.video_source_cap = None
            return
        self.start_stream()

    def start_stream(self):
        if self.running:
            return
        choice = self.source_var.get()
        src = None
        if choice == "Webcam 0":
            src = 0
        elif choice == "Webcam 1":
            src = 1
        elif choice == "IP Camera":
            src = self.ip_var.get().strip()
            if not src:
                self.root.after(0, lambda: messagebox.showwarning("IP Camera", "Please enter the IP/RTSP URL."))
                return
        elif choice == "Video File":
            if not self.current_source:
                self.root.after(0, lambda: messagebox.showwarning("Video File", "Choose a video file first."))
                return
            src = self.current_source

        self.cap = cv2.VideoCapture(src)
        if not self.cap or not self.cap.isOpened():
            error_msg = f"Unable to open source: {src}. Check URL or connection." if choice == "IP Camera" else "Unable to open source."
            self.root.after(0, lambda: messagebox.showerror("Camera", error_msg))
            if self.cap:
                self.cap.release()
                self.cap = None
            return

        self.running = True
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        self._update_canvas()

    def stop_stream(self):
        self.running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        if self.video_source_cap:
            self.video_source_cap.release()
            self.video_source_cap = None
        if self.writer:
            self.writer.release()
            self.writer = None
        if self.root.winfo_exists():
            self.canvas.delete("all")

    def toggle_recording(self):
        if not self.running:
            messagebox.showinfo("Recording", "Start the live feed first.")
            return
        self.recording = not self.recording
        if self.recording:
            self.btn_record.config(text="Stop Recording")
        else:
            self.btn_record.config(text="Start Recording")
            if self.writer:
                self.writer.release()
                self.writer = None

    def capture_photo(self):
        if self.last_frame is None:
            messagebox.showinfo("Capture", "No frame available yet.")
            return
        os.makedirs("captures", exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = f"captures/capture_{ts}.jpg"
        cv2.imwrite(path, self.last_frame)
        self._add_thumbnail(path)

    def playback_video(self):
        path = filedialog.askopenfilename(title="Choose video to play",
                                          initialdir=".",
                                          filetypes=[("Video files", "*.mp4 *.avi *.mkv *.mov"), ("All files", "*.*")])
        if not path:
            return
        self.stop_playback()
        playback_win = tk.Toplevel(self.root)
        playback_win.title("Video Playback")
        playback_win.geometry("640x480")
        playback_win.protocol("WM_DELETE_WINDOW", self.stop_playback)
        
        self.playback_canvas = tk.Canvas(playback_win, width=640, height=360)
        self.playback_canvas.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        
        self.playback_cap = cv2.VideoCapture(path)
        if not self.playback_cap.isOpened():
            self.root.after(0, lambda: messagebox.showerror("Playback", "Cannot open selected video."))
            playback_win.destroy()
            return
        self.playback_running = True
        self.playback_thread = threading.Thread(target=self._playback_loop, daemon=True)
        self.playback_thread.start()

    def stop_playback(self):
        self.playback_running = False
        if self.playback_cap:
            self.playback_cap.release()
            self.playback_cap = None
        if self.playback_canvas:
            self.playback_canvas.destroy()
            self.playback_canvas = None
        if self.playback_thread:
            self.playback_thread = None

    def _playback_loop(self):
        while self.playback_running and self.playback_cap:
            ret, frame = self.playback_cap.read()
            if not ret:
                self.playback_running = False
                self.root.after(0, self.stop_playback)
                break
            self.last_frame = frame  # Update last_frame for analysis
            h, w = frame.shape[:2]
            c_w = self.playback_canvas.winfo_width() or 640
            c_h = self.playback_canvas.winfo_height() or 360
            scale = min(c_w / w, c_h / h)
            nw, nh = int(w * scale), int(h * scale)
            resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            tk_img = ImageTk.PhotoImage(image=pil)
            self.playback_canvas.delete("all")
            self.playback_canvas.create_image(c_w//2, c_h//2, image=tk_img, anchor='center')
            self.playback_canvas.image = tk_img  # Keep reference
            time.sleep(1 / self.playback_cap.get(cv2.CAP_PROP_FPS))
        self.root.after(0, self.stop_playback)

    def open_settings(self):
        win = tk.Toplevel(self.root)
        win.title("Settings")
        win.configure(bg='#0b3d91')

        ttk.Label(win, text="OpenAI API Key:").grid(row=0, column=0, sticky='e', padx=6, pady=6)
        hf_var = tk.StringVar(value=self.config.hf_token)
        ttk.Entry(win, textvariable=hf_var, width=50).grid(row=0, column=1, padx=6, pady=6)

        ttk.Label(win, text="SMTP Server:").grid(row=1, column=0, sticky='e', padx=6, pady=6)
        smtp_server_var = tk.StringVar(value=self.config.smtp_server)
        ttk.Entry(win, textvariable=smtp_server_var).grid(row=1, column=1, padx=6, pady=6)

        ttk.Label(win, text="SMTP Port:").grid(row=2, column=0, sticky='e', padx=6, pady=6)
        smtp_port_var = tk.StringVar(value=str(self.config.smtp_port))
        ttk.Entry(win, textvariable=smtp_port_var).grid(row=2, column=1, padx=6, pady=6)

        use_tls_var = tk.BooleanVar(value=self.config.use_tls)
        ttk.Checkbutton(win, text="Use TLS", variable=use_tls_var).grid(row=3, column=1, sticky='w', padx=6, pady=6)

        ttk.Label(win, text="Sender Email:").grid(row=4, column=0, sticky='e', padx=6, pady=6)
        sender_email_var = tk.StringVar(value=self.config.sender_email)
        ttk.Entry(win, textvariable=sender_email_var).grid(row=4, column=1, padx=6, pady=6)

        ttk.Label(win, text="Sender Password/App Password:").grid(row=5, column=0, sticky='e', padx=6, pady=6)
        sender_pass_var = tk.StringVar(value=self.config.sender_password)
        ttk.Entry(win, textvariable=sender_pass_var, show="*").grid(row=5, column=1, padx=6, pady=6)

        def save_settings():
            self.config.hf_token = hf_var.get().strip()
            self.config.smtp_server = smtp_server_var.get().strip()
            try:
                self.config.smtp_port = int(smtp_port_var.get().strip())
            except:
                self.config.smtp_port = 587
            self.config.use_tls = use_tls_var.get()
            self.config.sender_email = sender_email_var.get().strip()
            self.config.sender_password = sender_pass_var.get().strip()
            messagebox.showinfo("Settings", "Saved.")
            win.destroy()
            

        ttk.Button(win, text="Save", command=save_settings).grid(row=6, column=1, sticky='e', padx=6, pady=10)
    def zoom_in(self):
        self.zoom_level = min(self.zoom_level + self.zoom_step, 3.0)  # Cap at 3x zoom

    def zoom_out(self):
        self.zoom_level = max(self.zoom_level - self.zoom_step, 0.5)  # Minimum 0.5x zoom
        
        
    def _safe_detect_and_annotate(self, frame_bgr):
        dets_meta = []
        summary_counts = {}
        annotated = frame_bgr.copy()
        is_threat = False
        if not self.model_loaded or self.model is None:
            return annotated, dets_meta, False, "No model"
        try:
            results = self.model.predict(source=frame_bgr, conf=self.detector_conf, verbose=False)
            res_list = results if isinstance(results, (list, tuple)) else [results]
            for res in res_list:
                try:
                    xyxy = res.boxes.xyxy.cpu().numpy()
                    confs = res.boxes.conf.cpu().numpy()
                    clsids = res.boxes.cls.cpu().numpy()
                except Exception:
                    continue
                for (box, conf, clsid) in zip(xyxy, confs, clsids):
                    x1, y1, x2, y2 = map(int, box)
                    name = self.class_names.get(int(clsid), str(int(clsid)))
                    dets_meta.append(((x1,y1,x2,y2), name, float(conf)))
                    summary_counts[name] = summary_counts.get(name, 0) + 1
            tracked = self.tracker.update([b for (b,_,_) in dets_meta])
            for (box, name, conf) in dets_meta:
                x1,y1,x2,y2 = box
                lowname = name.lower()
                is_obj_threat = (lowname in THREAT_OBJECTS) or any(k in lowname for k in THREAT_KEYWORDS)
                color = (0,0,255) if is_obj_threat else (0,255,0)
                if is_obj_threat:
                    is_threat = True
                cv2.rectangle(annotated, (x1,y1), (x2,y2), color, thickness=4)
                label = f"{name} {conf:.2f}"
                cx, cy = (x1+x2)//2, (y1+y2)//2
                best_id, best_d = None, 1e9
                for tid, (tx,ty) in tracked.items():
                    d = ((cx-tx)**2 + (cy-ty)**2) ** 0.5  # Euclidean distance
                    if d < best_d:
                        best_d, best_id = d, tid
                if best_d < 60:
                    label = f"ID {best_id} | " + label
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(annotated, (x1, y1-th-10), (x1+tw+6, y1), color, -1)
                cv2.putText(annotated, label, (x1+3, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
            if summary_counts.get("person",0) >= 3 and (summary_counts.get("backpack",0) >=1 or any(summary_counts.get(v,0)>=1 for v in ["truck","car","bus"])):
                is_threat = True
            parts = []
            for k,v in sorted(summary_counts.items(), key=lambda x: -x[1]):
                parts.append(f"{v} {k}{'' if v==1 else 's'}")
            summary = ", ".join(parts) if parts else "No notable objects"
            return annotated, dets_meta, is_threat, summary
        except Exception as e:
            print("Detection error:", e)
            return annotated, dets_meta, False, "Detection error"
        
    def _generate_report(self, frame_bgr, dets_with_meta, extra_summary="", is_threat=False):
        ts = datetime.now().strftime("%a %d%H%M %b %y").upper()  # e.g., SAT 161723 AUG 25
        counts = {}
        detections = []
        for (box, cls, conf) in dets_with_meta:
            counts[cls] = counts.get(cls, 0) + 1
            detections.append({'cls_name': cls, 'conf': conf, 'bbox': box})
        scene = "indoor/office-like" if counts.get("person",0) >= 2 else "open area"
        detected_list = ", ".join([f"{v} {k}{'' if v==1 else 's'}" for k,v in counts.items()]) if counts else "none"
        text_blob = " ".join(list(counts.keys()))
        threat_level, reasons = assess_threat(detections, text_blob)
        lines = [
            f"INCIDENT REPORT AS AT {ts}",
            f"Scene: {scene}",
            f"Observations: {extra_summary or detected_list}",
            f"Detailed counts: {detected_list}",
            f"Threat Assessment: {threat_level}",
            "Reasons: " + "; ".join(reasons) if reasons else "Reasons: No strong threat indicators",
        ]
        # Add recommendations based on threat level
        recommendations = []
        if threat_level == "HIGH":
            recommendations.append("Alert authorities immediately. Evacuate area if necessary. Deploy security personnel.")
        elif threat_level == "MEDIUM":
            recommendations.append("Increase monitoring. Notify supervisor. Prepare for potential escalation.")
        else:
            recommendations.append("Continue routine surveillance. No immediate action required.")
        lines.append("Recommendations: " + "; ".join(recommendations))
        # Detailed description
        if counts:
            persons = counts.get("person",0)
            setting = "office/group" if persons >= 2 else "open environment"
            attire_hint = "wearing various clothing, camouflage not confirmed"
            activity = "moving/discussing" if persons >= 2 else "standing/moving"
            lines.append("")
            lines.append(f"Auto-Description: {persons} person(s) detected; setting looks like {setting}; {attire_hint}; likely {activity}.")
        else:
            lines.append("")
            lines.append("Auto-Description: No objects detected.")
        return "\n".join(lines)
    
    def _set_report(self, text):
        try:
            cur = self.auto_report.index(tk.INSERT)  # Use self.auto_report instead of self.report_txt
            self.auto_report.delete("1.0", "end")
            self.auto_report.insert("1.0", text)
            self.auto_report.mark_set(tk.INSERT, cur)
        except Exception:
            self.auto_report.delete("1.0", "end")
            self.auto_report.insert("1.0", text)
        
    # --------------------------- Capture & Draw -------------------------------

    def _capture_loop(self):
        fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_path = None

        while self.running and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            self.last_frame = frame.copy()

             # Use _safe_detect_and_annotate for detection and annotation
            annotated, dets_meta, is_threat, summary = self._safe_detect_and_annotate(frame)

            # Write to recording if active
            if self.recording:
                if self.writer is None:
                    os.makedirs("recordings", exist_ok=True)
                    out_path = f"recordings/rec_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
                    self.writer = cv2.VideoWriter(out_path, fourcc, fps, (annotated.shape[1], annotated.shape[0]))
                self.writer.write(annotated)

            # Push to queue for UI thread
            try:
                if not self.frame_queue.full():
                    self.frame_queue.put(annotated, block=False)
            except queue.Full:
                pass

        # Cleanup
        if self.writer:
            self.writer.release()
            self.writer = None

    def _update_canvas(self):
        if not self.running or self.root.winfo_exists() == 0:
            return
        try:
            frame = self.frame_queue.get_nowait()
            self._draw_on_canvas(frame)
        except queue.Empty:
            pass
        self.root.after(15, self._update_canvas)

    def _draw_on_canvas(self, frame_bgr):
        # Fit to canvas
        self.last_frame = frame_bgr.copy()
        h, w = frame_bgr.shape[:2]
        c_w = self.canvas.winfo_width() or 640
        c_h = self.canvas.winfo_height() or 360
        scale = min(c_w / w, c_h / h) * self.zoom_level
        nw, nh = int(w * scale), int(h * scale)
        resized = cv2.resize(frame_bgr, (nw, nh), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        # Tk photo
        img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)  # keep as BGR for imencode? but we need Tk image
        # Use PhotoImage via PIL to preserve colors
        from PIL import Image, ImageTk
        pil = Image.fromarray(rgb)
        self.tk_img = ImageTk.PhotoImage(image=pil)
        self.canvas.delete("all")
        self.canvas.create_image(c_w//2, c_h//2, image=self.tk_img, anchor='center')

    # --------------------------- Thumbnails ----------------------------------

    def _add_thumbnail(self, path):
        self.snapshot_paths.insert(0, path)
        self.snapshot_paths = self.snapshot_paths[:2]
        # Update UI
        from PIL import Image, ImageTk
        for i in range(2):
            if i < len(self.snapshot_paths):
                img = Image.open(self.snapshot_paths[i])
                img.thumbnail((260, 160))
                tkimg = ImageTk.PhotoImage(img)
                self.thumb_labels[i].configure(image=tkimg)
                self.thumb_labels[i].image = tkimg
            else:
                self.thumb_labels[i].configure(image='', text='')

    # --------------------------- Analysis & Voice -----------------------------

    def generate_analysis_from_frame(self):
        if self.last_frame is None:
            messagebox.showinfo("Analysis", "No frame available.")
            return
        # Run detection for report
        _, dets_with_meta, is_threat, summary = self._safe_detect_and_annotate(self.last_frame)

        # Generate report using custom logic
        report_text = self._generate_report(self.last_frame, dets_with_meta, summary, is_threat)

        self.last_analysis = report_text
        # Update textbox (user can still edit)
        self._set_report(self.last_analysis)

        # Save a screenshot for record & thumbnails
        os.makedirs("captures", exist_ok=True)
        snap_path = f"captures/analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        cv2.imwrite(snap_path, self.last_frame)
        self._add_thumbnail(snap_path)

    def generate_voice_from_report(self):
        text = self.auto_report.get("1.0", tk.END).strip()
        if not text:
            messagebox.showinfo("Voice", "Report text is empty.")
            return
        os.makedirs("audio_reports", exist_ok=True)
        path = f"audio_reports/report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp3"
        try:
            tts = gTTS(text=text, lang='en')
            tts.save(path)
            self.last_audio_path = path
            messagebox.showinfo("Voice", f"Saved voice report:\n{path}")
        except Exception as e:
            messagebox.showerror("Voice", f"Failed to generate voice:\n{e}")

    def play_voice(self):
        if not self.last_audio_path or not os.path.exists(self.last_audio_path):
            messagebox.showinfo("Voice", "No voice file available. Generate first.")
            return
        try:
            # playsound is blocking; run in a thread
            threading.Thread(target=playsound, args=(self.last_audio_path,), daemon=True).start()
        except Exception as e:
            messagebox.showerror("Voice", f"Failed to play audio:\n{e}")

    # --------------------------- Email senders --------------------------------

    def send_report_email(self):
        to_email = self.to_email_var.get().strip()
        if not to_email:
            messagebox.showwarning("Email", "Enter recipient email.")
            return
        text = self.auto_report.get("1.0", tk.END).strip()
        if not text:
            messagebox.showwarning("Email", "Report text is empty.")
            return
        try:
            attachments = []
            # Attach the latest two screenshots if exist
            for p in self.snapshot_paths[:2]:
                if os.path.exists(p):
                    with open(p, "rb") as f:
                        attachments.append((os.path.basename(p), f.read()))
            send_email(self.config.smtp_server, self.config.smtp_port, self.config.use_tls,
                       self.config.sender_email, self.config.sender_password,
                       to_email,
                       subject="AI Surveillance - Detection Report",
                       body=text,
                       attachments=attachments)
            messagebox.showinfo("Email", "Report sent.")
        except Exception as e:
            messagebox.showerror("Email", f"Failed to send email:\n{e}")

    def send_voice_email(self):
        to_email = self.to_email_var.get().strip()
        if not to_email:
            messagebox.showwarning("Email", "Enter recipient email.")
            return
        if not self.last_audio_path or not os.path.exists(self.last_audio_path):
            messagebox.showwarning("Email", "No voice MP3 found. Generate voice first.")
            return
        try:
            with open(self.last_audio_path, "rb") as f:
                audio_bytes = f.read()
            send_email(self.config.smtp_server, self.config.smtp_port, self.config.use_tls,
                       self.config.sender_email, self.config.sender_password,
                       to_email,
                       subject="AI Surveillance - Voice Report",
                       body="Voice report attached.",
                       attachments=[(os.path.basename(self.last_audio_path), audio_bytes)])
            messagebox.showinfo("Email", "Voice email sent.")
        except Exception as e:
            messagebox.showerror("Email", f"Failed to send email:\n{e}")

    # --------------------------- Shutdown ------------------------------------

    def on_close(self):
        self.stop_stream()
        self.root.destroy()


def main():
    root = tk.Tk()
    app = SurveillanceApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
