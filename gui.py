"""
gui.py — Premium GUI for the Secure Face Recognition Authentication System.

Modern dark-themed tkinter application with:
  - Camera selection (local vs external IP/RTSP)
  - Live camera feed embedded in the GUI
  - Step-by-step pipeline progress with visual indicators
  - Enrollment and authentication modes
  - Auto key generation for encryption
"""

import sys
import os
import time
import threading
import tkinter as tk
from tkinter import ttk, messagebox
from typing import Optional
import cv2
import numpy as np

from PIL import Image, ImageTk

sys.path.insert(0, os.path.dirname(__file__))
from modules.ai_voice_bot import bot as voice_bot

import config
from modules.camera import CameraCapture
from modules.face_recognition_module import (
    get_live_embedding, match_face, rate_limiter, _get_fernet, enroll_user
)
from modules.anti_spoofing import AntiSpoofing
from modules.voice_greeting import VoiceGreeting
from modules.head_movement import HeadMovementChallenge
from modules.blink_detection import BlinkDetection
from modules.voice_challenge import VoiceChallenge
from modules.session import SessionManager, AuditLogger

# ─── Premium Color Palette ────────────────────────────────────────────────────
COLORS = {
    "bg_dark":       "#080c14",
    "bg_card":       "#0f1629",
    "bg_card_hover": "#151d35",
    "bg_input":      "#1a2340",
    "border":        "#1c2e50",
    "border_glow":   "#2563eb30",
    "text_primary":  "#f0f4fc",
    "text_secondary":"#8b9dc3",
    "text_muted":    "#4a5d82",
    "accent_blue":   "#2563eb",
    "accent_cyan":   "#0ea5e9",
    "accent_green":  "#22c55e",
    "accent_red":    "#f43f5e",
    "accent_amber":  "#f59e0b",
    "accent_purple": "#7c3aed",
    "accent_pink":   "#ec4899",
    "success":       "#22c55e",
    "error":         "#f43f5e",
    "warning":       "#f59e0b",
    "step_pending":  "#1e293b",
    "step_active":   "#2563eb",
    "step_done":     "#22c55e",
    "step_fail":     "#f43f5e",
    "gradient_start":"#2563eb",
    "gradient_end":  "#7c3aed",
}

FONT = "Segoe UI"


# ─── Custom Widgets ───────────────────────────────────────────────────────────

class GlowButton(tk.Canvas):
    """Modern button with hover glow and press animation."""

    def __init__(self, parent, text, command=None, width=220, height=48,
                 bg=COLORS["accent_blue"], fg="#ffffff", font_size=12, **kw):
        super().__init__(parent, width=width, height=height,
                         bg=parent["bg"], highlightthickness=0, **kw)
        self.command = command
        self._bg = bg
        self._fg = fg
        self._btn_w = width
        self._btn_h = height
        self._text = text
        self._fs = font_size
        self._hover = False
        self._pressed = False
        self._draw()
        self.bind("<Enter>", lambda e: self._set_hover(True))
        self.bind("<Leave>", lambda e: (self._set_hover(False), self._set_press(False)))
        self.bind("<ButtonPress-1>", lambda e: self._set_press(True))
        self.bind("<ButtonRelease-1>", self._on_release)

    def _set_hover(self, val):
        self._hover = val
        self._draw()
        self.config(cursor="hand2" if val else "")

    def _set_press(self, val):
        self._pressed = val
        self._draw()

    def _on_release(self, e):
        self._pressed = False
        self._draw()
        if self.command and 0 <= e.x <= self._btn_w and 0 <= e.y <= self._btn_h:
            self.command()

    def _draw(self):
        self.delete("all")
        if self._pressed:
            c = self._darken(self._bg, 25)
        elif self._hover:
            c = self._lighten(self._bg, 22)
        else:
            c = self._bg

        r = 14
        w, h = self._btn_w, self._btn_h

        # Subtle shadow on hover
        if self._hover and not self._pressed:
            sc = self._lighten(self._bg, 8)
            self.create_oval(-4, 2, w+4, h+6, fill=sc, outline="")

        # Rounded rectangle
        self.create_arc(0, 0, 2*r, 2*r, start=90, extent=90, fill=c, outline="")
        self.create_arc(w-2*r, 0, w, 2*r, start=0, extent=90, fill=c, outline="")
        self.create_arc(0, h-2*r, 2*r, h, start=180, extent=90, fill=c, outline="")
        self.create_arc(w-2*r, h-2*r, w, h, start=270, extent=90, fill=c, outline="")
        self.create_rectangle(r, 0, w-r, h, fill=c, outline="")
        self.create_rectangle(0, r, w, h-r, fill=c, outline="")

        # Text with slight vertical offset when pressed
        ty = h//2 + (1 if self._pressed else 0)
        self.create_text(w//2, ty, text=self._text, fill=self._fg,
                         font=(FONT, self._fs, "bold"))

    @staticmethod
    def _lighten(hex_color, amount):
        hc = hex_color.lstrip("#")
        r = min(255, int(hc[0:2], 16) + amount)
        g = min(255, int(hc[2:4], 16) + amount)
        b = min(255, int(hc[4:6], 16) + amount)
        return f"#{r:02x}{g:02x}{b:02x}"

    @staticmethod
    def _darken(hex_color, amount):
        hc = hex_color.lstrip("#")
        r = max(0, int(hc[0:2], 16) - amount)
        g = max(0, int(hc[2:4], 16) - amount)
        b = max(0, int(hc[4:6], 16) - amount)
        return f"#{r:02x}{g:02x}{b:02x}"


class StepIndicator(tk.Frame):
    """Visual pipeline progress with circular step badges."""

    STEPS = [
        ("1", "Detect"), ("2", "Match"), ("3", "Spoof"),
        ("4", "Confirm"), ("5", "Head"), ("6", "Blink"),
        ("7", "Voice"), ("8", "Token"),
    ]

    def __init__(self, parent, **kw):
        super().__init__(parent, bg=COLORS["bg_dark"], **kw)
        self._canvases = []
        self._labels = []
        self._lines = []
        self._states = ["pending"] * len(self.STEPS)

        for i, (num, name) in enumerate(self.STEPS):
            if i > 0:
                line = tk.Canvas(self, width=28, height=4,
                                 bg=COLORS["bg_dark"], highlightthickness=0)
                line.grid(row=0, column=i*2-1, padx=0, pady=(8, 0))
                line.create_rectangle(0, 1, 28, 3, fill=COLORS["step_pending"], outline="")
                self._lines.append(line)
            else:
                self._lines.append(None)

            f = tk.Frame(self, bg=COLORS["bg_dark"])
            f.grid(row=0, column=i*2, padx=3)

            size = 36
            cv = tk.Canvas(f, width=size, height=size,
                           bg=COLORS["bg_dark"], highlightthickness=0)
            cv.pack()
            # Draw circle + number
            cv.create_oval(2, 2, size-2, size-2,
                          fill=COLORS["step_pending"], outline=COLORS["border"], width=1)
            cv.create_text(size//2, size//2, text=num,
                          fill=COLORS["text_muted"], font=(FONT, 11, "bold"))
            self._canvases.append(cv)

            nl = tk.Label(f, text=name, font=(FONT, 7), bg=COLORS["bg_dark"],
                          fg=COLORS["text_muted"])
            nl.pack(pady=(3, 0))
            self._labels.append(nl)

    def set_state(self, idx, state):
        if 0 <= idx < len(self._canvases):
            self._states[idx] = state
            cv = self._canvases[idx]
            size = 36
            cv.delete("all")

            fill_map = {
                "pending": COLORS["step_pending"],
                "active": COLORS["step_active"],
                "done": COLORS["step_done"],
                "fail": COLORS["step_fail"],
            }
            fill = fill_map.get(state, COLORS["step_pending"])

            # Draw outline glow for active
            if state == "active":
                cv.create_oval(0, 0, size, size, fill="", outline=COLORS["accent_cyan"], width=2)

            cv.create_oval(2, 2, size-2, size-2, fill=fill, outline="", width=0)

            # Icon text
            num = self.STEPS[idx][0]
            icon_map = {"done": "✓", "fail": "✗"}
            display_text = icon_map.get(state, num)
            cv.create_text(size//2, size//2, text=display_text,
                          fill="#ffffff", font=(FONT, 11, "bold"))

            # Label color
            fg_map = {
                "active": COLORS["accent_cyan"],
                "done": COLORS["success"],
                "fail": COLORS["error"],
                "pending": COLORS["text_muted"],
            }
            self._labels[idx].config(fg=fg_map.get(state, COLORS["text_muted"]))

            # Update connecting line
            if idx > 0 and self._lines[idx] and state in ("done", "active"):
                line = self._lines[idx]
                line.delete("all")
                line_color = COLORS["step_done"] if state == "done" else COLORS["accent_blue"]
                line.create_rectangle(0, 1, 28, 3, fill=line_color, outline="")

    def reset(self):
        for i in range(len(self.STEPS)):
            self._states[i] = "pending"
            cv = self._canvases[i]
            size = 36
            cv.delete("all")
            cv.create_oval(2, 2, size-2, size-2,
                          fill=COLORS["step_pending"], outline=COLORS["border"], width=1)
            cv.create_text(size//2, size//2, text=self.STEPS[i][0],
                          fill=COLORS["text_muted"], font=(FONT, 11, "bold"))
            self._labels[i].config(fg=COLORS["text_muted"])
            if self._lines[i]:
                line = self._lines[i]
                line.delete("all")
                line.create_rectangle(0, 1, 28, 3, fill=COLORS["step_pending"], outline="")


# ─── Main Application ────────────────────────────────────────────────────────

class SecureFaceAuthApp:

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Secure Face Recognition System")
        self.root.configure(bg=COLORS["bg_dark"])
        self.root.resizable(True, True)

        # Center on screen
        sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
        w, h = 1000, 720
        root.geometry(f"{w}x{h}+{(sw-w)//2}+{(sh-h)//2}")
        root.minsize(900, 650)

        # State
        self.camera: Optional[CameraCapture] = None
        self.camera_running = False
        self.pipeline_running = False
        self.current_frame = None
        self._after_id = None

        # Enrollment state
        self._enroll_images = []
        self._enroll_cam: Optional[CameraCapture] = None
        self._enroll_feed_running = False
        self._enroll_after_id = None
        self._enroll_current_pose = 0
        self._enroll_face_detected = False
        self._enroll_current_frame = None
        self._enroll_uid = ""
        self._enroll_name = ""

        self._build_chrome()
        self._show_home()
        root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ─── Chrome (top bar + status bar) ───────────────────────────────────

    def _build_chrome(self):
        # Top bar
        top = tk.Frame(self.root, bg=COLORS["bg_card"], height=52)
        top.pack(fill="x"); top.pack_propagate(False)
        lf = tk.Frame(top, bg=COLORS["bg_card"]); lf.pack(side="left", padx=20)
        tk.Label(lf, text="SECURE", font=(FONT, 14, "bold"),
                 bg=COLORS["bg_card"], fg=COLORS["accent_blue"]).pack(side="left")
        tk.Label(lf, text="FACE", font=(FONT, 14, "bold"),
                 bg=COLORS["bg_card"], fg=COLORS["accent_cyan"]).pack(side="left", padx=(4,0))
        tk.Label(lf, text="AUTH", font=(FONT, 14),
                 bg=COLORS["bg_card"], fg=COLORS["text_muted"]).pack(side="left", padx=(4,0))
        # Version badge
        vb = tk.Label(top, text=" v1.0 ", font=(FONT, 8),
                      bg=COLORS["bg_input"], fg=COLORS["text_secondary"])
        vb.pack(side="right", padx=20)
        # Accent line under header
        accent_line = tk.Canvas(self.root, height=2, bg=COLORS["bg_dark"], highlightthickness=0)
        accent_line.pack(fill="x")
        accent_line.update_idletasks()
        aw = max(accent_line.winfo_width(), 1000)
        accent_line.create_rectangle(0, 0, aw//2, 2, fill=COLORS["accent_blue"], outline="")
        accent_line.create_rectangle(aw//2, 0, aw, 2, fill=COLORS["accent_purple"], outline="")

        # Content area
        self.content = tk.Frame(self.root, bg=COLORS["bg_dark"])
        self.content.pack(fill="both", expand=True)

        # Status bar
        sb = tk.Frame(self.root, bg=COLORS["bg_card"], height=30)
        sb.pack(fill="x", side="bottom"); sb.pack_propagate(False)
        self.status_label = tk.Label(sb, text="Ready", font=(FONT, 9),
                                     bg=COLORS["bg_card"], fg=COLORS["text_muted"], anchor="w")
        self.status_label.pack(side="left", padx=16)
        self.status_dot = tk.Label(sb, text="●", font=(FONT, 8),
                                   bg=COLORS["bg_card"], fg=COLORS["accent_green"])
        self.status_dot.pack(side="right", padx=16)

    def _clear(self):
        if self._after_id:
            self.root.after_cancel(self._after_id); self._after_id = None
        for w in self.content.winfo_children():
            w.destroy()

    def _status(self, text, color=COLORS["text_muted"]):
        self.status_label.config(text=text)
        self.status_dot.config(fg=color)

    def _gui(self, func, *args):
        """Thread-safe GUI update."""
        try:
            self.root.after(0, func, *args)
        except Exception:
            pass

    # ─── Home Screen ─────────────────────────────────────────────────────

    def _show_home(self):
        self._clear(); self._stop_camera(); self._stop_enroll_camera()
        self._status("Home", COLORS["accent_green"])

        # Outer wrapper, centered
        outer = tk.Frame(self.content, bg=COLORS["bg_dark"])
        outer.place(relx=0.5, rely=0.46, anchor="center")

        # Hero section
        tk.Label(outer, text="SECURE FACE", font=(FONT, 32, "bold"),
                 bg=COLORS["bg_dark"], fg=COLORS["text_primary"]).pack()
        tk.Label(outer, text="AUTHENTICATION", font=(FONT, 28),
                 bg=COLORS["bg_dark"], fg=COLORS["accent_cyan"]).pack(pady=(0, 6))
        tk.Label(outer, text="Multi-layer biometric identity verification",
                 font=(FONT, 11), bg=COLORS["bg_dark"],
                 fg=COLORS["text_secondary"]).pack(pady=(0, 28))

        # Features grid (2x2 inside a card)
        card = tk.Frame(outer, bg=COLORS["bg_card"], padx=24, pady=16,
                        highlightbackground=COLORS["border"], highlightthickness=1)
        card.pack(pady=(0, 28))
        features = [
            ("AES-128", "Encrypted face storage", COLORS["accent_blue"]),
            ("Anti-Spoof", "Texture liveness check", COLORS["accent_cyan"]),
            ("Voice AI", "Speaker verification", COLORS["accent_purple"]),
            ("Audit Log", "Full event tracking", COLORS["accent_green"]),
        ]
        for i, (title, desc, clr) in enumerate(features):
            ff = tk.Frame(card, bg=COLORS["bg_card"])
            ff.grid(row=i//2, column=i%2, padx=16, pady=8, sticky="w")
            tk.Label(ff, text="●", font=(FONT, 8), bg=COLORS["bg_card"],
                     fg=clr).pack(side="left", padx=(0, 6))
            tf = tk.Frame(ff, bg=COLORS["bg_card"])
            tf.pack(side="left")
            tk.Label(tf, text=title, font=(FONT, 10, "bold"),
                     bg=COLORS["bg_card"], fg=clr).pack(anchor="w")
            tk.Label(tf, text=desc, font=(FONT, 8),
                     bg=COLORS["bg_card"], fg=COLORS["text_muted"]).pack(anchor="w")

        # CTA buttons
        bf = tk.Frame(outer, bg=COLORS["bg_dark"]); bf.pack()
        GlowButton(bf, "▶  Authenticate", self._show_camera_select,
                   width=250, height=54, bg=COLORS["accent_blue"],
                   font_size=13).pack(side="left", padx=10)
        GlowButton(bf, "+  Enroll User", self._show_enroll,
                   width=250, height=54, bg=COLORS["accent_purple"],
                   font_size=13).pack(side="left", padx=10)

        # Footer
        tk.Label(outer,
                 text="8-step pipeline  ·  Real-time liveness  ·  Voice + Face biometrics",
                 font=(FONT, 9), bg=COLORS["bg_dark"],
                 fg=COLORS["text_muted"]).pack(pady=(24, 0))

    # ─── Camera Selection ────────────────────────────────────────────────

    def _show_camera_select(self, is_enroll=False):
        self._clear()
        self._status("Select camera source", COLORS["accent_amber"])

        c = tk.Frame(self.content, bg=COLORS["bg_dark"])
        c.place(relx=0.5, rely=0.43, anchor="center")

        title = "👤 Select Camera for Enrollment" if is_enroll else "📷 Select Camera for Authentication"
        tk.Label(c, text=title, font=(FONT, 22, "bold"),
                 bg=COLORS["bg_dark"], fg=COLORS["text_primary"]).pack(pady=(0, 6))
        tk.Label(c, text="Choose your camera type to begin", font=(FONT, 11),
                 bg=COLORS["bg_dark"], fg=COLORS["text_secondary"]).pack(pady=(0, 32))

        bf = tk.Frame(c, bg=COLORS["bg_dark"])
        bf.pack()
        
        GlowButton(bf, "💻  Internal Camera", lambda: self._start_local_camera(is_enroll),
                   width=240, height=52, bg=COLORS["accent_blue"], font_size=13).pack(side="left", padx=10)
        GlowButton(bf, "🌐  External Camera", lambda: self._show_external_camera_page(is_enroll),
                   width=240, height=52, bg=COLORS["accent_cyan"], font_size=13).pack(side="left", padx=10)
        
        back_cmd = self._show_enroll if is_enroll else self._show_home
        GlowButton(c, "←  Back", back_cmd,
                   width=160, height=38, bg=COLORS["step_pending"], font_size=10).pack(pady=(40, 0))

    def _show_external_camera_page(self, is_enroll=False):
        self._clear()
        self._status("External Camera Configuration", COLORS["accent_amber"])

        c = tk.Frame(self.content, bg=COLORS["bg_dark"])
        c.place(relx=0.5, rely=0.43, anchor="center")

        tk.Label(c, text="🌐  External Camera Setup", font=(FONT, 22, "bold"),
                 bg=COLORS["bg_dark"], fg=COLORS["text_primary"]).pack(pady=(0, 6))
        tk.Label(c, text="Enter the IP and Port of your external camera stream", font=(FONT, 11),
                 bg=COLORS["bg_dark"], fg=COLORS["text_secondary"]).pack(pady=(0, 32))

        ec = tk.Frame(c, bg=COLORS["bg_card"], padx=36, pady=28,
                      highlightbackground=COLORS["border"], highlightthickness=1)
        ec.pack()

        # IP
        ipf = tk.Frame(ec, bg=COLORS["bg_card"])
        ipf.pack(fill="x", pady=8)
        tk.Label(ipf, text="IP Address:", font=(FONT, 11, "bold"), bg=COLORS["bg_card"],
                 fg=COLORS["text_secondary"], width=10, anchor="w").pack(side="left")
        self.ip_var = tk.StringVar(value="192.168.1.100")
        tk.Entry(ipf, textvariable=self.ip_var, font=(FONT, 12), width=20,
                 bg=COLORS["bg_input"], fg=COLORS["text_primary"],
                 insertbackground=COLORS["text_primary"], relief="flat",
                 highlightthickness=1, highlightbackground=COLORS["border"]).pack(side="left", padx=(6,0))

        # Port
        pf2 = tk.Frame(ec, bg=COLORS["bg_card"])
        pf2.pack(fill="x", pady=8)
        tk.Label(pf2, text="Port:", font=(FONT, 11, "bold"), bg=COLORS["bg_card"],
                 fg=COLORS["text_secondary"], width=10, anchor="w").pack(side="left")
        self.port_var = tk.StringVar(value="8080")
        tk.Entry(pf2, textvariable=self.port_var, font=(FONT, 12), width=10,
                 bg=COLORS["bg_input"], fg=COLORS["text_primary"],
                 insertbackground=COLORS["text_primary"], relief="flat",
                 highlightthickness=1, highlightbackground=COLORS["border"]).pack(side="left", padx=(6,0))

        tk.Label(ec, text="Auto-tries multiple URL formats (IP Webcam, RTSP, DroidCam)",
                 font=(FONT, 9), bg=COLORS["bg_card"], fg=COLORS["text_muted"]).pack(pady=(16, 16))

        GlowButton(ec, "Connect", lambda: self._start_external_camera(is_enroll),
                   width=240, height=46, bg=COLORS["accent_cyan"], font_size=12).pack()

        GlowButton(c, "←  Back", lambda: self._show_camera_select(is_enroll),
                   width=160, height=38, bg=COLORS["step_pending"], font_size=10).pack(pady=(28, 0))

    # ─── Camera Start Handlers ───────────────────────────────────────────

    def _start_local_camera(self, is_enroll=False):
        self._status(f"Connecting to internal camera...", COLORS["accent_amber"])
        cam = CameraCapture(camera_source=0)
        if not cam.open():
            messagebox.showerror("Camera Error", "Could not open internal camera. Check connection.")
            self._status("Camera failed", COLORS["error"])
            return
            
        if is_enroll:
            self._enroll_cam = cam
            self._enroll_images = []
            self._enroll_current_pose = 0
            self._show_enroll_capture()
        else:
            self.camera = cam
            self._show_auth_pipeline()

    def _start_external_camera(self, is_enroll=False):
        ip = self.ip_var.get().strip()
        port = self.port_var.get().strip()
        if not ip or not port:
            messagebox.showerror("Invalid", "IP address and Port are required."); return

        self._status(f"Trying to connect to {ip}:{port}...", COLORS["accent_amber"])
        self.root.update_idletasks()

        # Try multiple URL formats in a thread so the GUI doesn't freeze
        def try_connect():
            cam = CameraCapture.try_connect_external(ip, port)
            if cam:
                self._gui(self._status, f"Connected to {ip}:{port}", COLORS["accent_green"])
                if is_enroll:
                    self._enroll_cam = cam
                    self._enroll_images = []
                    self._enroll_current_pose = 0
                    self._gui(self._show_enroll_capture)
                else:
                    self.camera = cam
                    self._gui(self._show_auth_pipeline)
            else:
                self._gui(lambda: messagebox.showerror(
                    "Connection Failed",
                    f"Could not connect to {ip}:{port}.\n\n"
                    "Tried 12+ URL formats (RTSP, HTTP, MJPEG, etc).\n\n"
                    "Check that:\n"
                    "• The camera app is running and streaming\n"
                    "• IP and port are correct\n"
                    "• Your PC and camera are on the same network"))
                self._gui(self._status, "Connection failed", COLORS["error"])

        threading.Thread(target=try_connect, daemon=True).start()

    # ─── Authentication Pipeline Screen ──────────────────────────────────

    def _show_auth_pipeline(self):
        self._clear()
        self._status("Camera connected — ready", COLORS["accent_green"])
        self.camera_running = True

        # Step indicator
        top = tk.Frame(self.content, bg=COLORS["bg_dark"])
        top.pack(fill="x", pady=(14, 6), padx=20)
        self.step_indicator = StepIndicator(top)
        self.step_indicator.pack(anchor="center")

        # Thin separator
        tk.Frame(self.content, bg=COLORS["border"], height=1).pack(fill="x", padx=20)

        # Middle: camera + log
        mid = tk.Frame(self.content, bg=COLORS["bg_dark"])
        mid.pack(fill="both", expand=True, padx=20, pady=8)

        # Camera feed card
        cf = tk.Frame(mid, bg=COLORS["bg_card"],
                      highlightbackground=COLORS["border"], highlightthickness=1)
        cf.pack(side="left", fill="both", expand=True, padx=(0, 6))
        cf.pack_propagate(False)

        ch = tk.Frame(cf, bg=COLORS["bg_card"])
        ch.pack(fill="x", padx=14, pady=(10, 4))
        tk.Label(ch, text="LIVE FEED", font=(FONT, 10, "bold"),
                 bg=COLORS["bg_card"], fg=COLORS["text_secondary"]).pack(side="left")
        self.cam_live_dot = tk.Label(ch, text="● REC", font=(FONT, 9, "bold"),
                                     bg=COLORS["bg_card"], fg=COLORS["accent_red"])
        self.cam_live_dot.pack(side="right")

        self.video_label = tk.Label(cf, bg="#000000")
        self.video_label.pack(fill="both", expand=True, padx=10, pady=(2, 6))

        # Pipeline challenge instruction overlay
        self.challenge_label = tk.Label(cf, text="", font=(FONT, 12, "bold"),
                                        bg=COLORS["bg_card"], fg=COLORS["accent_amber"])
        self.challenge_label.pack(pady=(0, 6))

        # Pipeline status panel
        ip = tk.Frame(mid, bg=COLORS["bg_card"], width=310,
                      highlightbackground=COLORS["border"], highlightthickness=1)
        ip.pack(side="right", fill="y", padx=(6, 0)); ip.pack_propagate(False)

        # Panel header
        ph = tk.Frame(ip, bg=COLORS["bg_card"])
        ph.pack(fill="x", padx=16, pady=(12, 6))
        tk.Label(ph, text="PIPELINE", font=(FONT, 10, "bold"),
                 bg=COLORS["bg_card"], fg=COLORS["accent_cyan"]).pack(side="left")
        tk.Label(ph, text="STATUS", font=(FONT, 10),
                 bg=COLORS["bg_card"], fg=COLORS["text_muted"]).pack(side="left", padx=(4,0))

        tk.Frame(ip, bg=COLORS["border"], height=1).pack(fill="x", padx=14)

        self.pipeline_log = tk.Text(ip, bg=COLORS["bg_card"], fg=COLORS["text_secondary"],
                                    font=(FONT, 9), wrap="word", relief="flat",
                                    highlightthickness=0, padx=16, pady=10,
                                    insertbackground=COLORS["bg_card"])
        self.pipeline_log.pack(fill="both", expand=True)
        self.pipeline_log.config(state="disabled")
        for tag, clr in [("info", COLORS["text_secondary"]), ("success", COLORS["success"]),
                          ("error", COLORS["error"]), ("warning", COLORS["warning"]),
                          ("step", COLORS["accent_cyan"])]:
            self.pipeline_log.tag_configure(tag, foreground=clr,
                                             font=(FONT, 10 if tag == "step" else 9,
                                                   "bold" if tag == "step" else "normal"))

        # Recording status label (at bottom of pipeline panel)
        tk.Frame(ip, bg=COLORS["border"], height=1).pack(fill="x", padx=14)
        self.recording_status_label = tk.Label(
            ip, text="", font=(FONT, 9, "bold"),
            bg=COLORS["bg_card"], fg=COLORS["accent_amber"],
            wraplength=270, justify="center", anchor="center")
        self.recording_status_label.pack(fill="x", padx=10, pady=(6, 8))

        # Bottom buttons
        bot = tk.Frame(self.content, bg=COLORS["bg_dark"])
        bot.pack(fill="x", padx=20, pady=(4, 12))
        self.start_btn = GlowButton(bot, "▶  Start Authentication",
                                     self._run_pipeline_thread,
                                     width=230, height=46, bg=COLORS["accent_blue"], font_size=11)
        self.start_btn.pack(side="left")
        GlowButton(bot, "⟲  Change Camera",
                   lambda: (self._stop_camera(), self._show_camera_select()),
                   width=180, height=46, bg=COLORS["bg_input"], font_size=11).pack(side="left", padx=(10,0))
        GlowButton(bot, "Home", self._show_home,
                   width=100, height=46, bg=COLORS["bg_input"], font_size=11).pack(side="right")

        self._update_camera_feed()

    def _log(self, text, tag="info"):
        try:
            self.pipeline_log.config(state="normal")
            self.pipeline_log.insert("end", text + "\n", tag)
            self.pipeline_log.see("end")
            self.pipeline_log.config(state="disabled")
        except tk.TclError:
            pass

    def _set_challenge_text(self, text):
        try:
            self.challenge_label.config(text=text)
        except tk.TclError:
            pass

    def _set_recording_status(self, text):
        """Update the recording status label in the pipeline panel."""
        try:
            self.recording_status_label.config(text=text)
        except (tk.TclError, AttributeError):
            pass

    def _render_frame(self, frame):
        try:
            # Draw cached bounding box if we have one
            bbox = getattr(self, '_last_face_bbox', None)
            display_frame = frame.copy()
            if bbox:
                x, y, w, h = bbox
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), (59, 130, 246), 2)

            rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            from PIL import Image, ImageTk
            img = Image.fromarray(rgb)
            lw = self.video_label.winfo_width()
            lh = self.video_label.winfo_height()
            if lw > 10 and lh > 10:
                img = img.resize((lw, lh), Image.BILINEAR)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.config(image=imgtk)
        except Exception:
            pass

    def _update_camera_feed(self):
        if not self.camera_running or self.camera is None:
            return

        frame = self.camera.read_frame()
        if frame is not None:
            self.current_frame = frame.copy()

            # Only run face detection every 3rd frame to reduce CPU load
            self._feed_frame_count = getattr(self, '_feed_frame_count', 0) + 1
            if self._feed_frame_count % 3 == 0:
                result = self.camera.detect_face(frame)
                if result:
                    _, bbox = result
                    self._last_face_bbox = bbox
                else:
                    self._last_face_bbox = None

            self._render_frame(frame)

        self._after_id = self.root.after(50, self._update_camera_feed)

    def _stop_camera(self):
        self.camera_running = False
        if self._after_id:
            self.root.after_cancel(self._after_id); self._after_id = None
        if self.camera:
            self.camera.release(); self.camera = None

    def _pause_camera_feed(self):
        """Pause the GUI feed loop so the pipeline thread can read frames exclusively."""
        self.camera_running = False
        if self._after_id:
            self.root.after_cancel(self._after_id); self._after_id = None

    def _resume_camera_feed(self):
        """Resume the GUI feed loop after the pipeline finishes."""
        if self.camera and self.camera.cap and self.camera.cap.isOpened():
            self.camera_running = True
            self._update_camera_feed()

    # ─── Pipeline Execution ──────────────────────────────────────────────

    def _run_pipeline_thread(self):
        if self.pipeline_running:
            return
        self.pipeline_running = True
        self._hide_result_banner()
        self.step_indicator.reset()
        self.pipeline_log.config(state="normal")
        self.pipeline_log.delete("1.0", "end")
        self.pipeline_log.config(state="disabled")
        threading.Thread(target=self._execute_pipeline, daemon=True).start()

    def _execute_pipeline(self):
        audit = AuditLogger()
        session_mgr = SessionManager()
        anti_spoof = AntiSpoofing()
        head_challenge = HeadMovementChallenge()
        blink_det = BlinkDetection()
        voice_chal = VoiceChallenge()

        uid = "unknown"
        dname = ""

        try:
            # ── Opening greeting ──
            self._gui(self._set_challenge_text, "🎤  hey, lets see who are you")
            self._gui(self._log, "▸ AI: hey, lets see who are you", "step")
            voice_bot.speak_sync("hey, lets see who are you")

            # ── Step 1: Face Detection ──
            self._gui(self.step_indicator.set_state, 0, "active")
            self._gui(self._log, "\n▸ Step 1: Detecting face...", "step")
            self._gui(self._status, "Step 1: Face detection", COLORS["accent_blue"])
            self._gui(self._set_challenge_text, "👁️  Looking for your face...")

            frame = None; face_bbox = None
            for _ in range(200):
                if self.current_frame is not None:
                    result = self.camera.detect_face(self.current_frame)
                    if result:
                        frame, face_bbox = result; break
                time.sleep(0.05)
            if frame is None or face_bbox is None:
                self._fail(0, "No face detected on camera.", audit, uid); return

            self._gui(self.step_indicator.set_state, 0, "done")
            self._gui(self._log, "  ✓ Face detected.", "success")

            # ── Step 2: Face Recognition ──
            self._gui(self.step_indicator.set_state, 1, "active")
            self._gui(self._log, "\n▸ Step 2: Matching face...", "step")
            self._gui(self._status, "Step 2: Face recognition", COLORS["accent_blue"])
            self._gui(self._set_challenge_text, "🔍  Checking if we know you...")

            allowed, reason = rate_limiter.check_allowed()
            if not allowed:
                self._fail(1, f"Rate limited: {reason}", audit, uid); return

            print("[AUTH] Extracting live embedding...")
            embedding = get_live_embedding(frame)
            if embedding is None:
                rate_limiter.record_failure()
                self._fail(1, "Could not extract face embedding.", audit, uid); return

            print("[AUTH] Matching face against database...")
            match = match_face(embedding)
            if match is None:
                status = rate_limiter.record_failure()
                self._fail(1, f"No matching user. {status}", audit, uid); return

            uid, dname, dist = match
            self._gui(self.step_indicator.set_state, 1, "done")
            self._gui(self._log, f"  ✓ Match: {dname} (dist: {dist:.4f})", "success")

            # Voice: we recognize you
            self._gui(self._set_challenge_text,
                      "🎤  hmmm i guess we know you\n     but we need to confirm if it's really you")
            voice_bot.speak_sync(
                "hmmm i guess we know you but we need to confirm if it's really you")
            time.sleep(0.5)

            # ── Step 3: Anti-Spoofing ──
            self._gui(self.step_indicator.set_state, 2, "active")
            self._gui(self._log, "\n▸ Step 3: Anti-spoofing check...", "step")
            self._gui(self._status, "Step 3: Anti-spoofing", COLORS["accent_blue"])
            self._gui(self._set_challenge_text, "🔍  let me check if you are a real person")
            voice_bot.speak_sync("let me check if you are a real person")

            is_real, pad_conf, pad_reason = anti_spoof.check_liveness(frame, face_bbox)
            if not is_real:
                self._fail(2, f"Spoofing: {pad_reason}", audit, uid); return

            self._gui(self.step_indicator.set_state, 2, "done")
            self._gui(self._log, f"  ✓ Liveness confirmed ({pad_conf:.2f})", "success")
            self._gui(self._set_challenge_text, "✓  you look real to me")
            voice_bot.speak_sync("you look real to me")
            time.sleep(0.3)

            # ── Step 4: Greeting (confirmed identity) ──
            self._gui(self.step_indicator.set_state, 3, "active")
            self._gui(self._log, "\n▸ Step 4: Identity confirmation...", "step")
            self._gui(self._status, "Step 4: Identity prompt", COLORS["accent_blue"])
            self._gui(self._set_challenge_text,
                      "🎤  now i need to run a few more tests to be sure")
            voice_bot.speak_sync("now i need to run a few more tests to be sure")
            self._gui(self.step_indicator.set_state, 3, "done")
            self._gui(self._log, "  ✓ Identity prompt delivered.", "success")
            time.sleep(0.3)

            # ── PAUSE GUI feed — pipeline needs exclusive camera access ──
            self._gui(self._pause_camera_feed)
            time.sleep(0.3)

            # ── Step 5: Head Movement ──
            self._gui(self.step_indicator.set_state, 4, "active")
            self._gui(self._log, "\n▸ Step 5: Head movement challenge...", "step")
            self._gui(self._status, "Step 5: Head movement", COLORS["accent_blue"])

            head_ok, head_reason = False, "Camera not available."
            if self.camera and self.camera.cap:
                for attempt in range(3):
                    # Speak the direction before starting the challenge
                    direction = head_challenge._get_random_direction()
                    head_challenge.last_direction = direction
                    
                    if attempt == 0:
                        self._gui(self._set_challenge_text, f"🔄  please {direction}")
                        voice_bot.speak_sync(f"please {direction}")
                    else:
                        self._gui(self._set_challenge_text, f"🔄  let's try that again. please {direction}")
                        voice_bot.speak_sync(f"let's try that again. please {direction}")
                        
                    time.sleep(0.3)

                    head_ok, head_reason = head_challenge.run_challenge(
                        self.camera, use_gui=True,
                        preset_direction=direction,
                        update_frame_callback=lambda f: self._gui(self._render_frame, f))
                        
                    if head_ok:
                        break
                        
                    if attempt < 2:
                        self._gui(self._log, f"  ! Retry {attempt+1}/3 failed: {head_reason}", "warning")
            else:
                head_ok, head_reason = False, "Camera not available."

            self._gui(self._set_challenge_text, "")
            if not head_ok:
                self._gui(self._resume_camera_feed)
                self._fail(4, f"Failed after 3 attempts: {head_reason}", audit, uid); return

            self._gui(self.step_indicator.set_state, 4, "done")
            self._gui(self._log, f"  ✓ {head_reason}", "success")
            self._gui(self._set_challenge_text, "✓  head movement detected")
            voice_bot.speak_sync("good")
            time.sleep(0.3)

            # ── Step 6: Blink Detection ──
            self._gui(self.step_indicator.set_state, 5, "active")
            self._gui(self._log, "\n▸ Step 6: Blink detection...", "step")
            self._gui(self._status, "Step 6: Blink challenge", COLORS["accent_blue"])
            blink_ok, blink_reason = False, "Camera not available."
            if self.camera and self.camera.cap:
                for attempt in range(3):
                    if attempt == 0:
                        self._gui(self._set_challenge_text, "👁️  now blink a few times for me")
                        voice_bot.speak_sync("now blink a few times for me")
                    else:
                        self._gui(self._set_challenge_text, "👁️  let's try that again. now blink a few times")
                        voice_bot.speak_sync("let's try that again. now blink a few times")

                    blink_ok, blink_reason = blink_det.run_challenge(
                        self.camera, pad_passed=is_real, use_gui=True,
                        update_frame_callback=lambda f: self._gui(self._render_frame, f))
                        
                    if blink_ok:
                        break
                    
                    if attempt < 2:
                        self._gui(self._log, f"  ! Retry {attempt+1}/3 failed: {blink_reason}", "warning")
            else:
                blink_ok, blink_reason = False, "Camera not available."

            self._gui(self._set_challenge_text, "")
            if not blink_ok:
                self._gui(self._resume_camera_feed)
                self._fail(5, f"Failed after 3 attempts: {blink_reason}", audit, uid); return

            self._gui(self.step_indicator.set_state, 5, "done")
            self._gui(self._log, f"  ✓ {blink_reason}", "success")

            # ── RESUME GUI feed ──
            self._gui(self._resume_camera_feed)

            # ── Step 7: Voice Challenge ──
            self._gui(self.step_indicator.set_state, 6, "active")
            self._gui(self._log, "\n▸ Step 7: Voice challenge...", "step")
            self._gui(self._status, "Step 7: Voice challenge", COLORS["accent_blue"])
            voice_ok, voice_reason = False, "Voice timeout."
            for attempt in range(3):
                if attempt == 0:
                    self._gui(self._set_challenge_text, "🎤  one last thing, i need to hear your voice")
                    voice_bot.speak_sync("one last thing, i need to hear your voice")
                else:
                    self._gui(self._set_challenge_text, "🎤  let's try that again. i need to hear your voice")
                    voice_bot.speak_sync("let's try that again. i need to hear your voice")
                    
                time.sleep(0.3)

                voice_ok, voice_reason = voice_chal.run_challenge(
                    user_id=uid,
                    update_ui_callback=lambda t: self._gui(self._set_recording_status, t))
                if voice_ok:
                    break
                    
                if attempt < 2:
                    self._gui(self._log, f"  ! Retry {attempt+1}/3 failed: {voice_reason}", "warning")

            self._gui(self._set_challenge_text, "")
            self._gui(self._set_recording_status, "")
            if not voice_ok:
                self._fail(6, f"Failed after 3 attempts: {voice_reason}", audit, uid); return

            self._gui(self.step_indicator.set_state, 6, "done")
            self._gui(self._log, f"  ✓ {voice_reason}", "success")

            # ── Step 8: Session Token ──
            self._gui(self.step_indicator.set_state, 7, "active")
            self._gui(self._log, "\n▸ Step 8: Generating session token...", "step")
            self._gui(self._status, "Generating session", COLORS["accent_blue"])

            token = session_mgr.create_token(uid)
            rate_limiter.record_success()
            audit.log_event(uid, "AUTH_SUCCESS", "all_steps", "success", "All passed.")

            self._gui(self.step_indicator.set_state, 7, "done")
            self._gui(self._log,
                      f"  ✓ Token: {token[:16]}...\n"
                      f"     Expires: {config.SESSION_EXPIRY_MINUTES} min", "success")

            # ── SUCCESS ──
            self._gui(self._status, "ACCESS GRANTED", COLORS["accent_green"])
            self._gui(self._log,
                      f"\n{'━'*30}\n✅ VERIFICATION COMPLETE\n"
                      f"   Welcome, {dname}!\n{'━'*30}", "success")
            self._gui(self._set_challenge_text,
                      f"🎤  oh its really you, welcome {dname}!")
            voice_bot.speak_sync(f"oh its really you, welcome {dname}")
            self._gui(self._show_result_banner, True, f"✅  ACCESS GRANTED — Welcome, {dname}   |   Token: {token[:20]}...")

        except Exception as e:
            print(f"[AUTH ERROR] {e}")
            self._gui(self._log, f"\n❌ Error: {e}", "error")
            self._gui(self._status, f"Error: {e}", COLORS["error"])
            audit.log_event(uid, "SYSTEM_ERROR", "pipeline", "failure", str(e))
            self._gui(self._resume_camera_feed)
            voice_bot.speak_sync("sorry, something went wrong. please try again")

        finally:
            self.pipeline_running = False
            try: head_challenge.close()
            except: pass
            try: blink_det.close()
            except: pass

    def _fail(self, step, reason, audit, uid):
        self._gui(self.step_indicator.set_state, step, "fail")
        self._gui(self._log, f"  ✗ {reason}", "error")
        self._gui(self._status, "Authentication failed", COLORS["error"])
        audit.log_event(uid, "AUTH_FAILURE", f"step_{step}", "failure", reason)

        # Speak apology synchronously (wait for it to finish before showing banner)
        self._gui(self._set_challenge_text,
                  "🎤  sorry, i guess we can't recognize you")
        voice_bot.speak_sync("sorry, i guess we can't recognize you")

        self._gui(self._set_challenge_text, "")
        self._gui(self._show_result_banner, False, "❌  ACCESS DENIED — Authentication failed. Try again.")

    def _hide_result_banner(self):
        if hasattr(self, '_result_banner') and getattr(self, '_result_banner', None):
            self._result_banner.destroy()
            self._result_banner = None

    def _show_result_banner(self, success, text):
        self._hide_result_banner()
        clr = COLORS["success"] if success else COLORS["error"]
        self._result_banner = tk.Frame(self.content, bg=clr, height=60)
        self._result_banner.pack(fill="x", side="bottom")
        self._result_banner.pack_propagate(False)
        tk.Label(self._result_banner, text=text, font=(FONT, 12, "bold"), bg=clr, fg="#ffffff").pack(expand=True)

    # ─── Enrollment ──────────────────────────────────────────────────────

    def _show_enroll(self):
        self._clear(); self._stop_camera(); self._stop_enroll_camera()
        self._status("User Enrollment", COLORS["accent_purple"])

        self._enroll_images = []
        self._enroll_current_pose = 0

        c = tk.Frame(self.content, bg=COLORS["bg_dark"])
        c.place(relx=0.5, rely=0.42, anchor="center")

        tk.Label(c, text="👤  Enroll New User", font=(FONT, 22, "bold"),
                 bg=COLORS["bg_dark"], fg=COLORS["text_primary"]).pack(pady=(0, 6))
        tk.Label(c, text="Register face biometrics for authentication", font=(FONT, 11),
                 bg=COLORS["bg_dark"], fg=COLORS["text_secondary"]).pack(pady=(0, 24))

        card = tk.Frame(c, bg=COLORS["bg_card"], padx=36, pady=28,
                        highlightbackground=COLORS["border"], highlightthickness=1)
        card.pack()

        # Check encryption key
        key_ok = True
        try:
            _get_fernet()
        except EnvironmentError:
            key_ok = False

        if not key_ok:
            tk.Label(card, text="⚠️  Encryption key not set!", font=(FONT, 13, "bold"),
                     bg=COLORS["bg_card"], fg=COLORS["error"]).pack(pady=(0, 10))
            tk.Label(card,
                     text="The FACE_DB_KEY environment variable is required.\n"
                          "Click the button below to auto-generate and set it.",
                     font=(FONT, 10), bg=COLORS["bg_card"], fg=COLORS["text_secondary"],
                     justify="center").pack(pady=(0, 16))

            GlowButton(card, "🔑  Generate & Set Key", self._auto_generate_key,
                       width=240, height=46, bg=COLORS["accent_amber"], font_size=12).pack()

            tk.Label(card,
                     text="\nOr set manually in your terminal:\n"
                          '  python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"\n'
                          f"  set {config.FACE_DB_KEY_ENV}=<your_key>",
                     font=("Consolas", 9), bg=COLORS["bg_card"], fg=COLORS["text_muted"],
                     justify="left").pack(padx=8, pady=(8, 0))
        else:
            # User ID (Auto-generated)
            import uuid
            auto_uid = f"USR-{uuid.uuid4().hex[:8].upper()}"
            
            tk.Label(card, text="User ID (Auto-generated)", font=(FONT, 10, "bold"),
                     bg=COLORS["bg_card"], fg=COLORS["text_secondary"], anchor="w").pack(fill="x", pady=(0, 2))
            self.enroll_uid_entry = tk.Entry(card, font=(FONT, 12), width=30,
                                              bg=COLORS["bg_dark"], fg=COLORS["text_muted"],
                                              relief="flat", highlightthickness=1, highlightbackground=COLORS["border"])
            self.enroll_uid_entry.insert(0, auto_uid)
            self.enroll_uid_entry.config(state="readonly")
            self.enroll_uid_entry.pack(fill="x", pady=(0, 12), ipady=6)

            # Display Name
            tk.Label(card, text="Display Name", font=(FONT, 10, "bold"),
                     bg=COLORS["bg_card"], fg=COLORS["text_secondary"], anchor="w").pack(fill="x", pady=(0, 2))
            self.enroll_name_entry = tk.Entry(card, font=(FONT, 12), width=30,
                                              bg=COLORS["bg_input"], fg=COLORS["text_primary"],
                                              insertbackground=COLORS["text_primary"], relief="flat",
                                              highlightthickness=1, highlightbackground=COLORS["border"])
            self.enroll_name_entry.pack(fill="x", pady=(0, 24), ipady=6)

            self.enroll_error_label = tk.Label(card, text="", font=(FONT, 10, "bold"),
                                               bg=COLORS["bg_card"], fg=COLORS["error"])
            self.enroll_error_label.pack(fill="x", pady=(0, 10))

            GlowButton(card, "Proceed to Camera Selection", self._proceed_enroll_camera_select,
                       width=260, height=46, bg=COLORS["accent_purple"], font_size=12).pack()

        GlowButton(c, "←  Back to Home", lambda: (self._stop_enroll_camera(), self._show_home()),
                   width=160, height=38, bg=COLORS["step_pending"], font_size=10).pack(pady=(20, 0))

    def _auto_generate_key(self):
        """Auto-generate a Fernet key and save it with restrictive permissions."""
        from modules.secure_storage import generate_and_save_key
        key = generate_and_save_key()

        messagebox.showinfo("Key Generated",
                            f"Encryption key has been generated and saved!\n\n"
                            f"Key: {key[:20]}...\n\n"
                            f"The key is stored securely in {config.FACE_DB_KEY_FILE}\n"
                            f"with restricted file permissions.")
        self._show_enroll()  # Refresh to show the form

    def _stop_enroll_camera(self):
        self._enroll_feed_running = False
        if self._enroll_after_id:
            self.root.after_cancel(self._enroll_after_id); self._enroll_after_id = None
        if self._enroll_cam:
            self._enroll_cam.release(); self._enroll_cam = None

    def _proceed_enroll_camera_select(self):
        uid = self.enroll_uid_entry.get().strip()
        name = self.enroll_name_entry.get().strip()
        
        self.enroll_error_label.config(text="")
        
        if not uid:
            self.enroll_error_label.config(text="⚠️  User ID is required.")
            return

        try:
            from modules.face_recognition_module import check_user_exists
            if check_user_exists(uid):
                self.enroll_error_label.config(text=f"⚠️  User ID '{uid}' is already registered.\nPlease choose a different ID.")
                return
        except Exception:
            pass # ignore missing DB locally for now, since it handles if DB is empty

        self._enroll_uid = uid
        self._enroll_name = name if name else uid
        self._show_camera_select(is_enroll=True)

    def _show_enroll_capture(self):
        self._clear()
        self._enroll_feed_running = True
        self._enroll_images = []
        self._enroll_current_pose = 0

        self._enroll_poses = [
            "stay in center",
            "turn right",
            "turn left",
            "up",
            "center again"
        ]

        # ── Top header ──
        top = tk.Frame(self.content, bg=COLORS["bg_dark"])
        top.pack(side="top", fill="x", padx=20, pady=(16, 8))
        tk.Label(top, text="📸  Enrollment Capture", font=(FONT, 18, "bold"),
                 bg=COLORS["bg_dark"], fg=COLORS["text_primary"]).pack(side="left")
        self.enroll_progress_label = tk.Label(
            top, text="Photo 1 of 5",
            font=(FONT, 12), bg=COLORS["bg_dark"], fg=COLORS["accent_purple"])
        self.enroll_progress_label.pack(side="right")

        # ── Voice instruction label (large, visible) ──
        self.enroll_pose_label = tk.Label(self.content,
            text="🎤  Starting...",
            font=(FONT, 16, "bold"),
            bg=COLORS["bg_dark"], fg=COLORS["accent_amber"])
        self.enroll_pose_label.pack(side="top", pady=(0, 8))

        # ── Bottom bar with Cancel only (no manual capture button) ──
        bot_bar = tk.Frame(self.content, bg=COLORS["bg_dark"])
        bot_bar.pack(side="bottom", fill="x", padx=20, pady=(4, 12))
        GlowButton(bot_bar, "✗  Cancel", lambda: self._cancel_enrollment(),
                   width=140, height=46, bg=COLORS["step_pending"], font_size=11).pack(side="right")

        self.enroll_captured_label = tk.Label(self.content,
            text="Captured: 0 / 5",
            font=(FONT, 11), bg=COLORS["bg_dark"], fg=COLORS["text_secondary"])
        self.enroll_captured_label.pack(side="bottom", pady=(4, 4))

        # ── Camera view (middle, expanding) ──
        mid = tk.Frame(self.content, bg=COLORS["bg_dark"])
        mid.pack(side="top", fill="both", expand=True, padx=20, pady=4)
        cam_card = tk.Frame(mid, bg=COLORS["bg_card"],
                            highlightbackground=COLORS["border"], highlightthickness=1)
        cam_card.pack(fill="both", expand=True)
        cam_card.pack_propagate(False)
        self.enroll_video_label = tk.Label(cam_card, bg=COLORS["bg_dark"])
        self.enroll_video_label.pack(fill="both", expand=True, padx=8, pady=8)
        self.enroll_face_status = tk.Label(cam_card, text="No face detected",
                                           font=(FONT, 10, "bold"),
                                           bg=COLORS["bg_card"], fg=COLORS["error"])
        self.enroll_face_status.pack(pady=(0, 8))

        self._enroll_face_detected = False
        self._enroll_current_frame = None
        self._enroll_cancelled = False

        # Start live camera feed
        self._update_enroll_feed()

        # Start the voice-driven enrollment in a background thread
        threading.Thread(target=self._voice_driven_enroll, daemon=True).start()

    def _cancel_enrollment(self):
        self._enroll_cancelled = True
        self._stop_enroll_camera()
        self._show_home()

    def _voice_driven_enroll(self):
        """Background thread: voice-driven enrollment sequence."""
        total = config.ENROLLMENT_CAPTURE_COUNT

        # ── Opening announcement ──
        self._gui(self._set_enroll_instruction,
                  "🎤  please follow me and take pics so we can\n"
                  "     get to know you next time you try to enter the system")
        voice_bot.speak_sync(
            "please follow me and take pics so we can get to know you "
            "next time you try to enter the system")

        if self._enroll_cancelled:
            return

        # ── Photo capture loop ──
        for i, pose in enumerate(self._enroll_poses):
            if self._enroll_cancelled:
                return

            # Display + speak the pose instruction
            self._gui(self._set_enroll_instruction, f"👉  {pose}")
            self._gui(self._set_enroll_progress, i + 1, total)
            voice_bot.speak_sync(pose)

            if self._enroll_cancelled:
                return

            # Wait for a face to appear, then auto-capture
            captured = False
            for attempt in range(60):     # ~3 seconds max
                if self._enroll_cancelled:
                    return
                if self._enroll_face_detected and self._enroll_current_frame is not None:
                    self._enroll_images.append(self._enroll_current_frame.copy())
                    self._gui(self._set_enroll_captured, len(self._enroll_images), total)
                    self._gui(self._flash_capture)
                    captured = True
                    break
                time.sleep(0.05)

            if not captured:
                # If no face after 3s, ask to retry
                self._gui(self._set_enroll_instruction, "⚠️  No face detected — adjusting...")
                voice_bot.speak_sync("I can't see your face, please adjust")
                # try again for this pose
                for attempt in range(100):
                    if self._enroll_cancelled:
                        return
                    if self._enroll_face_detected and self._enroll_current_frame is not None:
                        self._enroll_images.append(self._enroll_current_frame.copy())
                        self._gui(self._set_enroll_captured, len(self._enroll_images), total)
                        self._gui(self._flash_capture)
                        captured = True
                        break
                    time.sleep(0.05)

            if not captured:
                voice_bot.speak_sync("sorry, enrollment failed. please try again")
                self._gui(self._show_enroll)
                return

            # Prevent duplicate face enrollments
            if i == 0 and len(self._enroll_images) == 1:
                from modules.face_recognition_module import get_live_embedding, match_face
                emb = get_live_embedding(self._enroll_images[0])
                if emb is not None:
                    match = match_face(emb)
                    if match:
                        _, dname, _ = match
                        self._gui(self._set_enroll_instruction, f"⚠️  You are already enrolled as {dname}!")
                        voice_bot.speak_sync(f"ooh you're already there {dname}, please authenticate directly")
                        self._enroll_cancelled = True
                        self._gui(self._show_home)
                        return

            # Say "very good" between captures (not after last one)
            if i < len(self._enroll_poses) - 1:
                voice_bot.speak_sync("very good")

            time.sleep(0.3)   # brief natural pause

        if self._enroll_cancelled:
            return

        # ── Voice capture phase ──
        self._gui(self._set_enroll_instruction,
                  "🎤  now i will ask you a general question\n"
                  "     answer it with your voice so we can store your voice")
        voice_bot.speak_sync(
            "very good. now i will ask you a general question "
            "and you answer it with your voice so we can store your voice")

        if self._enroll_cancelled:
            return

        # Pick a random engaging question so they talk enough for a good voiceprint
        import random
        questions = [
            "What would your perfect relaxing weekend look like?",
            "If you could travel anywhere right now, where would it be and why?",
            "Could you describe your favorite movie or book without naming it?",
            "What is your favorite hobby and how did you get into it?",
            "Describe the best meal you've ever had in a few sentences."
        ]
        question = random.choice(questions)
        self._gui(self._set_enroll_instruction, f"🎤  \"{question}\"")
        voice_bot.speak_sync(question)

        if self._enroll_cancelled:
            return

        # Record the user's voice with a timer (to secure temp directory)
        duration = 10
        record_result = [None]
        def do_record():
            record_result[0] = voice_bot.record_audio(duration, "voice_enroll_capture.wav")

        rec_thread = threading.Thread(target=do_record, daemon=True)
        rec_thread.start()

        start = time.time()
        while rec_thread.is_alive():
            elapsed = time.time() - start
            remaining = max(0, duration - elapsed)
            bar_total = 20
            bar_filled = int((elapsed / duration) * bar_total)
            bar_empty = bar_total - bar_filled
            bar = "█" * bar_filled + "░" * bar_empty
            self._gui(self._set_enroll_instruction, 
                      f"🔴  Recording your voice... please answer!\n"
                      f"     {bar}  {remaining:.0f}s")
            self._gui(self._status, "Recording voice...", COLORS["error"])
            time.sleep(0.25)
            
        wav_file = record_result[0] or "voice_enroll_capture.wav"

        if self._enroll_cancelled:
            return

        # ── Process enrollment ──
        print("[ENROLL] Recording done. Processing enrollment...")
        self._gui(self._set_enroll_instruction, "⏳  Processing enrollment...")
        self._gui(self._status, "Processing enrollment...", COLORS["accent_amber"])

        # Stop camera on the GUI thread (touching tkinter from background = crash)
        self._gui(self._stop_enroll_camera)
        time.sleep(0.5)   # let the GUI thread process the stop

        uid = self._enroll_uid
        name = self._enroll_name
        images = self._enroll_images

        try:
            print(f"[ENROLL] Computing voiceprint from: {wav_file}")
            voiceprint_embedding = None
            try:
                from resemblyzer import VoiceEncoder, preprocess_wav
                encoder = VoiceEncoder()
                wav_data = preprocess_wav(wav_file)
                voiceprint_embedding = encoder.embed_utterance(wav_data).astype(np.float32)
                print(f"[ENROLL] Voiceprint computed: shape={voiceprint_embedding.shape}")
            except Exception as e:
                print(f"[ENROLL] Voiceprint extraction failed: {e}")

            # Securely wipe the voice recording after extracting voiceprint
            if wav_file and os.path.exists(wav_file):
                from modules.secure_storage import secure_delete as sd
                sd(wav_file)
                print(f"[SECURITY] Enrollment voice recording securely wiped.")

            print(f"[ENROLL] Calling enroll_user with {len(images)} images...")
            success = enroll_user(uid, name, images, voiceprint=voiceprint_embedding)
            print(f"[ENROLL] enroll_user returned: {success}")
            if success:
                voice_bot.speak_sync("excellent, we have successfully enrolled you")
                self._gui(self._show_home)
                self._gui(lambda: self._show_result_banner(True, f"✅  ENROLLMENT COMPLETE — User '{name}' enrolled securely!"))
            else:
                voice_bot.speak_sync("sorry, enrollment failed. please try again")
                self._gui(self._show_enroll)
                self._gui(lambda: self._show_result_banner(False, "❌  ENROLLMENT FAILED — Could not extract enough face encodings. Ensure good lighting."))
        except Exception as e:
            import traceback
            print(f"[ENROLL ERROR] {e}")
            traceback.print_exc()
            try:
                voice_bot.speak_sync("sorry, there was an error during enrollment")
            except Exception:
                pass
            self._gui(self._show_enroll)
            self._gui(lambda: self._show_result_banner(False, f"❌  ENROLLMENT ERROR — {e}"))

    # ── Enrollment GUI helpers (called from background thread via _gui) ──

    def _set_enroll_instruction(self, text):
        try:
            self.enroll_pose_label.config(text=text)
        except tk.TclError:
            pass

    def _set_enroll_progress(self, current, total):
        try:
            self.enroll_progress_label.config(text=f"Photo {current} of {total}")
        except tk.TclError:
            pass

    def _set_enroll_captured(self, captured, total):
        try:
            self.enroll_captured_label.config(text=f"Captured: {captured} / {total}")
        except tk.TclError:
            pass

    def _flash_capture(self):
        """Brief green flash on the face status label to confirm capture."""
        try:
            self.enroll_face_status.config(text="📸 Captured!", fg=COLORS["accent_green"])
        except tk.TclError:
            pass

    def _update_enroll_feed(self):
        if not self._enroll_feed_running or not self._enroll_cam:
            return
        frame = self._enroll_cam.read_frame()
        if frame is not None:
            self._enroll_current_frame = frame.copy()
            result = self._enroll_cam.detect_face(frame)
            if result:
                _, bbox = result
                x, y, w, h = bbox
                cv2.rectangle(frame, (x, y), (x+w, y+h), (139, 92, 246), 2)
                self._enroll_face_detected = True
                try: self.enroll_face_status.config(text="✓ Face detected", fg=COLORS["success"])
                except tk.TclError: pass
            else:
                self._enroll_face_detected = False
                try: self.enroll_face_status.config(text="✗ No face — adjust position", fg=COLORS["error"])
                except tk.TclError: pass

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            try:
                lw, lh = self.enroll_video_label.winfo_width(), self.enroll_video_label.winfo_height()
                if lw > 10 and lh > 10:
                    img = img.resize((lw, lh), Image.LANCZOS)
            except: pass
            imgtk = ImageTk.PhotoImage(image=img)
            try:
                self.enroll_video_label.imgtk = imgtk
                self.enroll_video_label.config(image=imgtk)
            except tk.TclError: return

        self._enroll_after_id = self.root.after(33, self._update_enroll_feed)

    # ─── Cleanup ─────────────────────────────────────────────────────────

    def _on_close(self):
        self._stop_camera()
        self._stop_enroll_camera()
        # Securely wipe all temporary files on exit
        try:
            from modules.secure_storage import cleanup_secure_temp, cleanup_residual_wav_files
            cleaned = cleanup_secure_temp()
            residual = cleanup_residual_wav_files()
            if cleaned or residual:
                print(f"[SECURITY] Cleaned up {cleaned + residual} temp file(s) on exit.")
            voice_bot.secure_cleanup()
        except Exception as e:
            print(f"[SECURITY] Cleanup error on exit: {e}")
        self.root.destroy()


# ─── Entry Point ─────────────────────────────────────────────────────────────

def main():
    root = tk.Tk()
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass
    SecureFaceAuthApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
