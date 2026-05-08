# 🔐 Secure Face Recognition System — Setup & Run Guide

This guide covers the prerequisites, installation, and configuration required to run the multi-layer biometric authentication system.

---

## 📋 Prerequisites

- **Python**: version 3.9 to 3.11 is recommended.
- **Hardware**:
  - **Webcam**: Standard USB or Integrated camera (640x480 minimum).
  - **Microphone**: For voice passphrase verification.
- **Operating System**: Windows (tested on Windows 10/11), Linux, or macOS.

---

## 🚀 Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/ramyguettal/secure_face_recognition.git
   cd secure_face_recognition
   ```

2. **Create a Virtual Environment** (Optional but Recommended):
   ```bash
   python -m venv venv
   # Windows
   .\venv\Scripts\activate
   # Linux/macOS
   source venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   > [!NOTE]
   > Installing `face_recognition` and `dlib` might require C++ Build Tools on Windows. If you encounter errors, ensure you have the [Visual Studio C++ Desktop Development](https://visualstudio.microsoft.com/visual-cpp-build-tools/) workload installed.

---

## 🔑 Configuration

### 1. Encryption Key
The system encrypts all biometric data at rest using **Fernet (AES-128)**.
- **Automatic**: If no key is found, the system will generate one during the first enrollment and save it to `data/.face_key`.
- **Manual**: You can set the `FACE_DB_KEY` environment variable with a valid Fernet key.

### 2. Central Config (`config.py`)
You can tweak security thresholds and timeouts in `config.py`:
- `FACE_MATCH_THRESHOLD`: Strictness of face recognition (default `0.5`).
- `VOICE_SPEAKER_THRESHOLD`: Strictness of voice verification (default `0.82`).
- `CAMERA_INDEX`: Change to `1` if using an external USB camera.

---

## 📝 First-Time Setup (Enrollment)

Before you can authenticate, you must enroll at least one user.

1. **Run the Enrollment Script**:
   ```bash
   python enroll.py
   ```
2. **Follow the Prompts**:
   - Enter a **User ID** (e.g., `admin_01`) and a **Display Name**.
   - Stay still for the camera to capture multiple angles.
   - Speak the enrollment phrase clearly when prompted.

---

## 🎮 Running the Application

The system supports two modes of operation:

### Mode A: Graphical Interface (GUI) — Recommended
This launches the premium dark-mode interface with a real-time video feed and step-by-step progress tracking.
```bash
python main.py
```

### Mode B: Command Line Interface (CLI)
Runs the entire pipeline in the terminal. Useful for headless servers or automated testing.
```bash
python main.py --cli
```

---

## 🛠️ Troubleshooting

- **Camera not opening**: Change `CAMERA_INDEX` in `config.py` to `1` or `2`.
- **Voice challenge fails**: Ensure your microphone is the default recording device in system settings and you are in a relatively quiet environment.
- **"Dlib" errors**: This is usually due to missing CMake or C++ compilers. Install CMake (`pip install cmake`) and try again.
- **Permissions**: On Windows, the system attempts to restrict access to the `data/.face_key` file. Run as Administrator if you see "Access Denied" errors during the first run.

---

## 📁 Project Structure

- `main.py`: Main entry point (GUI/CLI switcher).
- `enroll.py`: User registration and biometric capture.
- `gui.py`: The Tkinter-based premium dashboard.
- `modules/`: Core logic for face, voice, and liveness challenges.
- `data/`: Encrypted database and session logs (generated on first run).
