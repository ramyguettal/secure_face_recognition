# 🔐 Secure Face Recognition Authentication System

A multi-layer, secure face recognition authentication system in Python with 6 pipeline steps and 2 added security layers.

---

## Architecture

```
Step 1: Camera        → Webcam capture + MediaPipe face detection
Step 2: Recognition   → Fernet-encrypted 128-d embedding match (dlib)
Decision Gate         → Rate limiting with exponential backoff + lockout
Layer 1: Anti-Spoof   → Texture analysis + screen glow detection (PAD)
Step 3: Greeting      → Neutral TTS (no username spoken)
Step 4: Head Movement → Random direction challenge with MediaPipe Face Mesh
Step 5: Blink         → EAR detection with random timing + variance check
Step 6: Voice         → Random passphrase (Whisper STT) + Voiceprint Match (Resemblyzer)
Layer 2: Session      → Cryptographic token + JSON-lines audit log
Layer 3: Storage      → Encrypted DB, Encrypted Logs, 3-Pass Audio Wiper
```

---

## 🔒 Data Security & Privacy (Overhaul)

This system implements **Zero-Persistence** for sensitive biometric buffers and **Data-at-Rest Encryption** for all stored profiles:

1. **Fully Encrypted Database**: The SQLite database (`face_db.db`) stores **no plaintext PII**. `user_id` and `display_name` are encrypted via Fernet (AES-128). Lookups are performed using irreversible **HMAC-SHA256 hashes**.
2. **Ephemeral Voice Processing**: Voice recordings (`.wav`) are routed to a protected temporary directory (`data/.secure_tmp/`). Immediately after processing (STT and Voiceprint matching), the audio files are wiped using **DoD 5220.22-M standards** (3-pass random byte overwrite + zero fill) before OS deletion.
3. **Encrypted Audit Logs**: `audit.log.enc` stores access events. Each individual log line is an independently encrypted Fernet token, ensuring IP addresses and User IDs remain secure even if the log file is stolen.
4. **OS-Level Key Protection**: The master encryption key (`.face_key`) is dynamically assigned restrictive OS permissions (`icacls` on Windows, `chmod 600` on Unix), binding access strictly to the owner account.
5. **Auto-Generated User IDs**: Enrollment generates random UUIDs (`USR-XXXX`) instead of relying on manual inputs, preventing identity inference.

---

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note:** `face_recognition` requires `dlib`, which needs CMake and C++ build tools.
> On Windows: install Visual Studio Build Tools with "Desktop development with C++".
> Or install a pre-built wheel: `pip install dlib` from [dlib releases](https://github.com/sachadee/Dlib).

### 2. Encryption Key Management

The face embedding database is strictly encrypted with Fernet (AES-128). To protect against database breaches, the private key is heavily protected and never hardcoded into the source code.

This system provides a **hybrid key management** feature depending on your needs:

**Option A: Automated GUI Bridging (Recommended for testing / development)**
When you launch the GUI (`python main.py`), you can click **"🔑 Generate & Set Key"** on the Enrollment screen. 
This will automatically:
1. Generate an enterprise-grade AES-key.
2. Inject it into your active session environment variables.
3. Silently save it to a protected, hidden local file (`data/.face_key`) so it persists on your machine across restarts. 
> *Note: By design, our `.gitignore` permanently blocks `.face_key` from ever being uploaded to GitHub.*

**Option B: Strict Environment Variables (For maximum zero-touch security)**
For true zero-trust environments, you can manually generate the key in the terminal and assign it as an ephemeral variable so it never touches the hard drive text files:

```bash
# Generate a key:
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

Copy the output and set the environment variable:

**Windows (CMD/PowerShell):**
```cmd
set FACE_DB_KEY=<your_generated_key>
```
**Linux/macOS:**
```bash
export FACE_DB_KEY="<your_generated_key>"
```

> ⚠️ Store this key securely. If you lose it, enrolled face data cannot be decrypted.

### 3. Enroll a User

```bash
python main.py
```
*(Or launch `python enroll.py` for CLI)*

This will:
1. Auto-generate a secure User ID and ask for a Display Name.
2. Capture **5 face images** from your webcam (with guided poses).
3. Average the face embeddings for robustness.
4. Capture a **voiceprint** for speaker verification (securely wiped from disk immediately after).
5. Encrypt everything and store it in `data/face_db.db`.

### 4. Run Authentication

```bash
python main.py
```

The system will run the full pipeline. It short-circuits immediately if any step fails.

---

## Hardware Requirements

| Component     | Required | Notes                                      |
|---------------|----------|--------------------------------------------|
| Webcam        | ✅ Yes   | Any USB or built-in webcam                 |
| Microphone    | ✅ Yes   | For voice challenge (Step 6)               |
| Speakers      | Optional | For TTS greeting (falls back to text)      |
| IR/Depth Cam  | Optional | Intel RealSense for enhanced anti-spoofing |

---

## Security Vulnerabilities Addressed

| # | Vulnerability            | Severity | Fix Implemented                                 |
|---|--------------------------|----------|--------------------------------------------------|
| 1 | Face DB & PII breach     | HIGH     | Fernet AES encryption + HMAC-SHA256 Lookups      |
| 2 | Biometric file leftovers | HIGH     | Voice recordings wiped with DoD 3-pass overwrite |
| 3 | No rate limiting         | HIGH     | Exponential backoff (2^n) + 15-min lockout       |
| 3 | Photo/screen spoofing    | HIGH     | Texture + brightness + color PAD analysis        |
| 4 | Looped video (blink)     | HIGH     | Random timing + EAR variance check + PAD gate    |
| 5 | Voice replay / Deepfake  | HIGH     | Random phrase (Whisper STT) + Speaker Verification (Resemblyzer) |
| 6 | Head movement replay     | MEDIUM   | Random direction + continuous motion verification|
| 7 | Username in greeting     | MEDIUM   | Neutral greeting only; initial shown on screen   |
| 8 | No audit trail           | MEDIUM   | JSON-lines rotating log + security alerts        |

---

## Project Structure

```
secure_face_auth/
├── main.py                          # Entry point — full pipeline
├── config.py                        # All thresholds and constants
├── enroll.py                        # CLI enrollment tool
├── modules/
│   ├── __init__.py
│   ├── secure_storage.py            # Core Security: encryption, HMAC, DoD wiping
│   ├── camera.py                    # Step 1: webcam + face detection (DSHOW fixed)
│   ├── face_recognition_module.py   # Step 2: encrypted embedding match
│   ├── anti_spoofing.py             # Layer 1: PAD / liveness
│   ├── voice_greeting.py            # Step 3: neutral TTS greeting
│   ├── head_movement.py             # Step 4: head pose challenge
│   ├── blink_detection.py           # Step 5: EAR blink detection
│   ├── voice_challenge.py           # Step 6: Whisper STT & Resemblyzer voiceprint
│   ├── ai_voice_bot.py              # Text-to-Speech & Speech-to-Text logic
│   └── session.py                   # Layer 2: token + encrypted audit logger
├── data/
│   ├── face_db.db                   # Encrypted SQLite database (auto-created)
│   ├── audit.log.enc                # Encrypted rotating audit log
│   └── .secure_tmp/                 # Ephemeral folder for biometric buffers
├── models/
│   └── anti_spoof_model/            # (Optional) Pre-trained PAD weights
├── requirements.txt
└── README.md
```

---

## Configuration

All thresholds live in `config.py` — no magic numbers in the codebase:

| Parameter                      | Default | Description                        |
|--------------------------------|---------|------------------------------------|
| `FACE_MATCH_THRESHOLD`         | 0.5     | Euclidean distance match threshold |
| `PAD_REAL_THRESHOLD`           | 0.8     | Anti-spoof "real" probability min  |
| `EAR_BLINK_THRESHOLD`          | 0.2     | Eye Aspect Ratio blink cutoff     |
| `MAX_FAILED_BEFORE_BACKOFF`    | 5       | Failures before exponential delay  |
| `MAX_FAILED_BEFORE_LOCKOUT`    | 10      | Failures before 15-min lockout     |
| `VOICE_FUZZY_MATCH_RATIO`      | 0.85    | STT fuzzy match acceptance ratio   |
| `SESSION_EXPIRY_MINUTES`       | 30      | Session token validity             |

---

## Testing Individual Modules

Each module has a `if __name__ == "__main__"` block for standalone testing:

```bash
python -m modules.camera              # Test webcam + face detection
python -m modules.anti_spoofing       # Test PAD liveness
python -m modules.head_movement       # Test head pose challenge
python -m modules.blink_detection     # Test blink detection
python -m modules.voice_challenge     # Test voice passphrase
python -m modules.session             # Test session tokens + audit
```

---

## License

Educational / research use. Not for production deployment without additional hardening.
