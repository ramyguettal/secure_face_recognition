"""
ai_voice_bot.py — AI Voice Bot for TTS, audio recording, and transcription.

SECURITY:
  - Audio is recorded to a secure temp directory (data/.secure_tmp/)
  - All WAV files are securely wiped (multi-pass overwrite) after processing
  - No voice data persists on disk after the pipeline completes
"""

import threading
import queue
import time
import sys
import os
import subprocess
import sounddevice as sd
from scipy.io import wavfile
import speech_recognition as sr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from modules.secure_storage import (
    get_secure_temp_dir, secure_delete, cleanup_secure_temp
)


class AIVoiceBot:
    def __init__(self):
        self._q = queue.Queue()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

        self.fs = 44100
        self.r = sr.Recognizer()

        # Track temporary files for cleanup
        self._temp_files = []


    def _worker(self):
        """Background worker that processes TTS requests one at a time."""
        while True:
            item = self._q.get()
            if item is None:
                break

            text, done_event = item
            print(f"[VOICE BOT] Speaking: {text}")
            try:
                subprocess.run(
                    [
                        sys.executable, "-c",
                        "import sys, pyttsx3; "
                        "e=pyttsx3.init(); "
                        "voices = e.getProperty('voices'); "
                        "e.setProperty('voice', voices[0].id if len(voices) > 0 else voices[0].id); "
                        "e.setProperty('rate', e.getProperty('rate')-30); "
                        "e.say(sys.argv[1]); "
                        "e.runAndWait()",
                        text,
                    ],
                    creationflags=(
                        subprocess.CREATE_NO_WINDOW
                        if hasattr(subprocess, "CREATE_NO_WINDOW")
                        else 0
                    ),
                    timeout=30,
                )
            except Exception as e:
                print(f"[VOICE BOT ERROR] TTS subprocess error: {e}")
            finally:
                if done_event is not None:
                    done_event.set()

    # ── Public API ───────────────────────────────────────────────────────

    def speak(self, text: str):
        """Queue speech — returns immediately (fire-and-forget)."""
        self._q.put((text, None))

    def speak_sync(self, text: str):
        """Queue speech and BLOCK until the subprocess finishes speaking."""
        done = threading.Event()
        self._q.put((text, done))
        done.wait()          # truly blocks until the voice is done
        time.sleep(0.3)      # tiny natural pause after speech

    def record_audio(self, duration_sec: int, filename="temp_recording.wav"):
        """
        Record audio from the default microphone into the secure temp directory.

        The file is stored in data/.secure_tmp/ instead of the project root.
        The returned path should be passed to secure_delete() after processing.
        """
        # Route all recordings to the secure temp directory
        secure_dir = get_secure_temp_dir()
        basename = os.path.basename(filename)
        secure_path = os.path.join(secure_dir, basename)

        print(f"[VOICE BOT] Recording {duration_sec}s to secure temp...")
        recording = sd.rec(
            int(duration_sec * self.fs),
            samplerate=self.fs, channels=1, dtype="int16",
        )
        sd.wait()
        wavfile.write(secure_path, self.fs, recording)
        print("[VOICE BOT] Recording complete (secure temp).")

        # Track for cleanup
        self._temp_files.append(secure_path)
        return secure_path

    def transcribe(self, filename="temp_recording.wav") -> str:
        """Transcribe a WAV file to text using OpenAI Whisper (local, no ffmpeg needed)."""
        print("[VOICE BOT] Transcribing with Whisper...")
        try:
            import whisper
            import numpy as np
            from scipy.io import wavfile as wf
            from scipy.signal import resample

            if not hasattr(self, '_whisper_model'):
                print("[VOICE BOT] Loading Whisper model (base)...")
                # Force CPU — RTX 5070 (sm_120 / Blackwell) not yet supported
                # by stable PyTorch CUDA. Remove device="cpu" once supported.
                self._whisper_model = whisper.load_model("base", device="cpu")

            # Load WAV with scipy (no ffmpeg needed)
            sr_orig, data = wf.read(filename)
            # Convert to float32 mono
            if data.dtype == np.int16:
                audio = data.astype(np.float32) / 32768.0
            elif data.dtype == np.int32:
                audio = data.astype(np.float32) / 2147483648.0
            else:
                audio = data.astype(np.float32)
            # If stereo, take mean
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)
            # Resample to 16kHz (Whisper expects 16000 Hz)
            if sr_orig != 16000:
                num_samples = int(len(audio) * 16000 / sr_orig)
                audio = resample(audio, num_samples).astype(np.float32)

            # Pad/trim to 30s and generate mel spectrogram
            audio = whisper.pad_or_trim(audio)
            mel = whisper.log_mel_spectrogram(audio).to(self._whisper_model.device)

            # Decode
            options = whisper.DecodingOptions(language="en", fp16=False)
            result = whisper.decode(self._whisper_model, mel, options)
            text = result.text.strip()
            print(f"[VOICE BOT] Whisper result: '{text}'")
            return text
        except Exception as e:
            print(f"[VOICE BOT] Whisper transcription error: {e}")
            return ""

    def secure_cleanup(self):
        """Securely wipe all temporary audio files created by this bot."""
        cleaned = 0
        for filepath in self._temp_files:
            if os.path.exists(filepath):
                secure_delete(filepath)
                cleaned += 1
        self._temp_files.clear()
        if cleaned:
            print(f"[VOICE BOT] Securely wiped {cleaned} temp audio file(s).")

    def cleanup_file(self, filepath: str):
        """Securely wipe a specific audio file after processing."""
        if filepath and os.path.exists(filepath):
            secure_delete(filepath)
            if filepath in self._temp_files:
                self._temp_files.remove(filepath)
            print(f"[VOICE BOT] Securely wiped: {os.path.basename(filepath)}")


bot = AIVoiceBot()
