"""
voice_greeting.py — Step 3: Neutral TTS greeting.

Uses pyttsx3 (offline) to speak a neutral greeting.
Security: NEVER reveals the user's name or identity aloud.
"""

import sys
import os
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


class VoiceGreeting:
    """Handles text-to-speech for neutral greetings."""

    def __init__(self) -> None:
        self.engine = None
        self._initialize()

    def _initialize(self) -> None:
        """Initialize the TTS engine."""
        try:
            import pyttsx3
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', 160)  # Speaking speed
            self.engine.setProperty('volume', 0.9)
            
            # Set to male voice (voices[0] is typically male on most systems)
            voices = self.engine.getProperty('voices')
            if len(voices) > 0:
                self.engine.setProperty('voice', voices[0].id)
            
            print("[INFO] TTS engine initialized (male voice).")
        except Exception as e:
            print(f"[WARN] Could not initialize TTS engine: {e}")
            print("[WARN] Voice greeting will be displayed on screen only.")

    def speak_greeting(self, user_initial: Optional[str] = None) -> bool:
        """
        Speak a neutral greeting. Never reveals the user's name.

        Args:
            user_initial: Optional single character to display on screen (not spoken).

        Returns:
            True if greeting was delivered.
        """
        greeting = config.GREETING_TEXT
        print(f"\n{'=' * 50}")
        print(f"  🔊 {greeting}")
        if user_initial:
            print(f"  [User: {user_initial[0].upper()}***]")
        print(f"{'=' * 50}\n")

        if self.engine:
            try:
                self.engine.say(greeting)
                self.engine.runAndWait()
                return True
            except Exception as e:
                print(f"[WARN] TTS playback failed: {e}")
                return True  # Greeting was at least displayed
        return True  # Displayed on screen

    def speak_challenge_prompt(self, prompt_text: str) -> bool:
        """Speak a challenge prompt (used by other modules)."""
        print(f"  🔊 {prompt_text}")
        if self.engine:
            try:
                self.engine.say(prompt_text)
                self.engine.runAndWait()
                return True
            except Exception as e:
                print(f"[WARN] TTS playback failed: {e}")
        return True

    def speak_result(self, success: bool) -> bool:
        """Speak the final authentication result."""
        if success:
            msg = "Verification complete. Welcome."
        else:
            msg = "Verification failed. Access denied."

        print(f"\n  🔊 {msg}")
        if self.engine:
            try:
                self.engine.say(msg)
                self.engine.runAndWait()
            except Exception:
                pass
        return True

    def cleanup(self) -> None:
        """Clean up TTS resources."""
        if self.engine:
            try:
                self.engine.stop()
            except Exception:
                pass


if __name__ == "__main__":
    greeting = VoiceGreeting()
    greeting.speak_greeting(user_initial="A")
    greeting.speak_challenge_prompt("Please turn your head to the left.")
    greeting.speak_result(True)
    greeting.cleanup()
