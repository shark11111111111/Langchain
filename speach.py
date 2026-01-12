import pyttsx3
import threading

def _speak(text):
    engine=pyttsx3.init()
    engine.say(text)
    engine.runAndWait()
    engine.stop()

def speak(text):
    threading.Thread(target=_speak, args=(text,), daemon=True).start()