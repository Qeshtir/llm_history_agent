import whisper
import os


audio = whisper.load_audio("Varyag_3.mp3")

whisper_model = whisper.load_model("turbo", device="cuda")
result = whisper_model.transcribe(audio, language="Russian", fp16=False, verbose=False)["text"].strip()


with open("varyag_3.txt", "w", encoding="utf-8") as file:
    file.write(result)
