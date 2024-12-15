import whisper
import os


audio = whisper.load_audio("rurik_.mp3")

whisper_model = whisper.load_model("turbo", device="cuda")
result = whisper_model.transcribe(audio, language="Russian", fp16=False, verbose=False)[
    "text"
].strip()


with open("cleaned_docs/rurik_v2.txt", "w", encoding="utf-8") as file:
    file.write(result)
