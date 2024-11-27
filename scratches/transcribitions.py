import whisper
import os


audio = whisper.load_audio("rurik_.mp3")

whisper_model = whisper.load_model("turbo", device="cuda")
result = whisper_model.transcribe(audio, language="Russian", fp16=False, verbose=True)["text"].strip()


with open("transcription.txt", "w") as file:
    file.write(result)
