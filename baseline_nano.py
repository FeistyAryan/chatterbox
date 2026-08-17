import torch
import torchaudio as ta
from pathlib import Path

from chatterbox.tts_turbo import ChatterboxTurboTTS


MANIFEST = "/home/aryan/Desktop/yt_scraper/dataset_final.manifest.txt"

# Pick a reference clip >= 5 seconds.
# We'll just find the first suitable one automatically.
import torchaudio

reference_audio = None

for line in open(MANIFEST, encoding="utf-8"):
    audio_path, _ = line.rstrip("\n").split("|", 1)

    try:
        info = torchaudio.info(audio_path)
        duration = info.num_frames / info.sample_rate

        if duration >= 5.5:
            reference_audio = audio_path
            break
    except Exception:
        continue

if reference_audio is None:
    raise RuntimeError("Could not find a reference clip >= 5.5 seconds")

print("Reference:", reference_audio)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

model = ChatterboxTurboTTS.from_pretrained(
    device=device,
    nano=True,
)

sentences = [
    "नमस्ते।",
    "भारत एक बड़ा देश है।",
    "आज मौसम बहुत अच्छा है।",
]

output_dir = Path("baseline_outputs")
output_dir.mkdir(exist_ok=True)

for i, text in enumerate(sentences):
    print(f"\nGenerating {i + 1}/{len(sentences)}")
    print(text)

    wav = model.generate(
    text,
    audio_prompt_path=reference_audio,
    cfg_weight=0.0,
)

    output_path = output_dir / f"baseline_{i+1}.wav"

    ta.save(
        str(output_path),
        wav.cpu(),
        model.sr,
    )

    torch.cuda.synchronize()
    del wav
    torch.cuda.empty_cache()

    print("Saved:", output_path)
