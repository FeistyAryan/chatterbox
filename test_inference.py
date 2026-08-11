import os
import torch
import torchaudio as ta
from peft import PeftModel
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

# Force CPU to bypass the VRAM limit
device = "cpu"
print(f"🚀 Inference Device: {device}")

# 1. Load Multilingual Base Model on CPU
print("Loading Multilingual Base Model on CPU...")
model = ChatterboxMultilingualTTS.from_pretrained(device=device)

# 2. Attach our trained LoRA Checkpoint
lora_path = "./chatterbox_lora_best"
if os.path.exists(lora_path):
    print("Attaching Fine-Tuned LoRA Weights...")
    model.t3.tfmr = PeftModel.from_pretrained(model.t3.tfmr, lora_path)
    print("✅ LoRA Attached Successfully!")
else:
    print("⚠️ LoRA checkpoint not found! Running base model.")

# 3. Setup Inputs (Direct Devanagari)
text = "क्योंकि जितना मैं चाहता हूं इंडिया सुपर पावर बने पावर बने पावर बने पावर बने। जो हालात हैं इंडिया के, वो देख के लगता नहीं कि हम सुपर पावर बनेंगे।"
audio_prompt = "/home/aryan/Desktop/yt_scraper/output/fbWf6HjaNiA/segments_aligned/000025/final.wav"

print("\nGenerating Speech (This might take 30-60 seconds on CPU)...")
with torch.inference_mode():
    # Native generate method running entirely on CPU
    wav = model.generate(
        text=text,
        language_id="hi",               
        audio_prompt_path=audio_prompt, 
        temperature=0.6,               
        cfg_weight=0.5                 
    )

# 4. Save Output
output_path = "lora_best_test.wav"
ta.save(output_path, wav, model.sr)

print(f"🎉 Audio perfectly generated and saved as '{output_path}'!")