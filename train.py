import os
import gc
import random
import torch
import torchaudio as ta
from torch.utils.data import Dataset, DataLoader, random_split
from peft import LoraConfig, get_peft_model
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

# --- 1. MEMORY OPTIMIZATIONS ---
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Execution Device: {device}")

# --- 2. MULTILINGUAL DATASET ---
class MultilingualTTSDataset(Dataset):
    def __init__(self, manifest_path, mtl_model, max_audio_sec=8.0):
        self.data = []
        self.model = mtl_model
        self.target_sr = 16000  # S3Tokenizer strictly needs 16kHz

        print("Loading and Filtering Devanagari Dataset...")
        with open(manifest_path, 'r', encoding='utf-8') as f:
            for line in f:
                if "|" not in line:
                    continue
                audio_path, text = line.strip().split("|", 1)

                # Check audio length to prevent RTX 2050 OOM
                try:
                    info = ta.info(audio_path)
                    duration = info.num_frames / info.sample_rate
                    if duration <= max_audio_sec:
                        self.data.append({"audio": audio_path, "text": text})
                except Exception as e:
                    pass

        print(f"✅ Loaded {len(self.data)} valid short clips for training.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        audio_path = entry['audio']

        # Text -> Native Text Tokens with [hi] tag natively injected
        text_tokens = self.model.tokenizer.text_to_tokens(
            entry['text'], language_id="hi"
        ).squeeze(0)

        sot = torch.tensor([self.model.t3.hp.start_text_token])
        eot = torch.tensor([self.model.t3.hp.stop_text_token])
        text_tokens = torch.cat([sot, text_tokens, eot])

        # Audio -> Speech Tokens
        wav, sr = ta.load(audio_path)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)

        if sr != self.target_sr:
            wav = ta.functional.resample(wav, orig_freq=sr, new_freq=self.target_sr)

        with torch.no_grad():
            speech_tokens, _ = self.model.s3gen.tokenizer(wav)
            speech_tokens = speech_tokens.squeeze(0)

        # FIX #2: Speech ko bhi start/stop tokens chahiye — inference start_speech_token
        # se shuru hoti hai aur stop_speech_token par rukti hai. Inke bina model
        # kabhi rukna seekhta hi nahi (infinite loop bug).
        sos = torch.tensor([self.model.t3.hp.start_speech_token])
        eos = torch.tensor([self.model.t3.hp.stop_speech_token])
        speech_tokens = torch.cat([sos, speech_tokens, eos])

        return {
            "text_tokens": text_tokens,
            "speech_tokens": speech_tokens,
            "audio_path": audio_path
        }

# --- 3. COLLATE FUNCTION (PADDING) ---
def collate_fn(batch):
    # Get sequence lengths for loss calculation masking
    text_lens = torch.tensor([len(b["text_tokens"]) for b in batch], dtype=torch.long)
    speech_lens = torch.tensor([len(b["speech_tokens"]) for b in batch], dtype=torch.long)

    text_padded = torch.nn.utils.rnn.pad_sequence(
        [b["text_tokens"] for b in batch], batch_first=True, padding_value=0
    )
    speech_padded = torch.nn.utils.rnn.pad_sequence(
        [b["speech_tokens"] for b in batch], batch_first=True, padding_value=6562
    )

    audio_paths = [b["audio_path"] for b in batch]
    return text_padded, text_lens, speech_padded, speech_lens, audio_paths

# --- 4. MODEL & LORA SETUP (VRAM OPTIMIZED) ---
print("Loading Multilingual Base Model...")
base_model = ChatterboxMultilingualTTS.from_pretrained(device="cpu")  # Strictly CPU first

gc.collect()
torch.cuda.empty_cache()

if hasattr(base_model.t3.tfmr, "gradient_checkpointing_enable"):
    base_model.t3.tfmr.gradient_checkpointing_enable()

# SHRINK THE MODEL: Cast base T3 to 16-bit float (FP16).
# PEFT apne aap LoRA adapters ko float32 mein rakhta hai, isliye GradScaler safe hai.
base_model.t3.half()

lora_config = LoraConfig(
    r=4,
    lora_alpha=8,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
)
base_model.t3.tfmr = get_peft_model(base_model.t3.tfmr, lora_config)

# STRICT VRAM CONTROL: Only move T3 to GPU. Leave s3gen and ve on CPU!
base_model.t3.to(device)

full_dataset = MultilingualTTSDataset("/home/aryan/Desktop/yt_scraper/dataset_final.manifest.txt", base_model)

# FIX #3 ke liye: conditioning prompt hamesha KISI DOOSRI clip se aayega (same speaker),
# warna model text->speech seekhne ke bajaye prompt se copy karna seekh leta hai.
all_audio_paths = [d["audio"] for d in full_dataset.data]

# Train/Val split (val_loss overfitting pakadne ke liye)
val_size = max(8, int(0.1 * len(full_dataset)))
train_size = len(full_dataset) - val_size
train_set, val_set = random_split(
    full_dataset, [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)
print(f"📊 Split: {train_size} train / {val_size} val clips")

train_loader = DataLoader(train_set, batch_size=1, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_set, batch_size=1, shuffle=False, collate_fn=collate_fn)

# --- 5. LOSS CALCULATION ---
IGNORE_ID = -100

def compute_losses(text_tokens, text_lens, speech_tokens, speech_lens, audio_paths):
    # Conditioning prompt: same speaker ki koi alag random clip (CPU par process hota hai)
    cond_path = random.choice([p for p in all_audio_paths if p != audio_paths[0]])
    base_model.prepare_conditionals(cond_path)
    t3_cond = base_model.conds.t3.to(device=device, dtype=torch.float16)

    text_tokens = text_tokens.to(device)
    text_lens = text_lens.to(device)
    speech_tokens = speech_tokens.to(device)
    speech_lens = speech_lens.to(device)

    out = base_model.t3(
        t3_cond=t3_cond,
        text_tokens=text_tokens,
        text_token_lens=text_lens,
        speech_tokens=speech_tokens,
        speech_token_lens=speech_lens,
        training=True
    )

    # Masking (Ignore Padding Tokens for Loss Calculation)
    mask_text = torch.arange(text_tokens.size(1), device=device)[None, :] >= text_lens[:, None]
    mask_speech = torch.arange(speech_tokens.size(1), device=device)[None, :] >= speech_lens[:, None]
    masked_text = text_tokens.masked_fill(mask_text, IGNORE_ID)
    masked_speech = speech_tokens.masked_fill(mask_speech, IGNORE_ID)

    # FIX #1: Next-token shift. Position t ka logit token t+1 ko predict karta hai,
    # isliye logits[:, :-1] ko targets[:, 1:] se compare karna hai. Bina shift ke
    # model "current token wapas bolo" (copy task) seekh leta hai -> token looping.
    loss_text = torch.nn.functional.cross_entropy(
        out.text_logits[:, :-1].reshape(-1, out.text_logits.size(-1)),
        masked_text[:, 1:].reshape(-1),
        ignore_index=IGNORE_ID
    )
    loss_speech = torch.nn.functional.cross_entropy(
        out.speech_logits[:, :-1].reshape(-1, out.speech_logits.size(-1)),
        masked_speech[:, 1:].reshape(-1),
        ignore_index=IGNORE_ID
    )
    return loss_text, loss_speech

# --- 6. THE TRAINING LOOP ---
trainable_params = [p for p in base_model.t3.tfmr.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(trainable_params, lr=1e-4)
scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

epochs = 15
grad_accum_steps = 16
step = 0
best_val_loss = float("inf")

print("Starting VRAM-Safe TTS Fine-tuning Loop...")

for epoch in range(epochs):
    base_model.t3.tfmr.train()
    running_loss = 0.0
    running_count = 0
    epoch_train_loss = 0.0
    optimizer.zero_grad()

    for i, batch in enumerate(train_loader):
        with torch.amp.autocast("cuda", enabled=(device == "cuda"), dtype=torch.float16):
            loss_text, loss_speech = compute_losses(*batch)
            loss = (loss_text + loss_speech) / grad_accum_steps

        scaler.scale(loss).backward()
        running_loss += loss.item() * grad_accum_steps
        running_count += 1
        epoch_train_loss += loss.item() * grad_accum_steps

        if (i + 1) % grad_accum_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            step += 1
            print(f"Epoch [{epoch+1}/{epochs}] | Step [{step}] | "
                  f"Loss: {running_loss/running_count:.4f} "
                  f"(text: {loss_text.item():.3f}, speech: {loss_speech.item():.3f})")
            running_loss = 0.0
            running_count = 0

    # Leftover gradients flush karo (agar last batch group incomplete tha)
    if len(train_loader) % grad_accum_steps != 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    # --- VALIDATION ---
    base_model.t3.tfmr.eval()
    val_text, val_speech = 0.0, 0.0
    with torch.no_grad():
        for batch in val_loader:
            with torch.amp.autocast("cuda", enabled=(device == "cuda"), dtype=torch.float16):
                loss_text, loss_speech = compute_losses(*batch)
            val_text += loss_text.item()
            val_speech += loss_speech.item()
    val_text /= len(val_loader)
    val_speech /= len(val_loader)
    val_loss = val_text + val_speech
    train_loss = epoch_train_loss / len(train_loader)

    print(f"\n📈 Epoch {epoch+1}/{epochs} | train_loss: {train_loss:.4f} | "
          f"val_loss: {val_loss:.4f} (text: {val_text:.3f}, speech: {val_speech:.3f})\n")

    # Har epoch ka alag checkpoint (adapter sirf ~2MB ka hai) + latest + best
    base_model.t3.tfmr.save_pretrained(f"./checkpoints/epoch_{epoch+1:02d}")
    base_model.t3.tfmr.save_pretrained("./chatterbox_lora_final")
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        base_model.t3.tfmr.save_pretrained("./chatterbox_lora_best")
        print(f"⭐ New best val_loss ({val_loss:.4f}) — saved to ./chatterbox_lora_best")

print("Real Training Completed Successfully!")
