"""Fine-tune Chatterbox Nano's T3 on a Hindi manifest.

Key design decisions (verified against the repo source):

* Text tokenization uses Nano's own GPT-2 tokenizer with NO added start/stop
  tokens, because `ChatterboxTurboTTS.generate()` feeds raw tokenizer ids to
  `inference_turbo` without them. Training must match inference exactly.
  (`hp.start_text_token=255` / `stop_text_token=0` belong to the old 704-vocab
  EnTokenizer, not the GPT-2 vocab.)
* The autoregressive loss is shifted: logits at position k are trained against
  the token at position k+1. The repo's `T3.loss()` compares unshifted, which
  degenerates into an input-copy task.
* `text_emb` rows for Devanagari byte tokens are essentially untrained in the
  released checkpoint, so `text_emb` is trainable by default (weight_decay=0 so
  unused rows receive no update at all under AdamW).
* Speech tokens (S3 codec) are language-agnostic and stay frozen along with
  s3gen / voice encoder / heads.
* All tokens are precomputed once into a disk cache so epochs never touch
  audio, and s3gen / voice encoder are freed after setup to save RAM.
"""

import argparse
import copy
import gc
import json
import math
import random
import time
import traceback
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
import torchaudio as ta
from torch.utils.data import DataLoader, Dataset, Sampler, Subset

try:
    from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
except Exception:  # pragma: no cover - optional scheduler path
    get_linear_schedule_with_warmup = None
    get_cosine_schedule_with_warmup = None

from peft import LoraConfig, PeftModel, get_peft_model

import chatterbox.models.t3.t3 as t3_module
from chatterbox.tts_turbo import ChatterboxTurboTTS, punc_norm
from chatterbox.models.s3gen.const import S3GEN_SIL
from chatterbox.models.t3.modules.cond_enc import T3Cond

# T3.forward() asserts that start/stop text tokens (255 / 0) are present, but
# those ids are from the legacy EnTokenizer vocab. Nano inference never uses
# them, so training must not either. Disable the assert.
t3_module._ensure_BOT_EOT = lambda text_tokens, hp: None


# Repo-relative so the scripts work on any machine. `data/` is gitignored and
# populated by prepare_vaani_dataset.py -- see NANO_HINDI_FINETUNE.md.
REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "vaani_hindi"
DEFAULT_MANIFEST = DEFAULT_DATA_DIR / "manifest.txt"
DEFAULT_REFERENCE_AUDIO = DEFAULT_DATA_DIR / "wavs" / "000000.wav"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "experiments" / "vaani_full"
IGNORE_ID = -100
TARGET_SR = 16000
TEXT_PAD = 0  # masked out by lengths, value irrelevant
SPEECH_PAD = 0  # masked out by lengths, value irrelevant
S3_EOS = 6561  # first non-codec id; valid codec ids are [0, 6561)
NANO_T3_TEXT_VOCAB = 50276

FIXED_EVAL_PROMPTS = [
    "आज हम एक छोटा तकनीकी परीक्षण कर रहे हैं।",
    "कृपया साफ़ और धीरे बोलिए।",
    "यह प्रयोग हिंदी आवाज़ सुधारने के लिए है।",
]


def resolve_amp_dtype(choice: str, device: torch.device):
    """Autocast dtype, or None for full fp32.

    bf16 is preferred wherever the hardware has it: it carries fp32's exponent
    range, so it needs no GradScaler and cannot overflow the way fp16 did at
    lora_r=64. fp16 remains the fallback for pre-Ampere cards.
    """
    if device.type != "cuda" or choice == "fp32":
        return None
    if choice == "auto":
        choice = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    return torch.bfloat16 if choice == "bf16" else torch.float16


class LengthBucketSampler(Sampler):
    """Yield batches whose members have similar sequence lengths.

    Every batch is padded to its longest member, so pairing a 50-token clip with
    an 1100-token one throws away most of the compute on padding. Vaani clip
    lengths span roughly that whole range, so with batch_size>1 this is the
    difference between a fast epoch and a mostly-idle one.

    Shuffle globally, cut into chunks of `bucket_size` batches, sort each chunk
    by length, then shuffle the emitted batch order. Randomness is preserved at
    the batch level; only the *composition* of a batch is length-biased.
    """

    def __init__(self, lengths, batch_size, shuffle=True, bucket_size=50, seed=0):
        self.lengths = list(lengths)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.chunk = batch_size * bucket_size
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __len__(self):
        return math.ceil(len(self.lengths) / self.batch_size)

    def __iter__(self):
        idx = list(range(len(self.lengths)))
        if self.shuffle:
            rng = random.Random(self.seed + self.epoch)
            rng.shuffle(idx)
        batches = []
        for start in range(0, len(idx), self.chunk):
            chunk = sorted(idx[start:start + self.chunk], key=lambda i: self.lengths[i])
            batches.extend(chunk[i:i + self.batch_size]
                           for i in range(0, len(chunk), self.batch_size))
        if self.shuffle:
            random.Random(self.seed + self.epoch + 1).shuffle(batches)
        return iter(batches)


@dataclass
class RunState:
    epoch: int = 0
    best_val_speech_loss: float = float("inf")
    no_improve_epochs: int = 0
    train_indices: Optional[list[int]] = None
    val_indices: Optional[list[int]] = None


def bytes_to_mb(value: int) -> float:
    return value / (1024 * 1024)


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_manifest_rows(manifest_path: Path) -> list[tuple[Path, str]]:
    rows = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            raw_line = raw_line.strip()
            if not raw_line or "|" not in raw_line:
                continue
            audio_path, transcript = raw_line.split("|", 1)
            rows.append((Path(audio_path), transcript))
    if not rows:
        raise RuntimeError(f"No valid rows found in {manifest_path}")
    return rows


def freeze_module(module: torch.nn.Module):
    for param in module.parameters():
        param.requires_grad = False


def to_mono_resampled(wav: torch.Tensor, sr: int, target_sr: int = TARGET_SR) -> torch.Tensor:
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = ta.functional.resample(wav, orig_freq=sr, new_freq=target_sr)
    return wav.contiguous()


# ---------------------------------------------------------------------------
# Token cache
# ---------------------------------------------------------------------------

def build_or_load_token_cache(cache_path: Path, rows: list[tuple[Path, str]], model: ChatterboxTurboTTS,
                              multi_speaker: bool = False, cache_device: str = "cpu"):
    """Tokenize the whole manifest once (text + S3 speech tokens) and cache to disk.

    In multi-speaker mode a voice-encoder embedding is also computed per clip, so
    every training example can be conditioned on its OWN speaker.
    """
    if cache_path.exists():
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        if payload.get("num_rows") == len(rows) and payload.get("multi_speaker", False) == multi_speaker:
            print(f"Loaded token cache: {cache_path} ({len(rows)} rows, multi_speaker={multi_speaker})")
            return payload["items"]
        print("Token cache mismatch (row count or speaker mode); rebuilding.")

    hp = model.t3.hp
    items = []
    # The S3 tokenizer and voice encoder are the slow part (~0.235 s/clip on CPU).
    # At 82k clips that is over five hours, so run them on the GPU when there is
    # one; both are small and the training model is not resident yet.
    dev = torch.device(cache_device)
    model.s3gen.tokenizer.to(dev)
    if multi_speaker:
        model.ve.to(dev)
    print(f"Building token cache for {len(rows)} rows (one-time, S3 tokenizer on {dev})...")
    start = time.perf_counter()
    for idx, (audio_path, transcript) in enumerate(rows):
        if not audio_path.exists():
            raise FileNotFoundError(f"Missing audio file: {audio_path}")

        # Text: match generate() exactly -> punc_norm, no special tokens.
        text = punc_norm(transcript)
        text_tokens = model.tokenizer(
            text, return_tensors="pt", padding=False, truncation=False
        ).input_ids.squeeze(0).to(torch.long)
        if text_tokens.numel() > hp.max_text_tokens:
            raise ValueError(f"Row {idx}: transcript exceeds max_text_tokens ({hp.max_text_tokens}).")
        if int(text_tokens.min()) < 0 or int(text_tokens.max()) >= NANO_T3_TEXT_VOCAB:
            raise ValueError(f"Row {idx}: text token id out of vocabulary bounds.")

        # Speech: S3 codec tokens wrapped in BOS/EOS, matching inference_turbo
        # which starts from start_speech_token and stops on stop_speech_token.
        wav, sr = ta.load(str(audio_path))
        wav = to_mono_resampled(wav, sr, TARGET_SR)
        with torch.no_grad():
            speech_tokens, _ = model.s3gen.tokenizer.forward([wav.to(dev)], max_len=None)
        speech_tokens = speech_tokens.squeeze(0).to("cpu", torch.long)
        if int(speech_tokens.min()) < 0 or int(speech_tokens.max()) >= S3_EOS:
            raise ValueError(f"Row {idx}: speech token id out of codec bounds.")
        speech_tokens = torch.cat([
            torch.tensor([hp.start_speech_token], dtype=torch.long),
            speech_tokens,
            torch.tensor([hp.stop_speech_token], dtype=torch.long),
        ])
        if speech_tokens.numel() > hp.max_speech_tokens:
            raise ValueError(f"Row {idx}: speech sequence exceeds max_speech_tokens.")

        item = {
            "audio_path": str(audio_path),
            "transcript": transcript,
            "text_tokens": text_tokens,
            "speech_tokens": speech_tokens,
        }
        if multi_speaker:
            # 256-d voice-encoder embedding of this clip. It encodes speaker
            # identity, not content, so deriving it from the target audio is
            # safe -- unlike a speech cond prompt, which would leak the answer.
            with torch.no_grad():
                ve_embed = model.ve.embeds_from_wavs([wav.squeeze(0).numpy()], sample_rate=TARGET_SR)
            item["speaker_emb"] = torch.from_numpy(ve_embed).float().mean(dim=0).cpu()
        items.append(item)
        if (idx + 1) % 500 == 0 or (idx + 1) == len(rows):
            rate = (idx + 1) / (time.perf_counter() - start)
            eta = (len(rows) - idx - 1) / max(rate, 1e-6)
            print(f"  tokenized {idx + 1}/{len(rows)}  ({rate:.1f} clips/s, ETA {eta/60:.1f} min)")

    model.s3gen.tokenizer.to("cpu")
    if multi_speaker:
        model.ve.to("cpu")
    if dev.type == "cuda":
        torch.cuda.empty_cache()

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"num_rows": len(rows), "items": items, "multi_speaker": multi_speaker}, cache_path)
    print(f"Token cache built in {time.perf_counter() - start:.1f}s -> {cache_path}")
    return items


class CachedTokenDataset(Dataset):
    def __init__(self, items: list[dict]):
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        return self.items[idx]


def collate_batch(batch):
    text_lens = torch.tensor([item["text_tokens"].numel() for item in batch], dtype=torch.long)
    speech_lens = torch.tensor([item["speech_tokens"].numel() for item in batch], dtype=torch.long)

    text_tokens = torch.nn.utils.rnn.pad_sequence(
        [item["text_tokens"] for item in batch], batch_first=True, padding_value=TEXT_PAD
    )
    speech_tokens = torch.nn.utils.rnn.pad_sequence(
        [item["speech_tokens"] for item in batch], batch_first=True, padding_value=SPEECH_PAD
    )

    out = {
        "text_tokens": text_tokens,
        "text_lens": text_lens,
        "speech_tokens": speech_tokens,
        "speech_lens": speech_lens,
    }
    if "speaker_emb" in batch[0]:
        out["speaker_emb"] = torch.stack([item["speaker_emb"] for item in batch])
    return out


# ---------------------------------------------------------------------------
# Model setup
# ---------------------------------------------------------------------------

def build_nano_model(args, device: torch.device) -> ChatterboxTurboTTS:
    # Load on CPU first to avoid an unnecessary CUDA memory spike.
    model = ChatterboxTurboTTS.from_pretrained(device="cpu", nano=True)

    freeze_module(model.t3)
    freeze_module(model.s3gen)
    freeze_module(model.ve)

    # Nano's loader deletes GPT-2's wte to save memory for inference, but
    # transformers' gradient-checkpointing helper expects get_input_embeddings()
    # to exist. Recreate a minimal frozen stub only for that helper.
    if not hasattr(model.t3.tfmr, "wte"):
        model.t3.tfmr.wte = torch.nn.Embedding(1, model.t3.tfmr.config.hidden_size)
        freeze_module(model.t3.tfmr.wte)

    if args.gradient_checkpointing and hasattr(model.t3.tfmr, "gradient_checkpointing_enable"):
        model.t3.tfmr.gradient_checkpointing_enable()
        model.t3.tfmr.config.use_cache = False

    if args.train_mode == "lora":
        lora_cfg = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.lora_targets.split(","),
            bias="none",
            fan_in_fan_out=True,  # GPT-2 uses Conv1D layers
        )
        if args.resume_from:
            model.t3.tfmr = PeftModel.from_pretrained(model.t3.tfmr, args.resume_from, is_trainable=True)
        else:
            model.t3.tfmr = get_peft_model(model.t3.tfmr, lora_cfg)
        # PEFT re-enables adapters; make sure everything else stayed frozen.
        for name, param in model.t3.tfmr.named_parameters():
            param.requires_grad = "lora_" in name
        if args.train_text_emb:
            for param in model.t3.text_emb.parameters():
                param.requires_grad = True
            if args.resume_from:
                emb_path = Path(args.resume_from) / "text_emb.pt"
                if emb_path.exists():
                    model.t3.text_emb.load_state_dict(torch.load(emb_path, map_location="cpu"))
                    print(f"Resumed text_emb from {emb_path}")
    elif args.train_mode == "full":
        # Full fine-tune of T3 (backbone, embeddings, speech head). text_head
        # stays frozen: it only serves the auxiliary text loss.
        for param in model.t3.parameters():
            param.requires_grad = True
        if not args.train_text_head:
            # text_head only feeds the auxiliary text loss (weight 0.2), so
            # training its 38.6M params buys little. Opt in explicitly.
            freeze_module(model.t3.text_head)
        freeze_module(model.t3.tfmr.wte)  # stub
        if args.resume_from:
            state_path = Path(args.resume_from) / "t3_state.pt"
            model.t3.load_state_dict(torch.load(state_path, map_location="cpu"), strict=False)
            print(f"Resumed full T3 state from {state_path}")
    else:
        raise ValueError(f"Unknown train mode: {args.train_mode}")

    model.t3.to(device=device)
    model.s3gen.to("cpu").eval()
    model.ve.to("cpu").eval()
    model.t3.train()
    return model


def build_param_groups(model: ChatterboxTurboTTS, args):
    lora_params = [p for n, p in model.t3.named_parameters() if p.requires_grad and "lora_" in n]
    emb_params = [p for n, p in model.t3.named_parameters() if p.requires_grad and n.startswith("text_emb")]
    other_params = [
        p for n, p in model.t3.named_parameters()
        if p.requires_grad and "lora_" not in n and not n.startswith("text_emb")
    ]

    groups = []
    if lora_params:
        groups.append({"params": lora_params, "lr": args.lr, "weight_decay": 0.01, "name": "lora"})
    if emb_params:
        # weight_decay must be 0: with decay, every embedding row (including the
        # ~50k unused English rows) would shrink each step and damage the model.
        groups.append({"params": emb_params, "lr": args.emb_lr, "weight_decay": 0.0, "name": "text_emb"})
    if other_params:
        groups.append({"params": other_params, "lr": args.lr, "weight_decay": 0.0, "name": "full"})

    for g in groups:
        count = sum(p.numel() for p in g["params"])
        print(f"Param group '{g['name']}': {count:,} params, lr={g['lr']}, wd={g['weight_decay']}")
    return groups


def check_sanity(model: ChatterboxTurboTTS, args):
    trainable = [(n, p) for n, p in model.t3.named_parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("No trainable parameters found.")

    if args.train_mode == "lora":
        for name, _ in trainable:
            if "lora_" not in name and not name.startswith("text_emb"):
                raise RuntimeError(f"Unexpected trainable parameter in lora mode: {name}")
    if any(p.requires_grad for p in model.s3gen.parameters()):
        raise RuntimeError("S3Gen parameters are not frozen.")
    if any(p.requires_grad for p in model.ve.parameters()):
        raise RuntimeError("Voice encoder parameters are not frozen.")
    if not getattr(args, "train_text_head", False):
        if any(p.requires_grad for p in model.t3.text_head.parameters()):
            raise RuntimeError("text_head parameters are not frozen.")

    total = sum(p.numel() for _, p in trainable)
    print(f"Sanity check passed. Trainable params: {total:,}")
    return total


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def shifted_masked_ce(logits: torch.Tensor, tokens: torch.Tensor, lens: torch.Tensor) -> torch.Tensor:
    """Autoregressive CE: logits at position k predict the token at k+1.

    Positions >= (len - 1) have no next-token target and are masked out.
    """
    logits = logits[:, :-1]
    targets = tokens[:, 1:]
    mask = torch.arange(targets.size(1), device=targets.device)[None, :] >= (lens - 1)[:, None]
    targets = targets.masked_fill(mask, IGNORE_ID)
    return F.cross_entropy(logits.transpose(1, 2), targets, ignore_index=IGNORE_ID)


def batch_t3_cond(batch, base_cond):
    """Conditioning for this batch.

    With a per-clip speaker embedding we must NOT reuse the single fixed
    reference: telling the model "this is speaker A" while asking it to predict
    speaker B teaches it to ignore speaker conditioning entirely.

    The speech cond prompt is left empty on purpose. hp.speech_cond_prompt_len
    is 375 tokens (15s), longer than most clips, so prompting a clip with its
    own audio would hand the model the very tokens it must predict.
    """
    if "speaker_emb" not in batch:
        return base_cond
    return T3Cond(
        speaker_emb=batch["speaker_emb"],
        cond_prompt_speech_tokens=None,
        cond_prompt_speech_emb=None,
        emotion_adv=None,
    )


def compute_batch_losses(model: ChatterboxTurboTTS, batch, t3_cond, text_loss_weight: float):
    t3_cond = batch_t3_cond(batch, t3_cond)
    out = model.t3.forward(
        t3_cond=t3_cond,
        text_tokens=batch["text_tokens"],
        text_token_lens=batch["text_lens"],
        speech_tokens=batch["speech_tokens"],
        speech_token_lens=batch["speech_lens"],
        training=True,
    )
    loss_text = shifted_masked_ce(out.text_logits, batch["text_tokens"], batch["text_lens"])
    loss_speech = shifted_masked_ce(out.speech_logits, batch["speech_tokens"], batch["speech_lens"])
    total_loss = loss_speech + text_loss_weight * loss_text
    return loss_text, loss_speech, total_loss


def move_batch_to_device(batch, device: torch.device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def clip_gradients(model: ChatterboxTurboTTS, max_grad_norm: float) -> None:
    """Clip the global gradient norm.

    LoRA is intrinsically constrained by its low rank, so this rarely mattered
    before. A full fine-tune of 178M params has no such guardrail: one outlier
    batch can produce a gradient large enough to wreck the weights in a single
    step, and the damage does not show up until val loss jumps an epoch later.
    """
    if not max_grad_norm or max_grad_norm <= 0:
        return
    params = [p for p in model.t3.parameters() if p.requires_grad and p.grad is not None]
    if params:
        torch.nn.utils.clip_grad_norm_(params, max_grad_norm)


def grad_finite_check(model: ChatterboxTurboTTS) -> tuple[int, bool]:
    grad_params = 0
    all_finite = True
    for name, param in model.t3.named_parameters():
        if not param.requires_grad:
            continue
        if param.grad is None:
            all_finite = False
            print(f"Gradient missing: {name}")
            continue
        grad_params += 1
        if not torch.isfinite(param.grad).all():
            all_finite = False
            print(f"Non-finite gradient detected: {name}")
    return grad_params, all_finite


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(model, output_dir: Path, state: RunState, optimizer, scheduler, args, epoch_metrics, name: str, scaler=None):
    ckpt_dir = output_dir / "checkpoints" / name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if args.train_mode == "lora":
        model.t3.tfmr.save_pretrained(ckpt_dir)
        if args.train_text_emb:
            torch.save(model.t3.text_emb.state_dict(), ckpt_dir / "text_emb.pt")
    else:
        torch.save(model.t3.state_dict(), ckpt_dir / "t3_state.pt")

    trainer_state = {
        "state": asdict(state),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None and scaler.is_enabled() else None,
        "epoch_metrics": epoch_metrics,
        "config": _json_safe(vars(args)),
    }
    torch.save(trainer_state, ckpt_dir / "trainer_state.pt")
    with (ckpt_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(vars(args)), handle, ensure_ascii=False, indent=2)
    return ckpt_dir


def _json_safe(payload: dict):
    def convert(value):
        if isinstance(value, (Path, torch.dtype, torch.device)):
            return str(value)
        if isinstance(value, dict):
            return {k: convert(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [convert(v) for v in value]
        return value

    return convert(payload)


def append_metrics_log(output_dir: Path, epoch_metrics: dict):
    log_path = output_dir / "metrics.jsonl"
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(epoch_metrics, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Eval sample generation
# ---------------------------------------------------------------------------

def maybe_generate_samples(model, reference_cond, output_dir: Path, device: torch.device, epoch: int):
    """Generate fixed Hindi prompts with the current adapter. Uses
    inference_turbo (the nano path); s3gen vocoding runs on CPU."""
    sample_dir = output_dir / "samples" / f"epoch_{epoch:03d}"
    sample_dir.mkdir(parents=True, exist_ok=True)

    eval_cond = copy.deepcopy(reference_cond.t3).to(device=device)
    was_checkpointing = getattr(model.t3.tfmr.config, "gradient_checkpointing", False)
    model.t3.eval()

    for idx, prompt in enumerate(FIXED_EVAL_PROMPTS, start=1):
        try:
            text = punc_norm(prompt)
            text_tokens = model.tokenizer(
                text, return_tensors="pt", padding=False, truncation=False
            ).input_ids.to(device)

            with torch.inference_mode():
                speech_tokens = model.t3.inference_turbo(
                    t3_cond=eval_cond,
                    text_tokens=text_tokens,
                    temperature=0.8,
                    top_k=1000,
                    top_p=0.95,
                    repetition_penalty=1.2,
                    max_gen_len=300,
                )

            speech_tokens = speech_tokens.squeeze(0)
            speech_tokens = speech_tokens[speech_tokens < S3_EOS].to("cpu")
            if speech_tokens.numel() == 0:
                print(f"Sample {idx}: no valid tokens generated, skipping.")
                continue

            silence = torch.tensor([S3GEN_SIL] * 3, dtype=torch.long)
            speech_tokens = torch.cat([speech_tokens, silence])

            with torch.inference_mode():
                wav, _ = model.s3gen.inference(
                    speech_tokens=speech_tokens,
                    ref_dict=reference_cond.gen,
                    n_cfm_timesteps=2,
                )
            wav = wav.squeeze(0).detach().cpu()
            out_path = sample_dir / f"prompt_{idx:02d}.wav"
            ta.save(str(out_path), wav if wav.dim() == 2 else wav.unsqueeze(0), sample_rate=model.sr)
            print(f"Saved eval sample: {out_path}")
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                print(f"Skipping sample generation for prompt {idx} due to CUDA OOM.")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                break
            raise

    model.t3.train()
    del eval_cond
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Train / eval loops
# ---------------------------------------------------------------------------

def run_epoch(model, loader, optimizer, scaler, device, t3_cond, grad_accum_steps, train, scheduler,
              text_loss_weight, amp_dtype=None, max_grad_norm=0.0):
    autocast_ctx = (torch.autocast(device_type="cuda", dtype=amp_dtype)
                    if amp_dtype is not None else nullcontext())
    grad_ctx = torch.enable_grad() if train else torch.inference_mode()

    total_text = total_speech = total_loss = 0.0
    batch_count = 0
    oom_skipped = 0

    if train:
        model.t3.train()
        optimizer.zero_grad(set_to_none=True)
    else:
        model.t3.eval()

    for step_idx, batch in enumerate(loader):
        if step_idx == 0 or (step_idx + 1) % 25 == 0 or (step_idx + 1) == len(loader):
            print(f"{'train' if train else 'val'} batch {step_idx + 1}/{len(loader)}")
        batch = move_batch_to_device(batch, device)

        try:
            with grad_ctx:
                with autocast_ctx:
                    loss_text, loss_speech, loss_total = compute_batch_losses(model, batch, t3_cond, text_loss_weight)
                    scaled_loss = loss_total / grad_accum_steps if train else loss_total
            # backward() must be inside the guard too: peak memory lands during
            # the backward pass, not the forward, so that is where an oversized
            # batch actually dies.
            if train:
                if scaler is not None:
                    scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()
        except torch.cuda.OutOfMemoryError:
            # A single unlucky long sequence must not kill a multi-hour run.
            oom_skipped += 1
            print(f"  CUDA OOM on batch {step_idx + 1} -- skipping it "
                  f"(seq len {int(batch['speech_tokens'].size(1))}+{int(batch['text_tokens'].size(1))}). "
                  f"total skipped: {oom_skipped}")
            # A half-finished backward leaves the accumulation window holding
            # gradients for only part of the batch. Stepping on those would apply
            # a silently mis-scaled update, so discard the window and restart it.
            if train:
                optimizer.zero_grad(set_to_none=True)
            # Drop every reference to the partial graph BEFORE empty_cache(),
            # otherwise the tensors are still live and nothing is reclaimed.
            batch = None
            loss_text = loss_speech = loss_total = scaled_loss = None
            if device.type == "cuda":
                torch.cuda.empty_cache()
            continue

        total_text += float(loss_text.detach())
        total_speech += float(loss_speech.detach())
        total_loss += float(loss_total.detach())
        batch_count += 1

        if train:
            should_step = ((step_idx + 1) % grad_accum_steps == 0) or ((step_idx + 1) == len(loader))
            if should_step:
                if scaler is not None:
                    # Gradients must be unscaled before they can be clipped or
                    # inspected -- otherwise the norm is off by the loss scale.
                    scaler.unscale_(optimizer)
                    # With fp16, overflow steps are expected occasionally; let
                    # GradScaler skip them instead of crashing.
                    grad_params, grads_finite = grad_finite_check(model)
                    if grad_params == 0:
                        raise RuntimeError("No gradients found before optimizer step.")
                    clip_gradients(model, max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    grad_params, grads_finite = grad_finite_check(model)
                    if not grads_finite:
                        raise RuntimeError("Non-finite or missing gradients detected before optimizer step.")
                    clip_gradients(model, max_grad_norm)
                    optimizer.step()

                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)

        del batch, loss_text, loss_speech, loss_total
        if train:
            del scaled_loss

    denom = max(1, batch_count)
    if oom_skipped:
        print(f"  NOTE: {oom_skipped} batch(es) skipped this pass due to CUDA OOM.")
    return {"text": total_text / denom, "speech": total_speech / denom,
            "total": total_loss / denom, "oom_skipped": oom_skipped}


def build_scheduler(args, optimizer, steps_per_epoch: int):
    if args.scheduler == "none":
        return None
    builder = {"linear": get_linear_schedule_with_warmup,
               "cosine": get_cosine_schedule_with_warmup}.get(args.scheduler)
    if builder is None:
        raise RuntimeError(f"{args.scheduler} scheduler requested, but the transformers "
                           "scheduler utility is unavailable.")
    total_steps = max(1, steps_per_epoch * args.max_epochs)
    warmup = min(args.warmup_steps, max(0, total_steps - 1))
    print(f"Scheduler: {args.scheduler}, {warmup} warmup / {total_steps} total optimizer steps")
    return builder(optimizer, num_warmup_steps=warmup, num_training_steps=total_steps)


def log_epoch_summary(prefix: str, metrics: dict):
    print(f"{prefix} | text={metrics['text']:.4f} speech={metrics['speech']:.4f} total={metrics['total']:.4f}")


def parse_args():
    parser = argparse.ArgumentParser(description="Nano Hindi fine-tuning")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--reference-audio", type=Path, default=DEFAULT_REFERENCE_AUDIO)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--token-cache", type=Path, default=None, help="Defaults to <output-dir>/token_cache.pt")
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Run one batch only and do not save checkpoints.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Tuned for 24GB. Use 1 with --grad-accum-steps 8 on a 4GB card. "
                             "Confirm with --dry-run, which measures the worst-case batch.")
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--max-epochs", type=int, default=None, help="Required for actual training runs.")
    parser.add_argument("--train-mode", choices=["lora", "full"], default="full",
                        help="'full' trains all of T3. 'lora' is the low-VRAM fallback.")
    parser.add_argument("--lr", type=float, default=None,
                        help="Defaults to 2e-5 for --train-mode full, 2e-4 for lora.")
    parser.add_argument("--emb-lr", type=float, default=None,
                        help="text_emb learning rate. Defaults to --lr in full mode (uniform, the "
                             "standard choice) and 1e-4 in lora mode, where the backbone is frozen "
                             "and the embedding is one of only two things learning.")
    parser.add_argument("--text-loss-weight", type=float, default=0.2)
    parser.add_argument("--scheduler", choices=["none", "linear", "cosine"], default="cosine",
                        help="A full fine-tune is unstable at a flat LR; cosine + warmup is the safe "
                             "default.")
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--early-stop-patience", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=False,
                        help="Trades ~30%% speed for a large activation-memory saving. Off for 24GB; "
                             "turn it ON for a 4GB card.")
    parser.add_argument("--precision", choices=["auto", "bf16", "fp16", "fp32"], default="auto",
                        help="auto picks bf16 where supported. bf16 has fp32's exponent range, so it "
                             "needs no GradScaler and cannot overflow the way fp16 does.")
    parser.add_argument("--max-grad-norm", type=float, default=1.0,
                        help="Gradient clipping. Matters far more for a full fine-tune than for LoRA; "
                             "0 disables.")
    parser.add_argument("--length-bucketing", action=argparse.BooleanOptionalAction, default=True,
                        help="Batch clips of similar length together so padding does not dominate "
                             "compute. Only meaningful when --batch-size > 1.")
    parser.add_argument("--cache-device", choices=["auto", "cuda", "cpu"], default="auto",
                        help="Device for the one-time token-cache build. GPU is ~8x faster.")
    parser.add_argument("--generate-audio", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--generate-every", type=int, default=1, help="Generate eval samples every N epochs.")
    parser.add_argument("--train-text-emb", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--train-text-head", action=argparse.BooleanOptionalAction, default=False,
                        help="Also train text_head (38.6M params). It only feeds the auxiliary text "
                             "loss, so this buys little; off by default even in full mode.")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-targets", type=str, default="c_attn,c_proj,c_fc")
    parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--multi-speaker", action=argparse.BooleanOptionalAction, default=False,
                        help="Condition each clip on its own voice embedding. Required for "
                             "multi-speaker corpora; a single fixed reference would teach the "
                             "model to ignore speaker conditioning.")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    if not args.manifest.exists():
        raise FileNotFoundError(f"Manifest not found: {args.manifest}")
    if not args.reference_audio.exists():
        raise FileNotFoundError(f"Reference audio not found: {args.reference_audio}")
    if not args.dry_run and args.max_epochs is None:
        raise ValueError("--max-epochs is required for actual training runs.")
    if args.token_cache is None:
        args.token_cache = args.output_dir / "token_cache.pt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = resolve_amp_dtype(args.precision, device)
    use_amp = amp_dtype is not None
    # Only fp16 needs loss scaling; bf16 spans the same exponent range as fp32.
    needs_scaler = amp_dtype == torch.float16

    if args.lr is None:
        args.lr = 2e-5 if args.train_mode == "full" else 2e-4
        print(f"--lr not given; using {args.lr} for --train-mode {args.train_mode}")
    if args.emb_lr is None:
        # In full mode everything trains together, so a text_emb LR several times
        # the backbone's would let the embeddings drift faster than the layers
        # reading them can adapt.
        args.emb_lr = args.lr if args.train_mode == "full" else 1e-4
        print(f"--emb-lr not given; using {args.emb_lr} for --train-mode {args.train_mode}")

    print(f"Device: {device}"
          + (f" ({torch.cuda.get_device_name(0)}, "
             f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.0f}GB)"
             if device.type == "cuda" else ""))
    print(f"Precision: {str(amp_dtype).replace('torch.', '') if use_amp else 'fp32'}"
          f" (GradScaler: {needs_scaler})")
    print(f"Train mode: {args.train_mode}")
    print(f"Manifest: {args.manifest}")
    print(f"Output dir: {args.output_dir}")

    rows = load_manifest_rows(args.manifest)
    split_gen = torch.Generator().manual_seed(args.split_seed)
    perm = torch.randperm(len(rows), generator=split_gen).tolist()
    val_count = max(1, int(len(rows) * args.val_ratio))
    val_indices = perm[:val_count]
    train_indices = perm[val_count:]

    # Resume must reuse the ORIGINAL train/val split, and it has to happen
    # before the DataLoaders are built. The split comes from
    # randperm(len(rows)), so if the manifest grew since the checkpoint was
    # written, recomputing it silently reshuffles everything and leaks
    # previously-held-out val rows into train -- which makes the resumed val
    # loss (and therefore early stopping) meaningless.
    resumed_payload = None
    if args.resume_from is not None:
        trainer_state_path = args.resume_from / "trainer_state.pt"
        if trainer_state_path.exists():
            resumed_payload = torch.load(trainer_state_path, map_location="cpu", weights_only=False)
            saved_state = resumed_payload.get("state", {})
            saved_train = saved_state.get("train_indices")
            saved_val = saved_state.get("val_indices")
            if saved_train and saved_val:
                saved_rows = len(saved_train) + len(saved_val)
                max_saved_idx = max(max(saved_train), max(saved_val))
                if max_saved_idx >= len(rows):
                    raise RuntimeError(
                        f"Cannot resume: checkpoint's split references row {max_saved_idx} but the "
                        f"manifest now has only {len(rows)} rows. The manifest shrank or was "
                        f"reordered, so the saved split no longer maps to the same audio. "
                        f"Start a fresh run with a new --output-dir instead of resuming."
                    )
                if saved_rows != len(rows):
                    print(
                        f"\n*** MANIFEST GREW since this checkpoint: {saved_rows} -> {len(rows)} rows.\n"
                        f"*** Reusing the SAVED split, so the {len(rows) - saved_rows} new row(s) are "
                        f"IGNORED for this resumed run.\n"
                        f"*** This keeps val loss comparable across the resume. To actually train on "
                        f"the larger dataset, start a FRESH run (new --output-dir, no --resume-from).\n"
                    )
                train_indices = saved_train
                val_indices = saved_val
                print(f"Restored split from checkpoint: train {len(train_indices)}, val {len(val_indices)}")
        else:
            print("No trainer_state.pt found in resume directory; loading weights only.")

    print(f"Loaded {len(rows)} rows -> train {len(train_indices)}, val {len(val_indices)}")

    model = build_nano_model(args, device)
    trainable_count = check_sanity(model, args)

    cache_device = args.cache_device
    if cache_device == "auto":
        cache_device = "cuda" if torch.cuda.is_available() else "cpu"
    items = build_or_load_token_cache(args.token_cache, rows, model,
                                      multi_speaker=args.multi_speaker,
                                      cache_device=cache_device)

    # Report which text token ids the dataset actually uses.
    used_ids = set()
    for item in items:
        used_ids.update(item["text_tokens"].tolist())
    print(f"Unique text token ids in dataset: {len(used_ids)}")

    # Reference conditioning (built once; ve/s3gen run on CPU).
    with torch.no_grad():
        model.prepare_conditionals(str(args.reference_audio), exaggeration=0.0, norm_loudness=True)
    reference_cond = copy.deepcopy(model.conds)
    reference_cond_t3 = copy.deepcopy(reference_cond.t3).to(device=device)

    if args.multi_speaker:
        n_spk = sum(1 for it in items if "speaker_emb" in it)
        print(f"Multi-speaker conditioning ON: {n_spk}/{len(items)} clips have a voice embedding.")
        print("Speech cond prompt is disabled during training (see batch_t3_cond).")

    # Free what we no longer need to keep RAM in check.
    model.ve = None
    if not args.generate_audio:
        model.s3gen = None
    gc.collect()

    dataset = CachedTokenDataset(items)
    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)
    loader_kwargs = dict(
        num_workers=args.num_workers,
        collate_fn=collate_batch,
        pin_memory=(args.pin_memory and device.type == "cuda"),
    )

    # Total sequence length drives both memory and compute, so bucket on it.
    train_sampler = val_sampler = None
    if args.length_bucketing and args.batch_size > 1:
        train_lens = [items[i]["speech_tokens"].numel() + items[i]["text_tokens"].numel()
                      for i in train_indices]
        val_lens = [items[i]["speech_tokens"].numel() + items[i]["text_tokens"].numel()
                    for i in val_indices]
        train_sampler = LengthBucketSampler(train_lens, args.batch_size, shuffle=True, seed=args.seed)
        val_sampler = LengthBucketSampler(val_lens, args.batch_size, shuffle=False)
        pad_waste = 1 - (sum(train_lens) / len(train_lens)) / max(
            1, max(train_lens))
        print(f"Length bucketing ON (batch {args.batch_size}): clip lengths "
              f"{min(train_lens)}-{max(train_lens)} tokens, mean "
              f"{sum(train_lens) / len(train_lens):.0f}. Without bucketing every batch "
              f"would pad toward the max ({pad_waste:.0%} waste worst case).")
        train_loader = DataLoader(train_subset, batch_sampler=train_sampler, **loader_kwargs)
        val_loader = DataLoader(val_subset, batch_sampler=val_sampler, **loader_kwargs)
    else:
        train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, **loader_kwargs)
        val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, **loader_kwargs)

    param_groups = build_param_groups(model, args)
    optimizer = torch.optim.AdamW(param_groups, foreach=False)
    steps_per_epoch = max(1, math.ceil(len(train_loader) / args.grad_accum_steps))
    scheduler = build_scheduler(args, optimizer, steps_per_epoch) if not args.dry_run else None
    scaler = torch.amp.GradScaler("cuda", enabled=needs_scaler)

    run_state = RunState(train_indices=train_indices, val_indices=val_indices)
    if resumed_payload is not None:
        loaded_state = resumed_payload.get("state", {})
        run_state.epoch = int(loaded_state.get("epoch", 0))
        run_state.best_val_speech_loss = float(loaded_state.get("best_val_speech_loss", float("inf")))
        run_state.no_improve_epochs = int(loaded_state.get("no_improve_epochs", 0))
        print(f"Resumed trainer state: epoch={run_state.epoch}, best_val_speech_loss={run_state.best_val_speech_loss}")
        if resumed_payload.get("optimizer_state_dict") is not None:
            optimizer.load_state_dict(resumed_payload["optimizer_state_dict"])
        if scheduler is not None and resumed_payload.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(resumed_payload["scheduler_state_dict"])
        # Restoring the GradScaler keeps the AMP loss scale warm; without it the
        # first few steps after a resume can be skipped while it re-calibrates.
        if needs_scaler and resumed_payload.get("scaler_state_dict") is not None:
            scaler.load_state_dict(resumed_payload["scaler_state_dict"])
        if args.max_epochs is not None and run_state.epoch >= args.max_epochs:
            raise RuntimeError(
                f"Nothing to do: resuming at epoch {run_state.epoch} but --max-epochs is "
                f"{args.max_epochs}. Pass a larger --max-epochs to continue training."
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(vars(args)), handle, ensure_ascii=False, indent=2)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    if args.dry_run:
        print("\n=== DRY RUN ===")
        # Peak memory is set by the LONGEST batch, not a random one. Benchmarking
        # an average batch is exactly what hid a real OOM until 1250 steps into a
        # multi-hour run, so build the worst case on purpose.
        worst = sorted(
            train_indices,
            key=lambda i: items[i]["speech_tokens"].numel() + items[i]["text_tokens"].numel(),
            reverse=True,
        )[:args.batch_size]
        batch = move_batch_to_device(collate_batch([items[i] for i in worst]), device)
        print(f"Worst-case batch: {len(worst)} longest clips -> "
              f"{int(batch['speech_tokens'].size(1))} speech + "
              f"{int(batch['text_tokens'].size(1))} text tokens (padded).")
        optimizer.zero_grad(set_to_none=True)
        autocast_ctx = (torch.autocast(device_type="cuda", dtype=amp_dtype)
                        if use_amp else nullcontext())
        with autocast_ctx:
            loss_text, loss_speech, loss_total = compute_batch_losses(model, batch, reference_cond_t3, args.text_loss_weight)
        print(f"Dry run loss_text: {loss_text.item():.6f}")
        print(f"Dry run loss_speech: {loss_speech.item():.6f}")
        print(f"Dry run loss_total: {loss_total.item():.6f}")
        if not torch.isfinite(loss_total):
            raise RuntimeError("Dry-run loss is not finite.")

        if needs_scaler:
            scaler.scale(loss_total).backward()
            scaler.unscale_(optimizer)
        else:
            loss_total.backward()

        grad_params, grads_finite = grad_finite_check(model)
        print(f"Dry run params with grads: {grad_params}")
        print(f"Dry run gradients finite: {grads_finite}")
        if not grads_finite and needs_scaler:
            # GradScaler starts at a loss scale of 65536 and deliberately
            # overflows-then-halves on the first steps; run_epoch lets it skip
            # those. So an fp16 overflow here is calibration, not breakage.
            # Re-check in fp32 to tell the two apart.
            print("Non-finite under fp16 -- re-checking in fp32 to distinguish "
                  "AMP calibration from a real problem...")
            optimizer.zero_grad(set_to_none=True)
            loss_text, loss_speech, loss_total = compute_batch_losses(
                model, batch, reference_cond_t3, args.text_loss_weight)
            loss_total.backward()
            grad_params, grads_finite = grad_finite_check(model)
            print(f"Dry run fp32 gradients finite: {grads_finite}")
            if grads_finite:
                print("OK: gradients are finite in fp32, so this is normal AMP "
                      "loss-scale calibration. GradScaler will skip those steps.")
            optimizer.zero_grad(set_to_none=True)
        if not grads_finite:
            raise RuntimeError("Dry-run gradients are not finite.")

        if needs_scaler:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        if torch.cuda.is_available():
            print(f"Dry run peak CUDA allocated MB: {bytes_to_mb(torch.cuda.max_memory_allocated()):.2f}")
            print(f"Dry run peak CUDA reserved MB: {bytes_to_mb(torch.cuda.max_memory_reserved()):.2f}")
        print("DRY RUN COMPLETE")
        return

    best_ckpt_dir = None
    for epoch in range(run_state.epoch + 1, args.max_epochs + 1):
        epoch_start = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        print(f"\n=== Epoch {epoch}/{args.max_epochs} ===")
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        train_metrics = run_epoch(
            model, train_loader, optimizer, scaler if needs_scaler else None, device,
            reference_cond_t3, max(1, args.grad_accum_steps), True, scheduler, args.text_loss_weight,
            amp_dtype=amp_dtype, max_grad_norm=args.max_grad_norm,
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        val_metrics = run_epoch(
            model, val_loader, None, None, device,
            reference_cond_t3, 1, False, None, args.text_loss_weight,
            amp_dtype=amp_dtype,
        )

        epoch_duration = time.perf_counter() - epoch_start
        log_epoch_summary("train", train_metrics)
        log_epoch_summary("val  ", val_metrics)
        print(f"epoch_duration_sec: {epoch_duration:.2f}")
        if torch.cuda.is_available():
            print(f"peak_cuda_allocated_mb: {bytes_to_mb(torch.cuda.max_memory_allocated()):.2f}")

        epoch_metrics = {
            "epoch": epoch,
            "train_text_loss": train_metrics["text"],
            "train_speech_loss": train_metrics["speech"],
            "train_total_loss": train_metrics["total"],
            "val_text_loss": val_metrics["text"],
            "val_speech_loss": val_metrics["speech"],
            "val_total_loss": val_metrics["total"],
            "epoch_duration_sec": epoch_duration,
            "trainable_params": trainable_count,
            "train_oom_skipped": train_metrics.get("oom_skipped", 0),
            "val_oom_skipped": val_metrics.get("oom_skipped", 0),
        }
        append_metrics_log(args.output_dir, epoch_metrics)

        run_state.epoch = epoch
        val_speech = val_metrics["speech"]
        if val_speech < run_state.best_val_speech_loss:
            run_state.best_val_speech_loss = val_speech
            run_state.no_improve_epochs = 0
            best_ckpt_dir = save_checkpoint(model, args.output_dir, run_state, optimizer, scheduler, args, epoch_metrics, "best", scaler=scaler)
            print(f"Best validation checkpoint updated: {best_ckpt_dir}")
        else:
            run_state.no_improve_epochs += 1
            print(f"No val improvement for {run_state.no_improve_epochs} epoch(s).")

        save_checkpoint(model, args.output_dir, run_state, optimizer, scheduler, args, epoch_metrics, "latest", scaler=scaler)

        if args.generate_audio and (epoch % args.generate_every == 0):
            try:
                maybe_generate_samples(model, reference_cond, args.output_dir, device, epoch)
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    print("Skipping epoch audio generation due to CUDA OOM.")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                else:
                    raise

        if args.early_stop_patience is not None and run_state.no_improve_epochs >= args.early_stop_patience:
            print("Early stopping: validation speech loss stopped improving.")
            break

    print("\nTraining complete.")
    print(f"Best checkpoint: {best_ckpt_dir}")
    print(f"Best val speech loss: {run_state.best_val_speech_loss:.4f}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("\nTRAINING SCRIPT FAILED")
        print(f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
        raise
