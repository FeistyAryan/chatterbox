# Teaching Chatterbox Nano to speak Hindi

Fine-tunes Nano's **T3** text-to-speech-token model on a Hindi dataset. S3Gen (vocoder)
and the voice encoder stay frozen, so zero-shot voice cloning from a reference clip
keeps working as-is.

Scripts:

| Script | Purpose |
| --- | --- |
| `train_nano_exp1.py` | LoRA / full fine-tune of T3, with resume support |
| `infer_nano_exp1.py` | Generate audio from a checkpoint |
| `codec_roundtrip_test.py` | Diagnostic: proves the S3 speech codec preserves Hindi |
| `nano_smoke_test.py`, `baseline_nano.py` | Early exploration / baseline scratch scripts |

---

## Non-obvious things a reviewer should know

These were found by reading the model source, and each one silently produces a
broken-but-still-training model. They are the highest-value part of this diff to review.

1. **The repo's `T3.loss()` is unshifted.** It compares `logits[k]` against `tokens[k]`
   instead of `tokens[k+1]`, which turns training into an input-copying task rather than
   next-token prediction. `shifted_masked_ce()` does it correctly.

2. **`hp.start_text_token=255` / `hp.stop_text_token=0` must NOT be used for Nano.** Those
   ids belong to the legacy 704-token `EnTokenizer`. Nano uses a GPT-2 tokenizer
   (vocab 50276) and `generate()` feeds raw tokenizer ids with no wrapper tokens. Adding
   them during training creates a train/inference mismatch. `T3.forward()` asserts they're
   present, so `_ensure_BOT_EOT` is monkeypatched to a no-op.

3. **`text_emb` must be trainable.** Hindi hits the GPT-2 tokenizer's byte-fallback path,
   and those embedding rows are effectively untrained in the released checkpoint. It's
   trained with `weight_decay=0` specifically so AdamW does not decay the ~50k *unused*
   English rows — only the ~95 rows this dataset actually touches move.

4. **`t3.inference()` crashes on Nano** (`speech_pos_emb is None`). Use `inference_turbo()`
   everywhere.

5. **Speech token ranges:** valid S3 codec ids are `[0, 6561)`, with BOS=6561, EOS=6562, at
   25 tokens/sec. Because the vocab is 6563-way, the random-baseline cross-entropy is
   `ln(6563) ≈ 8.79` nats — useful for judging whether a loss value is actually good.

---

## Setup

```bash
source venv/bin/activate
```

The manifest is `audio_path|transcript`, one row per line, produced by the separate
`yt_scraper` pipeline (ASR-verify → align → clean → DNSMOS filter). Default paths point at
a local dataset; override with `--manifest` / `--reference-audio`.

## Verify before training

```bash
python train_nano_exp1.py --dry-run
```

Runs one batch and asserts the loss is finite, gradients are finite, and only the intended
parameters are trainable. Also prints peak VRAM so you can confirm it fits.

## Train

```bash
python train_nano_exp1.py --max-epochs 50
```

Defaults are tuned for a 4GB card: batch size 1, grad-accum 8, AMP fp16, gradient
checkpointing on, S3Gen/voice-encoder freed after setup. Peak usage ≈1.6GB.

Outputs, under `--output-dir` (default `experiments/nano_exp1/`):

- `checkpoints/best/` — lowest **val speech loss** (this is the one to deploy)
- `checkpoints/latest/` — most recent epoch (used for resuming)
- `metrics.jsonl` — per-epoch losses
- `token_cache.pt` — derived, gitignored, rebuilt automatically

## Stop and resume later

Every epoch writes `checkpoints/latest/`, which contains model weights **plus** optimizer,
scheduler, and AMP-scaler state. To continue:

```bash
python train_nano_exp1.py \
  --resume-from experiments/nano_exp1/checkpoints/latest \
  --max-epochs 100
```

`--max-epochs` is an absolute target, not "how many more" — it must exceed the resumed
epoch or the run exits with an explanatory error.

Verified working: resuming at epoch 50 continued the train speech loss smoothly
(3.8796 → 3.8007 → 3.7431) with no spike, confirming optimizer momentum was restored.

### Resume reuses the saved train/val split, deliberately

The split comes from `randperm(len(rows))`, so if the manifest grew since the checkpoint was
written, recomputing it would reshuffle everything and leak previously-held-out val rows
into train — making the resumed val loss (and early stopping) meaningless. So resume
restores the saved indices and **ignores new manifest rows**, printing a loud warning.

**To train on a larger dataset, start a fresh run with a new `--output-dir`** — don't
resume. A new dataset deserves a fresh split and a fresh token cache.

If the manifest *shrank* or was reordered, resume aborts instead of silently mismatching
indices to different audio.

## Inference

```bash
python infer_nano_exp1.py \
  --checkpoint experiments/nano_exp1/checkpoints/best \
  --text "आज मौसम बहुत अच्छा है।" --out out.wav
```

`--max-new-tokens` (default 300) caps generation. This matters: the high-level
`model.generate()` hardcodes an internal limit of 1000 tokens, and at low temperature a
weakly-trained model may never emit EOS, run to the cap, and OOM S3Gen's flow encoder. This
script calls `inference_turbo()` + `s3gen.inference()` directly to keep that bounded, and
reports whether EOS was actually reached.

---

## Current status

Trained on **24.4 minutes / 284 clips** (one speaker, one source video) as a pilot.

- Best val speech loss **5.946** at epoch 10 (from ~6.86 at init).
- Training past that overfits as expected on so little data: by epoch 50 train speech loss
  reached 3.88 while val rose monotonically to 7.25.
- Audio: voice cloning/prosody is clearly working; **Hindi words are not yet intelligible.**

### Why, and what was ruled out

Loss of ~5.9 against a random baseline of 8.79 means the model is far from converged — this
is a data-volume problem, not a bug. Both plausible "the pipeline is broken" theories were
tested and cleared:

- **Text tokenizer** — GPT-2 BPE on Devanagari is *inefficient* (byte-fallback, ~1.5
  tokens/char, only ~95 distinct ids used) but lossless. Under matched near-greedy decoding,
  epoch 50 emitted EOS confidently at 98 tokens where epoch 10 ran past a 300-token cap
  without terminating — the same tokenizer measurably improving with training.
- **Speech tokenizer (S3 codec)** — `codec_roundtrip_test.py` encodes real Hindi audio to
  codec tokens and decodes straight back, bypassing T3 entirely. Result is ~98-99%
  perceptually identical to the original, so the codec represents Hindi phonemes fine.

Do **not** swap in the multilingual model's 2454-token tokenizer: it occupies a different id
space than Nano's `text_emb` / `text_head`, so it would require reinitializing and
retraining both from scratch.

## Training on the Vaani corpus (multi-speaker)

`/home/aryan/Desktop/vaani_hindi_dataset/` holds **Vaani** (IISc/Google): 39 parquet
shards, 18 GB, ~82k rows. It is an **ASR** corpus of spontaneous field recordings, not a
TTS corpus, so most of it is unusable as-is. Measured per shard:

| Problem | Share |
| --- | --- |
| Noise-tagged (`<noise>`, `<static_noise>`, `<horn>`, `<people_talking>`) | 31% |
| Truncated mid-word (trailing `--`) | 29% |
| Not pure Hindi (Bhojpuri / Chhattisgarhi / Maithili / Magahi dialects) | 21% |
| `[unintelligible]` | 5% |

Transcripts also carry annotation markup and `{brace}` spans that are *either* a Devanagari
spelling correction (use it, drop the word before) *or* an English gloss (drop it, keep the
Hindi). `prepare_vaani_dataset.py` handles all of this:

```bash
python prepare_vaani_dataset.py --target-hours 20
```

Keep rate is ~19%. The 20 h run produced **10,069 clips / 6,189 speakers / mean 7.15 s**
(2.2 GB) plus a manifest in the usual `audio_path|transcript` format.

### Why `--multi-speaker` is mandatory here

Conditioning was previously built **once** from a single reference file and reused for every
example. That is correct for a single-speaker corpus and actively harmful across 6k
speakers: the model is told "this is speaker A" while being asked to predict speaker B, so
it learns to ignore speaker conditioning altogether and voice cloning degrades.

`--multi-speaker` instead gives every clip its own 256-d voice-encoder embedding, computed
once at tokenization time and cached alongside the tokens.

**The speech cond prompt is deliberately disabled in this mode.** `speech_cond_prompt_len`
is 375 tokens (15 s), longer than most clips, so prompting a clip with its own audio would
hand the model the exact tokens it must predict — it would learn to copy the prompt and
ignore the text. Prompting from a *different* clip by the same speaker isn't an option
either: Vaani averages ~1.27 clips per speaker. VALL-E-style prefix prompting would need the
text trimmed to match the audio suffix, and there's no word-level alignment. So the model
conditions on the speaker embedding alone, which carries identity rather than content and
therefore leaks nothing.

`infer_nano_exp1.py --multi-speaker` nulls the same fields so inference matches training.

Verified: embeddings are distinct per clip (cosine 0.65-0.78 across speakers), and speech
loss orders correctly — correct speaker 8.0098 < wrong speaker 8.0467 < zeros 8.0559 —
confirming the conditioning path is live rather than ignored.

### Measured cost on a 4GB card

Worst case (15 s clips): batch 1 = 1537 MB, batch 2 = 1804 MB, batch 4 = 2043 MB peak. All
fit. Token cache build is ~0.235 s/clip, so ~40 min one-time for 10k clips.

### Next step

More data. Rough estimate: **~20-50h for first recognizable words, ~50-150h for fluency.**
~200-300h of raw source video is available to run through the `yt_scraper` pipeline. At that
scale, switch to `--train-mode full` (with `--lr 2e-5`) and rent a larger GPU; the 4GB local
card is a pilot-scale machine.
