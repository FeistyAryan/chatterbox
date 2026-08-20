# Teaching Chatterbox Nano to speak Hindi

Fine-tunes Nano's **T3** text-to-speech-token model on Hindi. S3Gen (the vocoder) and the
voice encoder stay frozen, so zero-shot voice cloning keeps working — including
cross-lingual cloning from an English reference (measured below).

| Script | Purpose |
| --- | --- |
| `prepare_vaani_dataset.py` | Vaani parquet shards → wavs + manifest |
| `train_nano_exp1.py` | Full / LoRA fine-tune of T3, with resume support |
| `infer_nano_exp1.py` | Generate audio from a checkpoint |
| `codec_roundtrip_test.py` | Diagnostic: proves the S3 speech codec preserves Hindi |

---

## Non-obvious things a reviewer should know

Each of these silently produces a broken-but-still-training model. They are the
highest-value part of this work to review.

1. **The repo's `T3.loss()` is unshifted.** It compares `logits[k]` against `tokens[k]`
   instead of `tokens[k+1]`, turning training into an input-copying task rather than
   next-token prediction. `shifted_masked_ce()` does it correctly.

2. **`hp.start_text_token=255` / `hp.stop_text_token=0` must NOT be used for Nano.** Those
   ids belong to the legacy 704-token `EnTokenizer`. Nano uses a GPT-2 tokenizer
   (vocab 50276) and `generate()` feeds raw tokenizer ids with no wrapper tokens. Adding
   them during training creates a train/inference mismatch. `T3.forward()` asserts they're
   present, so `_ensure_BOT_EOT` is monkeypatched to a no-op.

3. **`text_emb` must be trainable.** Hindi hits the GPT-2 tokenizer's byte-fallback path,
   and those embedding rows are effectively untrained in the released checkpoint. It uses
   `weight_decay=0` specifically so AdamW does not decay the ~47k *unused* English rows —
   under AdamW, decay applies every step regardless of gradient.

4. **`t3.inference()` crashes on Nano** (`speech_pos_emb is None`). Use `inference_turbo()`
   everywhere.

5. **Speech token ranges:** valid S3 codec ids are `[0, 6561)`, with BOS=6561, EOS=6562, at
   25 tokens/sec. The vocab is 6563-way, so the random-baseline cross-entropy is
   **`ln(6563) = 8.79` nats**. Judge every loss against that number: `exp(loss)` is the
   effective number of choices the model is still deciding between.

---

## The dataset

`prepare_vaani_dataset.py` converts the Vaani corpus (IISc / Google) into an
`audio_path|transcript` manifest.

```bash
python prepare_vaani_dataset.py --keep-all --target-hours 1000 \
  --src /path/to/vaani/hindi \
  --out data/vaani_hindi
```

Full extraction, as committed to `data/vaani_hindi/` (gitignored, 19 GB):

| | |
| --- | --- |
| Clips | **81,951** |
| Hours | **166.77** |
| Speakers | **20,223** |
| Mean duration | 7.33 s |
| Keep rate | 100% (2 decode errors across the whole corpus) |

### `--keep-all` keeps the transcripts verbatim, on purpose

Vaani transcripts carry markup like `<noise>`, `<birds_chirping>`, `<pause>`,
`[breathing]`, and `{brace}` spans. **These are acoustic-event conditioning labels, not
words to pronounce.** `<noise>` co-occurs with a hissy background; `<birds_chirping>` with
birdsong. No audio aligns to the tag itself, so the model learns it as a background
condition it can reproduce on request — the same mechanism as Bark's `[laughter]`.
Stripping them would throw away controllability.

Audio is written as the **original bytes** — no decode, resample, or normalisation. The
clip on disk is byte-identical to the clip as downloaded.

The legacy filtered mode (drop the `--keep-all` flag) rejects noisy / truncated / dialectal
rows and strips markup; it keeps ~19%. It exists for building a clean single-style corpus,
but the verbatim path is what the trained models use.

**Cost of verbatim text:** 3,023 distinct token ids (vs 138 filtered) and a max transcript
of 734 tokens (vs 485). That difference is what caused an OOM 1250 steps into a run — see
*Sizing* below.

---

## Setup

```bash
source venv/bin/activate
```

## Verify before training — always

```bash
python train_nano_exp1.py --dry-run --manifest data/vaani_hindi/manifest.txt --multi-speaker
```

Runs one batch and asserts the loss is finite, gradients are finite, and only the intended
parameters are trainable.

**The dry run deliberately builds the worst-case batch** — the `--batch-size` longest clips
in the split, not a random one. Peak memory is set by the longest batch, and benchmarking
an average batch is exactly what hid a real OOM until 1250 steps into a multi-hour run.
The printed peak is the number to trust.

## Train

Full fine-tune is now the default. On a 24 GB card:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python train_nano_exp1.py \
  --manifest data/vaani_hindi/manifest.txt \
  --reference-audio data/vaani_hindi/wavs/000000.wav \
  --output-dir experiments/vaani_full \
  --multi-speaker --max-epochs 20
```

On a 4 GB card, the defaults will not fit — add:

```bash
  --batch-size 1 --grad-accum-steps 8 --gradient-checkpointing --train-mode lora
```

### Defaults, and why

| Flag | Default | Reason |
| --- | --- | --- |
| `--train-mode` | `full` | LoRA was a 4 GB workaround. With 24 GB, train all 140.2M params. |
| `--lr` | auto | 2e-5 for full, 2e-4 for lora. |
| `--emb-lr` | auto | Matches `--lr` in full mode. A text_emb LR several times the backbone's would let embeddings drift faster than the layers reading them can adapt. In lora mode it stays 1e-4, where the frozen backbone makes it one of only two things learning. |
| `--precision` | `auto` | Picks **bf16** where supported. bf16 has fp32's exponent range, so it needs no GradScaler and cannot overflow the way fp16 did at `lora_r=64`. |
| `--max-grad-norm` | `1.0` | LoRA is intrinsically constrained by its low rank. A full fine-tune has no such guardrail: one outlier batch can wreck the weights in a single step, and it does not show up until val loss jumps an epoch later. |
| `--scheduler` | `cosine` | A full fine-tune is unstable at a flat LR. |
| `--warmup-steps` | `500` | Same reason. |
| `--batch-size` | `8` | Sized for 24 GB. **Confirm with `--dry-run` before committing to a long run.** |
| `--gradient-checkpointing` | **off** | Costs ~30% speed to save activation memory. Unnecessary at 24 GB; turn it **on** for 4 GB. |
| `--length-bucketing` | on | See below. |
| `--train-text-head` | off | text_head only feeds the auxiliary text loss (weight 0.2), so training its 38.6M params buys little even in "full" mode. |

### Length bucketing

Every batch pads to its longest member. Vaani clip lengths span roughly 64–1174 tokens, so
pairing a short clip with a long one throws most of the compute away on padding. The
sampler shuffles globally, cuts into large chunks, sorts each chunk by length, emits
batches, then shuffles the batch *order*. Randomness is preserved at the batch level; only
the composition of a batch is length-biased. Only meaningful when `--batch-size > 1`.

### Token cache

All text tokens, speech tokens, and (in `--multi-speaker`) per-clip voice embeddings are
precomputed once into `<output-dir>/token_cache.pt`, so epochs never touch audio.

The S3 tokenizer is the slow part. It now runs on the **GPU** during the one-time build:
**0.065 s/clip vs 0.235 s/clip on CPU** — for 82k clips that is ~90 min instead of ~5.4 h.

The cache is keyed on **row count AND `multi_speaker`**. If either differs it silently
rebuilds — so a resume that forgets `--multi-speaker` rebuilds the cache *and* trains with
the wrong conditioning.

### OOM handling

A batch that does not fit is skipped, reported, and counted in `metrics.jsonl` as
`train_oom_skipped` rather than killing the run. The guard wraps **both forward and
backward** — peak memory lands during backward, so that is where an oversized batch
actually dies.

On an OOM the accumulation window is zeroed. A half-finished backward leaves gradients for
only part of the batch, and stepping on those would apply a silently mis-scaled update.

Note that a cuBLAS allocation failure (`CUBLAS_STATUS_ALLOC_FAILED`) is a plain
`RuntimeError`, **not** `torch.cuda.OutOfMemoryError`, so it is not caught — it means the
card is comprehensively out of memory, and recovery is not reliable.

## Outputs

Under `--output-dir`:

- `checkpoints/best/` — lowest **val speech loss** (deploy this one)
- `checkpoints/latest/` — most recent epoch (used for resuming)
- `metrics.jsonl` — per-epoch losses, OOM counts, duration
- `token_cache.pt` — derived, gitignored, rebuilt automatically

## Stop and resume

`checkpoints/latest/` holds model weights **plus** optimizer, scheduler, and AMP-scaler
state.

```bash
python train_nano_exp1.py --resume-from experiments/vaani_full/checkpoints/latest \
  --max-epochs 40 ...
```

`--max-epochs` is an **absolute target**, not "how many more" — it must exceed the resumed
epoch or the run exits with an explanatory error.

Verified: a full-mode restart continued train speech loss 6.3313 → 6.2037 → 6.1196 with no
spike, confirming optimizer momentum was restored.

### Resume reuses the saved train/val split, deliberately

The split comes from `randperm(len(rows))`. If the manifest grew since the checkpoint was
written, recomputing it would reshuffle everything and leak previously-held-out val rows
into train — making the resumed val loss and early stopping meaningless. So resume restores
the saved indices and **ignores new manifest rows**, printing a loud warning. If the
manifest shrank or was reordered, resume aborts rather than silently mismatching indices to
different audio.

**To train on a larger dataset, start a fresh run with a new `--output-dir`** — don't
resume. A new dataset deserves a fresh split and a fresh token cache.

---

## Multi-speaker conditioning

`--multi-speaker` gives every clip its own 256-d voice-encoder embedding, computed once at
tokenization time and cached with the tokens.

Without it, conditioning is built **once** from a single reference file and reused for
every example. That is correct for a single-speaker corpus and actively harmful across
20,223 speakers: the model is told "this is speaker A" while being asked to predict speaker
B, so it learns to ignore speaker conditioning altogether and voice cloning degrades.

**The speech cond prompt is deliberately disabled in this mode.**
`speech_cond_prompt_len` is 375 tokens (15 s), longer than most clips, so prompting a clip
with its own audio would hand the model the exact tokens it must predict — it would learn
to copy the prompt and ignore the text. Prompting from a *different* clip by the same
speaker isn't an option either: Vaani averages ~1.27 clips per speaker. VALL-E-style prefix
prompting would need the text trimmed to match the audio suffix, and there is no word-level
alignment.

So the model conditions on the speaker embedding alone. **That is not leakage**: a 256-d
vector encodes identity — timbre, pitch range, vocal tract — and cannot encode which words
to say. The 375-token cond prompt literally is the answer; the embedding is not.

`infer_nano_exp1.py --multi-speaker` nulls the same fields so inference matches training.

Verified: embeddings are distinct per clip (cosine 0.65–0.78 across speakers), and speech
loss orders correctly — correct speaker 8.0098 < wrong speaker 8.0467 < zeros 8.0559 —
confirming the conditioning path is live rather than ignored.

---

## Inference

```bash
python infer_nano_exp1.py \
  --checkpoint experiments/vaani_20h/checkpoints/best \
  --reference-audio my_voice.wav --multi-speaker \
  --text "आज मौसम बहुत अच्छा है।" --out out.wav
```

`--max-new-tokens` (default 300) caps generation. This matters: the high-level
`model.generate()` hardcodes an internal limit of 1000 tokens, and at low temperature a
weakly-trained model may never emit EOS, run to the cap, and OOM S3Gen's flow encoder. This
script calls `inference_turbo()` + `s3gen.inference()` directly to keep that bounded, and
reports whether EOS was actually reached.

The reference audio must be **longer than 5 seconds** (hard assert). S3Gen uses only the
first 10 s; the voice encoder averages over the whole clip. ~15–20 s is the sweet spot.

### Measured inference cost (446.31M params, fp32, nothing offloaded)

| | RTX 2050 (GPU) | Ryzen 7 7435HS (CPU) |
| --- | --- | --- |
| Model load | 5.01 s | 4.55 s |
| Real-time factor | **0.22** | **0.96** |
| Memory | **2098 MB VRAM** (incl. CUDA context) | 4.4 GB RSS |

The whole model loads with no offloading, no quantization, no `device_map`, and no meta
tensors; the LoRA adapter merges into the base weights so no PEFT wrapper remains at
inference.

CPU runs at real time on a laptop CPU, which makes GPU-free serving viable. Two caveats:
that is single-stream on 16 threads (concurrent requests contend for the same cores), and
4.4 GB is fp32 — bf16/int8 should roughly halve it but is untested.

Quote **2098 MB**, not the 1914 MB that `max_memory_allocated()` reports: that excludes the
CUDA context, which is exactly the gap that caused a training OOM.

### Cross-lingual voice cloning works

An 18.4 s **English** reference recording, generating **Hindi**, `--multi-speaker`:

| Metric | Cosine |
| --- | --- |
| Two different people (baseline) | 0.6552 |
| **Your voice → Hindi, vs your reference** | **0.8351** |
| Your voice → Hindi, vs a training speaker (leakage) | 0.6256 |
| Vaani speaker → Hindi, vs its reference (in-domain ceiling) | 0.8805 |

6/6 generations matched their intended reference. Leakage (0.6256) sits *below* the
different-person baseline, so 20k Hindi training speakers did **not** pull outputs toward a
generic average voice. 0.8351 against a 0.8805 in-domain ceiling is ~95%, on the hard case:
out-of-distribution English reference, no speech cond prompt.

This works because S3Gen is frozen and does most of the timbre rendering from `ref_dict`,
and it is language-agnostic — it maps codec tokens + reference timbre to a waveform without
caring that the tokens are Hindi and the reference is English. Only T3 was fine-tuned, and
T3 decides *what* is said, not *who* says it.

> Measuring this yourself: use the **pretrained** `model.ve`, not a bare `VoiceEncoder()`.
> A randomly-initialised encoder collapses every voice to nearly the same point and reports
> 0.999 for everything.

---

## Results so far

### Pilot — 24.4 min, 284 clips, one speaker

Best val speech loss 5.946 at epoch 10. Overfit as expected: by epoch 50 train reached 3.88
while val rose monotonically to 7.25. Voice cloning and prosody clearly worked; **Hindi
words were not intelligible.**

### Vaani 20 h — 9,900 clips, 7,397 speakers, LoRA r=32

| | |
| --- | --- |
| Trainable | 43,330,560 of 446.31M (LoRA 4.72M + text_emb 38.61M) |
| Best | **epoch 9, val speech loss 4.9685** (from 5.6972) |
| Epoch time | 30.9 min · **0 OOM skips** |

**This was the first run where val loss actually fell** — the pilot had val rising from
epoch 1. Same code, more data, opposite behaviour, which isolates data as the variable.

`exp(4.9685) ≈ 144` effective choices per step, against a random baseline of 6563. Real
progress, and still far from converged. Epochs 5→9 bought only 0.03 and then val turned up
while train kept falling — the run was **data-limited, not step-limited**, the reverse of
the pilot. All generated samples hit EOS cleanly and durations tracked text length.

### Next — 166.77 h, full fine-tune

That is the run the current defaults are built for.

---

## Sizing, and the lesson from the OOM

A batch died at step 1250/4455 of a 20 h run. Two causes:

1. Benchmarked on the *filtered* set (max clip 14.9 s, max text 485 tokens) but trained on
   *verbatim* (18.8 s, 734 tokens).
2. Read `max_memory_allocated()`, which excludes the ~330 MB CUDA context.

Contributing: Steam (204 MB) and an IDE (138 MB) held 342 MB of VRAM.
`nvidia-smi --query-compute-apps` **does not list graphics contexts**, so they were
invisible in the obvious check — use the full `nvidia-smi` table.

The fix was to benchmark the true worst case, selected by exact sequence length from the
token cache, which is now what `--dry-run` does automatically.

Measured on a 4 GB card, full fine-tune, batch 2 + gradient checkpointing: **2967 MB
peak**. Static cost of a full fine-tune is roughly 2.4 GB regardless of batch (715 MB
weights + 561 MB grads + 1122 MB Adam moments); everything above that is activations, which
scale with batch × sequence length.

**Do not extrapolate these to the 4090 — run `--dry-run` there and read the printed peak.**

---

## Things ruled out (don't re-litigate)

- **Text tokenizer** — GPT-2 BPE on Devanagari is *inefficient* (byte-fallback, ~1.5
  tokens/char) but lossless. Under matched near-greedy decoding, epoch 50 emitted EOS
  confidently at 98 tokens where epoch 10 ran past a 300-token cap without terminating —
  the same tokenizer measurably improving with training.
- **Speech tokenizer (S3 codec)** — `codec_roundtrip_test.py` encodes real Hindi audio to
  codec tokens and decodes straight back, bypassing T3 entirely. Result is ~98-99%
  perceptually identical, so the codec represents Hindi phonemes fine.
- **Don't swap in the multilingual model's 2454-token tokenizer.** It occupies a different
  id space than Nano's `text_emb` / `text_head`, so it would require reinitializing and
  retraining both from scratch.

## Known opportunity: vocab pruning

Only **3,023** of 50,276 text token ids appear in the verbatim corpus. `text_emb` +
`text_head` together are 77.22M params — **17% of the whole model**. Pruning to ~4,096 rows
would save ~71M params, which matters for the "smallest Indian TTS" goal. Not done yet;
it invalidates existing checkpoints.
