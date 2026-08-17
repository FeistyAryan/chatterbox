import sys
import traceback
from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio as ta

from chatterbox.tts_turbo import ChatterboxTurboTTS


MANIFEST = Path("/home/aryan/Desktop/yt_scraper/dataset_final.manifest.txt")
REFERENCE_AUDIO = Path(
    "/home/aryan/Desktop/yt_scraper/output/fbWf6HjaNiA/segments_aligned/000030/final.wav"
)
TARGET_SR = 16000
IGNORE_ID = -100


def load_first_manifest_example(manifest_path: Path):
    with manifest_path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            raw_line = raw_line.strip()
            if not raw_line or "|" not in raw_line:
                continue
            audio_path, text = raw_line.split("|", 1)
            return Path(audio_path), text
    raise RuntimeError(f"No valid manifest rows found in {manifest_path}")


def to_mono_resampled(wav: torch.Tensor, sr: int, target_sr: int) -> tuple[torch.Tensor, int]:
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = ta.functional.resample(wav, orig_freq=sr, new_freq=target_sr)
    return wav.contiguous(), target_sr


def add_special_tokens(tokenizer_output: torch.Tensor, start_token: int, stop_token: int) -> torch.Tensor:
    start = torch.tensor([[start_token]], dtype=torch.long, device=tokenizer_output.device)
    stop = torch.tensor([[stop_token]], dtype=torch.long, device=tokenizer_output.device)
    return torch.cat([start, tokenizer_output, stop], dim=1)


def masked_ce(logits: torch.Tensor, targets: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    mask = torch.arange(targets.size(1), device=targets.device)[None, :] >= lengths[:, None]
    masked_targets = targets.masked_fill(mask, IGNORE_ID)

    # T3.forward() returns [B, seq, vocab], but cross_entropy expects [B, vocab, seq].
    return F.cross_entropy(
        logits.transpose(1, 2),
        masked_targets,
        ignore_index=IGNORE_ID,
    )


def print_stage(name: str, ok: bool, detail: str = ""):
    status = "PASS" if ok else "FAIL"
    line = f"{name}: {status}"
    if detail:
        line += f" | {detail}"
    print(line)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Manifest: {MANIFEST}")
    print(f"Reference audio: {REFERENCE_AUDIO}")

    summary = {
        "data pipeline": False,
        "speech tokenizer": False,
        "text tokenizer": False,
        "conditioning": False,
        "T3 forward": False,
        "loss": False,
        "backward": False,
        "gradients": False,
        "optimizer step": False,
    }

    model = None
    optimizer = None

    try:
        audio_path, text = load_first_manifest_example(MANIFEST)
        if not audio_path.exists():
            raise FileNotFoundError(f"Target audio not found: {audio_path}")
        if not REFERENCE_AUDIO.exists():
            raise FileNotFoundError(f"Reference audio not found: {REFERENCE_AUDIO}")

        print("\nTarget example:")
        print(f"Audio: {audio_path}")
        print(f"Text : {text}")

        print("\nLoading Nano...")
        model = ChatterboxTurboTTS.from_pretrained(device=device, nano=True)
        model.t3.train()
        model.s3gen.eval()
        model.ve.eval()

        # Freeze the non-T3 paths. The smoke test is only about proving the T3
        # training path, not training the codec or voice encoder.
        for p in model.s3gen.parameters():
            p.requires_grad = False
        for p in model.ve.parameters():
            p.requires_grad = False

        total_params = sum(p.numel() for p in model.t3.parameters())
        trainable_params = sum(p.numel() for p in model.t3.parameters() if p.requires_grad)
        print("\nT3 parameters:")
        print(f"Total: {total_params}")
        print(f"Trainable: {trainable_params}")

        # ------------------------------------------------------------------
        # Data pipeline: load target audio and convert to the required SR.
        # ------------------------------------------------------------------
        wav, sr = ta.load(audio_path)
        wav, _ = to_mono_resampled(wav, sr, TARGET_SR)

        print("\nPrepared target audio:")
        print(f"Shape: {tuple(wav.shape)}")
        print(f"Sample rate: {TARGET_SR}")
        summary["data pipeline"] = True

        # ------------------------------------------------------------------
        # Speech tokenizer: convert audio to S3 tokens, then add the repo's
        # explicit start/stop tokens.
        # ------------------------------------------------------------------
        with torch.no_grad():
            speech_tokens, speech_token_lens = model.s3gen.tokenizer.forward([wav], max_len=None)

        speech_tokens = speech_tokens.to(device)
        speech_token_lens = speech_token_lens.to(device)

        print("\nRaw speech tokens:")
        print(f"Shape: {tuple(speech_tokens.shape)}")
        print(f"Length: {speech_token_lens.tolist()}")
        print(f"First 20: {speech_tokens[0, :20].tolist()}")

        speech_tokens = add_special_tokens(
            speech_tokens,
            model.t3.hp.start_speech_token,
            model.t3.hp.stop_speech_token,
        )
        speech_token_lens = speech_token_lens + 2

        print("\nTraining speech sequence:")
        print(f"Shape: {tuple(speech_tokens.shape)}")
        print(f"Length: {speech_token_lens.tolist()}")
        print(f"First token: {speech_tokens[0, 0].item()}")
        print(f"Last token : {speech_tokens[0, -1].item()}")
        summary["speech tokenizer"] = True

        # ------------------------------------------------------------------
        # Text tokenizer: use Nano's actual tokenizer, then add the required
        # T3 start/stop tokens externally.
        # ------------------------------------------------------------------
        text_batch = model.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        raw_text_tokens = text_batch.input_ids.to(device)

        print("\nRaw text tokens:")
        print(f"Shape: {tuple(raw_text_tokens.shape)}")
        print(f"Length: {raw_text_tokens.shape[1]}")

        text_tokens = add_special_tokens(
            raw_text_tokens,
            model.t3.hp.start_text_token,
            model.t3.hp.stop_text_token,
        )
        text_token_lens = torch.tensor([text_tokens.shape[1]], dtype=torch.long, device=device)

        print("\nTraining text sequence:")
        print(f"Shape: {tuple(text_tokens.shape)}")
        print(f"Length: {text_token_lens.tolist()}")
        print(f"First token: {text_tokens[0, 0].item()}")
        print(f"Last token : {text_tokens[0, -1].item()}")
        summary["text tokenizer"] = True

        # ------------------------------------------------------------------
        # Conditioning: use a valid >5 second reference clip, independent of
        # the target example.
        # ------------------------------------------------------------------
        print("\nPreparing conditioning...")
        with torch.no_grad():
            model.prepare_conditionals(
                str(REFERENCE_AUDIO),
                exaggeration=0.0,
                norm_loudness=True,
            )
        t3_cond = model.conds.t3
        summary["conditioning"] = True

        # ------------------------------------------------------------------
        # Forward pass through T3.
        # ------------------------------------------------------------------
        print("\nRunning T3 forward...")
        out = model.t3.forward(
            t3_cond=t3_cond,
            text_tokens=text_tokens,
            text_token_lens=text_token_lens,
            speech_tokens=speech_tokens,
            speech_token_lens=speech_token_lens,
            training=True,
        )

        print("\nT3 output shapes:")
        print(f"Text logits  : {tuple(out.text_logits.shape)}")
        print(f"Text target  : {tuple(text_tokens.shape)}")
        print(f"Speech logits: {tuple(out.speech_logits.shape)}")
        print(f"Speech target: {tuple(speech_tokens.shape)}")
        summary["T3 forward"] = True

        # ------------------------------------------------------------------
        # Loss: use transposed logits because T3.forward() returns [B, seq, vocab].
        # ------------------------------------------------------------------
        print("\nCalculating loss...")
        loss_text = masked_ce(out.text_logits, text_tokens, text_token_lens)
        loss_speech = masked_ce(out.speech_logits, speech_tokens, speech_token_lens)
        loss = loss_text + loss_speech

        print("\nLoss:")
        print(f"Text loss  : {loss_text.item()}")
        print(f"Speech loss: {loss_speech.item()}")
        print(f"Total loss : {loss.item()}")

        if not torch.isfinite(loss):
            raise RuntimeError("Loss is not finite")
        summary["loss"] = True

        # ------------------------------------------------------------------
        # Backward + optimizer step.
        # ------------------------------------------------------------------
        print("\nRunning backward...")
        # Use a zero-state optimizer here. AdamW allocates per-parameter state
        # on the first step and can OOM on a 4 GB card after the backward pass.
        # This keeps the smoke test focused on proving that a parameter update
        # can occur at all.
        optimizer = torch.optim.SGD(
            model.t3.parameters(),
            lr=1e-5,  # Temporary smoke-test LR; not a recommended training LR.
            momentum=0.0,
            weight_decay=0.0,
            foreach=False,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        summary["backward"] = True
        print("Backward SUCCESS.")

        grad_params = 0
        grad_norm_squared = 0.0
        grad_names = []
        for name, param in model.t3.named_parameters():
            if param.grad is not None:
                grad_params += 1
                grad_names.append(name)
                grad_norm = param.grad.detach().float().norm().item()
                grad_norm_squared += grad_norm * grad_norm

        grad_norm = grad_norm_squared ** 0.5
        print("\nGradients:")
        print(f"Parameters with gradients: {grad_params}")
        print(f"Gradient norm: {grad_norm}")
        print(f"Sample gradient params: {grad_names[:10]}")

        expected_grad_params = [
            ("text_head.weight", getattr(model.t3.text_head.weight, "grad", None) is not None),
            ("speech_head.weight", getattr(model.t3.speech_head.weight, "grad", None) is not None),
        ]
        for name, ok in expected_grad_params:
            print(f"Gradient check {name}: {'PASS' if ok else 'FAIL'}")

        if grad_params == 0:
            raise RuntimeError("No gradients were produced")
        if not all(ok for _, ok in expected_grad_params):
            raise RuntimeError("Expected T3 parameters did not receive gradients")
        summary["gradients"] = True

        print("\nRunning optimizer step...")
        optimizer.step()
        summary["optimizer step"] = True
        print("Optimizer step SUCCESS.")

        print("\n========================================")
        print("SMOKE TEST SUCCESS")
        print("========================================")

    except RuntimeError as exc:
        print("\nSMOKE TEST FAILED")
        print(f"RuntimeError: {exc}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        traceback.print_exc()
        print("\nPartial summary:")
        for name, ok in summary.items():
            print_stage(name, ok)
        sys.exit(1)
    except Exception as exc:
        print("\nSMOKE TEST FAILED")
        print(f"{type(exc).__name__}: {exc}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        traceback.print_exc()
        print("\nPartial summary:")
        for name, ok in summary.items():
            print_stage(name, ok)
        sys.exit(1)

    print("\nFinal summary:")
    for name, ok in summary.items():
        print_stage(name, ok)


if __name__ == "__main__":
    main()
