"""Sanity check: does the S3 speech tokenizer preserve Hindi intelligibility?

Takes REAL ground-truth speech tokens from the training cache (no T3 involved
at all) and decodes them straight back to audio via s3gen. If this round-trip
sounds clean, the codec can represent Hindi fine and T3's text->token mapping
is the only thing that needs more training. If it's mangled, the codec itself
is a bottleneck.
"""
import argparse
from pathlib import Path

import torch
import torchaudio as ta

from chatterbox.tts_turbo import ChatterboxTurboTTS

S3_EOS = 6561


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=Path("experiments/nano_exp1/token_cache.pt"))
    parser.add_argument("--reference-audio", type=Path,
                         default=Path("/home/aryan/Desktop/yt_scraper/output/fbWf6HjaNiA/segments_aligned/000030/final.wav"))
    parser.add_argument("--indices", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--out-dir", type=Path, default=Path("experiments/nano_exp1/codec_roundtrip"))
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading Nano on {args.device}...")
    model = ChatterboxTurboTTS.from_pretrained(device=args.device, nano=True)

    with torch.no_grad():
        model.prepare_conditionals(str(args.reference_audio), exaggeration=0.0, norm_loudness=True)

    payload = torch.load(args.cache, map_location="cpu", weights_only=False)
    items = payload["items"]
    print(f"Cache has {len(items)} rows.")

    for idx in args.indices:
        item = items[idx]
        transcript = item["transcript"]
        orig_path = Path(item["audio_path"])
        speech_tokens = item["speech_tokens"].clone()
        # strip BOS/EOS wrapper added during caching
        speech_tokens = speech_tokens[(speech_tokens < S3_EOS)]

        print(f"\n[{idx}] transcript: {transcript}")
        print(f"    orig audio: {orig_path}")
        print(f"    {speech_tokens.numel()} ground-truth codec tokens")

        speech_tokens = speech_tokens.to(args.device)
        with torch.inference_mode():
            wav, _ = model.s3gen.inference(
                speech_tokens=speech_tokens,
                ref_dict=model.conds.gen,
                n_cfm_timesteps=2,
            )
        wav = wav.squeeze(0).detach().cpu()
        wav = wav if wav.dim() == 2 else wav.unsqueeze(0)

        out_path = args.out_dir / f"roundtrip_{idx}.wav"
        ta.save(str(out_path), wav, model.sr)
        print(f"    saved: {out_path}")

        if args.device == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
