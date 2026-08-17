"""Generate Hindi speech with a fine-tuned Nano checkpoint from train_nano_exp1.py.

Usage:
    python infer_nano_exp1.py --checkpoint experiments/nano_exp1/checkpoints/best \
        --text "आज मौसम बहुत अच्छा है।" --out out.wav
"""

import argparse
from pathlib import Path

import torch
import torchaudio as ta
from peft import PeftModel

from chatterbox.tts_turbo import ChatterboxTurboTTS, punc_norm
from chatterbox.models.s3gen.const import S3GEN_SIL

DEFAULT_REFERENCE_AUDIO = Path(
    "/home/aryan/Desktop/yt_scraper/output/fbWf6HjaNiA/segments_aligned/000030/final.wav"
)
S3_EOS = 6561


def parse_args():
    parser = argparse.ArgumentParser(description="Nano Hindi inference")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Checkpoint dir (best/latest).")
    parser.add_argument("--reference-audio", type=Path, default=DEFAULT_REFERENCE_AUDIO)
    parser.add_argument("--text", type=str, action="append", required=True, help="Repeatable.")
    parser.add_argument("--out", type=Path, default=Path("nano_exp1_out.wav"))
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--repetition-penalty", type=float, default=1.2)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=1000)
    parser.add_argument("--max-new-tokens", type=int, default=300, help="Caps generation to avoid runaway/OOM.")
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"Loading Nano on {args.device}...")
    model = ChatterboxTurboTTS.from_pretrained(device=args.device, nano=True)

    ckpt = args.checkpoint
    full_state = ckpt / "t3_state.pt"
    if full_state.exists():
        print(f"Loading full T3 state from {full_state}")
        model.t3.load_state_dict(torch.load(full_state, map_location=args.device), strict=False)
    else:
        print(f"Loading LoRA adapter from {ckpt}")
        model.t3.tfmr = PeftModel.from_pretrained(model.t3.tfmr, ckpt, is_trainable=False)
        try:
            model.t3.tfmr = model.t3.tfmr.merge_and_unload()
            print("Adapter merged into base weights.")
        except Exception as exc:
            print(f"Could not merge adapter (running with wrapper): {exc}")
        emb_path = ckpt / "text_emb.pt"
        if emb_path.exists():
            model.t3.text_emb.load_state_dict(torch.load(emb_path, map_location=args.device))
            print("Loaded fine-tuned text_emb.")
    model.t3.to(args.device).eval()

    with torch.no_grad():
        model.prepare_conditionals(str(args.reference_audio), exaggeration=0.0, norm_loudness=True)

    for idx, text in enumerate(args.text, start=1):
        print(f"\nGenerating {idx}/{len(args.text)}: {text}")
        norm_text = punc_norm(text)
        text_tokens = model.tokenizer(
            norm_text, return_tensors="pt", padding=False, truncation=False
        ).input_ids.to(args.device)

        with torch.inference_mode():
            speech_tokens = model.t3.inference_turbo(
                t3_cond=model.conds.t3,
                text_tokens=text_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                max_gen_len=args.max_new_tokens,
            )
        num_generated = speech_tokens.size(1)
        speech_tokens = speech_tokens.squeeze(0)
        speech_tokens = speech_tokens[speech_tokens < S3_EOS]
        hit_eos = num_generated < args.max_new_tokens
        print(f"Generated {num_generated} tokens (hit EOS: {hit_eos}), {speech_tokens.numel()} valid codec tokens.")

        silence = torch.tensor([S3GEN_SIL] * 3, dtype=torch.long, device=args.device)
        speech_tokens = torch.cat([speech_tokens, silence])

        with torch.inference_mode():
            wav, _ = model.s3gen.inference(
                speech_tokens=speech_tokens,
                ref_dict=model.conds.gen,
                n_cfm_timesteps=2,
            )
        wav = wav.squeeze(0).detach().cpu()
        wav = wav if wav.dim() == 2 else wav.unsqueeze(0)

        out_path = args.out if len(args.text) == 1 else args.out.with_stem(f"{args.out.stem}_{idx}")
        ta.save(str(out_path), wav, model.sr)
        print(f"Saved: {out_path}")

        if args.device == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
