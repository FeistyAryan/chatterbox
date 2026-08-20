"""Turn the Vaani Hindi parquet shards into a Chatterbox training manifest.

Writes wavs plus an `audio_path|transcript` manifest in the format
train_nano_exp1.py expects.

Two modes:

* `--keep-all` (recommended): every clip is used exactly as downloaded --
  original audio bytes, transcripts verbatim including the <tag> markup, which
  is acoustic-event conditioning the model can learn to reproduce.
* default: drops noisy/truncated/dialectal rows and strips markup, for when a
  clean single-style corpus is wanted instead.

Usage:
    python prepare_vaani_dataset.py --keep-all --target-hours 20
"""

import argparse
import io
import json
import re
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf

DEFAULT_SRC = Path("/home/aryan/Desktop/vaani_hindi_dataset/hindi")
DEFAULT_OUT = Path("/home/aryan/Desktop/vaani_hindi_dataset/prepared")

TAG_RE = re.compile(r"<[^>]*>|\[[^\]]*\]")
DEVA = lambda c: "ऀ" <= c <= "ॿ"
# "word {correction}" where the brace holds Devanagari -> use the correction.
CORRECTION_RE = re.compile(r"\S*\s*\{\s*([ऀ-ॿ][^}]*?)\s*\}")
# "word {gloss}" where the brace holds Latin -> drop the gloss, keep the word.
GLOSS_RE = re.compile(r"\s*\{[^}ऀ-ॿ]*\}")
ANY_BRACE_RE = re.compile(r"\{[^}]*\}")
NOISE_TAG_RE = re.compile(
    r"<\s*/?\s*(static_)?noise|<\s*/?\s*horn|<\s*/?\s*talking|<\s*/?\s*people_talking"
    r"|baby_cry|bird|<\s*/?\s*music|<\s*/?\s*cough|<\s*/?\s*laugh|<\s*/?\s*clap"
    r"|<\s*/?\s*vehicle|<\s*/?\s*wind|<\s*/?\s*bell|<\s*/?\s*ring",
    re.IGNORECASE,
)


def clean_transcript(text: str) -> str:
    """Strip annotation markup and resolve brace glosses/corrections."""
    # Corrections first: the braced Devanagari replaces the token before it.
    text = CORRECTION_RE.sub(lambda m: " " + m.group(1), text)
    text = GLOSS_RE.sub("", text)       # English glosses: drop
    text = ANY_BRACE_RE.sub("", text)   # anything left over
    text = TAG_RE.sub(" ", text)        # <noise>, [breathing], ...
    text = text.replace("--", " ")
    text = re.sub(r"[|]+", "।", text)
    text = re.sub(r"\s+([।,.?!])", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def devanagari_ratio(text: str) -> float:
    alpha = [c for c in text if c.isalpha()]
    if not alpha:
        return 0.0
    return sum(1 for c in alpha if DEVA(c)) / len(alpha)


def row_is_usable(transcript, languages, min_chars):
    """Reject rows that are noisy, truncated, dialectal, or unintelligible."""
    if languages != '["Hindi"]':
        return None, "not pure Hindi"
    if transcript is None:
        return None, "empty"
    if "--" in transcript:
        return None, "truncated"
    low = transcript.lower()
    if "unintelligible" in low or "inaudible" in low:
        return None, "unintelligible"
    if NOISE_TAG_RE.search(transcript):
        return None, "noise-tagged"
    clean = clean_transcript(transcript)
    if len(clean) < min_chars:
        return None, "too short"
    if devanagari_ratio(clean) < 0.90:
        return None, "not Devanagari"
    return clean, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--target-hours", type=float, default=20.0)
    ap.add_argument("--min-dur", type=float, default=2.0)
    ap.add_argument("--max-dur", type=float, default=15.0)
    ap.add_argument("--min-chars", type=int, default=12)
    ap.add_argument("--max-per-speaker", type=int, default=4,
                    help="Cap clips per speaker so no voice dominates.")
    ap.add_argument("--keep-all", action="store_true",
                    help="Use every clip exactly as downloaded: no quality filtering "
                         "and transcripts kept verbatim, tags included. The <tag> markup "
                         "is acoustic-event conditioning the model can learn to reproduce.")
    ap.add_argument("--shards", type=int, default=0,
                    help="Spread the quota across the first N shards for speaker/region "
                         "diversity. 0 = use all shards.")
    args = ap.parse_args()

    wav_dir = args.out / "wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / "manifest.txt"

    shards = sorted(args.src.glob("*.parquet"))
    if not shards:
        sys.exit(f"No parquet shards found in {args.src}")

    target_sec = args.target_hours * 3600
    kept_sec = 0.0
    kept = 0
    seen = 0
    rejects = {}
    per_speaker = {}
    rows_out = []

    n_shards = len(shards) if args.shards <= 0 else min(args.shards, len(shards))
    shards = shards[:n_shards]
    # Even quota per shard keeps region/speaker coverage broad instead of
    # taking everything from the first shard or two.
    sec_per_shard = target_sec / n_shards
    print(f"Scanning {len(shards)} shards, target {args.target_hours}h "
          f"({'ALL clips, no filtering' if args.keep_all else 'filtered'}) ...")
    for shard_i, shard in enumerate(shards):
        if kept_sec >= target_sec:
            break
        shard_budget = min(target_sec, sec_per_shard * (shard_i + 1))
        pf = pq.ParquetFile(shard)
        for rg in range(pf.num_row_groups):
            if kept_sec >= shard_budget or kept_sec >= target_sec:
                break
            tbl = pf.read_row_group(rg)
            d = tbl.to_pydict()
            for i in range(tbl.num_rows):
                seen += 1
                if args.keep_all:
                    # Transcript VERBATIM. The <tag> markup is acoustic-event
                    # conditioning, not speech: <noise> co-occurs with a hissy
                    # background, <birds_chirping> with birdsong. No audio aligns
                    # to the tag itself, so the model learns it as a background
                    # condition it can reproduce on request rather than as
                    # something to pronounce. Stripping it would throw that away.
                    clean = (d["transcript"][i] or "").strip()
                    if not clean:
                        rejects["empty transcript"] = rejects.get("empty transcript", 0) + 1
                        continue
                else:
                    clean, why = row_is_usable(d["transcript"][i], d["languages"][i], args.min_chars)
                    if why:
                        rejects[why] = rejects.get(why, 0) + 1
                        continue

                fn = d["file_name"][i]
                parts = fn.split("_")
                spk = parts[5] if len(parts) > 5 else "unknown"
                if not args.keep_all and per_speaker.get(spk, 0) >= args.max_per_speaker:
                    rejects["speaker cap"] = rejects.get("speaker cap", 0) + 1
                    continue

                raw = d["audio"][i]["bytes"]
                out_wav = wav_dir / f"{kept:06d}.wav"

                if args.keep_all:
                    # Write the source bytes untouched -- no decode, no resample,
                    # no normalisation. The clip on disk is the clip as downloaded.
                    try:
                        dur = sf.info(io.BytesIO(raw)).duration
                    except Exception:
                        rejects["decode error"] = rejects.get("decode error", 0) + 1
                        continue
                    out_wav.write_bytes(raw)
                else:
                    try:
                        wav, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
                    except Exception:
                        rejects["decode error"] = rejects.get("decode error", 0) + 1
                        continue
                    if wav.ndim > 1:
                        wav = wav.mean(axis=1)
                    dur = len(wav) / sr
                    if not (args.min_dur <= dur <= args.max_dur):
                        rejects["duration"] = rejects.get("duration", 0) + 1
                        continue
                    peak = float(np.max(np.abs(wav))) if wav.size else 0.0
                    if peak < 1e-6:
                        rejects["digital silence"] = rejects.get("digital silence", 0) + 1
                        continue
                    wav = wav / peak * 0.95
                    sf.write(out_wav, wav, sr, subtype="PCM_16")
                rows_out.append(f"{out_wav}|{clean}")
                per_speaker[spk] = per_speaker.get(spk, 0) + 1
                kept += 1
                kept_sec += dur

                if kept % 500 == 0:
                    print(f"  kept {kept} clips / {kept_sec/3600:.2f}h  (scanned {seen})")

    with manifest_path.open("w", encoding="utf-8") as fh:
        fh.write("\n".join(rows_out) + "\n")

    stats = {
        "clips": kept,
        "hours": round(kept_sec / 3600, 3),
        "rows_scanned": seen,
        "keep_rate": round(kept / max(seen, 1), 4),
        "distinct_speakers": len(per_speaker),
        "mean_dur_sec": round(kept_sec / max(kept, 1), 2),
        "rejects": rejects,
    }
    (args.out / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))

    print(f"\nManifest: {manifest_path}")
    print(f"Clips: {kept} | Hours: {kept_sec/3600:.2f} | Speakers: {len(per_speaker)}")
    print(f"Mean duration: {stats['mean_dur_sec']}s | Keep rate: {100*stats['keep_rate']:.1f}%")
    print("Rejections:")
    for k, v in sorted(rejects.items(), key=lambda x: -x[1]):
        print(f"  {v:7d}  {k}")


if __name__ == "__main__":
    main()
