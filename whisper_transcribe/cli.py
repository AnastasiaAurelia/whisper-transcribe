"""whisper-transcribe: timestamped .srt/.txt transcripts with faster-whisper."""

import argparse
import os
import sys
import time
from pathlib import Path

from .checks import check_transcript
from .formats import write_outputs

AUDIO_EXTS = {
    ".mp3", ".wav", ".m4a", ".flac", ".ogg", ".opus", ".aac", ".wma",
    ".webm", ".mp4", ".mkv", ".mov",
}


def fmt_duration(seconds):
    if seconds is None:
        return "-"
    m, s = divmod(int(round(seconds)), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def collect_inputs(path):
    path = Path(path).expanduser()
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in AUDIO_EXTS)
    raise FileNotFoundError(f"no such file or directory: {path}")


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        prog="whisper-transcribe",
        description="Transcribe an audio file or a folder of audio files into "
        "timestamped .srt and .txt using faster-whisper (offline by default).",
    )
    p.add_argument("input", help="audio file, or folder (non-recursive) of audio files")
    p.add_argument("-o", "--output", default="transcripts",
                   help="output folder (default: ./transcripts)")
    p.add_argument("-l", "--language", default="en",
                   help="language code, e.g. en, id, ja; 'auto' to detect (default: en)")
    p.add_argument("-m", "--model", default="base",
                   help="model size (tiny, base, small, medium, large-v3, ...) "
                   "or path to a local CTranslate2 model folder (default: base)")
    p.add_argument("--device", default="cpu", help="cpu, cuda, or auto (default: cpu)")
    p.add_argument("--compute-type", default="int8", help="default: int8")
    p.add_argument("--threads", type=int, default=os.cpu_count() or 4,
                   help="CPU threads (default: all cores)")
    p.add_argument("--beam-size", type=int, default=5, help="default: 5")
    p.add_argument("--no-vad", action="store_true",
                   help="disable voice-activity filtering (enabled by default)")
    p.add_argument("--skip-existing", action="store_true",
                   help="skip files whose .srt and .txt already exist in the output folder")
    p.add_argument("--allow-download", action="store_true",
                   help="allow downloading the model from Hugging Face if it is not cached "
                   "(default: offline, local cache only)")
    return p.parse_args(argv)


def load_model(args):
    if not args.allow_download:
        # Must be set before huggingface_hub is imported.
        os.environ["HF_HUB_OFFLINE"] = "1"
    from faster_whisper import WhisperModel

    try:
        return WhisperModel(
            args.model,
            device=args.device,
            compute_type=args.compute_type,
            cpu_threads=args.threads,
            local_files_only=not args.allow_download,
        )
    except Exception as e:
        hint = ""
        if not args.allow_download:
            hint = (
                f"\nModel '{args.model}' is not available locally. Offline mode only uses "
                "models already in the Hugging Face cache (~/.cache/huggingface/hub) or a "
                "local model folder passed with --model. Re-run with --allow-download to "
                "fetch it once."
            )
        sys.exit(f"error: could not load model '{args.model}': {e}{hint}")


def transcribe_one(model, path, args):
    """Transcribe one file; return a result dict (never raises)."""
    result = {"file": path.name, "duration": None, "segments": 0,
              "language": None, "warnings": [], "error": None}
    t0 = time.time()
    try:
        language = None if args.language == "auto" else args.language
        segs, info = model.transcribe(
            str(path), language=language, beam_size=args.beam_size,
            vad_filter=not args.no_vad,
        )
        result["duration"] = info.duration
        result["language"] = info.language
        print(f"  language={info.language} p={info.language_probability:.2f} "
              f"duration={fmt_duration(info.duration)}", flush=True)

        segments, logprobs = [], []
        for i, s in enumerate(segs, 1):
            segments.append((s.start, s.end, s.text.strip()))
            logprobs.append(s.avg_logprob)
            if i % 100 == 0:
                print(f"  {s.end:.0f}/{info.duration:.0f}s", flush=True)

        srt, txt = write_outputs(segments, args.output, path.stem)
        result["segments"] = len(segments)
        result["warnings"] = check_transcript(
            [t for _, _, t in segments], info.duration, info.language, logprobs,
            detected_language=info.language if language is None else None,
            language_probability=info.language_probability if language is None else None,
        )
        print(f"  wrote {srt.name} and {txt.name} in {time.time() - t0:.0f}s", flush=True)
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
    return result


def print_report(results):
    print("\nSummary")
    print(f"{'status':<7} {'duration':>9} {'segments':>8}  file")
    for r in results:
        status = "ERROR" if r["error"] else "WARN" if r["warnings"] else "ok"
        print(f"{status:<7} {fmt_duration(r['duration']):>9} {r['segments']:>8}  {r['file']}")
        if r["error"]:
            print(f"{'':<7} error: {r['error']}")
        for w in r["warnings"]:
            print(f"{'':<7} warning: {w}")
    n_err = sum(1 for r in results if r["error"])
    n_warn = sum(1 for r in results if r["warnings"] and not r["error"])
    print(f"\n{len(results)} file(s): {len(results) - n_err - n_warn} ok, "
          f"{n_warn} with warnings, {n_err} failed")


def main(argv=None):
    args = parse_args(argv)
    try:
        inputs = collect_inputs(args.input)
    except FileNotFoundError as e:
        sys.exit(f"error: {e}")
    if not inputs:
        sys.exit(f"error: no audio files ({', '.join(sorted(AUDIO_EXTS))}) in {args.input}")

    out = Path(args.output)
    if args.skip_existing:
        todo = [p for p in inputs
                if not ((out / f"{p.stem}.srt").exists() and (out / f"{p.stem}.txt").exists())]
        for p in sorted(set(inputs) - set(todo)):
            print(f"skip (already transcribed): {p.name}")
        inputs = todo
        if not inputs:
            return 0

    model = load_model(args)
    results = []
    for n, path in enumerate(inputs, 1):
        print(f"[{n}/{len(inputs)}] {path.name}", flush=True)
        results.append(transcribe_one(model, path, args))
    print_report(results)
    return 1 if any(r["error"] for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
