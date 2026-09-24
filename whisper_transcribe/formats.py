"""Timestamp formatting and .srt / .txt writers."""

from pathlib import Path


def ts(t, sep=","):
    """Seconds -> 'HH:MM:SS<sep>mmm' (SRT uses ',' as separator)."""
    h, r = divmod(t, 3600)
    m, s = divmod(r, 60)
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d}{sep}{int((s % 1) * 1000):03d}"


def to_srt(segments):
    """segments: iterable of (start, end, text)."""
    return "".join(
        f"{i}\n{ts(start)} --> {ts(end)}\n{text}\n\n"
        for i, (start, end, text) in enumerate(segments, 1)
    )


def to_txt(segments):
    """One line per segment: '[HH:MM:SS] text'."""
    return "".join(f"[{ts(start, '.')[:8]}] {text}\n" for start, _end, text in segments)


def write_outputs(segments, out_dir, stem):
    """Write <stem>.srt and <stem>.txt into out_dir; return both paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    srt_path = out_dir / f"{stem}.srt"
    txt_path = out_dir / f"{stem}.txt"
    srt_path.write_text(to_srt(segments), encoding="utf-8")
    txt_path.write_text(to_txt(segments), encoding="utf-8")
    return srt_path, txt_path
