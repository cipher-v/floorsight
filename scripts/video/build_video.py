"""Build the demo video from a section list (runs on the AWS instance; never on the owner PC).

    /opt/floorsight/py scripts/video/build_video.py spec.json OUT_DIR [--no-polly]

spec.json: {"voice": "Matthew", "sections": [
    {"id": "title", "card": ["Floorsight", "subtitle ..."], "min_s": 6, "narration": "..."},
    {"id": "app", "video": "/path/session.webm", "start": 12.0, "end": 80.0, "speed": 1.0, "narration": "...",
     "caption_top": "optional fixed caption"},
    {"id": "map", "image": "/path/figure.jpg", "min_s": 6, "narration": "..."}, ...]}
Each section becomes a 1920x1080 30 fps H.264 segment whose length is max(picture, narration). The narration comes
from Amazon Polly (neural voice) and the captions (SRT, burned in) from the same text, so they always match. Cards are
drawn with OpenCV. Writes OUT_DIR/floorsight-demo.mp4, floorsight-demo.srt and build-receipt.json.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import textwrap
from pathlib import Path

import cv2
import numpy as np

W, H, FPS = 1920, 1080, 30
TOP, CAPTION_BAND = 64, 180  # fixed top caption bar, bottom band kept free for the burned-in captions


def run(cmd: list[str]) -> str:
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"{cmd[0]} failed: {r.stderr[-800:]}")
    return r.stdout


def duration(path: Path) -> float:
    return float(run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(path)]).strip())


def card(lines: list[str], path: Path, seconds: float) -> None:
    img = np.full((H, W, 3), (35, 28, 22), np.uint8)
    cv2.rectangle(img, (0, H - 90), (W, H), (60, 48, 38), -1)
    y = 380 if len(lines) <= 2 else 300
    for i, ln in enumerate(lines):
        scale, thick = (2.6, 5) if i == 0 else (1.25, 2)
        (tw, th), _ = cv2.getTextSize(ln, cv2.FONT_HERSHEY_DUPLEX, scale, thick)
        if tw > W - 120:  # long line: shrink to fit the frame
            scale *= (W - 120) / tw
            (tw, th), _ = cv2.getTextSize(ln, cv2.FONT_HERSHEY_DUPLEX, scale, thick)
        cv2.putText(img, ln, ((W - tw) // 2, y), cv2.FONT_HERSHEY_DUPLEX, scale, (245, 245, 245) if i == 0 else (205, 215, 225), thick, cv2.LINE_AA)
        y += th + (60 if i == 0 else 34)
    png = path.with_suffix(".png")
    cv2.imwrite(str(png), img)
    run(["ffmpeg", "-v", "error", "-y", "-loop", "1", "-i", str(png), "-t", f"{seconds:.2f}", "-r", str(FPS),
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "medium", "-crf", "20", str(path)])


def drawtext_safe(text: str) -> str:
    """Escape a caption for ffmpeg drawtext. This ffmpeg build drops one trailing character per extra UTF-8 byte
    ("Önal & Dandıl" loses 2) and trims trailing spaces, so pad with spaces and a final dot that gets dropped."""
    t = text.replace("'", "’")
    extra = len(t.encode("utf-8")) - len(t)
    return t.replace(":", "\\:") + (" " * (extra - 1) + "." if extra else "")


def fit(caption_top: str | None) -> list[str]:
    """Scale the picture into the area between the top caption bar (if any) and the bottom caption band."""
    top = TOP if caption_top else 0
    ah = H - top - CAPTION_BAND
    return [f"scale={W}:{ah}:force_original_aspect_ratio=decrease", f"pad={W}:{H}:(ow-iw)/2:{top}+({ah}-ih)/2:color=0x16191e"]


def clip(src: str, start: float, end: float, speed: float, path: Path, caption_top: str | None) -> None:
    vf = [f"setpts=PTS/{speed}", *fit(caption_top), f"fps={FPS}"]
    if caption_top:
        t = drawtext_safe(caption_top)
        vf.append(f"drawbox=x=0:y=0:w=iw:h=64:color=0x101418@0.85:t=fill,drawtext=text='{t}':x=24:y=18:fontsize=30:fontcolor=white")
    run(["ffmpeg", "-v", "error", "-y", "-ss", f"{start:.2f}", "-to", f"{end:.2f}", "-i", src, "-an", "-vf", ",".join(vf),
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "medium", "-crf", "20", str(path)])


def still(src: str, path: Path, seconds: float, caption_top: str | None) -> None:
    """A still image (figure, diagram) letterboxed to 1920x1080 for `seconds`."""
    vf = [*fit(caption_top), f"fps={FPS}"]
    if caption_top:
        t = drawtext_safe(caption_top)
        vf.append(f"drawbox=x=0:y=0:w=iw:h=64:color=0x101418@0.85:t=fill,drawtext=text='{t}':x=24:y=18:fontsize=30:fontcolor=white")
    run(["ffmpeg", "-v", "error", "-y", "-loop", "1", "-i", src, "-t", f"{seconds:.2f}", "-vf", ",".join(vf),
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "medium", "-crf", "20", str(path)])


def polly(text: str, voice: str, path: Path) -> None:
    import boto3

    r = boto3.client("polly").synthesize_speech(Text=text, VoiceId=voice, Engine="neural", OutputFormat="mp3", SampleRate="24000")
    path.write_bytes(r["AudioStream"].read())


def srt_time(t: float) -> str:
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d},{int(round((s - int(s)) * 1000)):03d}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("spec", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--no-polly", action="store_true", help="silent narration track (captions only)")
    a = ap.parse_args()
    spec = json.loads(a.spec.read_text(encoding="utf-8"))
    a.out.mkdir(parents=True, exist_ok=True)
    seg_dir = a.out / "segments"
    seg_dir.mkdir(exist_ok=True)
    voice = spec.get("voice", "Matthew")
    segs, srt, t0, receipt = [], [], 0.0, {"sections": []}
    for i, s in enumerate(spec["sections"]):
        sid = f"{i:02d}-{s['id']}"
        narr = s.get("narration", "").strip()
        audio = seg_dir / f"{sid}.mp3"
        a_len = 0.0
        if narr and not a.no_polly:
            polly(narr, voice, audio)
            a_len = duration(audio)
        pic = seg_dir / f"{sid}-pic.mp4"
        if "card" in s:
            card(s["card"], pic, max(float(s.get("min_s", 4)), a_len + 0.6))
        elif "image" in s:
            still(s["image"], pic, max(float(s.get("min_s", 4)), a_len + 0.6), s.get("caption_top"))
        else:
            clip(s["video"], float(s["start"]), float(s["end"]), float(s.get("speed", 1.0)), pic, s.get("caption_top"))
        p_len = duration(pic)
        total = max(p_len, a_len + 0.6)
        seg = seg_dir / f"{sid}.mp4"
        pad = max(0.0, total - p_len)
        vf = f"tpad=stop_mode=clone:stop_duration={pad:.2f}" if pad > 0.01 else "null"
        if a_len > 0:
            run(["ffmpeg", "-v", "error", "-y", "-i", str(pic), "-i", str(audio), "-filter_complex", f"[0:v]{vf}[v];[1:a]apad[a]",
                 "-map", "[v]", "-map", "[a]", "-t", f"{total:.2f}", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
                 "-c:a", "aac", "-b:a", "128k", "-ar", "48000", "-ac", "2", str(seg)])
        else:
            run(["ffmpeg", "-v", "error", "-y", "-i", str(pic), "-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo", "-filter_complex", f"[0:v]{vf}[v]",
                 "-map", "[v]", "-map", "1:a", "-t", f"{total:.2f}", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
                 "-c:a", "aac", "-b:a", "128k", "-ar", "48000", "-ac", "2", str(seg)])
        if narr:  # captions spread over the narration, 2 lines of up to 56 chars each
            chunks = textwrap.wrap(narr, 96)
            span = (a_len if a_len > 0 else total) / max(1, len(chunks))
            for k, ch in enumerate(chunks):
                srt.append((t0 + k * span, t0 + (k + 1) * span - 0.05, "\n".join(textwrap.wrap(ch, 56))))
        receipt["sections"].append({"id": sid, "picture_s": round(p_len, 2), "narration_s": round(a_len, 2), "segment_s": round(total, 2)})
        segs.append(seg)
        t0 += total
    (a.out / "concat.txt").write_text("".join(f"file '{p.resolve()}'\n" for p in segs))
    joined = a.out / "joined.mp4"
    run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(a.out / "concat.txt"), "-c", "copy", str(joined)])
    srt_path = a.out / "floorsight-demo.srt"
    srt_path.write_text("".join(f"{n + 1}\n{srt_time(s)} --> {srt_time(e)}\n{txt}\n\n" for n, (s, e, txt) in enumerate(srt)), encoding="utf-8")
    final = a.out / "floorsight-demo.mp4"
    style = "FontName=DejaVu Sans,FontSize=14,PrimaryColour=&H00FFFFFF,BackColour=&H99000000,BorderStyle=4,Outline=0,Shadow=0,MarginV=10"
    run(["ffmpeg", "-v", "error", "-y", "-i", str(joined), "-vf", f"subtitles={srt_path}:force_style='{style}'",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "medium", "-crf", "20", "-c:a", "copy", "-movflags", "+faststart", str(final)])
    receipt.update(total_s=round(duration(final), 2), voice=None if a.no_polly else voice, opencv=cv2.__version__, captions=str(srt_path))
    (a.out / "build-receipt.json").write_text(json.dumps(receipt, indent=1))
    print("VIDEO", json.dumps(receipt)[:1500])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
