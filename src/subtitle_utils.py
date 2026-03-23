from pathlib import Path
from typing import Iterable, Set

from src.filtering import mask_text
from src.models import TranscriptSegment


def _to_srt_time(seconds: float) -> str:
    millis = int(round(seconds * 1000))
    hours = millis // 3_600_000
    millis -= hours * 3_600_000
    minutes = millis // 60_000
    millis -= minutes * 60_000
    secs = millis // 1000
    millis -= secs * 1000
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"


def write_masked_srt(
    segments: Iterable[TranscriptSegment], bad_terms: Set[str], output_path: Path
) -> Path:
    lines = []
    for idx, segment in enumerate(segments, start=1):
        lines.append(str(idx))
        lines.append(f"{_to_srt_time(segment.start)} --> {_to_srt_time(segment.end)}")
        lines.append(mask_text(segment.text.strip(), bad_terms))
        lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path

