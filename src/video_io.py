import shutil
import subprocess
import urllib.parse
import urllib.request
from pathlib import Path


_DIRECT_VIDEO_EXTENSIONS = {
    ".mp4",
    ".mov",
    ".m4v",
    ".mkv",
    ".webm",
    ".avi",
    ".flv",
    ".wmv",
    ".mpeg",
    ".mpg",
}


def _run_ffmpeg(cmd: list[str], cwd: Path | None = None) -> None:
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        stderr = result.stderr.strip() or "ffmpeg failed"
        command_preview = " ".join(cmd)
        cwd_text = str(cwd) if cwd else "<current-dir>"
        raise RuntimeError(f"{stderr}\nCommand: {command_preview}\nWorking directory: {cwd_text}")


def _is_valid_video_container(video_path: Path) -> bool:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_type",
        "-of",
        "default=nokey=1:noprint_wrappers=1",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0 and "video" in (result.stdout or "")


def _validate_video_file(video_path: Path) -> None:
    if not video_path.exists() or video_path.stat().st_size < 1024:
        raise RuntimeError("Downloaded file is missing or too small to be a valid video.")
    if not _is_valid_video_container(video_path):
        raise RuntimeError(
            "Downloaded file is not a valid video container. "
            "This usually means the URL returned an HTML page or blocked response instead of media."
        )


def _is_direct_media_link(parsed_url: urllib.parse.ParseResult) -> bool:
    suffix = Path(parsed_url.path).suffix.lower()
    return suffix in _DIRECT_VIDEO_EXTENSIONS


def _is_same_file(path_a: Path, path_b: Path) -> bool:
    try:
        return path_a.exists() and path_b.exists() and path_a.samefile(path_b)
    except OSError:
        return path_a.resolve() == path_b.resolve()


def download_video_from_url(video_url: str, work_dir: Path) -> Path:
    parsed = urllib.parse.urlparse(video_url)

    if parsed.scheme in {"", "file"}:
        path = Path(parsed.path if parsed.scheme else video_url).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Video not found: {path}")
        local_path = work_dir / f"input_video{path.suffix if path.suffix else '.mp4'}"
        shutil.copy2(path, local_path)
        _validate_video_file(local_path)
        return local_path

    output_path = work_dir / "input_video.mp4"

    yt_dlp_error: Exception | None = None

    # Preferred approach for hosted video pages and signed URLs.
    try:
        from yt_dlp import YoutubeDL

        options = {
            "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
            "outtmpl": str(work_dir / "input_video.%(ext)s"),
            "merge_output_format": "mp4",
            "quiet": True,
            "no_warnings": True,
            "noplaylist": True,
        }
        with YoutubeDL(options) as ydl:
            info = ydl.extract_info(video_url, download=True)
            file_path = ydl.prepare_filename(info)
            path = Path(file_path)
            if path.suffix != ".mp4":
                # yt-dlp may keep original extension depending on source.
                mp4_candidate = path.with_suffix(".mp4")
                if mp4_candidate.exists():
                    path = mp4_candidate
            if _is_same_file(path, output_path):
                _validate_video_file(output_path)
                return output_path

            shutil.copy2(path, output_path)
            _validate_video_file(output_path)
            return output_path
    except Exception as exc:
        yt_dlp_error = exc

    # Fallback only for direct downloadable file URLs.
    request = urllib.request.Request(video_url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(request) as response, output_path.with_suffix(".part").open("wb") as out_file:
            content_type = response.headers.get_content_type() or ""
            if not content_type.startswith("video/") and not _is_direct_media_link(parsed):
                raise RuntimeError(
                    f"URL content-type '{content_type}' is not a direct video stream."
                )
            shutil.copyfileobj(response, out_file)
        output_path.with_suffix(".part").replace(output_path)
        _validate_video_file(output_path)
        return output_path
    except Exception as fallback_error:
        if yt_dlp_error is not None:
            raise RuntimeError(
                "Could not download a valid video from this URL. "
                f"yt-dlp failed with: {yt_dlp_error}. "
                f"Direct-download fallback failed with: {fallback_error}."
            ) from fallback_error
        raise RuntimeError(
            f"Could not download a valid video from this URL: {fallback_error}"
        ) from fallback_error


def extract_audio_wav(video_path: Path, audio_output_path: Path) -> Path:
    video_path = video_path.resolve()
    audio_output_path = audio_output_path.resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"Input video not found for audio extraction: {video_path}")

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-acodec",
        "pcm_s16le",
        str(audio_output_path),
    ]
    _run_ffmpeg(cmd)
    return audio_output_path


def mute_occurrences_in_audio(video_path: Path, output_path: Path, mute_spans: list[tuple[float, float]]) -> Path:
    video_path = video_path.resolve()
    output_path = output_path.resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"Input video not found for muting: {video_path}")

    if not mute_spans:
        shutil.copy2(video_path, output_path)
        return output_path

    checks = "+".join([f"between(t,{start:.3f},{end:.3f})" for start, end in mute_spans])
    # FFmpeg filtergraph treats commas as argument separators, so expression commas must be escaped.
    volume_expr = f"if({checks},0,1)".replace(",", r"\,")
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-af",
        f"volume={volume_expr}:eval=frame",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        str(output_path),
    ]
    _run_ffmpeg(cmd)
    if not output_path.exists():
        raise RuntimeError(f"Muted video was not generated: {output_path}")
    return output_path


def _merge_spans(spans: list[tuple[float, float]], bridge_seconds: float = 0.05) -> list[tuple[float, float]]:
    if not spans:
        return []

    ordered = sorted((max(0.0, float(start)), max(0.0, float(end))) for start, end in spans)
    merged: list[list[float]] = [[ordered[0][0], max(ordered[0][0], ordered[0][1])]]

    for start, end in ordered[1:]:
        end = max(start, end)
        current = merged[-1]
        if start <= current[1] + bridge_seconds:
            current[1] = max(current[1], end)
        else:
            merged.append([start, end])

    return [(item[0], item[1]) for item in merged]


def blur_occurrences_in_video(
    video_path: Path,
    output_path: Path,
    blur_spans: list[tuple[float, float]],
    blur_strength: float = 40.0,
) -> Path:
    video_path = video_path.resolve()
    output_path = output_path.resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"Input video not found for blur: {video_path}")

    merged_spans = _merge_spans(blur_spans)
    if not merged_spans:
        shutil.copy2(video_path, output_path)
        return output_path

    checks = "+".join([f"between(t,{start:.3f},{end:.3f})" for start, end in merged_spans])
    enable_expr = checks.replace(",", r"\,")
    sigma = max(8.0, float(blur_strength))

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vf",
        f"gblur=sigma={sigma:.1f}:steps=2:enable={enable_expr}",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "20",
        "-c:a",
        "copy",
        str(output_path),
    ]
    _run_ffmpeg(cmd)
    if not output_path.exists():
        raise RuntimeError(f"Blurred video was not generated: {output_path}")
    return output_path


def burn_subtitles(video_path: Path, srt_path: Path, output_path: Path) -> Path:
    video_path = video_path.resolve()
    srt_path = srt_path.resolve()
    output_path = output_path.resolve()

    if not video_path.exists():
        raise FileNotFoundError(f"Input video not found for subtitle burn: {video_path}")
    if not srt_path.exists():
        raise FileNotFoundError(f"Subtitle file not found: {srt_path}")
    if srt_path.stat().st_size == 0:
        # No transcript segments were produced; keep the muted video as-is.
        shutil.copy2(video_path, output_path)
        return output_path

    subtitle_filter_path = str(srt_path).replace("\\", r"\\").replace(":", r"\:")

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vf",
        f"subtitles={subtitle_filter_path}",
        "-c:a",
        "copy",
        str(output_path),
    ]
    _run_ffmpeg(cmd)
    if not output_path.exists():
        raise RuntimeError(f"Subtitle-burn output was not generated: {output_path}")
    return output_path
