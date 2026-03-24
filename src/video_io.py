import os
import shutil
import ssl
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


def _is_enabled_env(name: str) -> bool:
    raw = os.environ.get(name, "")
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _should_disable_ssl_verification() -> bool:
    return _is_enabled_env("CONTENT_FILTERING_INSECURE_SSL") or _is_enabled_env("CONTENT_FILTER_INSECURE_SSL")


def _get_yt_dlp_cookies_file() -> str | None:
    """Return the path from YT_DLP_COOKIES_FILE if set and the file exists."""
    raw = os.environ.get("YT_DLP_COOKIES_FILE", "").strip()
    if not raw:
        return None
    path = Path(raw).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(
            "YT_DLP_COOKIES_FILE is set but the file does not exist: "
            f"{path}. In Kubernetes/headless environments, mount a Netscape-format "
            "cookies.txt and point YT_DLP_COOKIES_FILE to that mounted path."
        )
    return str(path)


def _get_yt_dlp_cookies_from_browser() -> str | None:
    """Return the browser name from YT_DLP_COOKIES_FROM_BROWSER if set."""
    raw = os.environ.get("YT_DLP_COOKIES_FROM_BROWSER", "").strip().lower()
    return raw or None


def _is_auth_required_error(exc: Exception) -> bool:
    """Return True when the yt-dlp error message indicates bot/auth gating."""
    message = str(exc).lower()
    return any(token in message for token in ("sign in", "cookies", "bot", "login required"))


def _is_unsupported_cookie_platform_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return "unsupported platform" in message and "linux" in message


def _build_ssl_context() -> ssl.SSLContext:
    if _should_disable_ssl_verification():
        return ssl._create_unverified_context()

    context = ssl.create_default_context()
    try:
        import certifi

        context.load_verify_locations(cafile=certifi.where())
    except Exception:
        # Keep platform defaults when certifi is unavailable.
        pass
    return context


def _is_certificate_error(exc: Exception) -> bool:
    message = str(exc).lower()
    ssl_tokens = (
        "certificate verify failed",
        "certificateverifyfailed",
        "unable to get local issuer certificate",
        "ssl",
    )
    return any(token in message for token in ssl_tokens)


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


def _download_via_yt_dlp(video_url: str, work_dir: Path, output_path: Path) -> tuple[Path | None, Exception | None]:
    """Try yt-dlp with a bounded retry plan for auth-gated YouTube failures."""
    from yt_dlp import YoutubeDL

    base_options = {
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "outtmpl": str(work_dir / "input_video.%(ext)s"),
        "merge_output_format": "mp4",
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "nocheckcertificate": _should_disable_ssl_verification(),
    }

    cookies_file = _get_yt_dlp_cookies_file()
    if cookies_file:
        base_options["cookiefile"] = cookies_file
    else:
        browser = _get_yt_dlp_cookies_from_browser()
        if browser:
            # yt-dlp expects a tuple: (browser_name,) or (browser_name, profile, keyring, container)
            base_options["cookiesfrombrowser"] = (browser,)

    attempts: list[tuple[str, dict]] = [
        ("default", {}),
        (
            "youtube_auth_fallback",
            {
                "extractor_args": {
                    "youtube": {
                        "player_client": ["ios", "android", "web"],
                    }
                }
            },
        ),
    ]

    attempt_errors: list[str] = []
    for stage, extra in attempts:
        options = dict(base_options)
        options.update(extra)
        try:
            with YoutubeDL(options) as ydl:
                info = ydl.extract_info(video_url, download=True)
                file_path = ydl.prepare_filename(info)
                path = Path(file_path)
                if path.suffix != ".mp4":
                    mp4_candidate = path.with_suffix(".mp4")
                    if mp4_candidate.exists():
                        path = mp4_candidate

                if _is_same_file(path, output_path):
                    _validate_video_file(output_path)
                    return output_path, None

                shutil.copy2(path, output_path)
                _validate_video_file(output_path)
                return output_path, None
        except Exception as exc:
            attempt_errors.append(f"{stage}: {exc}")
            # Extra attempts only help for auth/bot-gated errors.
            if not _is_auth_required_error(exc):
                break

    if not attempt_errors:
        return None, None
    return None, RuntimeError("; ".join(attempt_errors))


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
        downloaded, yt_dlp_error = _download_via_yt_dlp(video_url, work_dir, output_path)
        if downloaded is not None:
            return downloaded
    except FileNotFoundError as exc:
        # Config errors (like a missing cookies file) should fail fast and stay actionable.
        raise RuntimeError(str(exc)) from exc
    except Exception as exc:
        yt_dlp_error = exc

    # Fallback only for direct downloadable file URLs.
    request = urllib.request.Request(video_url, headers={"User-Agent": "Mozilla/5.0"})
    ssl_context = _build_ssl_context()
    try:
        with urllib.request.urlopen(request, context=ssl_context) as response, output_path.with_suffix(".part").open(
            "wb"
        ) as out_file:
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
        insecure_hint = (
            " If this environment has custom/intercepted TLS certs, install/update trust roots or try "
            "CONTENT_FILTERING_INSECURE_SSL=1 as a temporary workaround."
        )
        if yt_dlp_error is not None:
            suffix = ""
            if not _should_disable_ssl_verification() and (
                _is_certificate_error(yt_dlp_error) or _is_certificate_error(fallback_error)
            ):
                suffix = insecure_hint
            elif _is_unsupported_cookie_platform_error(yt_dlp_error):
                suffix = (
                    " Browser-cookie mode is unsupported in this runtime (for example, safari on Linux). "
                    "In Kubernetes/headless environments, use YT_DLP_COOKIES_FILE=/path/to/cookies.txt "
                    "with a Netscape-format cookies export instead of YT_DLP_COOKIES_FROM_BROWSER."
                )
            elif _is_auth_required_error(yt_dlp_error):
                suffix = (
                    " YouTube requires authentication to download this video. "
                    "Export your browser cookies to a Netscape-format file and set "
                    "YT_DLP_COOKIES_FILE=/path/to/cookies.txt, or set "
                    "YT_DLP_COOKIES_FROM_BROWSER=chrome (or firefox/safari) to read "
                    "cookies directly from your browser profile."
                )
            raise RuntimeError(
                "Could not download a valid video from this URL. "
                f"yt-dlp failed with: {yt_dlp_error}. "
                f"Direct-download fallback failed with: {fallback_error}.{suffix}"
            ) from fallback_error
        suffix = ""
        if not _should_disable_ssl_verification() and _is_certificate_error(fallback_error):
            suffix = insecure_hint
        raise RuntimeError(
            f"Could not download a valid video from this URL: {fallback_error}{suffix}"
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


def extract_frame_at_timestamp(video_path: Path, output_path: Path, timestamp_seconds: float) -> Path:
    video_path = video_path.resolve()
    output_path = output_path.resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"Input video not found for frame extraction: {video_path}")

    ts = max(0.0, float(timestamp_seconds))
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{ts:.3f}",
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        str(output_path),
    ]
    _run_ffmpeg(cmd)
    if not output_path.exists():
        raise RuntimeError(f"Frame extraction failed: {output_path}")
    return output_path

