import json
import uuid
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from src.filtering import find_filtered_occurrences
from src.reporting import write_report
from src.scene_detection import detect_scene_events
from src.subtitle_utils import write_masked_srt
from src.term_config import (
    get_scene_detection_config,
    load_term_config,
    resolve_terms_from_categories,
)
from src.transcription import transcribe_with_word_timestamps
from src.video_io import (
    blur_occurrences_in_video,
    burn_subtitles,
    download_video_from_url,
    extract_frame_at_timestamp,
    extract_audio_wav,
    mute_occurrences_in_audio,
)


def _build_scene_gallery(
    input_video: Path,
    scene_events: list,
    gallery_dir: Path,
    max_items: int = 30,
) -> list[tuple[str, str]]:
    gallery_dir.mkdir(parents=True, exist_ok=True)
    items: list[tuple[str, str]] = []
    for idx, event in enumerate(scene_events[:max_items], start=1):
        midpoint = max(0.0, (event.start + event.end) / 2.0)
        frame_path = gallery_dir / f"scene_{idx:03d}_{event.category}.jpg"
        try:
            extract_frame_at_timestamp(input_video, frame_path, midpoint)
        except Exception:
            continue
        caption = (
            f"{event.category} | {event.source} | "
            f"detected={event.detected_value:.3f} threshold={event.threshold_used:.3f} | "
            f"t={event.start:.2f}-{event.end:.2f}s | {event.reason}"
        )
        items.append((str(frame_path), caption))
    return items


def _collect_effective_scene_thresholds(scene_config: dict) -> dict[str, float]:
    collected: dict[str, float] = {}
    for key, value in scene_config.items():
        if isinstance(value, (int, float)) and ("threshold" in key or "interval" in key):
            collected[key] = float(value)
    return dict(sorted(collected.items()))


def process_video_url(
    video_url: str,
    selected_categories: Iterable[str],
    selected_scene_categories: Iterable[str],
    config_path: str = "config/bad_terms_categories.json",
    model_size: str = "small",
    mute_padding_seconds: float = 0.08,
    mute_scene_audio: bool = False,
    blur_scene_video: bool = True,
    blur_strength: float = 40.0,
    scene_nudity_threshold: float | None = None,
    scene_immodesty_threshold: float | None = None,
    scene_frame_interval_seconds: float | None = None,
    scene_threshold_overrides: dict[str, float] | None = None,
    artifacts_root: str = "artifacts",
) -> tuple[str, str, str, list[tuple[str, str]]]:
    if not video_url.strip():
        raise ValueError("Please provide a video URL.")

    config = load_term_config(config_path)
    bad_terms, caption_terms, context_exclusions = resolve_terms_from_categories(
        config,
        selected_categories,
    )
    if not bad_terms and not caption_terms:
        raise ValueError("No active terms found for selected categories.")

    job_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    artifacts_root_path = Path(artifacts_root).expanduser().resolve()
    job_dir = artifacts_root_path / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    input_video = download_video_from_url(video_url.strip(), job_dir)
    audio_wav = extract_audio_wav(input_video, job_dir / "audio.wav")
    segments = transcribe_with_word_timestamps(
        str(audio_wav),
        language="en",
        model_size=model_size,
    )

    scene_config = dict(get_scene_detection_config(config) or {})
    if scene_nudity_threshold is not None:
        scene_config["nudity_score_threshold"] = min(0.99, max(0.01, float(scene_nudity_threshold)))
    if scene_immodesty_threshold is not None:
        scene_config["immodesty_score_threshold"] = min(
            0.99, max(0.01, float(scene_immodesty_threshold))
        )
    if scene_frame_interval_seconds is not None:
        scene_config["frame_interval_seconds"] = max(0.1, float(scene_frame_interval_seconds))
    if scene_threshold_overrides:
        for key, value in scene_threshold_overrides.items():
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            if "threshold" in key:
                scene_config[key] = min(0.99, max(0.01, numeric))
            elif "interval" in key:
                scene_config[key] = max(0.05, numeric)
            else:
                scene_config[key] = numeric
    pipeline_warnings: list[str] = []
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always", RuntimeWarning)
        scene_events = detect_scene_events(
            input_video,
            segments,
            scene_config,
            selected_scene_categories,
        )
    for w in caught_warnings:
        pipeline_warnings.append(str(w.message))

    # Phrases that are not covered by single-word muting get their own span.
    phrase_terms = {t for t in (caption_terms or set()) if len(t.split()) >= 2}
    occurrences = find_filtered_occurrences(
        segments,
        bad_terms,
        phrase_terms=phrase_terms or None,
        safety_padding_seconds=max(0.0, float(mute_padding_seconds)),
        context_exclusions=context_exclusions,
    )
    scene_spans = [(event.start, event.end) for event in scene_events]
    mute_spans = [(o.start, o.end) for o in occurrences]
    if mute_scene_audio or blur_scene_video:
        mute_spans.extend(scene_spans)

    muted_video = mute_occurrences_in_audio(input_video, job_dir / "muted.mp4", mute_spans)
    if not Path(muted_video).exists():
        raise RuntimeError(f"Muted video was not created: {muted_video}")

    video_for_subtitles = muted_video
    if blur_scene_video and scene_spans:
        blurred_video = blur_occurrences_in_video(
            muted_video,
            job_dir / "blurred.mp4",
            scene_spans,
            blur_strength=blur_strength,
        )
        video_for_subtitles = blurred_video

    masked_srt = write_masked_srt(segments, caption_terms or bad_terms, job_dir / "masked.srt")
    output_video = burn_subtitles(video_for_subtitles, masked_srt, job_dir / "output_censored.mp4")

    json_path, csv_path, report_dict = write_report(
        occurrences,
        job_dir,
        scene_events=scene_events,
    )
    scene_gallery_items = _build_scene_gallery(input_video, scene_events, job_dir / "scene_gallery")

    report_summary = {
        "job_id": job_id,
        "config_path": str(Path(config_path).expanduser().resolve()),
        "selected_categories": sorted(set(selected_categories or [])),
        "selected_scene_categories": sorted(set(selected_scene_categories or [])),
        "mute_scene_audio": bool(mute_scene_audio),
        "blur_scene_video": bool(blur_scene_video),
        "blur_strength": float(blur_strength),
        "output_video": str(output_video),
        "report_json": str(json_path),
        "report_csv": str(csv_path),
        "scene_events_csv": str(job_dir / "scene_events.csv"),
        "total_filtered": report_dict["total_filtered"],
        "counts": report_dict["counts"],
        "scene_total": report_dict.get("scene_total", 0),
        "scene_by_category": report_dict.get("scene_by_category", {}),
        "scene_gallery_dir": str(job_dir / "scene_gallery"),
        "scene_gallery_count": len(scene_gallery_items),
        "effective_scene_config": _collect_effective_scene_thresholds(scene_config),
        "warnings": pipeline_warnings,
    }
    return (
        str(output_video),
        json.dumps(report_summary, indent=2),
        str(report_dict["counts"]),
        scene_gallery_items,
    )
