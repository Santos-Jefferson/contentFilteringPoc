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
    get_config_language,
    get_scene_detection_config,
    load_term_config,
    resolve_processing_profile,
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


def _create_job_dir(artifacts_root: str) -> tuple[str, Path]:
    job_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    artifacts_root_path = Path(artifacts_root).expanduser().resolve()
    job_dir = artifacts_root_path / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    return job_id, job_dir


def _resolve_processing_inputs(
    config: dict,
    processing_profile: str | None,
    selected_categories: Iterable[str] | None,
    selected_scene_categories: Iterable[str] | None,
    mute_scene_audio: bool | None,
    blur_scene_video: bool | None,
):
    profile = resolve_processing_profile(config, processing_profile)
    resolved_categories = sorted(set(selected_categories or profile.selected_categories))
    resolved_scene_categories = sorted(set(selected_scene_categories or profile.selected_scene_categories))
    resolved_mute_scene_audio = profile.mute_scene_audio if mute_scene_audio is None else bool(mute_scene_audio)
    resolved_blur_scene_video = profile.blur_scene_video if blur_scene_video is None else bool(blur_scene_video)
    return profile, resolved_categories, resolved_scene_categories, resolved_mute_scene_audio, resolved_blur_scene_video


def process_video_job(
    video_url: str,
    selected_categories: Iterable[str] | None = None,
    selected_scene_categories: Iterable[str] | None = None,
    config_path: str = "config/bad_terms_categories.json",
    processing_profile: str | None = None,
    model_size: str = "small",
    mute_padding_seconds: float = 0.08,
    mute_scene_audio: bool | None = None,
    blur_scene_video: bool | None = None,
    blur_strength: float = 40.0,
    scene_nudity_threshold: float | None = None,
    scene_immodesty_threshold: float | None = None,
    scene_frame_interval_seconds: float | None = None,
    scene_threshold_overrides: dict[str, float] | None = None,
    artifacts_root: str = "artifacts",
) -> dict:
    if not video_url.strip():
        raise ValueError("Please provide a video URL.")

    config = load_term_config(config_path)
    language = get_config_language(config)
    profile, resolved_categories, resolved_scene_categories, resolved_mute_scene_audio, resolved_blur_scene_video = (
        _resolve_processing_inputs(
            config,
            processing_profile,
            selected_categories,
            selected_scene_categories,
            mute_scene_audio,
            blur_scene_video,
        )
    )
    bad_terms, caption_terms, context_exclusions = resolve_terms_from_categories(
        config,
        resolved_categories,
    )
    if not bad_terms and not caption_terms:
        raise ValueError("No active terms found for selected categories.")

    job_id, job_dir = _create_job_dir(artifacts_root)

    input_video = download_video_from_url(video_url.strip(), job_dir)
    audio_wav = extract_audio_wav(input_video, job_dir / "audio.wav")
    segments = transcribe_with_word_timestamps(
        str(audio_wav),
        language=language,
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
    scene_events = []
    if resolved_scene_categories:
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always", RuntimeWarning)
            scene_events = detect_scene_events(
                input_video,
                segments,
                scene_config,
                resolved_scene_categories,
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
    if resolved_mute_scene_audio:
        mute_spans.extend(scene_spans)

    output_video: str | None = None
    masked_srt: Path | None = None
    if profile.emit_masked_subtitles:
        masked_srt = write_masked_srt(segments, caption_terms or bad_terms, job_dir / "masked.srt")

    if profile.emit_output_video:
        muted_video = mute_occurrences_in_audio(input_video, job_dir / "muted.mp4", mute_spans)
        if not Path(muted_video).exists():
            raise RuntimeError(f"Muted video was not created: {muted_video}")

        video_for_subtitles = muted_video
        if resolved_blur_scene_video and scene_spans:
            blurred_video = blur_occurrences_in_video(
                muted_video,
                job_dir / "blurred.mp4",
                scene_spans,
                blur_strength=blur_strength,
            )
            video_for_subtitles = blurred_video

        if masked_srt is not None:
            output_video = str(burn_subtitles(video_for_subtitles, masked_srt, job_dir / "output_censored.mp4"))
        else:
            output_video = str(video_for_subtitles)

    json_path, csv_path, report_dict = write_report(
        occurrences,
        job_dir,
        scene_events=scene_events,
    )
    scene_gallery_items: list[tuple[str, str]] = []
    if profile.emit_scene_gallery and scene_events:
        scene_gallery_items = _build_scene_gallery(input_video, scene_events, job_dir / "scene_gallery")

    report_summary = {
        "job_id": job_id,
        "job_dir": str(job_dir),
        "config_path": str(Path(config_path).expanduser().resolve()),
        "language": language,
        "processing_profile": profile.name,
        "processing_profile_description": profile.description,
        "selected_categories": resolved_categories,
        "selected_scene_categories": resolved_scene_categories,
        "mute_scene_audio": bool(resolved_mute_scene_audio),
        "blur_scene_video": bool(resolved_blur_scene_video),
        "blur_strength": float(blur_strength),
        "emit_output_video": bool(profile.emit_output_video),
        "emit_masked_subtitles": bool(profile.emit_masked_subtitles),
        "emit_scene_gallery": bool(profile.emit_scene_gallery),
        "output_video": output_video,
        "masked_srt": str(masked_srt) if masked_srt is not None else None,
        "report_json": str(json_path),
        "report_csv": str(csv_path),
        "scene_events_csv": str(job_dir / "scene_events.csv"),
        "total_filtered": report_dict["total_filtered"],
        "counts": report_dict["counts"],
        "scene_total": report_dict.get("scene_total", 0),
        "scene_by_category": report_dict.get("scene_by_category", {}),
        "scene_gallery_dir": str(job_dir / "scene_gallery"),
        "scene_gallery_count": len(scene_gallery_items),
        "scene_gallery_items": scene_gallery_items,
        "effective_scene_config": _collect_effective_scene_thresholds(scene_config),
        "warnings": pipeline_warnings,
    }
    summary_path = job_dir / "job_summary.json"
    report_summary["job_summary"] = str(summary_path)
    summary_path.write_text(json.dumps(report_summary, indent=2), encoding="utf-8")
    return report_summary


def process_video_url(
    video_url: str,
    selected_categories: Iterable[str] | None,
    selected_scene_categories: Iterable[str] | None,
    config_path: str = "config/bad_terms_categories.json",
    processing_profile: str | None = None,
    model_size: str = "small",
    mute_padding_seconds: float = 0.08,
    mute_scene_audio: bool | None = None,
    blur_scene_video: bool | None = None,
    blur_strength: float = 40.0,
    scene_nudity_threshold: float | None = None,
    scene_immodesty_threshold: float | None = None,
    scene_frame_interval_seconds: float | None = None,
    scene_threshold_overrides: dict[str, float] | None = None,
    artifacts_root: str = "artifacts",
) -> tuple[str | None, str, str, list[tuple[str, str]]]:
    summary = process_video_job(
        video_url=video_url,
        selected_categories=selected_categories,
        selected_scene_categories=selected_scene_categories,
        config_path=config_path,
        processing_profile=processing_profile,
        model_size=model_size,
        mute_padding_seconds=mute_padding_seconds,
        mute_scene_audio=mute_scene_audio,
        blur_scene_video=blur_scene_video,
        blur_strength=blur_strength,
        scene_nudity_threshold=scene_nudity_threshold,
        scene_immodesty_threshold=scene_immodesty_threshold,
        scene_frame_interval_seconds=scene_frame_interval_seconds,
        scene_threshold_overrides=scene_threshold_overrides,
        artifacts_root=artifacts_root,
    )
    return (
        summary.get("output_video"),
        json.dumps(summary, indent=2),
        str(summary.get("counts", {})),
        summary.get("scene_gallery_items", []),
    )
