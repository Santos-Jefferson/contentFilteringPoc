import subprocess
import tempfile
from pathlib import Path
from typing import Iterable, List, Set

from src.models import SceneDetectionEvent, TranscriptSegment


_FRAME_ONLY_CATEGORIES = {
    "shown_with_nudity",
    "nudity_without_sex",
    "implied_nudity",
    "female_male_nudity",
    "immodesty_female",
    "immodesty_male",
}

_SEVERE_NUDITY_LABELS = {
    "ANUS_EXPOSED",
    "FEMALE_GENITALIA_EXPOSED",
    "MALE_GENITALIA_EXPOSED",
}

_IMMODESTY_FEMALE_LABELS = {
    "FEMALE_BREAST_COVERED",
    "FEMALE_BREAST_EXPOSED",
    "FEMALE_GENITALIA_COVERED",
    "FEMALE_GENITALIA_EXPOSED",
    "BUTTOCKS_COVERED",
    "BUTTOCKS_EXPOSED",
}

_IMMODESTY_MALE_LABELS = {
    "MALE_GENITALIA_COVERED",
    "MALE_GENITALIA_EXPOSED",
    "BUTTOCKS_COVERED",
    "BUTTOCKS_EXPOSED",
}


def detect_scene_events(
    video_path: Path,
    segments: Iterable[TranscriptSegment],
    scene_config: dict,
    selected_scene_categories: Iterable[str],
) -> List[SceneDetectionEvent]:
    selected = set(selected_scene_categories or [])
    if not selected:
        return []

    events: List[SceneDetectionEvent] = []
    events.extend(_detect_text_scene_events(segments, scene_config, selected))
    events.extend(_detect_frame_scene_events(video_path, scene_config, selected))
    return _merge_overlapping_events(events)


def _detect_text_scene_events(
    segments: Iterable[TranscriptSegment],
    scene_config: dict,
    selected: Set[str],
) -> List[SceneDetectionEvent]:
    text_rules = scene_config.get("text_rules", {})
    events: List[SceneDetectionEvent] = []

    for segment in segments:
        segment_text = (segment.text or "").lower()
        if not segment_text:
            continue

        for category in selected:
            cues = text_rules.get(category, [])
            if any(cue.lower() in segment_text for cue in cues):
                events.append(
                    SceneDetectionEvent(
                        category=category,
                        start=float(segment.start),
                        end=float(segment.end),
                        score=0.55,
                        source="transcript",
                    )
                )
    return events


def _detect_frame_scene_events(
    video_path: Path,
    scene_config: dict,
    selected: Set[str],
) -> List[SceneDetectionEvent]:
    if not (selected & _FRAME_ONLY_CATEGORIES):
        return []

    try:
        from nudenet import NudeDetector
    except Exception:
        import warnings
        needs_frame = selected & _FRAME_ONLY_CATEGORIES
        warnings.warn(
            f"nudenet is not installed. Frame-level detection is disabled for: "
            f"{sorted(needs_frame)}. Run `pip install nudenet` to enable it.",
            RuntimeWarning,
            stacklevel=2,
        )
        return []

    interval = float(scene_config.get("frame_interval_seconds", 1.0))
    threshold = float(scene_config.get("nudity_score_threshold", 0.5))
    # Covered body parts (e.g. bikini) score lower than exposed ones in NudeNet;
    # use a separate, lower threshold so immodesty categories are not missed.
    immodesty_threshold = float(scene_config.get("immodesty_score_threshold", 0.3))
    interval = max(0.25, interval)
    detector = NudeDetector()

    events: List[SceneDetectionEvent] = []
    with tempfile.TemporaryDirectory() as tmp_dir_name:
        tmp_dir = Path(tmp_dir_name)
        pattern = tmp_dir / "frame_%06d.jpg"
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-vf",
            f"fps=1/{interval}",
            str(pattern),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            return []

        frames = sorted(tmp_dir.glob("frame_*.jpg"))
        for index, frame_path in enumerate(frames):
            detections = detector.detect(str(frame_path))
            matched_categories, score = _map_nudity_to_categories(
                detections, threshold, immodesty_threshold, selected
            )
            if not matched_categories:
                continue

            start = index * interval
            end = start + interval
            for category in matched_categories:
                events.append(
                    SceneDetectionEvent(
                        category=category,
                        start=float(start),
                        end=float(end),
                        score=float(score),
                        source="frame",
                    )
                )

    return events


def _max_label_score(detections: list[dict], label_set: Set[str]) -> float:
    """Return the highest NudeNet score for detections whose class is in *label_set*."""
    scores = [
        float(item.get("score", 0.0))
        for item in detections
        if str(item.get("class", "")).upper() in label_set
    ]
    return max(scores) if scores else 0.0


def _map_nudity_to_categories(
    detections: list[dict],
    threshold: float,
    immodesty_threshold: float,
    selected: Set[str],
) -> tuple[Set[str], float]:
    if not detections:
        return set(), 0.0

    overall_score = max(float(item.get("score", 0.0)) for item in detections)
    all_labels = {str(item.get("class", "")).upper() for item in detections}
    matched: Set[str] = set()

    # ── Immodesty categories: use per-label scores with a lower threshold ──
    # Covered labels (FEMALE_BREAST_COVERED, BUTTOCKS_COVERED, …) are typical
    # for a bikini but tend to score below the explicit-nudity threshold.
    if "immodesty_female" in selected:
        score = _max_label_score(detections, _IMMODESTY_FEMALE_LABELS)
        if score >= immodesty_threshold:
            matched.add("immodesty_female")

    if "immodesty_male" in selected:
        score = _max_label_score(detections, _IMMODESTY_MALE_LABELS)
        if score >= immodesty_threshold:
            matched.add("immodesty_male")

    # ── Explicit-nudity categories: require the stricter global threshold ──
    if overall_score >= threshold:
        if "shown_with_nudity" in selected:
            matched.add("shown_with_nudity")
        if "female_male_nudity" in selected and all_labels & _SEVERE_NUDITY_LABELS:
            matched.add("female_male_nudity")
        if "nudity_without_sex" in selected:
            matched.add("nudity_without_sex")
        if "implied_nudity" in selected and overall_score >= max(0.5, threshold - 0.1):
            matched.add("implied_nudity")

    return matched, overall_score


def _merge_overlapping_events(events: List[SceneDetectionEvent]) -> List[SceneDetectionEvent]:
    if not events:
        return []

    merged: List[SceneDetectionEvent] = []
    for category in sorted({event.category for event in events}):
        group = sorted((event for event in events if event.category == category), key=lambda e: e.start)
        current = group[0]
        for nxt in group[1:]:
            if nxt.start <= current.end + 0.05:
                current.end = max(current.end, nxt.end)
                current.score = max(current.score, nxt.score)
                if current.source != "frame" and nxt.source == "frame":
                    current.source = "frame"
            else:
                merged.append(current)
                current = nxt
        merged.append(current)

    return merged

