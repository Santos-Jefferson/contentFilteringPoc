import subprocess
import tempfile
import warnings
import importlib
import json
import re
from pathlib import Path
from typing import Any, Iterable, List, Set

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

_VIOLENCE_FRAME_CATEGORIES = {
    "violence_non_graphic",
    "violence_graphic",
    "violence_gore",
    "violence_disturbing_images",
    "violence_animal_violence",
}

_VIOLENCE_LABELS_BY_CATEGORY = {
    "violence_non_graphic": [
        "physical fight",
        "punching or kicking",
        "person holding a weapon",
    ],
    "violence_graphic": [
        "bloody injury",
        "visible blood on body",
        "graphic violence",
    ],
    "violence_gore": [
        "gore",
        "severed body part",
        "disembowelment",
    ],
    "violence_disturbing_images": [
        "mass grave",
        "disturbing dead body",
        "burning dead body",
    ],
    "violence_animal_violence": [
        "animal abuse",
        "animal being hurt",
        "injured animal from violence",
    ],
}

_VIOLENCE_NEGATIVE_LABELS = [
    "normal conversation",
    "person talking to camera",
    "outdoor landscape",
    "city street scene",
    "family photo",
    "sports game",
]


_VIOLENCE_THRESHOLD_DEFAULTS = {
    "violence_non_graphic": 0.26,
    "violence_graphic": 0.24,
    "violence_gore": 0.22,
    "violence_disturbing_images": 0.24,
    "violence_animal_violence": 0.24,
}

_VIOLENCE_BACKENDS = {"clip", "qwen_vl"}

_QWEN_VIOLENCE_PROMPT_TEMPLATE = (
    "Analyze this frame for violence moderation. Return ONLY JSON with numeric scores between 0 and 1 "
    "for these keys: {keys}. Also include non_violent and reason keys."
)


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
    events.extend(_detect_violence_frame_events(video_path, scene_config, selected))
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
            matched_cue = next((cue for cue in cues if cue.lower() in segment_text), None)
            if matched_cue:
                events.append(
                    SceneDetectionEvent(
                        category=category,
                        start=float(segment.start),
                        end=float(segment.end),
                        score=0.55,
                        source="transcript",
                        detected_value=1.0,
                        threshold_used=1.0,
                        reason=f"Transcript cue matched: {matched_cue}",
                    )
                )
    return events


def _detect_frame_scene_events(
    video_path: Path,
    scene_config: dict,
    selected: Set[str],
) -> List[SceneDetectionEvent]:
    if not video_path.exists() or video_path.is_dir():
        return []

    if not (selected & _FRAME_ONLY_CATEGORIES):
        return []

    try:
        from nudenet import NudeDetector
    except Exception:
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
            category_evidence = _map_nudity_to_categories(
                detections, threshold, immodesty_threshold, selected
            )
            if not category_evidence:
                continue

            start = index * interval
            end = start + interval
            for category, evidence in category_evidence.items():
                detected_value, threshold_used, reason = evidence
                events.append(
                    SceneDetectionEvent(
                        category=category,
                        start=float(start),
                        end=float(end),
                        score=float(detected_value),
                        source="frame",
                        detected_value=float(detected_value),
                        threshold_used=float(threshold_used),
                        reason=reason,
                    )
                )

    return events


def _detect_violence_frame_events(
    video_path: Path,
    scene_config: dict,
    selected: Set[str],
) -> List[SceneDetectionEvent]:
    if not video_path.exists() or video_path.is_dir():
        return []

    selected_violence = selected.intersection(_VIOLENCE_FRAME_CATEGORIES)
    if not selected_violence:
        return []

    backend_name = _get_violence_backend(scene_config)
    classifier = _load_violence_backend(scene_config, backend_name)
    if classifier is None:
        warnings.warn(
            "Violence frame model is unavailable. Install required model dependencies "
            "to enable ML-based violence scene detection.",
            RuntimeWarning,
            stacklevel=2,
        )
        return []

    base_interval = float(scene_config.get("frame_interval_seconds", 1.0))
    interval = float(scene_config.get("violence_frame_interval_seconds", base_interval))
    margin_threshold = float(scene_config.get("violence_score_margin_threshold", 0.1))
    interval = max(0.25, interval)

    events: List[SceneDetectionEvent] = []
    with tempfile.TemporaryDirectory() as tmp_dir_name:
        tmp_dir = Path(tmp_dir_name)
        pattern = tmp_dir / "violence_frame_%06d.jpg"
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

        frames = sorted(tmp_dir.glob("violence_frame_*.jpg"))
        candidate_labels = _build_violence_candidate_labels(selected_violence)
        for index, frame_path in enumerate(frames):
            clip_best_label_by_category: dict[str, str] = {}
            if backend_name == "qwen_vl":
                frame_scores, neutral_score, backend_reason = _run_qwen_violence_classifier(
                    classifier,
                    str(frame_path),
                    selected_violence,
                    scene_config,
                )
            else:
                raw_scores = _run_violence_classifier(classifier, str(frame_path), candidate_labels)
                if not raw_scores:
                    continue
                frame_scores = _clip_scores_to_category_scores(raw_scores, selected_violence)
                neutral_score = _max_score_from_labels(raw_scores, _VIOLENCE_NEGATIVE_LABELS)
                for category in selected_violence:
                    labels = _VIOLENCE_LABELS_BY_CATEGORY.get(category, [])
                    best_label, _ = _max_score_label(raw_scores, labels)
                    clip_best_label_by_category[category] = best_label
                backend_reason = "CLIP zero-shot label scores"

            if not frame_scores:
                continue

            frame_start = index * interval
            frame_end = frame_start + interval
            for category in selected_violence:
                category_score = float(frame_scores.get(category, 0.0))
                if category_score <= 0.0:
                    continue

                if backend_name == "qwen_vl":
                    detected_label = category
                    detected_value = category_score
                else:
                    detected_label = clip_best_label_by_category.get(category, "")
                    detected_value = category_score
                    if not detected_label:
                        continue

                threshold_key = f"{category}_threshold"
                threshold_default = _VIOLENCE_THRESHOLD_DEFAULTS.get(category, 0.25)
                threshold_used = float(scene_config.get(threshold_key, threshold_default))
                if detected_value < threshold_used:
                    continue
                score_margin = detected_value - neutral_score
                if score_margin < margin_threshold:
                    continue

                reason = (
                    f"{backend_reason}; matched '{detected_label}' score {detected_value:.3f} "
                    f"(threshold {threshold_used:.3f}, neutral {neutral_score:.3f}, margin {score_margin:.3f})"
                )
                events.append(
                    SceneDetectionEvent(
                        category=category,
                        start=float(frame_start),
                        end=float(frame_end),
                        score=float(detected_value),
                        source="frame",
                        detected_value=float(detected_value),
                        threshold_used=float(threshold_used),
                        reason=reason,
                    )
                )

    return events


def _get_violence_backend(scene_config: dict) -> str:
    configured = str(scene_config.get("violence_backend", "clip")).strip().lower()
    if configured in _VIOLENCE_BACKENDS:
        return configured
    return "clip"


def _load_violence_backend(scene_config: dict, backend_name: str):
    if backend_name == "qwen_vl":
        return _load_qwen_violence_classifier(scene_config)
    return _load_clip_violence_classifier(scene_config)


def _load_clip_violence_classifier(scene_config: dict):
    model_name = str(scene_config.get("violence_model_name", "openai/clip-vit-base-patch32"))
    try:
        transformers = importlib.import_module("transformers")
        pipeline_fn = getattr(transformers, "pipeline")

        return pipeline_fn(
            task="zero-shot-image-classification",
            model=model_name,
        )
    except Exception:
        return None


def _load_qwen_violence_classifier(scene_config: dict):
    model_name = str(scene_config.get("violence_qwen_model_name", "Qwen/Qwen2.5-VL-3B-Instruct"))
    try:
        transformers = importlib.import_module("transformers")
        pipeline_fn = getattr(transformers, "pipeline")
        return pipeline_fn(
            task="image-text-to-text",
            model=model_name,
        )
    except Exception:
        return None


def _build_violence_candidate_labels(selected_violence: Set[str]) -> list[str]:
    labels: list[str] = []
    for category in sorted(selected_violence):
        labels.extend(_VIOLENCE_LABELS_BY_CATEGORY.get(category, []))
    labels.extend(_VIOLENCE_NEGATIVE_LABELS)
    # Preserve order while removing duplicates.
    return list(dict.fromkeys(labels))


def _run_violence_classifier(classifier, image_path: str, labels: list[str]) -> dict[str, float]:
    try:
        output = classifier(image_path, candidate_labels=labels)
    except Exception:
        return {}

    if not isinstance(output, list):
        return {}
    scores: dict[str, float] = {}
    for item in output:
        label = str(item.get("label", ""))
        score = float(item.get("score", 0.0))
        if label:
            scores[label] = max(scores.get(label, 0.0), score)
    return scores


def _run_qwen_violence_classifier(
    classifier,
    image_path: str,
    selected_violence: Set[str],
    scene_config: dict,
) -> tuple[dict[str, float], float, str]:
    prompt = str(
        scene_config.get(
            "violence_qwen_prompt",
            _QWEN_VIOLENCE_PROMPT_TEMPLATE.format(keys=sorted(selected_violence)),
        )
    )
    max_new_tokens = int(scene_config.get("violence_qwen_max_new_tokens", 220))

    outputs: Any = None
    call_attempts = [
        lambda: classifier(text=prompt, images=image_path, max_new_tokens=max_new_tokens),
        lambda: classifier(image_path, prompt=prompt, max_new_tokens=max_new_tokens),
        lambda: classifier(
            [{"role": "user", "content": [{"type": "image", "image": image_path}, {"type": "text", "text": prompt}]}],
            max_new_tokens=max_new_tokens,
        ),
    ]
    for attempt in call_attempts:
        try:
            outputs = attempt()
            break
        except Exception:
            continue

    if outputs is None:
        return {}, 0.0, "Qwen-VL call failed"

    generated_text = _extract_generated_text(outputs)
    parsed = _parse_qwen_scores_json(generated_text, selected_violence)
    if not parsed:
        return {}, 0.0, "Qwen-VL produced non-JSON or invalid output"

    neutral_score = float(parsed.get("non_violent", 0.0))
    scores = {key: float(parsed.get(key, 0.0)) for key in selected_violence}
    reason = str(parsed.get("reason", "Qwen-VL structured frame analysis"))
    return scores, neutral_score, reason


def _extract_generated_text(outputs: Any) -> str:
    if isinstance(outputs, str):
        return outputs
    if isinstance(outputs, dict):
        if "generated_text" in outputs:
            return str(outputs["generated_text"])
        return json.dumps(outputs)
    if isinstance(outputs, list) and outputs:
        first = outputs[0]
        if isinstance(first, dict):
            if "generated_text" in first:
                return str(first["generated_text"])
            if "text" in first:
                return str(first["text"])
        return str(first)
    return ""


def _parse_qwen_scores_json(text: str, selected_violence: Set[str]) -> dict[str, float | str] | None:
    if not text:
        return None

    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()

    candidates = [cleaned]
    first_brace = cleaned.find("{")
    last_brace = cleaned.rfind("}")
    if 0 <= first_brace < last_brace:
        candidates.append(cleaned[first_brace : last_brace + 1])

    loaded: dict[str, Any] | None = None
    for candidate in candidates:
        try:
            obj = json.loads(candidate)
        except Exception:
            continue
        if isinstance(obj, dict):
            loaded = obj
            break

    if not loaded:
        return None

    # Some prompts return nested structures like {"scores": {...}, "reason": ...}
    score_source: dict[str, Any]
    if isinstance(loaded.get("scores"), dict):
        score_source = dict(loaded.get("scores") or {})
    else:
        score_source = loaded

    normalized: dict[str, float | str] = {}
    for key in selected_violence:
        normalized[key] = _to_probability(score_source.get(key, 0.0))
    normalized["non_violent"] = _to_probability(score_source.get("non_violent", loaded.get("non_violent", 0.0)))
    normalized["reason"] = str(loaded.get("reason", "Qwen-VL structured frame analysis"))
    return normalized


def _to_probability(value: Any) -> float:
    try:
        num = float(value)
    except Exception:
        return 0.0
    return max(0.0, min(1.0, num))


def _clip_scores_to_category_scores(scores: dict[str, float], selected_violence: Set[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for category in selected_violence:
        labels = _VIOLENCE_LABELS_BY_CATEGORY.get(category, [])
        out[category] = _max_score_from_labels(scores, labels)
    return out


def _max_score_label(scores: dict[str, float], labels: list[str]) -> tuple[str, float]:
    best_label = ""
    best_score = 0.0
    for label in labels:
        value = float(scores.get(label, 0.0))
        if value > best_score:
            best_label = label
            best_score = value
    return best_label, best_score


def _max_score_from_labels(scores: dict[str, float], labels: list[str]) -> float:
    best = 0.0
    for label in labels:
        best = max(best, float(scores.get(label, 0.0)))
    return best


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
) -> dict[str, tuple[float, float, str]]:
    if not detections:
        return {}

    overall_score = max(float(item.get("score", 0.0)) for item in detections)
    all_labels = {str(item.get("class", "")).upper() for item in detections}
    matched: dict[str, tuple[float, float, str]] = {}

    # ── Immodesty categories: use per-label scores with a lower threshold ──
    # Covered labels (FEMALE_BREAST_COVERED, BUTTOCKS_COVERED, …) are typical
    # for a bikini but tend to score below the explicit-nudity threshold.
    if "immodesty_female" in selected:
        score = _max_label_score(detections, _IMMODESTY_FEMALE_LABELS)
        if score >= immodesty_threshold:
            matched["immodesty_female"] = (
                score,
                immodesty_threshold,
                "Frame labels matched female immodesty set",
            )

    if "immodesty_male" in selected:
        score = _max_label_score(detections, _IMMODESTY_MALE_LABELS)
        if score >= immodesty_threshold:
            matched["immodesty_male"] = (
                score,
                immodesty_threshold,
                "Frame labels matched male immodesty set",
            )

    # ── Explicit-nudity categories: require the stricter global threshold ──
    if overall_score >= threshold:
        if "shown_with_nudity" in selected:
            matched["shown_with_nudity"] = (
                overall_score,
                threshold,
                "Overall nudity score exceeded threshold",
            )
        if "female_male_nudity" in selected and all_labels & _SEVERE_NUDITY_LABELS:
            matched["female_male_nudity"] = (
                overall_score,
                threshold,
                "Severe nudity label detected",
            )
        if "nudity_without_sex" in selected:
            matched["nudity_without_sex"] = (
                overall_score,
                threshold,
                "Overall nudity score exceeded threshold",
            )
        if "implied_nudity" in selected and overall_score >= max(0.5, threshold - 0.1):
            implied_threshold = max(0.5, threshold - 0.1)
            matched["implied_nudity"] = (
                overall_score,
                implied_threshold,
                "Implied nudity score exceeded threshold",
            )

    return matched


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

