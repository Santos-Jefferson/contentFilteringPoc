import asyncio
import json
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.models import BatchManifestEntry, BatchManifestResult, BatchRunSummary
from src.pipeline import process_video_job


DEFAULT_CONFIG_PATH = "config/bad_terms_categories.json"


def _coerce_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    items: list[str] = []
    seen: set[str] = set()
    for raw in value:
        item = str(raw).strip()
        if not item or item in seen:
            continue
        items.append(item)
        seen.add(item)
    return items


def _coerce_float_dict(value: Any) -> dict[str, float]:
    if not isinstance(value, dict):
        return {}
    parsed: dict[str, float] = {}
    for key, raw in value.items():
        try:
            parsed[str(key)] = float(raw)
        except (TypeError, ValueError):
            continue
    return parsed


def _load_manifest_payload(manifest_path: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    path = Path(manifest_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Batch manifest not found: {path}")

    loaded = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(loaded, list):
        return {}, [item for item in loaded if isinstance(item, dict)]
    if not isinstance(loaded, dict):
        raise ValueError("Batch manifest must be a JSON object with a 'jobs' array or a plain JSON array of jobs.")

    jobs = loaded.get("jobs", [])
    if not isinstance(jobs, list):
        raise ValueError("Batch manifest field 'jobs' must be a JSON array.")
    return loaded, [item for item in jobs if isinstance(item, dict)]


def _manifest_entry_from_payload(defaults: dict[str, Any], raw_job: dict[str, Any]) -> BatchManifestEntry:
    merged = dict(defaults or {})
    merged.update(raw_job or {})

    video_url = str(merged.get("video_url", "")).strip()
    if not video_url:
        raise ValueError("Each batch job must include a non-empty 'video_url'.")

    return BatchManifestEntry(
        video_url=video_url,
        title=str(merged.get("title", "")).strip(),
        source_id=str(merged.get("source_id", merged.get("external_id", ""))).strip(),
        processing_profile=str(merged.get("processing_profile", "")).strip() or None,
        selected_categories=_coerce_string_list(merged.get("selected_categories", [])),
        selected_scene_categories=_coerce_string_list(merged.get("selected_scene_categories", [])),
        model_size=str(merged.get("model_size", "small")).strip() or "small",
        mute_padding_seconds=float(merged.get("mute_padding_seconds", 0.08)),
        mute_scene_audio=merged.get("mute_scene_audio"),
        blur_scene_video=merged.get("blur_scene_video"),
        blur_strength=float(merged.get("blur_strength", 40.0)),
        scene_threshold_overrides=_coerce_float_dict(merged.get("scene_threshold_overrides", {})),
    )


def load_manifest(manifest_path: str) -> tuple[dict[str, Any], list[BatchManifestEntry]]:
    payload, raw_jobs = _load_manifest_payload(manifest_path)
    defaults = payload.get("defaults", {}) if isinstance(payload.get("defaults", {}), dict) else {}
    entries = [_manifest_entry_from_payload(defaults, raw_job) for raw_job in raw_jobs]
    return payload, entries


def _create_batch_summary_dir(artifacts_root: str) -> tuple[str, Path]:
    batch_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_batch_" + uuid.uuid4().hex[:8]
    path = Path(artifacts_root).expanduser().resolve() / "batches" / batch_id
    path.mkdir(parents=True, exist_ok=True)
    return batch_id, path


async def _run_manifest_entry(
    entry: BatchManifestEntry,
    config_path: str,
    artifacts_root: str,
) -> BatchManifestResult:
    try:
        summary = await asyncio.to_thread(
            process_video_job,
            video_url=entry.video_url,
            selected_categories=entry.selected_categories or None,
            selected_scene_categories=entry.selected_scene_categories or None,
            config_path=config_path,
            processing_profile=entry.processing_profile,
            model_size=entry.model_size,
            mute_padding_seconds=entry.mute_padding_seconds,
            mute_scene_audio=entry.mute_scene_audio,
            blur_scene_video=entry.blur_scene_video,
            blur_strength=entry.blur_strength,
            scene_threshold_overrides=entry.scene_threshold_overrides or None,
            artifacts_root=artifacts_root,
        )
        return BatchManifestResult(
            video_url=entry.video_url,
            title=entry.title,
            source_id=entry.source_id,
            status="succeeded",
            processing_profile=summary.get("processing_profile"),
            job_id=str(summary.get("job_id", "")),
            job_dir=str(summary.get("job_dir", "")),
            output_video=str(summary.get("output_video") or ""),
            summary_path=str(summary.get("job_summary", "")),
            total_filtered=int(summary.get("total_filtered", 0)),
            scene_total=int(summary.get("scene_total", 0)),
        )
    except Exception as exc:
        return BatchManifestResult(
            video_url=entry.video_url,
            title=entry.title,
            source_id=entry.source_id,
            status="failed",
            processing_profile=entry.processing_profile,
            error=str(exc),
        )


def write_batch_summary(summary: BatchRunSummary, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "batch_summary.json"
    payload = asdict(summary)
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return summary_path


async def run_manifest_async(
    manifest_path: str,
    config_path: str | None = None,
    artifacts_root: str = "artifacts",
    concurrency: int | None = None,
) -> BatchRunSummary:
    payload, entries = load_manifest(manifest_path)
    resolved_config_path = str(config_path or payload.get("config_path", DEFAULT_CONFIG_PATH))
    resolved_artifacts_root = str(payload.get("artifacts_root", artifacts_root))
    resolved_concurrency = max(1, int(concurrency or payload.get("concurrency", 1)))

    batch_id, batch_dir = _create_batch_summary_dir(resolved_artifacts_root)
    semaphore = asyncio.Semaphore(resolved_concurrency)

    async def _guarded_run(entry: BatchManifestEntry) -> BatchManifestResult:
        async with semaphore:
            return await _run_manifest_entry(entry, resolved_config_path, resolved_artifacts_root)

    results = await asyncio.gather(*[_guarded_run(entry) for entry in entries])
    summary = BatchRunSummary(
        batch_id=batch_id,
        manifest_path=str(Path(manifest_path).expanduser().resolve()),
        total_jobs=len(entries),
        succeeded=sum(1 for item in results if item.status == "succeeded"),
        failed=sum(1 for item in results if item.status != "succeeded"),
        results=list(results),
    )
    summary_path = write_batch_summary(summary, batch_dir)
    summary.summary_path = str(summary_path)
    summary_path.write_text(json.dumps(asdict(summary), indent=2), encoding="utf-8")
    return summary


def run_manifest(
    manifest_path: str,
    config_path: str | None = None,
    artifacts_root: str = "artifacts",
    concurrency: int | None = None,
) -> BatchRunSummary:
    return asyncio.run(
        run_manifest_async(
            manifest_path=manifest_path,
            config_path=config_path,
            artifacts_root=artifacts_root,
            concurrency=concurrency,
        )
    )

