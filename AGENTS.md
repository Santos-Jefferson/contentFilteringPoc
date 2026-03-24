# AGENTS Guide

## Project Purpose
- This app censors video from a URL/local path by muting transcript matches, optionally flagging/muting/blurring scene intervals, and burning masked subtitles (`main.py`, `src/pipeline.py`).
- UI-first entrypoint is `python3 main.py` (Gradio), but core behavior lives in `process_video_url(...)` in `src/pipeline.py`.

## Architecture and Data Flow
- Flow is linear and file-backed: download -> extract audio -> ASR -> detect terms/scenes -> mute/blur -> write SRT -> burn subtitles -> write reports (`src/pipeline.py`).
- `src/video_io.py` owns all FFmpeg/ffprobe calls and URL download logic (`yt-dlp` first, urllib fallback for direct media URLs). Local file paths (bare path or `file://`) are also accepted by `download_video_from_url(...)`.
- `src/transcription.py` is intentionally thin: wraps `faster_whisper.WhisperModel.transcribe(..., word_timestamps=True, vad_filter=True)` and converts to dataclasses.
- `process_video_job(...)` in `src/pipeline.py` is now the machine-facing seam; `process_video_url(...)` is the UI wrapper that formats the same job summary for Gradio.
- Filtering is split by term type in `src/filtering.py`: single-token muting via `bad_terms`; multi-word phrase muting via `phrase_terms`. `resolve_terms_from_categories(...)` returns `(token_terms, caption_terms, context_exclusions)` — `token_terms` (from `word_terms`) drives audio muting; `caption_terms` (word + phrase terms) drives subtitle masking. The pipeline splits out `phrase_terms = {t for t in caption_terms if len(t.split()) >= 2}` before calling `find_filtered_occurrences(...)`.
- Scene detection merges three sources in `src/scene_detection.py`: transcript cue hits (`_detect_text_scene_events`), NudeNet frame labels (`_detect_frame_scene_events`), and violence frame models (`_detect_violence_frame_events`, CLIP or Qwen-VL), then merges overlaps by category. RuntimeWarnings from this step are collected via `warnings.catch_warnings` in the pipeline and surfaced in `job_summary["warnings"]`.
- Blur spans are pre-merged via `_merge_spans(...)` in `src/video_io.py` (0.05 s bridge tolerance) before the blur pass; mute spans are **not** pre-merged and are passed individually to the FFmpeg `volume` filter.
- Reporting is artifact-driven: JSON + CSV term occurrences and scene events via `write_report(...)` in `src/reporting.py`.
- Batch catalog runs live in `src/batch_runner.py` and `batch_manifest.py`; they read a JSON manifest, fan jobs out with bounded concurrency using `asyncio.to_thread` (thread-pool, not multiprocessing), and write `artifacts/batches/<batch_id>/batch_summary.json`.

## Config-Driven Behavior (Critical)
- Canonical config is `config/bad_terms_categories.json`; UI defaults and category lists are loaded from it (`src/term_config.py`, `main.py`).
- Processing profiles also live in config (`default_processing_profile`, `processing_profiles.*`) and are resolved via `resolve_processing_profile(...)`; the default profile is currently `language_first_mvp`.
- `MANUAL_PROFILE_NAME = "manual_custom"` (sentinel in `src/term_config.py`) is used when no named profile is selected; it uses config-default categories and scene categories rather than a profile payload.
- Audio categories provide `word_terms`, `phrase_terms`, and optional `context_exclusions` (example: suppress `hell` when nearby word includes `heaven`).
- If no audio categories are selected, `resolve_terms_from_categories(...)` defaults to all configured categories.
- Scene thresholds/intervals are JSON-overridable in UI and merged in pipeline; keys containing `threshold` are clamped to 0.01-0.99 and `interval` to >=0.05.
- Scene category defaults come from `scene_detection.categories.*.enabled_by_default`; these drive initial checkbox state in Gradio.
- Profiles can disable heavy outputs (`emit_output_video`, `emit_masked_subtitles`, `emit_scene_gallery`) for catalog triage jobs; `catalog_review_fast` intentionally skips final video rendering but still writes reports.

## Developer Workflows
- Setup/run (from `README.md`): create venv, `pip install -r requirements.txt`, then `python3 main.py`.
- Lightweight validation is `python3 smoke_test.py`; this avoids heavy runtime model downloads and exercises parsing/filtering/reporting logic, phrase detection, violence transcript cues, SRT generation, and batch manifest loading.
- Batch/catalog processing runs through `python3 batch_manifest.py manifest.json --concurrency 2`; manifest defaults can set `processing_profile`, `model_size`, and per-job overrides.
- Artifacts are per run in `artifacts/<timestamp_uuid>/` and include `output_censored.mp4`, `masked.srt`, `filtered_report.json`, `filtered_occurrences.csv`, `scene_events.csv`, and `scene_gallery/`. When blur is applied and scene spans are found, `blurred.mp4` is also written as an intermediate before subtitle burn.
- Every pipeline job now also writes `job_summary.json` into its artifact directory for automation to consume.

## Integration and Runtime Expectations
- Required system tools: `ffmpeg` and `ffprobe` must be in `PATH` (`src/video_io.py`).
- Optional frame nudity detection requires `nudenet`; if missing, runtime warning is emitted and frame nudity categories are skipped.
- Violence frame backend is selected by `scene_detection.violence_backend` (`clip` or `qwen_vl`) and loaded dynamically through `transformers` pipelines.
- Download behavior: `yt-dlp` is preferred; direct URL fallback validates content-type/container and fails fast on HTML/invalid media responses.
- Local file input: `video_url` accepts bare filesystem paths or `file://` URIs; `download_video_from_url(...)` copies the file into the job directory before processing.
- TLS: `certifi` is used when available; set `CONTENT_FILTERING_INSECURE_SSL=1` (or `CONTENT_FILTER_INSECURE_SSL=1`) to bypass certificate verification in proxied/restricted environments.
- YouTube / bot-gated platforms: set `YT_DLP_COOKIES_FILE=/path/to/cookies.txt` (Netscape format, takes precedence) or `YT_DLP_COOKIES_FROM_BROWSER=chrome` (also `firefox`, `safari`) when yt-dlp fails with "Sign in to confirm you're not a bot". `cookiesfrombrowser` requires local browser profile access and won't work in headless/remote environments.

## Codebase-Specific Conventions and Gotchas
- Use dataclasses from `src/models.py` (`TranscriptSegment`, `WordToken`, `FilteredOccurrence`, `SceneDetectionEvent`) across modules; avoid dict-shaped ad hoc payloads in core logic.
- Additional dataclasses (`ProcessingProfile`, `BatchManifestEntry`, `BatchManifestResult`, `BatchRunSummary`) carry config/profile/batch state; prefer them over loose manifest dicts once parsing leaves IO boundaries.
- `src/subtitle_utils.py` and `src/subtitles.py` are currently duplicates; pipeline imports `write_masked_srt` from `src/subtitle_utils.py`. `src/subtitles.py` is dead code — do not add logic there.
- `process_video_job(...)` is the correct seam for tests/automation; `process_video_url(...)` is the Gradio wrapper only. UI (`main.py`) should stay orchestration-focused.
- Current behavior: scene spans are only added to mute spans when `mute_scene_audio` is enabled; blur no longer implicitly mutes scene audio, which matters for language-first / review-first profiles.
- Frame-nudity categories that require `nudenet` are enumerated in `_FRAME_ONLY_CATEGORIES` in `src/scene_detection.py` (e.g. `shown_with_nudity`, `implied_nudity`, `immodesty_female`, `immodesty_male`); transcript-cue and violence-frame detection continue without it.
- Context exclusion window in `_is_context_excluded(...)` spans 2 tokens before and 2 tokens after the matched index (±2 neighbors).

