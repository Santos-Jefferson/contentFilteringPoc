# Content Filtering POC (Python + Gradio)

This POC downloads a video from a URL, transcribes English speech with word timestamps, mutes only filtered words (not whole frames), masks captions with `***`, and writes audit reports to calibrate the bad-word list.

## Features

- URL input (direct files and many video platforms via `yt-dlp`)
- English ASR with word-level timestamps (`faster-whisper`)
- Per-word muting windows using FFmpeg
- Configurable mute padding (before/after each detected bad word)
- Category-based filtering loaded from `config/bad_terms_categories.json`
- Scene-detection categories (transcript cues + optional frame sampling)
- Collapsed category groups with per-group select-all toggles
- VidAngel-style violence scene taxonomy (implied, non-graphic, graphic, gore, disturbing images, animal violence)
- ML-based violence frame detection (CLIP zero-shot labels) with per-category thresholds
- Strong blur on detected scene intervals (configurable intensity)
- Frame evidence gallery with per-event reason and threshold vs detected value
- Masked captions (`***`) burned into the output video
- JSON + CSV report of actual filtered words and timings

## Prerequisites

- Python 3.10+
- FFmpeg installed and available in `PATH`

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python3 main.py
```

Open the Gradio URL shown in your terminal.

Edit `config/bad_terms_categories.json` to manage bad terms and categories, then select categories in the UI instead of typing terms manually.

The UI now supports config-driven **processing profiles**. The default profile (`language_first_mvp`) keeps the MVP focused on language cleanup first, while `catalog_review_fast` skips final video rendering so you can triage titles asynchronously and build a catalog faster.

You can also select scene-detection categories, mute detected scene intervals, and apply strong blur to those same intervals.

The Gradio UI focuses on URL input first, with collapsed category groups for quick selection and subcategory checkboxes.

Violence categories are detected from transcript cues defined in `scene_detection.text_rules` and can be tuned in the config file.
For visual violence categories, the app supports two backends via `scene_detection.violence_backend`:

- `clip` (uses `violence_model_name`)
- `qwen_vl` (uses `violence_qwen_model_name`)

Both backends apply per-category thresholds (for example `violence_graphic_threshold`) plus a neutral-scene margin gate (`violence_score_margin_threshold`) to reduce false positives.

Use the **Mute padding per word (seconds)** slider to tune how much audio is muted around each detected word. If a word is detected but still audible, increase this value.

Use **Reload from config** to refresh category lists and threshold values from the selected config file without restarting Gradio. You can edit all threshold/interval values directly in the JSON overrides box.

Frame-based nudity detection is optional and requires:

```bash
pip install nudenet
```

Without `nudenet`, the app still runs scene detection using transcript text cues.

## Artifacts

Each run writes to `artifacts/<job_id>/`:

- `output_censored.mp4`
- `masked.srt`
- `filtered_report.json`
- `filtered_occurrences.csv`
- `scene_events.csv`
- `scene_gallery/` (representative flagged frames for manual review)
- `blurred.mp4` (only when scene blur is enabled and scene spans are detected)

## Smoke Test (no heavy dependencies)

```bash
python3 smoke_test.py
```

This only validates parsing/masking/report logic with synthetic transcript data.

## Batch manifest mode

To process many titles asynchronously and build catalog artifacts, create a JSON manifest with shared defaults plus a `jobs` array:

```json
{
  "defaults": {
	"processing_profile": "catalog_review_fast",
	"model_size": "tiny"
  },
  "jobs": [
	{
	  "title": "Example Show S01E01",
	  "source_id": "example-s01e01",
	  "video_url": "https://example.com/video.mp4"
	}
  ]
}
```

Run it with:

```bash
python3 batch_manifest.py manifest.json --concurrency 2
```

This writes normal per-job artifacts plus a batch summary JSON under `artifacts/batches/<batch_id>/batch_summary.json`.

## SSL download troubleshooting

If video download fails with errors like `CERTIFICATE_VERIFY_FAILED`, first update your environment trust store and reinstall dependencies so `certifi` is available.

As a temporary workaround in restricted/proxied environments, you can disable TLS certificate checks for downloads:

```bash
CONTENT_FILTERING_INSECURE_SSL=1 python3 main.py
```

Use this only when necessary.

