# Content Filtering POC (Python + Gradio)

This POC downloads a video from a URL, transcribes English speech with word timestamps, mutes only filtered words (not whole frames), masks captions with `***`, and writes audit reports to calibrate the bad-word list.

## Features

- URL input (direct files and many video platforms via `yt-dlp`)
- English ASR with word-level timestamps (`faster-whisper`)
- Per-word muting windows using FFmpeg
- Configurable mute padding (before/after each detected bad word)
- Category-based filtering loaded from `config/bad_terms_categories.json`
- Scene-detection categories (transcript cues + optional frame sampling)
- Strong blur on detected scene intervals (configurable intensity)
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

You can also select scene-detection categories, mute detected scene intervals, and apply strong blur to those same intervals.

Use the **Mute padding per word (seconds)** slider to tune how much audio is muted around each detected word. If a word is detected but still audible, increase this value.

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
- `blurred.mp4` (only when scene blur is enabled and scene spans are detected)

## Smoke Test (no heavy dependencies)

```bash
python3 smoke_test.py
```

This only validates parsing/masking/report logic with synthetic transcript data.

