import csv
import json
from pathlib import Path
from typing import Iterable

from src.filtering import filtered_word_counts
from src.models import FilteredOccurrence, SceneDetectionEvent


def write_report(
    occurrences: Iterable[FilteredOccurrence],
    output_dir: Path,
    scene_events: Iterable[SceneDetectionEvent] | None = None,
) -> tuple[Path, Path, dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    occurrences_list = list(occurrences)
    scene_events_list = list(scene_events or [])
    counts = filtered_word_counts(occurrences_list)
    scene_counts: dict[str, int] = {}
    for event in scene_events_list:
        scene_counts[event.category] = scene_counts.get(event.category, 0) + 1

    report_json = {
        "total_filtered": len(occurrences_list),
        "unique_terms": sorted(counts.keys()),
        "counts": counts,
        "occurrences": [
            {
                "word": o.original_word,
                "normalized_word": o.normalized_word,
                "start": round(o.start, 3),
                "end": round(o.end, 3),
                "probability": round(o.probability, 4),
            }
            for o in occurrences_list
        ],
        "scene_total": len(scene_events_list),
        "scene_by_category": scene_counts,
        "scene_events": [
            {
                "category": e.category,
                "start": round(e.start, 3),
                "end": round(e.end, 3),
                "score": round(e.score, 4),
                "source": e.source,
            }
            for e in scene_events_list
        ],
    }

    json_path = output_dir / "filtered_report.json"
    json_path.write_text(json.dumps(report_json, indent=2), encoding="utf-8")

    csv_path = output_dir / "filtered_occurrences.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["word", "normalized_word", "start", "end", "probability"],
        )
        writer.writeheader()
        for item in report_json["occurrences"]:
            writer.writerow(item)

    scene_csv_path = output_dir / "scene_events.csv"
    with scene_csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["category", "start", "end", "score", "source"],
        )
        writer.writeheader()
        for item in report_json["scene_events"]:
            writer.writerow(item)

    return json_path, csv_path, report_json

