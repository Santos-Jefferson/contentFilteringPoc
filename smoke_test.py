from pathlib import Path
import tempfile

from src.filtering import find_filtered_occurrences, mask_text, parse_bad_terms
from src.models import TranscriptSegment, WordToken
from src.reporting import write_report
from src.scene_detection import detect_scene_events
from src.subtitle_utils import write_masked_srt
from src.term_config import (
    get_scene_detection_config,
    get_default_scene_categories,
    get_scene_category_names,
    load_term_config,
    resolve_terms_from_categories,
)


def main() -> None:
    config = load_term_config("config/bad_terms_categories.json")
    token_terms, caption_terms, context_exclusions = resolve_terms_from_categories(
        config,
        ["profanity", "childish_language"],
    )
    assert "poop" in token_terms
    assert "goddamn" in token_terms
    assert "shut up" in caption_terms
    assert "heaven" in context_exclusions.get("hell", set())

    scene_names = get_scene_category_names(config)
    scene_defaults = get_default_scene_categories(config)
    assert "sexual_assault" in scene_names
    assert "sexual_assault" in scene_defaults
    assert "immodesty_female" in scene_defaults
    assert "violence_implied" in scene_names
    assert "violence_graphic" in scene_names
    assert "violence_gore" in scene_names

    terms = parse_bad_terms("damn,hell,pooped")
    segments = [
        TranscriptSegment(
            start=0.0,
            end=2.0,
            text="This is damn bad",
            words=[
                WordToken("This", 0.0, 0.2),
                WordToken("is", 0.2, 0.35),
                WordToken("damn", 0.35, 0.6, 0.92),
                WordToken("bad", 0.6, 0.9),
            ],
        ),
        TranscriptSegment(
            start=2.0,
            end=4.0,
            text="What the hell",
            words=[
                WordToken("What", 2.1, 2.3),
                WordToken("the", 2.3, 2.45),
                WordToken("hell", 2.45, 2.8, 0.87),
            ],
        ),
        TranscriptSegment(
            start=4.0,
            end=6.0,
            text="Then you threw the pooped underwear away",
            words=[
                WordToken("Then", 4.0, 4.2),
                WordToken("you", 4.2, 4.35),
                WordToken("threw", 4.35, 4.6),
                WordToken("the", 4.6, 4.73),
                WordToken("pooped", 4.73, 5.03, 0.91),
                WordToken("underwear", 5.03, 5.45),
                WordToken("away", 5.45, 5.7),
            ],
        ),
    ]

    occurrences = find_filtered_occurrences(segments, terms, safety_padding_seconds=0.08)
    assert len(occurrences) == 3, "Expected exactly three filtered terms"

    pooped_occurrence = next(o for o in occurrences if o.normalized_word == "pooped")
    assert round(pooped_occurrence.start, 2) == 4.65
    assert round(pooped_occurrence.end, 2) == 5.11

    masked_line = mask_text("Then you threw the pooped underwear away", terms)
    assert "***" in masked_line and "pooped" not in masked_line.lower()

    # ── phrase detection ──────────────────────────────────────────────────
    phrase_segments = [
        TranscriptSegment(
            start=0.0,
            end=3.0,
            text="Just shut up already",
            words=[
                WordToken("Just", 0.0, 0.2),
                WordToken("shut", 0.2, 0.45, 0.90),
                WordToken("up", 0.45, 0.65, 0.88),
                WordToken("already", 0.65, 1.0),
            ],
        ),
        TranscriptSegment(
            start=3.0,
            end=6.0,
            text="And you suck too",
            words=[
                WordToken("And", 3.0, 3.15),
                WordToken("you", 3.15, 3.3, 0.85),
                WordToken("suck", 3.3, 3.6, 0.82),
                WordToken("too", 3.6, 3.8),
            ],
        ),
    ]
    phrase_only_terms: set = set()  # no single-word terms
    phrase_terms = {"shut up", "you suck"}
    phrase_occurrences = find_filtered_occurrences(
        phrase_segments,
        phrase_only_terms,
        phrase_terms=phrase_terms,
        safety_padding_seconds=0.05,
    )
    assert len(phrase_occurrences) == 2, (
        f"Expected 2 phrase occurrences, got {len(phrase_occurrences)}: {phrase_occurrences}"
    )
    shut_up = next(o for o in phrase_occurrences if o.normalized_word == "shut up")
    assert round(shut_up.start, 2) == 0.15  # 0.2 - 0.05
    assert round(shut_up.end, 2) == 0.70   # 0.65 + 0.05
    you_suck = next(o for o in phrase_occurrences if o.normalized_word == "you suck")
    assert round(you_suck.start, 2) == 3.10  # 3.15 - 0.05
    assert round(you_suck.end, 2) == 3.65   # 3.6 + 0.05

    # ── violence transcript cues ───────────────────────────────────────────
    violence_segments = [
        TranscriptSegment(
            start=0.0,
            end=2.0,
            text="The witness gave a graphic description of violence and detailed talk of suicide",
            words=[],
        ),
        TranscriptSegment(
            start=2.0,
            end=4.0,
            text="There was blood splatter and a bloody body in the room",
            words=[],
        ),
        TranscriptSegment(
            start=4.0,
            end=6.0,
            text="The scene showed brain matter after someone was decapitated",
            words=[],
        ),
    ]
    violence_events = detect_scene_events(
        Path("."),
        violence_segments,
        get_scene_detection_config(config),
        ["violence_implied", "violence_graphic", "violence_gore"],
    )
    event_categories = {event.category for event in violence_events}
    assert "violence_implied" in event_categories
    assert "violence_graphic" in event_categories
    assert "violence_gore" in event_categories
    implied_event = next(event for event in violence_events if event.category == "violence_implied")
    assert implied_event.detected_value == 1.0
    assert implied_event.threshold_used == 1.0
    assert "Transcript cue matched" in implied_event.reason

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        srt = write_masked_srt(segments, terms, tmp_path / "masked.srt")
        json_path, csv_path, report = write_report(occurrences, tmp_path)

        assert srt.exists(), "SRT file was not generated"
        assert json_path.exists(), "JSON report was not generated"
        assert csv_path.exists(), "CSV report was not generated"
        assert report["counts"].get("damn") == 1
        assert report["counts"].get("hell") == 1
        assert report["counts"].get("pooped") == 1

        srt_text = srt.read_text(encoding="utf-8")
        assert "pooped" not in srt_text.lower()
        assert "***" in srt_text

    print("Smoke test passed")


if __name__ == "__main__":
    main()
