import re
from collections import Counter
from typing import Dict, Iterable, List, Optional, Set

from src.models import FilteredOccurrence, TranscriptSegment


_NORMALIZE_RE = re.compile(r"[^a-z0-9']+")


def normalize_word(word: str) -> str:
    return _NORMALIZE_RE.sub("", word.lower()).strip("'")


def parse_bad_terms(raw_terms: str) -> Set[str]:
    terms = set()
    for chunk in re.split(r"[,\n;]+", raw_terms):
        normalized = normalize_word(chunk.strip())
        if normalized:
            terms.add(normalized)
    return terms


def find_filtered_occurrences(
    segments: Iterable[TranscriptSegment],
    bad_terms: Set[str],
    phrase_terms: Optional[Set[str]] = None,
    safety_padding_seconds: float = 0.04,
    context_exclusions: Dict[str, Set[str]] | None = None,
) -> List[FilteredOccurrence]:
    """Detect bad-word and phrase occurrences across transcript segments.

    Args:
        segments: Transcript segments with word-level timestamps.
        bad_terms: Single normalised tokens that should be muted.
        phrase_terms: Multi-word phrases (space-separated, lower-cased) that
            should be muted as a continuous span.  Single-word entries are
            silently ignored (they are handled via ``bad_terms``).
        safety_padding_seconds: Extra seconds added before/after each match.
        context_exclusions: Per-term neighbouring words that suppress a match.
    """
    segments_list = list(segments)
    occurrences: List[FilteredOccurrence] = []

    # ── single-word detection ──────────────────────────────────────────────
    for segment in segments_list:
        normalized_words = [normalize_word(token.word) for token in segment.words]
        for idx, token in enumerate(segment.words):
            normalized = normalized_words[idx]
            if not normalized or normalized not in bad_terms:
                continue
            if _is_context_excluded(normalized_words, idx, normalized, context_exclusions):
                continue
            start = max(0.0, token.start - safety_padding_seconds)
            end = max(start, token.end + safety_padding_seconds)
            occurrences.append(
                FilteredOccurrence(
                    original_word=token.word,
                    normalized_word=normalized,
                    start=start,
                    end=end,
                    probability=token.probability,
                )
            )

    # ── phrase detection ───────────────────────────────────────────────────
    if phrase_terms:
        for segment in segments_list:
            occurrences.extend(
                _find_phrase_occurrences_in_segment(
                    segment, phrase_terms, safety_padding_seconds
                )
            )

    return occurrences


def _find_phrase_occurrences_in_segment(
    segment: TranscriptSegment,
    phrase_terms: Set[str],
    safety_padding_seconds: float,
) -> List[FilteredOccurrence]:
    """Slide a window over the segment tokens looking for phrase matches."""
    tokens = segment.words
    n = len(tokens)
    if n == 0:
        return []

    normalized_tokens = [normalize_word(t.word) for t in tokens]
    found: List[FilteredOccurrence] = []

    for phrase in phrase_terms:
        phrase_words = phrase.lower().split()
        if len(phrase_words) < 2:
            continue  # single words handled by the word-term pass

        phrase_norm = [normalize_word(w) for w in phrase_words]
        phrase_len = len(phrase_norm)
        if phrase_len > n:
            continue

        for i in range(n - phrase_len + 1):
            if normalized_tokens[i : i + phrase_len] == phrase_norm:
                span_tokens = tokens[i : i + phrase_len]
                start = max(0.0, span_tokens[0].start - safety_padding_seconds)
                end = max(start, span_tokens[-1].end + safety_padding_seconds)
                prob = min(t.probability for t in span_tokens) if span_tokens else 0.0
                found.append(
                    FilteredOccurrence(
                        original_word=" ".join(t.word for t in span_tokens),
                        normalized_word=phrase,
                        start=start,
                        end=end,
                        probability=prob,
                    )
                )

    return found


def mask_text(text: str, bad_terms: Set[str]) -> str:
    masked = text
    for term in sorted(bad_terms, key=len, reverse=True):
        if not term:
            continue
        pattern = re.compile(rf"\b{re.escape(term)}\b", re.IGNORECASE)
        masked = pattern.sub("***", masked)
    return masked


def _is_context_excluded(
    normalized_words: List[str],
    index: int,
    normalized: str,
    context_exclusions: Dict[str, Set[str]] | None,
) -> bool:
    if not context_exclusions:
        return False
    blocked_neighbors = context_exclusions.get(normalized)
    if not blocked_neighbors:
        return False

    left = max(0, index - 2)
    right = min(len(normalized_words), index + 3)
    local_window = set(normalized_words[left:right])
    local_window.discard(normalized)
    return bool(local_window.intersection(blocked_neighbors))


def filtered_word_counts(occurrences: Iterable[FilteredOccurrence]) -> dict:
    counts = Counter(o.normalized_word for o in occurrences)
    return dict(counts)

