from dataclasses import dataclass, field
from typing import List


@dataclass
class WordToken:
    word: str
    start: float
    end: float
    probability: float = 0.0


@dataclass
class TranscriptSegment:
    start: float
    end: float
    text: str
    words: List[WordToken] = field(default_factory=list)


@dataclass
class FilteredOccurrence:
    original_word: str
    normalized_word: str
    start: float
    end: float
    probability: float


@dataclass
class SceneDetectionEvent:
    category: str
    start: float
    end: float
    score: float
    source: str
    detected_value: float = 0.0
    threshold_used: float = 0.0
    reason: str = ""


