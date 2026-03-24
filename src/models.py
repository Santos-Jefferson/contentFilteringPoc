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


@dataclass
class ProcessingProfile:
    name: str
    description: str = ""
    selected_categories: List[str] = field(default_factory=list)
    selected_scene_categories: List[str] = field(default_factory=list)
    mute_scene_audio: bool = False
    blur_scene_video: bool = False
    emit_output_video: bool = True
    emit_masked_subtitles: bool = True
    emit_scene_gallery: bool = True


@dataclass
class BatchManifestEntry:
    video_url: str
    title: str = ""
    source_id: str = ""
    processing_profile: str | None = None
    selected_categories: List[str] = field(default_factory=list)
    selected_scene_categories: List[str] = field(default_factory=list)
    model_size: str = "small"
    mute_padding_seconds: float = 0.08
    mute_scene_audio: bool | None = None
    blur_scene_video: bool | None = None
    blur_strength: float = 40.0
    scene_threshold_overrides: dict[str, float] = field(default_factory=dict)


@dataclass
class BatchManifestResult:
    video_url: str
    title: str = ""
    source_id: str = ""
    status: str = "queued"
    processing_profile: str | None = None
    job_id: str = ""
    job_dir: str = ""
    output_video: str = ""
    summary_path: str = ""
    total_filtered: int = 0
    scene_total: int = 0
    error: str = ""


@dataclass
class BatchRunSummary:
    batch_id: str
    manifest_path: str
    summary_path: str = ""
    total_jobs: int = 0
    succeeded: int = 0
    failed: int = 0
    results: List[BatchManifestResult] = field(default_factory=list)


