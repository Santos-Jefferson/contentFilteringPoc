from typing import List

from src.models import TranscriptSegment, WordToken


def transcribe_with_word_timestamps(
    audio_path: str,
    language: str = "en",
    model_size: str = "small",
    compute_type: str = "int8",
) -> List[TranscriptSegment]:
    from faster_whisper import WhisperModel

    model = WhisperModel(model_size, compute_type=compute_type)
    segments, _ = model.transcribe(
        audio_path,
        language=language,
        word_timestamps=True,
        vad_filter=True,
    )

    parsed_segments: List[TranscriptSegment] = []
    for segment in segments:
        words = []
        if segment.words:
            for word in segment.words:
                if word.start is None or word.end is None:
                    continue
                words.append(
                    WordToken(
                        word=word.word,
                        start=float(word.start),
                        end=float(word.end),
                        probability=float(word.probability or 0.0),
                    )
                )
        parsed_segments.append(
            TranscriptSegment(
                start=float(segment.start),
                end=float(segment.end),
                text=segment.text,
                words=words,
            )
        )
    return parsed_segments

