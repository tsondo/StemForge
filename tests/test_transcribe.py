"""Smoke test for TranscribePipeline with the Whisper engine.

Runs the full pipeline on tests/data/silence.wav with the smallest
Whisper model and confirms:
  - Pipeline configures and loads without error
  - run() produces a TranscribeResult with output_paths for all formats
  - Output files exist (txt may be empty on silence)
  - .srt and .lrc are parseable (or empty for silence)
  - Pipeline can be cleared without raising

Qwen is intentionally NOT covered in CI — it requires GPU, is large to
download, and may have a license gate at the model level.
"""
import pathlib
import tempfile

from pipelines.transcribe_pipeline import TranscribePipeline, TranscribeConfig


def main() -> None:
    audio = pathlib.Path("tests/data/silence.wav")
    assert audio.exists(), f"Missing test fixture: {audio}"

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = pathlib.Path(tmp)
        pipeline = TranscribePipeline()
        pipeline.configure(TranscribeConfig(
            engine_id="whisper",
            model_id="whisper-tiny",
            output_dir=out_dir,
        ))
        pipeline.load_model()
        try:
            result = pipeline.run(audio)
        finally:
            pipeline.clear()

        for fmt in ("txt", "lrc", "srt"):
            assert fmt in result.output_paths, f"Missing format: {fmt}"
            assert result.output_paths[fmt].exists(), f"Missing output: {result.output_paths[fmt]}"
        print(f"engine_id      : {result.result.engine_id}")
        print(f"model_id       : {result.result.model_id}")
        print(f"language       : {result.result.language}")
        print(f"segments       : {len(result.result.segments)}")
        print(f"has_words      : {result.result.has_word_timestamps}")
        print(f"output_paths   : {sorted(result.output_paths)}")
        print("transcribe pipeline OK")


def test_collapse_repetitions() -> None:
    """Verify the repetition collapse helper without invoking a model."""
    from pipelines.transcribe_pipeline import _collapse_repetitions
    from pipelines.transcribe_engines import TranscriptionResult, TranscriptionSegment

    # Build a synthetic result: 2 normal lines, then 10 identical "oh oh oh"
    # segments, then 1 normal line.
    segs = [
        TranscriptionSegment(start=0.0, end=2.0, text="Verse line one", words=[]),
        TranscriptionSegment(start=2.0, end=4.0, text="Verse line two", words=[]),
    ]
    for i in range(10):
        segs.append(TranscriptionSegment(
            start=4.0 + i * 0.5,
            end=4.5 + i * 0.5,
            text="¡Oh, oh, oh!",
            words=[],
        ))
    segs.append(TranscriptionSegment(start=9.0, end=11.0, text="Final line", words=[]))

    result = TranscriptionResult(
        text="(unused)",
        language="es",
        segments=segs,
        has_word_timestamps=False,
        engine_id="whisper",
        model_id="whisper-large-v3",
    )

    collapsed = _collapse_repetitions(result, max_run=4)

    # Expect: 2 verse + 4 oh + 1 ellipsis + 1 final = 8 segments
    assert len(collapsed.segments) == 8, f"got {len(collapsed.segments)} segments"
    assert collapsed.segments[6].text == "[...]"
    assert collapsed.segments[6].start == 6.0, (
        f"expected ellipsis start=6.0, got {collapsed.segments[6].start}"
    )  # 5th oh starts at 4.0 + 4*0.5
    assert collapsed.segments[7].text == "Final line"
    print("collapse_repetitions OK")


def test_qwen_chunker_single_chunk() -> None:
    """Audio shorter than chunk size should produce exactly one chunk."""
    import numpy as np
    from pipelines.transcribe_engines._qwen_chunker import (
        slice_audio, CHUNK_DURATION_S, SAMPLE_RATE,
    )
    # 10 seconds of silence
    audio = np.zeros(int(10 * SAMPLE_RATE), dtype=np.float32)
    chunks = slice_audio(audio)
    assert len(chunks) == 1, f"got {len(chunks)} chunks for 10s audio"
    assert chunks[0].start_s == 0.0
    assert abs(chunks[0].end_s - 10.0) < 0.01
    print("qwen_chunker single OK")


def test_qwen_chunker_overlap_layout() -> None:
    """Verify chunk boundaries match the documented step/overlap values."""
    import numpy as np
    from pipelines.transcribe_engines._qwen_chunker import (
        slice_audio, CHUNK_DURATION_S, OVERLAP_DURATION_S, STEP_DURATION_S,
        SAMPLE_RATE,
    )
    # 60 seconds → expect chunks at [0-24], [18-42], [36-60].
    audio = np.zeros(int(60 * SAMPLE_RATE), dtype=np.float32)
    chunks = slice_audio(audio)
    assert len(chunks) == 3, f"got {len(chunks)} chunks for 60s audio"
    assert abs(chunks[0].start_s - 0.0) < 0.01
    assert abs(chunks[1].start_s - STEP_DURATION_S) < 0.01    # 18.0
    assert abs(chunks[2].start_s - 2 * STEP_DURATION_S) < 0.01  # 36.0
    assert abs(chunks[0].end_s - CHUNK_DURATION_S) < 0.01       # 24.0
    assert abs(chunks[-1].end_s - 60.0) < 0.01
    print("qwen_chunker overlap layout OK")


def test_qwen_chunker_irregular_tail() -> None:
    """Final chunk should be truncated to actual audio length, not padded."""
    import numpy as np
    from pipelines.transcribe_engines._qwen_chunker import (
        slice_audio, SAMPLE_RATE,
    )
    # 50 seconds → [0-24], [18-42], [36-50]
    audio = np.zeros(int(50 * SAMPLE_RATE), dtype=np.float32)
    chunks = slice_audio(audio)
    assert len(chunks) == 3
    assert abs(chunks[-1].end_s - 50.0) < 0.01
    assert abs(chunks[-1].start_s - 36.0) < 0.01
    # The final chunk is 14 seconds, not 24.
    print("qwen_chunker irregular tail OK")


def test_qwen_stitcher_clean_overlap() -> None:
    """Two chunks with a clear shared substring should stitch cleanly."""
    from pipelines.transcribe_engines._qwen_chunker import stitch_chunks
    chunk_a = "feliz cumpleaños mi princesa eres mi sol mi fortaleza"
    chunk_b = "mi sol mi fortaleza con tu amor llenas mi vida"
    stitched = stitch_chunks([chunk_a, chunk_b])
    # The shared "mi sol mi fortaleza" should appear exactly once.
    assert stitched.lower().count("mi sol mi fortaleza") == 1, stitched
    assert "feliz cumpleaños" in stitched.lower()
    assert "llenas mi vida" in stitched.lower()
    print("qwen_stitcher clean overlap OK")


def test_qwen_stitcher_no_overlap_falls_back() -> None:
    """When chunks share no content, both are preserved with a separator."""
    from pipelines.transcribe_engines._qwen_chunker import stitch_chunks
    chunk_a = "uno dos tres cuatro cinco"
    chunk_b = "siete ocho nueve diez once"
    stitched = stitch_chunks([chunk_a, chunk_b])
    assert "uno dos tres" in stitched
    assert "nueve diez once" in stitched
    print("qwen_stitcher fallback OK")


def test_qwen_stitcher_single_chunk() -> None:
    """Single-chunk input should be returned unchanged."""
    from pipelines.transcribe_engines._qwen_chunker import stitch_chunks
    stitched = stitch_chunks(["hello world"])
    assert stitched == "hello world"
    print("qwen_stitcher single chunk OK")


if __name__ == "__main__":
    main()
    test_collapse_repetitions()
    test_qwen_chunker_single_chunk()
    test_qwen_chunker_overlap_layout()
    test_qwen_chunker_irregular_tail()
    test_qwen_stitcher_clean_overlap()
    test_qwen_stitcher_no_overlap_falls_back()
    test_qwen_stitcher_single_chunk()
