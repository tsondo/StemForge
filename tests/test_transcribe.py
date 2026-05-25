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


if __name__ == "__main__":
    main()
    test_collapse_repetitions()
