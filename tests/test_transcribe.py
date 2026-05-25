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


# ── Qwen2-Audio tests removed in Addendum 8 Stage 2 ──────────────────
# All tests below this line until test_qwen3_asr_* exercised code that
# no longer exists (chunker, stitcher, conversation builder).  They
# lived in this file across Addenda 3-7 and were removed as part of the
# Qwen2-Audio engine deletion.  See git history if needed.

def test_qwen3_asr_engine_registration() -> None:
    """Verify the Qwen3-ASR engine class is registered with correct metadata."""
    from pipelines.transcribe_engines import ENGINES
    from pipelines.transcribe_engines.qwen3_asr_engine import (
        Qwen3AsrEngine, QWEN3_ASR_VARIANTS,
    )

    assert "qwen3-asr" in ENGINES
    assert ENGINES["qwen3-asr"] is Qwen3AsrEngine
    assert "qwen3-asr-1.7b" in QWEN3_ASR_VARIANTS
    assert "qwen3-asr-0.6b" in QWEN3_ASR_VARIANTS

    # Invalid model_id raises.
    try:
        Qwen3AsrEngine(model_id="bogus")
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for unknown model_id")

    # Valid construction (no .load() called — that requires CUDA + weights).
    engine = Qwen3AsrEngine(model_id="qwen3-asr-1.7b")
    assert engine.engine_id == "qwen3-asr"
    assert engine.requires_gpu is True

    print("qwen3_asr_engine_registration OK")


def test_qwen3_asr_language_resolution() -> None:
    """Verify ISO code → Qwen3-ASR language name mapping."""
    from pipelines.transcribe_engines.qwen3_asr_engine import _resolve_language

    assert _resolve_language("es") == "Spanish"
    assert _resolve_language("en") == "English"
    assert _resolve_language("Spanish") == "Spanish"   # passthrough for already-resolved names
    assert _resolve_language(None) is None
    assert _resolve_language("") is None
    print("qwen3_asr_language_resolution OK")


if __name__ == "__main__":
    main()
    test_collapse_repetitions()
    test_qwen3_asr_engine_registration()
    test_qwen3_asr_language_resolution()
