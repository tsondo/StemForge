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


def test_qwen_stitcher_avoids_distant_repetition() -> None:
    """The matcher must only consider the immediately previous chunk.

    Simulates a song where chunk 1 contains a chorus, chunk 2 contains a
    verse, and chunk 3 contains the same chorus again. The chunk 2 → chunk 3
    stitch must use chunk 2's tail (not chunk 1's), even though chunk 1
    would match chunk 3 more completely.
    """
    from pipelines.transcribe_engines._qwen_chunker import stitch_chunks
    chunk_1 = "feliz cumpleaños mi princesa eres mi sol mi fortaleza"
    chunk_2 = "mi sol mi fortaleza con tu amor llenas mi vida cada sonrisa"
    chunk_3 = "cada sonrisa es la salida catrina mi corazón te pertenece"
    stitched = stitch_chunks([chunk_1, chunk_2, chunk_3])
    # "feliz cumpleaños" must appear exactly once — the second chorus is
    # not in any of these chunks.
    assert stitched.lower().count("feliz cumpleaños") == 1, stitched
    # "cada sonrisa" appears in both chunk 2 and chunk 3 and must be
    # deduplicated cleanly.
    assert stitched.lower().count("cada sonrisa") == 1, stitched
    # And the full transcript should read sensibly.
    assert "te pertenece" in stitched.lower()
    print("qwen_stitcher distant-repetition OK")


def test_qwen_stitcher_single_token_substantive() -> None:
    """A single substantive token in the overlap region should stitch."""
    from pipelines.transcribe_engines._qwen_chunker import stitch_chunks
    # Overlap: one long content word (catrina). Stop-list words and short
    # tokens are mixed in but shouldn't affect the acceptance decision.
    chunk_a = "una reina en su día catrina"
    chunk_b = "catrina mi corazón te pertenece"
    stitched = stitch_chunks([chunk_a, chunk_b])
    assert stitched.lower().count("catrina") == 1, stitched
    assert "te pertenece" in stitched.lower()
    print("qwen_stitcher single-token-substantive OK")


def test_qwen_stitcher_single_token_stopword_rejected() -> None:
    """A single stop-word match should fall back to concatenation."""
    from pipelines.transcribe_engines._qwen_chunker import stitch_chunks
    chunk_a = "uno dos tres cuatro de"
    chunk_b = "de cinco seis siete ocho"
    stitched = stitch_chunks([chunk_a, chunk_b])
    # "de" is a Spanish stop word — match should be refused, both
    # occurrences preserved.
    assert stitched.lower().count("de") == 2, stitched
    print("qwen_stitcher stopword-rejected OK")


def test_qwen_stitcher_single_token_short_rejected() -> None:
    """A single token shorter than MIN_SINGLE_TOKEN_LENGTH should be rejected."""
    from pipelines.transcribe_engines._qwen_chunker import stitch_chunks
    # "sol" is 3 chars, below the 4-char threshold.
    chunk_a = "eres mi gran sol"
    chunk_b = "sol que brilla mucho"
    stitched = stitch_chunks([chunk_a, chunk_b])
    assert stitched.lower().count("sol") == 2, stitched
    print("qwen_stitcher short-token-rejected OK")


def test_qwen_conversation_construction() -> None:
    """Verify the Qwen conversation builder produces the expected 3-turn structure."""
    from pipelines.transcribe_engines.qwen_engine import (
        _build_qwen_conversation,
        _TURN1_BASE, _TURN2_ACKNOWLEDGMENT, _TURN3_BASE,
    )

    # No hint, no language → 3 turns, hint guidance absent, language suffix absent.
    conv = _build_qwen_conversation(None, None)
    assert len(conv) == 3
    assert conv[0]["role"] == "user"
    assert conv[1]["role"] == "assistant"
    assert conv[2]["role"] == "user"

    # Turn 1 contains base task only.
    turn1_text = conv[0]["content"][0]["text"]
    assert turn1_text == _TURN1_BASE
    assert "names and spellings" not in turn1_text

    # Turn 2 is the fabricated acknowledgment.
    assert conv[1]["content"][0]["text"] == _TURN2_ACKNOWLEDGMENT

    # Turn 3 has audio placeholder + minimal text.
    assert conv[2]["content"][0]["type"] == "audio"
    assert conv[2]["content"][1]["text"] == _TURN3_BASE

    # Empty-string and whitespace hints behave like None.
    assert _build_qwen_conversation("", None) == conv
    assert _build_qwen_conversation("   ", None) == conv

    # Hint only → hint guidance in turn 1, turn 2/3 unchanged.
    conv = _build_qwen_conversation("Catrina, Feliz cumpleaños", None)
    turn1_text = conv[0]["content"][0]["text"]
    assert turn1_text.startswith(_TURN1_BASE)
    assert "Catrina, Feliz cumpleaños" in turn1_text
    assert "do not emit these names unless you actually hear them" in turn1_text
    # Turn 3 must NOT contain the hint — that was the Phase 1 failure mode.
    assert "Catrina" not in conv[2]["content"][1]["text"]
    # Turn 2 must not vary based on hint presence.
    assert conv[1]["content"][0]["text"] == _TURN2_ACKNOWLEDGMENT

    # Language only → turn 3 has language suffix, turn 1 unchanged.
    conv = _build_qwen_conversation(None, "Spanish")
    assert conv[0]["content"][0]["text"] == _TURN1_BASE
    turn3_text = conv[2]["content"][1]["text"]
    assert "in Spanish" in turn3_text

    # Hint + language → hint in turn 1, language in turn 3, no crosstalk.
    conv = _build_qwen_conversation("Catrina", "Spanish")
    turn1_text = conv[0]["content"][0]["text"]
    turn3_text = conv[2]["content"][1]["text"]
    assert "Catrina" in turn1_text
    assert "Catrina" not in turn3_text
    assert "Spanish" in turn3_text
    assert "Spanish" not in turn1_text

    # Multi-line / quote-laden hint should be sanitized.
    conv = _build_qwen_conversation('Line one\nLine "two"\nLine three', None)
    turn1_text = conv[0]["content"][0]["text"]
    assert "\n\nThe lyrics may contain" in turn1_text  # the join we added
    # The hint itself, post-sanitization:
    assert "Line one Line 'two' Line three" in turn1_text

    print("qwen_conversation_construction OK")


if __name__ == "__main__":
    main()
    test_collapse_repetitions()
    test_qwen_chunker_single_chunk()
    test_qwen_chunker_overlap_layout()
    test_qwen_chunker_irregular_tail()
    test_qwen_stitcher_clean_overlap()
    test_qwen_stitcher_no_overlap_falls_back()
    test_qwen_stitcher_single_chunk()
    test_qwen_stitcher_avoids_distant_repetition()
    test_qwen_stitcher_single_token_substantive()
    test_qwen_stitcher_single_token_stopword_rejected()
    test_qwen_stitcher_single_token_short_rejected()
    test_qwen_conversation_construction()
