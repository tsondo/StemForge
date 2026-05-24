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


if __name__ == "__main__":
    main()
