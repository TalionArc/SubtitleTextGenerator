from __future__ import annotations

from pathlib import Path

from lecture_subtitle_batcher.engine import (
    EngineRun,
    EngineRunner,
    _has_complete_output_after_nonzero_exit,
)


class StubRuntime:
    def __init__(self, root: Path) -> None:
        self.root = root

    def engine_executable(self) -> Path:
        return self.root / "faster-whisper-xxl.exe"

    def model_path(self, key: str) -> Path:
        return self.root / "models" / f"faster-whisper-{key}"


def test_build_turbo_command_is_explicit_cuda_fp16(tmp_path: Path) -> None:
    runner = EngineRunner(StubRuntime(tmp_path))  # type: ignore[arg-type]
    command = runner.build_command(
        EngineRun(
            video=tmp_path / "한국어 강의.mp4",
            model_key="large-v3-turbo",
            output_dir=tmp_path / "out",
            batch_size=4,
            glossary=("CTranslate2", "Whisper"),
        )
    )
    assert command[1].endswith("한국어 강의.mp4")
    assert command[command.index("--device") + 1] == "cuda"
    assert command[command.index("--compute_type") + 1] == "float16"
    assert command[command.index("--language") + 1] == "ko"
    assert command[command.index("--model") + 1] == "large-v3-turbo"
    assert command[command.index("--model_dir") + 1] == str(tmp_path / "models")
    assert "--batched" in command
    assert command[command.index("--batch_size") + 1] == "4"
    assert "CTranslate2, Whisper" in command


def test_build_refine_command_uses_one_clip_invocation(tmp_path: Path) -> None:
    runner = EngineRunner(StubRuntime(tmp_path))  # type: ignore[arg-type]
    command = runner.build_command(
        EngineRun(
            video=tmp_path / "v.mp4",
            model_key="large-v3",
            output_dir=tmp_path / "out",
            clip_timestamps=(1.25, 3.5, 8.0, 10.0),
        )
    )
    assert "--batched" not in command
    assert command.count("--clip_timestamps") == 1
    assert command[command.index("--clip_timestamps") + 1] == "1.25,3.5,8,10"


def test_nonzero_exit_is_recoverable_only_after_confirmed_complete_json(tmp_path: Path) -> None:
    result = tmp_path / "result.json"
    result.write_text('{"segments": [{"start": 0, "end": 1, "text": "ok"}]}', encoding="utf-8")
    complete_log = "100%\nSubtitles are written to 'out' directory.\nOperation finished in: 0:00:10"
    assert _has_complete_output_after_nonzero_exit(complete_log.casefold(), [result])
    assert not _has_complete_output_after_nonzero_exit("100% but crashed", [result])
    assert not _has_complete_output_after_nonzero_exit(complete_log.casefold(), [])
