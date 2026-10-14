from __future__ import annotations

from pathlib import Path

import pytest

from subtitle_text_generator.engine import (
    EngineExecutionError,
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

    def ffmpeg_executable(self) -> Path:
        return self.root / "ffmpeg.exe"


def test_measure_mean_volume_reads_volumedetect_output(tmp_path: Path) -> None:
    runner = EngineRunner(StubRuntime(tmp_path))  # type: ignore[arg-type]
    commands: list[list[str]] = []
    output = "[Parsed_volumedetect_0 @ 0x1] mean_volume: -33.7 dB\n[...] max_volume: 0.0 dB"

    def fake_tool(command: list[str], cancelled) -> tuple[int, str]:
        commands.append(command)
        return 0, output

    runner._run_tool = fake_tool  # type: ignore[method-assign]
    assert runner.measure_mean_volume(tmp_path / "녹음.m4a", cancelled=lambda: False) == -33.7
    assert commands[0][0] == str(tmp_path / "ffmpeg.exe")
    assert commands[0][commands[0].index("-map") + 1] == "0:a:0"
    assert "volumedetect" in commands[0]

    runner._run_tool = lambda command, cancelled: (1, output)  # type: ignore[method-assign]
    assert runner.measure_mean_volume(tmp_path / "x.mp4", cancelled=lambda: False) is None
    runner._run_tool = lambda command, cancelled: (0, "no audio")  # type: ignore[method-assign]
    assert runner.measure_mean_volume(tmp_path / "x.mp4", cancelled=lambda: False) is None


def test_normalize_audio_writes_target_only_after_success(tmp_path: Path) -> None:
    runner = EngineRunner(StubRuntime(tmp_path))  # type: ignore[arg-type]
    target = tmp_path / "job" / "normalized.wav"

    def succeed(command: list[str], cancelled) -> tuple[int, str]:
        assert command[command.index("-af") + 1] == "dynaudnorm"
        assert command[command.index("-ar") + 1] == "16000"
        Path(command[-1]).write_bytes(b"RIFF" + bytes(96))
        return 0, ""

    runner._run_tool = succeed  # type: ignore[method-assign]
    runner.normalize_audio(tmp_path / "a.m4a", target, "dynaudnorm", cancelled=lambda: False)
    assert target.read_bytes().startswith(b"RIFF")
    assert not list(target.parent.glob("*.partial.wav"))

    target.unlink()

    def fail(command: list[str], cancelled) -> tuple[int, str]:
        Path(command[-1]).write_bytes(b"broken")
        return 1, "Invalid argument"

    runner._run_tool = fail  # type: ignore[method-assign]
    with pytest.raises(EngineExecutionError, match="음량 보정 실패"):
        runner.normalize_audio(tmp_path / "a.m4a", target, "dynaudnorm", cancelled=lambda: False)
    assert not target.exists()
    assert not list(target.parent.glob("*.partial.wav"))


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
