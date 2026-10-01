from __future__ import annotations

import hashlib
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

from subtitle_text_generator.runtime import (
    ResumableDownloader,
    RuntimeSetupError,
    gpu_summary,
)


def test_resumable_downloader_continues_part_file(tmp_path: Path) -> None:
    payload = (b"0123456789abcdef" * 70_000)[:1_000_000]

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            start = 0
            header = self.headers.get("Range")
            if header:
                start = int(header.removeprefix("bytes=").removesuffix("-"))
                self.send_response(206)
                self.send_header("Content-Range", f"bytes {start}-{len(payload) - 1}/{len(payload)}")
            else:
                self.send_response(200)
            chunk = payload[start:]
            self.send_header("Content-Length", str(len(chunk)))
            self.end_headers()
            self.wfile.write(chunk)

        def log_message(self, _format: str, *_args: object) -> None:
            return None

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        target = tmp_path / "asset.bin"
        partial = target.with_name(target.name + ".part")
        partial.write_bytes(payload[:123_456])
        downloader = ResumableDownloader(lambda *_: None, lambda: False)
        result = downloader.fetch(
            f"http://127.0.0.1:{server.server_port}/asset",
            target,
            expected_size=len(payload),
            expected_sha256=hashlib.sha256(payload).hexdigest(),
            label="test",
        )
    finally:
        server.shutdown()
        server.server_close()
    assert result.read_bytes() == payload
    assert not partial.exists()


def test_gpu_summary_does_not_allow_silent_cpu_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "subtitle_text_generator.runtime.subprocess.run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=1, stdout="", stderr="driver error"),
    )
    with pytest.raises(RuntimeSetupError, match="CUDA"):
        gpu_summary()
