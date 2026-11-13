"""Tests for llama-server command building, binary discovery and process startup failures."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from dom.backend import LlamaServerProcess, ServerConfig, ServerStartError, build_command
from dom.backend.process import build_env, pick_free_port
from dom.config import InferenceSettings, LlamaServerNotFoundError, find_llama_server
from dom.config.binary import server_executable_name


def test_default_settings_only_override_what_is_broken() -> None:
    assert InferenceSettings().to_server_args() == [
        "--ctx-size",
        "8192",
        "--reasoning-budget",
        "512",
    ]


def test_settings_emit_flags_only_for_set_values() -> None:
    settings = InferenceSettings(ctx_size=None, reasoning_budget=None, temperature=0.7, top_k=20)
    assert settings.to_server_args() == ["--temp", "0.7", "--top-k", "20"]


def test_build_command_uses_router_mode(tmp_path: Path) -> None:
    binary = tmp_path / "llama-server"
    command = build_command(binary, tmp_path / "models", 4321, InferenceSettings(ctx_size=1024))
    assert command[0] == str(binary)
    assert "-m" not in command
    assert "--model" not in command
    joined = " ".join(command[1:])
    assert "--host 127.0.0.1" in joined
    assert "--port 4321" in joined
    assert f"--models-dir {(tmp_path / 'models').resolve()}" in joined
    assert "--no-models-autoload" in command
    assert "--models-max 1" in joined
    assert "--offline" in command
    assert "--no-webui" in command
    assert "--ctx-size 1024" in joined


def test_build_env_strips_llama_vars_and_sets_key_and_cache(tmp_path: Path) -> None:
    base = {"PATH": "/bin", "LLAMA_ARG_CTX_SIZE": "99", "llama_cache": "x", "HOME": "/h"}
    env = build_env(tmp_path / "cache", "secret", base)
    assert env["PATH"] == "/bin"
    assert env["HOME"] == "/h"
    assert "LLAMA_ARG_CTX_SIZE" not in env
    assert "llama_cache" not in env
    assert env["LLAMA_API_KEY"] == "secret"
    assert env["LLAMA_CACHE"] == str((tmp_path / "cache").resolve())


def test_pick_free_port_returns_a_usable_port() -> None:
    port = pick_free_port()
    assert 0 < port < 65536  # noqa: PLR2004


# ---- finding the binary ------------------------------------------------------------------


def _make_file(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    return path


def test_configured_path_wins(tmp_path: Path) -> None:
    configured = _make_file(tmp_path / "custom" / "my-llama-server")
    _make_file(tmp_path / "bin" / server_executable_name())
    found = find_llama_server(configured, tmp_path / "bin", which=lambda _n: "/usr/bin/x")
    assert found == configured


def test_bin_dir_before_path(tmp_path: Path) -> None:
    in_bin = _make_file(tmp_path / "bin" / server_executable_name())
    found = find_llama_server(None, tmp_path / "bin", which=lambda _n: "/usr/bin/llama-server")
    assert found == in_bin


def test_falls_back_to_path(tmp_path: Path) -> None:
    found = find_llama_server(
        tmp_path / "missing", tmp_path / "bin", which=lambda _n: "/opt/llama-server"
    )
    assert found == Path("/opt/llama-server")


def test_not_found_has_friendly_message(tmp_path: Path) -> None:
    with pytest.raises(LlamaServerNotFoundError) as info:
        find_llama_server(None, tmp_path / "bin", which=lambda _n: None)
    message = str(info.value)
    assert "llama-server" in message
    assert str(tmp_path / "bin") in message
    assert "Traceback" not in message


def test_executable_name_per_platform() -> None:
    assert server_executable_name("win32") == "llama-server.exe"
    assert server_executable_name("linux") == "llama-server"


# ---- process failures --------------------------------------------------------------------


def _config(tmp_path: Path, binary: Path) -> ServerConfig:
    return ServerConfig(
        binary=binary,
        models_dir=tmp_path / "models",
        cache_dir=tmp_path / "cache",
        log_path=tmp_path / "logs" / "llama-server.log",
        settings=InferenceSettings(),
        startup_timeout=10.0,
    )


async def test_missing_binary_raises_server_start_error(tmp_path: Path) -> None:
    process = LlamaServerProcess(_config(tmp_path, tmp_path / "does-not-exist"))
    with pytest.raises(ServerStartError, match="Could not start"):
        await process.start()
    assert not process.is_running


@pytest.mark.skipif(sys.platform == "win32", reason="uses a shebang script as a fake binary")
async def test_binary_that_exits_immediately_raises(tmp_path: Path) -> None:
    fake = tmp_path / "fake-llama-server"
    fake.write_text(
        f"#!{sys.executable}\nimport sys\nprint('error: model dir missing')\nsys.exit(3)\n"
    )
    fake.chmod(0o755)
    process = LlamaServerProcess(_config(tmp_path, fake))
    with pytest.raises(ServerStartError, match="exit code 3"):
        await process.start()
    assert not process.is_running
    assert "model dir missing" in process.log_tail()
