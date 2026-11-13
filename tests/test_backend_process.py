"""Tests for llama-server command building, binary discovery and process startup failures."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from dom.backend import LlamaServerProcess, ServerConfig, ServerStartError, build_command
from dom.backend.presets import PresetFile
from dom.backend.process import build_env, pick_free_port
from dom.config import (
    InferenceSettings,
    LlamaServerNotFoundError,
    ServerBuild,
    find_server_builds,
)
from dom.config.binary import server_executable_name


def test_default_settings_only_override_what_is_broken() -> None:
    assert InferenceSettings().to_preset_items() == [
        ("ctx-size", "8192"),
        ("reasoning-budget", "512"),
    ]


def test_settings_emit_entries_only_for_set_values() -> None:
    settings = InferenceSettings(
        ctx_size=None, reasoning_budget=None, device="none", temperature=0.7, top_k=20
    )
    assert settings.to_preset_items() == [("device", "none"), ("temp", "0.7"), ("top-k", "20")]


def test_build_command_uses_router_mode(tmp_path: Path) -> None:
    binary = tmp_path / "llama-server"
    presets = tmp_path / "presets.ini"
    command = build_command(binary, tmp_path / "models", 4321, presets)
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
    assert f"--models-preset {presets.resolve()}" in joined
    assert "--models-memory-margin 1024" in joined
    # Inference settings live in the presets file: command-line flags would override it.
    assert "--ctx-size" not in command


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


# ---- finding the builds ------------------------------------------------------------------


def _make_file(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    return path


def test_configured_path_wins(tmp_path: Path) -> None:
    configured = _make_file(tmp_path / "custom" / "my-llama-server")
    _make_file(tmp_path / "bin" / "cpu" / server_executable_name())
    builds = find_server_builds(configured, tmp_path / "bin", which=lambda _n: "/usr/bin/x")
    assert builds == [ServerBuild("configured", configured)]


def test_finds_every_build_in_bin_dir(tmp_path: Path) -> None:
    cpu = _make_file(tmp_path / "bin" / "cpu" / server_executable_name())
    vulkan = _make_file(tmp_path / "bin" / "vulkan" / server_executable_name())
    flat = _make_file(tmp_path / "bin" / server_executable_name())
    (tmp_path / "bin" / "empty").mkdir()
    builds = find_server_builds(None, tmp_path / "bin", which=lambda _n: "/usr/bin/llama-server")
    assert builds == [
        ServerBuild("cpu", cpu),
        ServerBuild("vulkan", vulkan),
        ServerBuild("default", flat),
    ]


def test_falls_back_to_path(tmp_path: Path) -> None:
    builds = find_server_builds(
        tmp_path / "missing", tmp_path / "bin", which=lambda _n: "/opt/llama-server"
    )
    assert builds == [ServerBuild("path", Path("/opt/llama-server"))]


def test_not_found_has_friendly_message(tmp_path: Path) -> None:
    with pytest.raises(LlamaServerNotFoundError) as info:
        find_server_builds(None, tmp_path / "bin", which=lambda _n: None)
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
        presets=PresetFile(tmp_path / "presets.ini", InferenceSettings()),
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


def test_downloaded_engine_comes_before_the_bundled_one(tmp_path: Path) -> None:
    bundled = _make_file(tmp_path / "app" / "llama" / "cpu" / server_executable_name())
    builds = find_server_builds(None, [tmp_path / "data" / "bin", tmp_path / "app" / "llama"])
    assert builds == [ServerBuild("cpu", bundled)]
    downloaded = _make_file(tmp_path / "data" / "bin" / "vulkan" / server_executable_name())
    builds = find_server_builds(None, [tmp_path / "data" / "bin", tmp_path / "app" / "llama"])
    assert builds == [ServerBuild("vulkan", downloaded)]
