"""Tests for app paths."""

from __future__ import annotations

from typing import TYPE_CHECKING

from dom.config import default_paths
from dom.config.settings import DEFAULT_MAX_REPLY_TOKENS, GenerationSettings

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def test_data_dir_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DOM_DEV", raising=False)
    monkeypatch.setenv("DOM_DATA_DIR", str(tmp_path / "portable"))
    paths = default_paths()
    assert paths.data_dir == tmp_path / "portable"
    assert paths.models_dir == tmp_path / "portable" / "models"
    assert paths.engine_dirs == [tmp_path / "portable" / "bin"]  # not frozen: nothing bundled


def test_generation_settings_request_fields() -> None:
    assert GenerationSettings().to_request_fields() == {"max_tokens": DEFAULT_MAX_REPLY_TOKENS}
    assert GenerationSettings(max_reply_tokens=None).to_request_fields() == {}
    fields = GenerationSettings(temperature=0.7, max_reply_tokens=None, thinking=False)
    assert fields.to_request_fields() == {
        "temperature": 0.7,
        "chat_template_kwargs": {"enable_thinking": False},
    }
