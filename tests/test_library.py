"""Tests for catalog suggestions and the "Get models" controller."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

import pytest

from dom.catalog import CATALOG, suggest
from dom.catalog.recommend import memory_for_models, needed_bytes
from dom.hardware import GpuDevice, HardwareProfile
from dom.hardware.models import GIB
from dom.ui.library import CatalogRow, LibraryController
from tests.fakes import TEST_HARDWARE, FakeDownloads

if TYPE_CHECKING:
    from collections.abc import Sequence


def _hw(ram_gib: int, gpu: GpuDevice | None = None) -> HardwareProfile:
    return replace(
        TEST_HARDWARE,
        memory=replace(TEST_HARDWARE.memory, total_bytes=ram_gib * GIB),
        gpus=(gpu,) if gpu else (),
    )


def test_catalog_is_well_formed() -> None:
    keys = [m.key for m in CATALOG]
    assert len(keys) == len(set(keys))
    for model in CATALOG:
        assert model.params_b <= 35  # noqa: PLR2004 -- the agreed size limit
        assert model.quants
        for quant in model.quants:
            assert quant.filename.endswith(".gguf")
            assert quant.size_bytes > 0


def test_small_computer_gets_small_models() -> None:
    items = {s.model.key: s for s in suggest(CATALOG, _hw(8), set())}
    assert items["qwen3.5-0.8b"].fits
    assert not items["qwen3.6-35b-a3b"].fits
    recommended = [s for s in items.values() if s.recommended]
    assert len(recommended) == 1
    assert recommended[0].model.params_b <= 9  # noqa: PLR2004


def test_everything_fits_a_large_gpu_machine() -> None:
    gpu = GpuDevice("Vulkan0", "NVIDIA GeForce RTX 4090", 24 * GIB, 23 * GIB, integrated=False)
    items = suggest(CATALOG, _hw(64, gpu), set())
    assert all(s.fits for s in items)
    assert [s.model.key for s in items if s.recommended] == ["qwen3.6-35b-a3b"]


def test_integrated_gpu_memory_does_not_count() -> None:
    igpu = GpuDevice("Vulkan0", "Intel(R) Iris(R) Xe Graphics", 11 * GIB, 7 * GIB, integrated=True)
    assert memory_for_models(_hw(16, igpu)) == memory_for_models(_hw(16))


def test_installed_file_is_reported_and_kept() -> None:
    items = {s.model.key: s for s in suggest(CATALOG, _hw(8), {"Qwen3.5-0.8B-Q4_K_M.gguf"})}
    item = items["qwen3.5-0.8b"]
    assert item.installed
    assert item.quant is not None
    assert item.quant.label == "Q4_K_M"


def test_preferred_quant_is_skipped_when_it_would_be_too_slow() -> None:
    items = {s.model.key: s for s in suggest(CATALOG, _hw(16), set())}
    quant = items["qwen3.5-4b"].quant
    assert quant is not None
    assert needed_bytes(quant) <= memory_for_models(_hw(16))
    assert items["qwen3.5-4b"].est_tokens_per_s is not None
    assert items["qwen3.5-4b"].est_tokens_per_s >= 8  # noqa: PLR2004


@dataclass
class LibraryRecorder:
    rows: list[CatalogRow] = field(default_factory=list[CatalogRow])
    progress: list[tuple[str, int, int]] = field(default_factory=list[tuple[str, int, int]])
    errors: list[str] = field(default_factory=list[str])
    hardware: str = ""

    def show_catalog(self, rows: Sequence[CatalogRow], hardware_text: str) -> None:
        self.rows = list(rows)
        self.hardware = hardware_text

    def show_download_progress(self, key: str, done: int, total: int) -> None:
        self.progress.append((key, done, total))

    def show_library_error(self, message: str) -> None:
        self.errors.append(message)

    def row(self, key: str) -> CatalogRow:
        return next(r for r in self.rows if r.suggestion.model.key == key)


@pytest.fixture
def downloads() -> FakeDownloads:
    return FakeDownloads(partial={"Qwen3.5-2B-Q8_0.gguf": 1000})


def _controller(
    downloads: FakeDownloads, installed: list[int]
) -> tuple[LibraryController, LibraryRecorder]:
    async def on_installed() -> None:
        installed.append(1)

    ctrl = LibraryController(CATALOG, downloads, lambda: _hw(16), on_installed)
    view = LibraryRecorder()
    ctrl.attach(view)
    return ctrl, view


async def test_refresh_lists_the_catalog(downloads: FakeDownloads) -> None:
    ctrl, view = _controller(downloads, [])
    await ctrl.refresh()
    assert len(view.rows) == len(CATALOG)
    assert view.row("qwen3.5-2b").partial_bytes == 1000  # noqa: PLR2004
    assert "16 GB RAM" in view.hardware


async def test_download_reports_progress_and_rescans(downloads: FakeDownloads) -> None:
    installed: list[int] = []
    ctrl, view = _controller(downloads, installed)
    await ctrl.download("qwen3.5-0.8b")
    assert downloads.downloaded == ["Qwen3.5-0.8B-Q8_0.gguf"]
    assert view.progress[-1] == ("qwen3.5-0.8b", 2, 2)
    assert installed == [1]
    assert view.row("qwen3.5-0.8b").suggestion.installed


async def test_failed_download_is_explained(downloads: FakeDownloads) -> None:
    downloads.error = "The download was interrupted. Try again to resume it."
    installed: list[int] = []
    ctrl, view = _controller(downloads, installed)
    await ctrl.download("qwen3.5-0.8b")
    assert view.errors == ["The download was interrupted. Try again to resume it."]
    assert installed == []


async def test_cancel_stops_a_download(downloads: FakeDownloads) -> None:
    downloads.block = asyncio.Event()
    installed: list[int] = []
    ctrl, view = _controller(downloads, installed)
    task = asyncio.create_task(ctrl.download("qwen3.5-0.8b"))
    await asyncio.sleep(0.01)
    assert ctrl.is_downloading("qwen3.5-0.8b")
    assert view.row("qwen3.5-0.8b").downloading
    await ctrl.cancel("qwen3.5-0.8b")
    await task
    assert not ctrl.is_downloading("qwen3.5-0.8b")
    assert installed == []
    assert downloads.downloaded == []


async def test_models_that_do_not_fit_cannot_be_downloaded(downloads: FakeDownloads) -> None:
    ctrl, _view = _controller(downloads, [])
    await ctrl.download("qwen3.6-35b-a3b")
    assert downloads.downloaded == []
