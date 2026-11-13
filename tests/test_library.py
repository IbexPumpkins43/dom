"""Tests for Hugging Face browsing: file names, suggestions, search parsing, the models folder
and the models screen's controller."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

import httpx
import pytest

from dom.catalog import HubModel, quant_label, suggest, to_catalog
from dom.catalog.quants import estimated_size, is_single_model_file
from dom.catalog.recommend import memory_for_models
from dom.downloads import DownloadError
from dom.downloads.http import make_client
from dom.downloads.hub import HubSearch, parse_model, parse_page, search_url
from dom.downloads.library import SOURCES_FILE, ModelLibrary
from dom.hardware import GpuDevice, HardwareProfile
from dom.hardware.models import GIB
from dom.ui.library import LOCAL_PREFIX, Fit, LibraryController, ModelCard, ScreenState
from tests.fakes import TEST_HARDWARE, FakeDownloads, FakeSource

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path


def _hw(ram_gib: int, gpu: GpuDevice | None = None) -> HardwareProfile:
    return replace(
        TEST_HARDWARE,
        memory=replace(TEST_HARDWARE.memory, total_bytes=ram_gib * GIB),
        gpus=(gpu,) if gpu else (),
    )


def _model(repo: str, params: float, *labels: str, downloads: int = 1000) -> HubModel:
    stem = repo.rsplit("/", maxsplit=1)[-1].removesuffix("-GGUF")
    return HubModel(
        repo=repo,
        downloads=downloads,
        likes=10,
        params=round(params * 1e9),
        files=tuple(f"{stem}-{label}.gguf" for label in labels),
    )


SMALL = _model("org/Small-1B-GGUF", 1.0, "Q4_K_M", "Q8_0", "BF16")
MEDIUM = _model("org/Medium-8B-GGUF", 8.0, "Q8_0", "Q4_K_M", "Q6_K", "IQ2_XXS")
HUGE = _model("org/Huge-200B-GGUF", 200.0, "Q4_K_M", "Q2_K")

# ---- file names and sizes ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "label"),
    [
        ("Qwen3.5-4B-Q4_K_M.gguf", "Q4_K_M"),
        ("Qwen3.5-4B-UD-Q4_K_XL.gguf", "UD-Q4_K_XL"),
        ("Q4_K_M/Model-q4_k_m.gguf", "Q4_K_M"),
        ("gemma-4-E2B-it-Q8_0.gguf", "Q8_0"),
        ("model.IQ3_XXS.gguf", "IQ3_XXS"),
        ("Model-BF16.gguf", "BF16"),
        ("gpt-oss-20b-MXFP4.gguf", "MXFP4"),
        ("README.md", None),
        ("model.gguf", None),
    ],
)
def test_quant_label(name: str, label: str | None) -> None:
    assert quant_label(name) == label


@pytest.mark.parametrize(
    ("path", "usable"),
    [
        ("Qwen3.5-4B-Q4_K_M.gguf", True),
        ("BF16/Qwen3.5-27B-BF16-00001-of-00002.gguf", False),
        ("mmproj-F16.gguf", False),
        ("Model-imatrix.gguf", False),
        ("model.gguf", False),
        ("README.md", False),
    ],
)
def test_single_model_files(path: str, usable: bool) -> None:
    assert is_single_model_file(path) is usable


def test_size_is_estimated_from_parameters() -> None:
    # Qwen3.5-4B-Q4_K_M is 2.74 GB on Hugging Face with ~4.2B parameters.
    size = estimated_size(4_200_000_000, "Q4_K_M")
    assert size is not None
    assert 2.4e9 < size < 2.9e9  # noqa: PLR2004
    assert estimated_size(4_000_000_000, "UNKNOWN") is None


def test_small_models_prefer_quality_larger_ones_q4_k_m() -> None:
    small = to_catalog(SMALL)
    medium = to_catalog(MEDIUM)
    assert small is not None
    assert medium is not None
    assert small.quants[0].label == "Q8_0"
    assert medium.quants[0].label == "Q4_K_M"
    assert to_catalog(replace(SMALL, params=None)) is None


def test_suggestion_fits_the_computer() -> None:
    catalogs = [c for c in (to_catalog(m) for m in (SMALL, MEDIUM, HUGE)) if c is not None]
    items = {s.model.key: s for s in suggest(catalogs, _hw(16), set())}
    assert items[SMALL.repo].fits
    medium = items[MEDIUM.repo].quant
    assert medium is not None
    assert medium.label == "Q4_K_M"  # slow on a 16 GB laptop, but not degraded to IQ2_XXS
    assert not items[HUGE.repo].fits


def test_low_bit_versions_only_when_nothing_better_fits() -> None:
    model = to_catalog(_model("org/Big-30B-GGUF", 30.0, "Q4_K_M", "Q3_K_M", "IQ2_XXS"))
    assert model is not None
    [item] = suggest([model], _hw(20), set())  # Q4_K_M needs ~20 GB; 20 GB RAM leaves ~17
    assert item.quant is not None
    assert item.quant.label == "Q3_K_M"


def test_integrated_gpu_memory_does_not_count() -> None:
    igpu = GpuDevice("Vulkan0", "Intel(R) Iris(R) Xe Graphics", 11 * GIB, 7 * GIB, integrated=True)
    assert memory_for_models(_hw(16, igpu)) == memory_for_models(_hw(16))


# ---- Hugging Face search -------------------------------------------------------------------


def _entry(repo: str, **overrides: object) -> dict[str, object]:
    entry: dict[str, object] = {
        "id": repo,
        "downloads": 5,
        "likes": 2,
        "gated": False,
        "gguf": {"total": 4_000_000_000, "chat_template": "{{ messages }}", "architecture": "x"},
        "siblings": [
            {"rfilename": "README.md"},
            {"rfilename": "M-Q4_K_M.gguf"},
            {"rfilename": "mmproj-F16.gguf"},
            {"rfilename": "BF16/M-BF16-00001-of-00002.gguf"},
        ],
    }
    entry.update(overrides)
    return entry


def test_parse_model_keeps_single_file_chat_models() -> None:
    model = parse_model(_entry("org/M-GGUF"))
    assert model is not None
    assert model.files == ("M-Q4_K_M.gguf",)
    assert model.params == 4_000_000_000  # noqa: PLR2004
    assert model.name == "M"
    assert model.author == "org"


@pytest.mark.parametrize(
    "overrides",
    [
        {"gated": "auto"},
        {"gguf": {"total": 1}},  # no chat template: not a chat model
        {"siblings": [{"rfilename": "M-00001-of-00003.gguf"}]},
        {"gguf": None},
    ],
)
def test_parse_model_skips_what_dom_cannot_offer(overrides: dict[str, object]) -> None:
    assert parse_model(_entry("org/M-GGUF", **overrides)) is None


def test_parse_page_follows_the_link_header() -> None:
    link = '<https://huggingface.co/api/models?cursor=abc>; rel="next"'
    page = parse_page([_entry("a/A-GGUF"), {"nonsense": 1}], link)
    assert [m.repo for m in page.models] == ["a/A-GGUF"]
    assert page.next_url == "https://huggingface.co/api/models?cursor=abc"
    assert parse_page([], None).next_url is None
    with pytest.raises(DownloadError):
        parse_page({"error": "x"}, None)


def _client(handler: httpx.MockTransport) -> httpx.AsyncClient:
    client = make_client()
    client._transport = handler  # pyright: ignore[reportPrivateUsage] -- inject fake network into the allow-listed client
    return client


async def test_hub_search_asks_for_gguf_models_matching_the_query() -> None:
    seen: list[httpx.URL] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url)
        return httpx.Response(200, json=[_entry("org/Qwen-GGUF")])

    page = await HubSearch(lambda: _client(httpx.MockTransport(handler))).search(" qwen ")
    assert [m.repo for m in page.models] == ["org/Qwen-GGUF"]
    assert seen[0].host == "huggingface.co"
    assert seen[0].params["search"] == "qwen"
    assert seen[0].params["filter"] == "gguf"
    assert "search=" not in search_url("")


async def test_hub_search_failure_is_plain_english() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    with pytest.raises(DownloadError, match="Could not reach Hugging Face"):
        await HubSearch(lambda: _client(httpx.MockTransport(handler))).search("x")


# ---- the models folder ---------------------------------------------------------------------

CONTENT = b"gguf" * 1000


def _hub_files(request: httpx.Request) -> httpx.Response:
    import hashlib  # noqa: PLC0415 -- only this fake needs it

    path = request.url.path
    if path == "/api/models/org/M-GGUF/tree/main/Q4_K_M":
        sha = hashlib.sha256(CONTENT).hexdigest()
        return httpx.Response(
            200,
            json=[
                {
                    "type": "file",
                    "path": "Q4_K_M/M-Q4_K_M.gguf",
                    "size": len(CONTENT),
                    "lfs": {"oid": sha},
                }
            ],
        )
    if path == "/org/M-GGUF/resolve/main/Q4_K_M/M-Q4_K_M.gguf":
        return httpx.Response(200, content=CONTENT)
    return httpx.Response(404)


async def test_download_from_a_folder_saves_by_name_and_remembers_the_repo(
    tmp_path: Path,
) -> None:
    library = ModelLibrary(tmp_path, lambda: _client(httpx.MockTransport(_hub_files)))
    await library.download("org/M-GGUF", "Q4_K_M/M-Q4_K_M.gguf", lambda _d, _t: None)
    assert (tmp_path / "M-Q4_K_M.gguf").read_bytes() == CONTENT
    assert await library.installed_files() == {"M-Q4_K_M.gguf"}
    assert await library.sources() == {"M-Q4_K_M.gguf": "org/M-GGUF"}

    await library.delete("M-Q4_K_M.gguf")
    assert await library.installed_files() == set()
    assert json.loads((tmp_path / SOURCES_FILE).read_text(encoding="utf-8")) == {}


@pytest.mark.parametrize("name", ["../evil.gguf", "notes.txt", ".dom-sources.json", "a/b.gguf"])
async def test_library_refuses_odd_file_names(tmp_path: Path, name: str) -> None:
    with pytest.raises(DownloadError):
        await ModelLibrary(tmp_path).delete(name)


# ---- the models screen's controller --------------------------------------------------------


@dataclass
class ScreenRecorder:
    cards: list[ModelCard] = field(default_factory=list[ModelCard])
    states: list[ScreenState] = field(default_factory=list[ScreenState])
    progress: list[tuple[str, int, int]] = field(default_factory=list[tuple[str, int, int]])
    errors: list[str] = field(default_factory=list[str])

    def show_cards(self, cards: Sequence[ModelCard], state: ScreenState) -> None:
        self.cards = list(cards)
        self.states.append(state)

    def show_download_progress(self, key: str, done: int, total: int) -> None:
        self.progress.append((key, done, total))

    def show_library_error(self, message: str) -> None:
        self.errors.append(message)

    def card(self, key: str) -> ModelCard:
        return next(c for c in self.cards if c.key == key)


@dataclass
class Screen:
    controller: LibraryController
    view: ScreenRecorder
    source: FakeSource
    downloads: FakeDownloads
    changed: list[int]
    released: list[str]


@pytest.fixture
def screen() -> Screen:
    source = FakeSource(models=[MEDIUM, SMALL, HUGE], more=[_model("org/Extra-2B-GGUF", 2, "Q8_0")])
    downloads = FakeDownloads()
    changed: list[int] = []
    released: list[str] = []

    async def on_changed() -> None:
        changed.append(1)

    async def before_delete(model_id: str) -> None:
        released.append(model_id)

    controller = LibraryController(source, downloads, lambda: _hw(16), on_changed, before_delete)
    view = ScreenRecorder()
    controller.attach(view)
    return Screen(controller, view, source, downloads, changed, released)


async def test_opening_lists_popular_models_once(screen: Screen) -> None:
    await screen.controller.open()
    assert [c.key for c in screen.view.cards] == [MEDIUM.repo, SMALL.repo, HUGE.repo]
    assert screen.view.states[0].loading
    assert not screen.view.states[-1].loading
    assert screen.view.states[-1].can_load_more
    await screen.controller.open()
    assert screen.source.queries == [""]  # not searched again


async def test_cards_judge_each_model(screen: Screen) -> None:
    await screen.controller.open()
    small, huge = screen.view.card(SMALL.repo), screen.view.card(HUGE.repo)
    assert small.fit == Fit.GOOD
    assert small.selected == "Small-1B-Q8_0.gguf"
    assert [v.label for v in small.versions] == ["Q4_K_M", "Q8_0", "BF16"]  # smallest first
    assert huge.fit == Fit.TOO_LARGE
    assert small.params_b == 1.0


async def test_search_and_load_more(screen: Screen) -> None:
    await screen.controller.open()
    await screen.controller.load_more()
    assert screen.view.cards[-1].key == "org/Extra-2B-GGUF"
    await screen.controller.search("small")
    assert [c.key for c in screen.view.cards] == [SMALL.repo]
    assert screen.view.states[-1].query == "small"


async def test_search_failure_is_shown(screen: Screen) -> None:
    screen.source.error = "Could not reach Hugging Face."
    await screen.controller.open()
    assert screen.view.states[-1].error == "Could not reach Hugging Face."
    assert screen.view.cards == []


async def test_download_uses_the_chosen_version(screen: Screen) -> None:
    await screen.controller.open()
    await screen.controller.choose_version(SMALL.repo, "Small-1B-Q4_K_M.gguf")
    assert screen.view.card(SMALL.repo).selected == "Small-1B-Q4_K_M.gguf"
    await screen.controller.download(SMALL.repo)
    assert screen.downloads.downloaded == [(SMALL.repo, "Small-1B-Q4_K_M.gguf")]
    assert screen.view.progress[-1] == (SMALL.repo, 2, 2)
    assert screen.changed == [1]
    card = screen.view.card(SMALL.repo)
    assert card.installed == "Small-1B-Q4_K_M.gguf"
    assert screen.view.cards[0].key == SMALL.repo  # downloaded models come first


async def test_too_large_models_are_not_downloaded(screen: Screen) -> None:
    await screen.controller.open()
    await screen.controller.download(HUGE.repo)
    assert screen.downloads.downloaded == []


async def test_failed_download_is_explained(screen: Screen) -> None:
    screen.downloads.error = "The download was damaged."
    await screen.controller.open()
    await screen.controller.download(SMALL.repo)
    assert screen.view.errors == ["The download was damaged."]
    assert screen.changed == []


async def test_cancel_stops_a_download(screen: Screen) -> None:
    screen.downloads.block = asyncio.Event()
    await screen.controller.open()
    task = asyncio.ensure_future(screen.controller.download(SMALL.repo))
    await asyncio.sleep(0.01)
    assert screen.view.card(SMALL.repo).downloading
    await screen.controller.cancel(SMALL.repo)
    await task
    assert not screen.view.card(SMALL.repo).downloading
    assert screen.downloads.downloaded == []


async def test_downloaded_files_not_in_the_results_still_show(screen: Screen) -> None:
    screen.downloads.files = {"Mine-Q4_K_M.gguf", "Small-1B-Q8_0.gguf"}
    screen.downloads.origin = {"Mine-Q4_K_M.gguf": "someone/Mine-GGUF"}
    await screen.controller.open()
    keys = [c.key for c in screen.view.cards]
    assert keys[:2] == [LOCAL_PREFIX + "Mine-Q4_K_M.gguf", SMALL.repo]
    local = screen.view.cards[0]
    assert local.installed == "Mine-Q4_K_M.gguf"
    assert local.author == "someone"
    await screen.controller.search("medium")
    assert all(not c.key.startswith(LOCAL_PREFIX) for c in screen.view.cards)


async def test_delete_releases_the_model_first(screen: Screen) -> None:
    screen.downloads.files = {"Small-1B-Q8_0.gguf"}
    await screen.controller.open()
    await screen.controller.delete("Small-1B-Q8_0.gguf")
    assert screen.released == ["Small-1B-Q8_0"]
    assert screen.downloads.deleted == ["Small-1B-Q8_0.gguf"]
    assert screen.changed == [1]
    assert screen.view.card(SMALL.repo).installed is None


async def test_failed_delete_is_explained(screen: Screen) -> None:
    screen.downloads.files = {"Small-1B-Q8_0.gguf"}
    screen.downloads.delete_error = "Could not delete Small-1B-Q8_0.gguf: in use"
    await screen.controller.open()
    await screen.controller.delete("Small-1B-Q8_0.gguf")
    assert screen.view.errors == ["Could not delete Small-1B-Q8_0.gguf: in use"]


def test_one_bit_versions_are_never_suggested() -> None:
    model = to_catalog(_model("org/Big-27B-GGUF", 27.0, "Q4_K_M", "UD-IQ2_XXS", "UD-IQ1_M"))
    assert model is not None
    [item] = suggest([model], _hw(16), set())  # only the 1-2 bit files would fit
    assert item.quant is None


def test_ud_files_rank_with_their_base_type() -> None:
    model = to_catalog(_model("org/M-8B-GGUF", 8.0, "UD-IQ1_M", "UD-Q4_K_XL", "Q4_K_M", "Q3_K_M"))
    assert model is not None
    assert [q.label for q in model.quants][:2] == ["Q4_K_M", "UD-Q4_K_XL"]


def test_mixture_of_experts_names_give_active_parameters() -> None:
    model = to_catalog(_model("org/Qwen3.6-35B-A3B-GGUF", 35.0, "Q4_K_M"))
    dense = to_catalog(_model("org/Gemma-12B-GGUF", 12.0, "Q4_K_M"))
    assert model is not None
    assert dense is not None
    assert model.active_params_b == 3.0  # noqa: PLR2004
    assert model.is_moe
    assert not dense.is_moe


async def test_too_large_cards_show_the_usual_version(screen: Screen) -> None:
    await screen.controller.open()
    huge = screen.view.card(HUGE.repo)
    assert huge.selected == "Huge-200B-Q4_K_M.gguf"
    assert huge.fit == Fit.TOO_LARGE
