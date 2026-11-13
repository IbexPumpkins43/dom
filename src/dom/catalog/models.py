"""Models that can be downloaded: Hugging Face repositories and their GGUF files."""

from __future__ import annotations

from dataclasses import dataclass

GGUF_REPO_SUFFIXES = ("-GGUF", "-gguf", "_GGUF", "_gguf")


@dataclass(frozen=True, slots=True)
class HubModel:
    """A GGUF chat model repository on Hugging Face, as listed by its search."""

    repo: str
    downloads: int
    likes: int
    params: int | None
    files: tuple[str, ...]  # paths of the repo's single-file GGUF models (one per quant)
    architecture: str | None = None
    context_length: int | None = None

    @property
    def author(self) -> str:
        """Who published the repository (e.g. ``unsloth``)."""
        return self.repo.split("/", 1)[0]

    @property
    def name(self) -> str:
        """The repository name without the author or a "-GGUF" suffix."""
        name = self.repo.split("/", 1)[-1]
        for suffix in GGUF_REPO_SUFFIXES:
            name = name.removesuffix(suffix)
        return name


@dataclass(frozen=True, slots=True)
class CatalogQuant:
    """One downloadable file of a model (a quantisation level)."""

    label: str
    filename: str  # path in the repository; saved under its base name
    size_bytes: int  # estimated until the download starts (then read from Hugging Face)

    @property
    def local_name(self) -> str:
        """The file's name in the models folder."""
        return self.filename.rsplit("/", 1)[-1]

    @property
    def model_id(self) -> str:
        """The id llama-server gives the file (its name without ``.gguf``)."""
        return self.local_name.removesuffix(".gguf")


@dataclass(frozen=True, slots=True)
class CatalogModel:
    """A model dom can download, with its files in order of preference."""

    key: str
    name: str
    maker: str
    repo: str
    params_b: float
    active_params_b: float
    quants: tuple[CatalogQuant, ...]

    @property
    def is_moe(self) -> bool:
        """Mixture-of-experts: only part of the weights is used per token, so it runs faster."""
        return self.active_params_b < self.params_b
