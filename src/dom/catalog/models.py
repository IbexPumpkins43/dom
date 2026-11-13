"""Catalog entries."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CatalogQuant:
    """One downloadable file of a model (a quantisation level)."""

    label: str
    filename: str
    size_bytes: int

    @property
    def model_id(self) -> str:
        """The id llama-server gives the file (its name without ``.gguf``)."""
        return self.filename.removesuffix(".gguf")


@dataclass(frozen=True, slots=True)
class CatalogModel:
    """A model dom offers, with its files in order of preference."""

    key: str
    name: str
    maker: str
    repo: str
    params_b: float
    active_params_b: float
    summary: str
    quants: tuple[CatalogQuant, ...]

    @property
    def is_moe(self) -> bool:
        """Mixture-of-experts: only part of the weights is used per token, so it runs faster."""
        return self.active_params_b < self.params_b
