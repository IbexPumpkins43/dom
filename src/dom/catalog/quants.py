"""Reading a GGUF file name: its quantisation, rough size, and which version to prefer.

Hugging Face's search gives a repository's file names and the model's parameter count, but not
file sizes. A file's size is close to parameters x bits per weight, which is enough to judge
whether it fits this computer; the exact size and hash are read when a download starts.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from dom.catalog.models import CatalogModel, CatalogQuant

if TYPE_CHECKING:
    from dom.catalog.models import HubModel

# Average bits per weight of llama.cpp's quantisation types (from llama.cpp's quantize table,
# rounded). Unsloth's "UD-…_XL" files keep some layers at higher precision: a little larger.
BITS_PER_WEIGHT: dict[str, float] = {
    "F32": 32.0,
    "F16": 16.0,
    "BF16": 16.0,
    "Q8_0": 8.5,
    "Q6_K": 6.6,
    "Q5_1": 6.0,
    "Q5_K_M": 5.7,
    "Q5_K_S": 5.5,
    "Q5_0": 5.5,
    "Q4_1": 5.0,
    "Q4_K_M": 4.9,
    "Q4_K_S": 4.6,
    "Q4_0": 4.6,
    "IQ4_NL": 4.5,
    "IQ4_XS": 4.3,
    "MXFP4": 4.3,
    "MXFP4_MOE": 4.3,
    "Q3_K_L": 4.3,
    "Q3_K_M": 3.9,
    "IQ3_M": 3.7,
    "Q3_K_S": 3.5,
    "IQ3_S": 3.4,
    "IQ3_XS": 3.3,
    "IQ3_XXS": 3.1,
    "Q2_K": 3.0,
    "IQ2_M": 2.7,
    "IQ2_S": 2.5,
    "IQ2_XS": 2.3,
    "IQ2_XXS": 2.1,
    "IQ1_M": 1.8,
    "IQ1_S": 1.6,
}
_XL_EXTRA_BITS = 0.4
_LABEL = re.compile(
    r"(?<![A-Za-z0-9])(UD-)?(IQ\d_(?:XXS|XS|S|M|NL)|Q\d_K_(?:XL|L|M|S)|Q\d_K|Q\d_[01]"
    r"|MXFP4_MOE|MXFP4|BF16|F16|F32)(?![A-Za-z0-9])",
    re.IGNORECASE,
)
_ACTIVE = re.compile(r"(?:^|[-_.])A(\d+(?:\.\d+)?)B(?:$|[-_.])", re.IGNORECASE)
_SPLIT_PART = re.compile(r"-\d{5}-of-\d{5}\.gguf$", re.IGNORECASE)
_NOT_A_MODEL = ("mmproj", "imatrix")  # vision projectors and quantisation data, not chat models

# Which version to offer first. Small models: higher quality costs little speed. Others: Q4_K_M,
# the usual balance of quality and speed, then smaller files, then larger ones.
SMALL_MODEL_PARAMS = 4.5e9
_SMALL_FIRST = ("Q8_0", "Q6_K", "Q5_K_M")
PREFERENCE: tuple[str, ...] = (
    "Q4_K_M", "UD-Q4_K_XL", "Q4_K_S", "IQ4_XS", "IQ4_NL", "Q4_0", "MXFP4_MOE", "MXFP4",
    "Q3_K_M", "UD-Q3_K_XL", "Q3_K_S", "IQ3_M", "IQ3_XXS", "Q2_K", "UD-Q2_K_XL", "IQ2_M",
    "IQ2_XXS", "Q5_K_M", "UD-Q5_K_XL", "Q5_K_S", "Q6_K", "UD-Q6_K_XL", "Q8_0", "UD-Q8_K_XL",
    "BF16", "F16", "F32",
)  # fmt: skip


def quant_label(path: str) -> str | None:
    """The quantisation in a GGUF file name, e.g. ``Q4_K_M`` or ``UD-Q4_K_XL``."""
    matches = list(_LABEL.finditer(path.rsplit("/", 1)[-1]))
    if not matches:
        return None
    ud, kind = matches[-1].group(1), matches[-1].group(2).upper()
    return f"UD-{kind}" if ud else kind


def bits_per_weight(label: str) -> float | None:
    """Average bits per weight of a quantisation label (``None`` if unknown)."""
    kind = label.removeprefix("UD-")
    if kind.endswith("_XL"):
        base = BITS_PER_WEIGHT.get(kind.removesuffix("_XL") + "_M")
        return base + _XL_EXTRA_BITS if base is not None else None
    return BITS_PER_WEIGHT.get(kind)


def is_single_model_file(path: str) -> bool:
    """True for a whole GGUF chat model (not split into parts, not a projector or imatrix)."""
    name = path.rsplit("/", 1)[-1].lower()
    return (
        name.endswith(".gguf")
        and not _SPLIT_PART.search(name)
        and not any(word in name for word in _NOT_A_MODEL)
        and quant_label(name) is not None
    )


def estimated_size(params: int, label: str) -> int | None:
    """Rough file size in bytes for ``params`` weights at ``label``."""
    bits = bits_per_weight(label)
    return round(params * bits / 8) if bits is not None else None


def _rank(label: str, params: int) -> tuple[int, bool, str]:
    """Sort key: place in the preference order (a "UD-" file ranks with its base type, just
    after it), then the label."""
    order = (_SMALL_FIRST if params <= SMALL_MODEL_PARAMS else ()) + PREFERENCE
    base = label.removeprefix("UD-")
    if label in order:
        index = order.index(label)
    elif base in order:
        index = order.index(base)
    else:
        index = len(order)
    return (index, label.startswith("UD-"), label)


def active_params_b(name: str, params_b: float) -> float:
    """Billions of weights used per token: mixture-of-experts names say so (``35B-A3B`` uses
    3B); other models use all of them."""
    match = _ACTIVE.search(name)
    if match is None:
        return params_b
    return min(params_b, float(match.group(1)))


def to_catalog(model: HubModel) -> CatalogModel | None:
    """``model`` as a catalog model (files in order of preference), or ``None`` if its size
    cannot be estimated (no parameter count, or no file with a known quantisation)."""
    if model.params is None or model.params <= 0:
        return None
    quants: dict[str, CatalogQuant] = {}
    for path in model.files:
        label = quant_label(path)
        if label is None or label in quants:
            continue
        size = estimated_size(model.params, label)
        if size is not None:
            quants[label] = CatalogQuant(label, path, size)
    if not quants:
        return None
    ordered = sorted(quants.values(), key=lambda q: _rank(q.label, model.params or 0))
    params_b = model.params / 1e9
    return CatalogModel(
        key=model.repo,
        name=model.name,
        maker=model.author,
        repo=model.repo,
        params_b=params_b,
        active_params_b=active_params_b(model.repo, params_b),
        quants=tuple(ordered),
    )
