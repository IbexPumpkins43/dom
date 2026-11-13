"""The models dom offers.

Text-chat models of at most 35B parameters, from GGUF repositories by unsloth on Hugging Face.
File names and sizes were read from the Hugging Face API (November 2026); the exact size and
SHA-256 are looked up again when a download starts, so a re-uploaded file is still verified.
Within each model, files are listed in order of preference: small models prefer higher-quality
quants (the speed cost is small), larger ones prefer Q4_K_M (the usual quality/speed balance).
"""

from __future__ import annotations

from dom.catalog.models import CatalogModel, CatalogQuant


def _q(label: str, filename: str, size: int) -> CatalogQuant:
    return CatalogQuant(label, filename, size)


CATALOG: tuple[CatalogModel, ...] = (
    CatalogModel(
        "qwen3.5-0.8b", "Qwen3.5 0.8B", "Qwen", "unsloth/Qwen3.5-0.8B-GGUF", 0.8, 0.8,
        "Tiny and very fast. Fine for simple questions; makes mistakes on harder ones.",
        (_q("Q8_0", "Qwen3.5-0.8B-Q8_0.gguf", 811843840),
         _q("Q4_K_M", "Qwen3.5-0.8B-Q4_K_M.gguf", 532517120)),
    ),
    CatalogModel(
        "qwen3.5-2b", "Qwen3.5 2B", "Qwen", "unsloth/Qwen3.5-2B-GGUF", 2, 2,
        "Small and fast; a good choice for older or low-memory computers.",
        (_q("Q8_0", "Qwen3.5-2B-Q8_0.gguf", 2012012800),
         _q("Q4_K_M", "Qwen3.5-2B-Q4_K_M.gguf", 1280835840)),
    ),
    CatalogModel(
        "gemma-4-e2b", "Gemma 4 E2B", "Google", "unsloth/gemma-4-E2B-it-GGUF", 5.1, 2,
        "Google's small model, built to run well on laptops and phones.",
        (_q("Q8_0", "gemma-4-E2B-it-Q8_0.gguf", 5048352864),
         _q("Q4_K_M", "gemma-4-E2B-it-Q4_K_M.gguf", 3106738272)),
    ),
    CatalogModel(
        "qwen3.5-4b", "Qwen3.5 4B", "Qwen", "unsloth/Qwen3.5-4B-GGUF", 4, 4,
        "A capable all-rounder that still runs quickly on most computers.",
        (_q("Q8_0", "Qwen3.5-4B-Q8_0.gguf", 4482403488),
         _q("Q6_K", "Qwen3.5-4B-Q6_K.gguf", 3525956768),
         _q("Q4_K_M", "Qwen3.5-4B-Q4_K_M.gguf", 2740937888)),
    ),
    CatalogModel(
        "gemma-4-e4b", "Gemma 4 E4B", "Google", "unsloth/gemma-4-E4B-it-GGUF", 8, 4,
        "Google's larger efficient model: good writing at a modest speed cost.",
        (_q("Q8_0", "gemma-4-E4B-it-Q8_0.gguf", 8192953472),
         _q("Q4_K_M", "gemma-4-E4B-it-Q4_K_M.gguf", 4977171584)),
    ),
    CatalogModel(
        "qwen3.5-9b", "Qwen3.5 9B", "Qwen", "unsloth/Qwen3.5-9B-GGUF", 9, 9,
        "Noticeably smarter than the small models; wants a reasonably modern computer.",
        (_q("Q4_K_M", "Qwen3.5-9B-Q4_K_M.gguf", 5680522464),
         _q("Q6_K", "Qwen3.5-9B-Q6_K.gguf", 7458301152),
         _q("Q8_0", "Qwen3.5-9B-Q8_0.gguf", 9527502048)),
    ),
    CatalogModel(
        "gemma-4-12b", "Gemma 4 12B", "Google", "unsloth/gemma-4-12b-it-GGUF", 12, 12,
        "Strong writing and reasoning; best with a graphics card or 16 GB+ of RAM.",
        (_q("Q4_K_M", "gemma-4-12b-it-Q4_K_M.gguf", 7121861440),
         _q("Q6_K", "gemma-4-12b-it-Q6_K.gguf", 9786022720),
         _q("Q8_0", "gemma-4-12b-it-Q8_0.gguf", 12669647680)),
    ),
    CatalogModel(
        "gpt-oss-20b", "gpt-oss 20B", "OpenAI", "unsloth/gpt-oss-20b-GGUF", 21, 3.6,
        "OpenAI's open model. Large, but only part of it works per word, so it runs fast.",
        (_q("Q4_K_M", "gpt-oss-20b-Q4_K_M.gguf", 11624759488),),
    ),
    CatalogModel(
        "gemma-4-26b-a4b", "Gemma 4 26B A4B", "Google", "unsloth/gemma-4-26B-A4B-it-GGUF", 26, 4,
        "High quality and fast for its size; needs about 24 GB of memory.",
        (_q("Q4_K_M", "gemma-4-26B-A4B-it-UD-Q4_K_M.gguf", 16947541728),
         _q("Q8_0", "gemma-4-26B-A4B-it-Q8_0.gguf", 26859861728)),
    ),
    CatalogModel(
        "qwen3.8-27b", "Qwen3.8 27B", "Qwen", "unsloth/Qwen3.8-27B-GGUF", 27, 27,
        "One of the best models of this size, but slow without a big graphics card.",
        (_q("Q4_K_M", "Qwen3.8-27B-UD-Q4_K_M.gguf", 16564259744),
         _q("Q8_0", "Qwen3.8-27B-Q8_0.gguf", 29047086048)),
    ),
    CatalogModel(
        "qwen3.6-35b-a3b", "Qwen3.6 35B A3B", "Qwen", "unsloth/Qwen3.6-35B-A3B-GGUF", 35, 3,
        "Very capable and surprisingly fast; needs about 32 GB of memory.",
        (_q("Q4_K_M", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf", 22134528992),
         _q("Q8_0", "Qwen3.6-35B-A3B-Q8_0.gguf", 36903140320)),
    ),
)  # fmt: skip
