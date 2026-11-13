"""Inference settings: the single place every llama-server flag comes from."""

from __future__ import annotations

from dataclasses import dataclass, fields

# Context size used instead of llama-server's default. The default (0 = the model's maximum)
# reserves 262,144 tokens for Qwen3.5 and used about 4 GB of RAM for a 0.8B model on CPU.
DEFAULT_CTX_SIZE = 8192

# Thinking-token budget used instead of llama-server's default (unlimited). With no budget the
# Qwen3.5-0.8B test model can "think" in a loop for minutes without ever answering.
DEFAULT_REASONING_BUDGET = 512


@dataclass(frozen=True, slots=True)
class InferenceSettings:
    """Model runtime and sampling settings passed to ``llama-server``.

    ``None`` means "use llama-server's own default" and emits no flag. Later milestones
    (automatic tuning) produce instances of this class; nothing else may build flags.
    """

    ctx_size: int | None = DEFAULT_CTX_SIZE
    gpu_layers: str | None = None
    threads: int | None = None
    n_predict: int | None = None
    reasoning_budget: int | None = DEFAULT_REASONING_BUDGET
    temperature: float | None = None
    top_k: int | None = None
    top_p: float | None = None
    min_p: float | None = None
    presence_penalty: float | None = None

    def to_server_args(self) -> list[str]:
        """Return the command-line flags for every setting that is not ``None``."""
        args: list[str] = []
        for field in fields(self):
            value: object = getattr(self, field.name)
            if value is not None:
                args += [_FLAGS[field.name], str(value)]
        return args


# Checked against `llama-server --help` for the pinned build (v0.9.0 / b12110).
_FLAGS: dict[str, str] = {
    "ctx_size": "--ctx-size",
    "gpu_layers": "--gpu-layers",
    "threads": "--threads",
    "n_predict": "--n-predict",
    "reasoning_budget": "--reasoning-budget",
    "temperature": "--temp",
    "top_k": "--top-k",
    "top_p": "--top-p",
    "min_p": "--min-p",
    "presence_penalty": "--presence-penalty",
}
