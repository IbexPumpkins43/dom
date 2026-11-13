# CLAUDE.md

## Project

**dom** is a desktop AI harness that makes local AI simple for people with no experience. Python + PySide6 (Qt) GUI,
llama.cpp (`llama-server`) running locally as the inference backend. Everything runs on the
user's machine: no cloud APIs, no telemetry. The app talks to `127.0.0.1` only, except for
downloads the user explicitly starts (a catalog model from Hugging Face, or the llama.cpp
engine from GitHub releases). All such traffic goes through `dom.downloads.make_client`, which
refuses any host outside `ALLOWED_HOSTS`; nothing runs in the background. The dev-only
`scripts/fetch_test_assets.py` uses the same code to fetch the llama.cpp builds and two test
models (Qwen3.5-0.8B Q4_K_M and Q8_0) into `.dev/`.

## Commands

- Install / sync: `uv sync`
- Fetch test binary + models: `uv run python scripts/fetch_test_assets.py`
- Run app in dev mode (uses `.dev/`): set `DOM_DEV=1`, then run the app
- Run app: `uv run python -m dom`
- Format: `uv run ruff format .`
- Lint: `uv run ruff check . --fix`
- Type check: `uv run pyright`
- Tests: `uv run pytest` (integration: `uv run pytest -m integration`)

Run format, lint, type check and tests before saying any task is done.

## Environment

- uv manages a project-local virtual environment in `.venv/` (git-ignored). Never install
  packages globally or with plain `pip`.
- Add dependencies with `uv add <pkg>` (dev tools: `uv add --dev <pkg>`) so `pyproject.toml` and
  `uv.lock` stay in sync. Run everything through `uv run`.

## Code style (non-negotiable)

- **4-space indentation**, never tabs. Line length 100. Double quotes. `ruff format` is the
  source of truth for formatting.
- **Strict static typing:** pyright `typeCheckingMode = "strict"`. Every function, method and
  attribute is fully annotated, including return types (`-> None`).
- No `Any`, no bare `# type: ignore`, no `cast()` to silence errors. If one is truly unavoidable
  (for example an untyped third-party API), use `# pyright: ignore[specificRule]` with a comment
  explaining why.
- Use modern syntax: `list[str]`, `X | None`, `type` aliases, `typing.Protocol` for interfaces.
- Domain objects are `@dataclass(frozen=True, slots=True)`. Use `enum.StrEnum` for fixed sets.
- No mutable global state. Dependencies are passed in (constructor injection) and wired up in
  `app.py`.
- Errors: define specific exception classes per layer. Never use bare `except:` or swallow
  exceptions silently. Anything that reaches the UI is turned into a plain-English message.
- Use `pathlib.Path` for all paths. Use `logging` for logs (no `print`).
- Docstrings on public classes and functions (Google style, short).

## Suggested ruff config (pyproject.toml)

```toml
[tool.ruff]
line-length = 100
indent-width = 4
target-version = "py312"

[tool.ruff.format]
indent-style = "space"
quote-style = "double"

[tool.ruff.lint]
select = ["E", "W", "F", "I", "B", "UP", "N", "SIM", "RUF", "ANN", "PL", "PT", "TCH", "C4", "RET"]

[tool.pyright]
typeCheckingMode = "strict"
pythonVersion = "3.12"
```

## Rules

- GUI is PySide6 (`pyside6-essentials` only; no QtWebEngine), with `qasync` running asyncio on
  the Qt event loop. Chosen over Flet/NiceGUI because it is native on Windows ARM64. Check the
  installed PySide6 stubs/docs before using a Qt API; don't rely on memory. `qasync` is untyped,
  so its API is described by the local stubs in `typings/`.
- Check `llama-server` flags and endpoints against `llama-server --help` and the llama.cpp server
  README for the installed build.
- Every inference parameter lives in `config/` (`InferenceSettings`). Don't hard-code flags
  anywhere else. Its values reach llama-server only through the generated presets file
  (`backend/presets.py`), never as router command-line flags: those override presets.
- Automatic tuning (`tuning/`) chooses per-model settings from `hardware/` detection and the
  router's memory measurements; keep `tuning/` pure and unit-tested.
- The UI never blocks on network or disk I/O.
- Never leave orphaned `llama-server` processes.
- Must work on Windows x64, Windows ARM64 and Linux.
- Keep to the current milestone's scope. Suggest extras, don't build them.
- If a requirement looks wrong or can't be done, say so and ask. Don't silently work around it.
- Commit after each working step, with clear messages.
