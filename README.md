# dōm

A desktop app that makes local AI simple. Everything runs on your machine: dom starts
[llama.cpp](https://github.com/ggml-org/llama.cpp)'s `llama-server` in the background and talks to
it over `127.0.0.1` only. No cloud, no telemetry.

**Milestone 1** is the foundation: a chat window with a model picker, streaming replies rendered
as Markdown, a Stop button, and chat history that survives restarts.

**Milestone 2a** adds automatic setup: dom detects the computer's RAM, CPU and GPU, runs the GPU
(Vulkan) build of llama.cpp when a GPU is found, and tunes each model when it is loaded:

- **Context size**: the largest that fits in the memory that is free right now (4k–32k tokens,
  never more than the model was trained for), based on llama.cpp's own measurement of the model.
- **CPU or GPU**: a dedicated GPU is always used. On a GPU built into the processor (Intel Iris /
  Arc, AMD Radeon graphics, Qualcomm Adreno) dom runs a short speed test the first time a model is
  loaded and keeps whichever is faster; the result is remembered.
- **Thinking budget**: scaled with model size.
- Models that cannot fit are greyed out in the picker with an explanation.

Works on Windows x64, Windows ARM64 and Linux. The GUI is PySide6 (Qt), which has native
Windows ARM64 builds.

## Getting started (development)

```sh
uv sync                                       # create .venv and install everything
uv run python scripts/fetch_test_assets.py    # llama-server + two small test models into .dev/
DOM_DEV=1 uv run python -m dom                # run the app using .dev/
```

On Windows PowerShell use `$env:DOM_DEV = "1"; uv run python -m dom`.

`fetch_test_assets.py` downloads the CPU and Vulkan (GPU) builds of the latest stable llama.cpp
release for your OS/CPU into `.dev/bin/cpu/` and `.dev/bin/vulkan/` (the exact version is written
to `VERSION` in each) and the Q4_K_M and Q8_0 quants of `unsloth/Qwen3.5-0.8B-GGUF` into
`.dev/models/`. Every file is checked against the
SHA-256 its source publishes. Interrupted downloads resume, and re-runs skip anything already
verified.

On Linux, Qt needs the `libxcb-cursor0` system package when running under X11.

## Where things live

| | Development (`DOM_DEV=1`) | Normal |
|---|---|---|
| `llama-server` | `.dev/bin/<build>/` | `DOM_LLAMA_SERVER` if set, else `<data>/bin/<build>/` (or `<data>/bin/`), else `PATH` |
| Models (`*.gguf`) | `.dev/models/` | `<data>/models/` |
| Chat history | `.dev/data/dom.sqlite3` | `<data>/dom.sqlite3` |
| Logs | `.dev/data/logs/` | `<data>/logs/` (`dom.log`, `llama-server.log`) |
| Tuning files | `.dev/data/` | `<data>/presets.ini` (rewritten by dom), `<data>/benchmarks.json` |

`<data>` is the OS user-data folder: `~/.local/share/dom` on Linux, `%LOCALAPPDATA%\dom` on
Windows.

## Checks

```sh
uv run ruff format --check .
uv run ruff check .
uv run pyright
uv run pytest                    # unit tests
uv run pytest -m integration     # real llama-server + test models (needs the fetch script first)
```

## Manual test checklist

Run with `DOM_DEV=1 uv run python -m dom` after fetching the test assets.

1. **Models are listed.** Open the model picker at the top. Both `Qwen3.5-0.8B-Q4_K_M` and
   `Qwen3.5-0.8B-Q8_0` are there, marked "not loaded".
2. **Loading and switching.** Pick `Qwen3.5-0.8B-Q4_K_M`. It shows "loading…", then "ready", and
   the message box becomes usable. Now pick `Qwen3.5-0.8B-Q8_0`: it goes loading → ready, and
   opening the picker again shows the Q4_K_M as "not loaded" (only one model is loaded at a time).
3. **Streaming and Stop.** Type a message and press Enter (Shift+Enter adds a new line). A
   collapsed "Thinking…" line appears first, then the reply streams in and is shown as Markdown.
   While it streams, the input is disabled and the button reads **Stop**. Send a long request
   ("Write a long essay about bread") and press Stop: the reply stops and what arrived so far is
   kept.
4. **History survives restarts.** Close the window and start the app again. The conversation(s)
   are in the left sidebar, newest first. Clicking one shows its messages. "New chat" starts an
   empty conversation.
5. **Missing `llama-server`.** Run with the binary unavailable, e.g. temporarily rename
   `.dev/bin` to `.dev/bin-off` and start with `DOM_DEV=1`. A red banner explains that
   `llama-server` is missing and which folder to put it in. There is no crash or stack trace.
   Rename the folder back and click **Try again**: the engine starts.
6. **No leftover processes.** After closing the window, check that no `llama-server` is running:
   - Linux: `pgrep -a llama-server` prints nothing.
   - Windows: `Get-Process llama-server -ErrorAction SilentlyContinue` prints nothing (or check
     Task Manager).
7. **Automatic setup (milestone 2a).** Hover over "AI engine running": the tooltip lists the
   detected hardware. Pick a model: next to the picker dom shows what it is doing ("Checking how
   much memory…", and on a computer with a built-in GPU, the one-time speed test), then e.g.
   "Running on Intel(R) Iris(R) Xe Graphics (GPU) · 32k context" or "Running on CPU · 16k
   context". Close and reopen dom and pick the same model: the speed test does not run again.
   `dom.log` records the detected hardware, the speed-test results and the chosen settings.

Optional: kill the `llama-server` process while the app is running. The banner says the AI engine
stopped unexpectedly and offers **Restart**, which also reloads the model you had picked.
