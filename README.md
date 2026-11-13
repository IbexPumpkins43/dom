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

**Milestone 2b** adds downloads, only ever started by the user:

- **Get models** (left sidebar) lists 11 chat models from 0.8B to 35B parameters (Qwen, Gemma,
  gpt-oss), each judged against this computer: whether it fits, which file size to use, about
  how fast it will answer, and one **Recommended** pick. Downloads come from Hugging Face, can be
  cancelled and resumed, and are checked against the published SHA-256 before they are used.
  The window opens by itself the first time there are no models.
- If the AI engine (llama.cpp) is missing, a **Download AI engine** button installs the CPU and
  GPU builds from GitHub.
- dom contacts nothing except Hugging Face and GitHub, and only for these downloads. Every
  request is checked against an allow-list.

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
| `llama-server` | `.dev/bin/<build>/` | `DOM_LLAMA_SERVER` if set, else `<data>/bin/<build>/` (downloaded by dom), else the copy installed with dom, else `PATH` |
| Models (`*.gguf`) | `.dev/models/` | `<data>/models/` |
| Chat history | `.dev/data/dom.sqlite3` | `<data>/dom.sqlite3` |
| Logs | `.dev/data/logs/` | `<data>/logs/` (`dom.log`, `llama-server.log`) |
| Tuning files | `.dev/data/` | `<data>/presets.ini` (rewritten by dom), `<data>/benchmarks.json` |

`<data>` is the OS user-data folder: `~/.local/share/dom` on Linux, `%LOCALAPPDATA%\dom` on
Windows. Set `DOM_DATA_DIR` to use another folder (for testing, or a portable install).

## Checks

```sh
uv run ruff format --check .
uv run ruff check .
uv run pyright
uv run pytest                    # unit tests
uv run pytest -m integration     # real llama-server + test models (needs the fetch script first)
```

## Building installers (milestone 2c)

```sh
uv run python scripts/build_release.py
```

builds dom for the computer it runs on, with llama.cpp (CPU and Vulkan builds, SHA-256 checked)
inside the app folder, licence texts in `licenses/`, and:

- **Linux**: `dist/dom-<version>-linux-<arch>.tar.gz`. Unpack it and run `install.sh` (installs
  for the current user into `~/.local/opt/dom` and adds a menu entry; `uninstall.sh` removes it).
  Under X11, Qt needs the usual X libraries (`libxcb-cursor0`, `libxkbcommon-x11-0`, …).
- **Windows**: `dist/dom-<version>-windows-<x64|arm64>-setup.exe`, a per-user installer (no
  administrator rights) made with [Inno Setup](https://jrsoftware.org/isinfo.php) if `iscc` is
  installed; otherwise a `.zip` of the app folder.

PyInstaller cannot cross-compile, so each platform is built on its own machine.
`.github/workflows/release.yml` does this for Linux x64/ARM64 and Windows x64/ARM64, then installs
each package and runs `dom --self-test` (start the engine, load the smallest model in the models
folder, get an answer, stop) against the 0.8B test model.

Chats, models and settings live in the user-data folder, so uninstalling or upgrading keeps them.

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
8. **Downloads (milestone 2b).** Click **Get models**. Every model shows whether it fits, its
   download size and speed estimate, and one is marked **Recommended**. Download the smallest one,
   cancel half way, click **Resume**: it continues where it stopped, then shows **Downloaded** and
   appears in the model picker. To test the engine download, start with an empty data folder
   (e.g. `DOM_DATA_DIR=/tmp/dom-test uv run python -m dom`, see below): the banner offers
   **Download AI engine**, and after it finishes the engine starts by itself.

Optional: kill the `llama-server` process while the app is running. The banner says the AI engine
stopped unexpectedly and offers **Restart**, which also reloads the model you had picked.
