# dom, Milestone 1: Local chat app (GUI + llama.cpp backend + persistent history)

## Context

I'm building a desktop AI harness whose whole purpose is to make local AI easy for people with no
experience. Right now running a local model means installing tools by hand, finding the right
files and then spending hours tuning parameters, and if you get them wrong the model performs
badly. Later milestones will automate all of that (hardware detection, choosing models, tuning
parameters). **This milestone is only the foundation:** a working chat app that I can build those
features on.

Nothing exists yet. Start from an empty directory. Read `CLAUDE.md` first; its rules apply to
everything below.

## Goal for this milestone

A desktop app that looks and feels like the ChatGPT / Claude web UI:

1. **Model picker:** a list of the local GGUF models found in the app's models folder, showing
   each one's status (unloaded / loading / loaded / failed). Picking one loads it.
2. **Chat view:** the conversation between the user and the AI, with user and assistant messages
   visually distinct. Assistant replies stream in token by token and render Markdown.
3. **Input box:** multi-line text field. Enter sends, Shift+Enter adds a newline. Disabled while a
   reply is streaming or no model is loaded, with a Stop button to cancel a reply mid-stream.
4. **Persistent history:** conversations and messages survive closing and reopening the app. A
   left sidebar lists past conversations (newest first) with a "New chat" button. Clicking one
   reopens it.

## Tech stack (decided, don't revisit without asking)

- **Python 3.12+**, managed with **uv** (`pyproject.toml`, `uv.lock`).
- **GUI: Flet 1.x** (latest stable). Flet 1.0 was a ground-up rewrite with breaking changes from
  0.28, so most examples online and in your training data are **outdated**. Before writing UI code,
  check the current API: use the Flet MCP server if it's connected, otherwise the docs at
  flet.dev. Never guess at a Flet API.
- **Inference: llama.cpp's `llama-server`**, run as a child process in **router mode** (started
  with no `-m`, using `--models-dir`), and reached over its local HTTP API. Use
  `--host 127.0.0.1`, a free port chosen at startup, `--no-models-autoload` and `--models-max 1`.
  Use `/models`, `/models/load` and `/models/unload` to manage models, and the OpenAI-compatible
  `/v1/chat/completions` with `stream: true` for chat. **Check the exact flags and endpoint shapes
  against `llama-server --help` and the llama.cpp server README for the installed build. Don't
  assume them from memory.** Do not use `llama-cpp-python`.
- **HTTP client:** `httpx` (async).
- **Storage:** SQLite through the standard library `sqlite3`, kept in the OS user-data directory
  (use `platformdirs`). Use a schema version table so later migrations are possible.
- **Tooling:** `ruff` (lint + format), `pyright` in strict mode, `pytest` (+ `pytest-asyncio` if
  needed).

## Architecture

Keep the layers separate so the UI can be replaced and the backend can be tested without a GUI:

```
src/dom/
    domain/      # frozen dataclasses & enums: Message, Conversation, ModelInfo, ModelStatus, Role
    storage/     # ConversationRepository protocol + SQLite implementation
    backend/     # LlamaServerProcess (start/stop/health), LlamaClient (models, load, stream chat)
    config/      # app paths, llama-server binary location, inference defaults (one place)
    ui/          # Flet views/components; depends only on protocols from the layers above
    app.py       # composition root: wires real implementations together
tests/
```

- `ui/` must depend on `typing.Protocol` interfaces, not on concrete classes, so tests can use
  fakes.
- **Every inference parameter** (context size, GPU layers, sampling settings) comes from a single
  `InferenceSettings` object in `config/`. For this milestone leave them at llama-server's defaults,
  except where a default is clearly broken for chat. This object is where automatic tuning will
  plug in later, so don't scatter flags around.
- **Locating `llama-server`:** check a configured path first, then a `bin/` folder inside the app
  data directory, then `PATH`. If it's not found, the UI shows a clear, friendly message saying
  what's missing and where to put it, never a stack trace. Having the *app* download the binary
  for users is **out of scope** for now. For development, see "Test assets" below.
- **Process lifecycle:** start the server on app launch, wait for `/health`, and stop it cleanly
  on exit, including when the window is closed. No orphaned `llama-server` processes. If the
  server crashes, show that in the UI and offer a restart.
- Network calls and streaming must never block the UI thread.

## Test assets (you may download these)

You're allowed to download the following for development and testing. Do it with a script,
`scripts/fetch_test_assets.py` (typed and linted like the rest of the code, run with
`uv run python scripts/fetch_test_assets.py`), so I can rerun it on any of my machines:

1. **`llama-server` binary:** the latest release from
   [ggml-org/llama.cpp on GitHub](https://github.com/ggml-org/llama.cpp/releases). Pick the asset
   that matches the current OS and CPU architecture (Windows x64, Windows ARM64, Linux x64). Use
   the **CPU build** by default so it runs everywhere. Unpack it into `.dev/bin/` and write the
   release tag to `.dev/bin/VERSION` so the build is pinned and known. Look up release asset names
   through the GitHub API rather than hard-coding them, because they change between releases.
2. **Test models:** two quants of
   [unsloth/Qwen3.5-0.8B-GGUF](https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF), so the model
   picker and model switching can be tested:
   - **Q4_K_M** (about 533 MB)
   - **Q8_0** (about 812 MB)

   Find the exact filenames from the repo's file list (Hugging Face API) instead of guessing them,
   download them into `.dev/models/`, and check each one's SHA-256 against the hash Hugging Face
   reports.

Rules for the script:
- Skip anything that's already present and verified, so reruns are fast.
- Show download progress. Downloads are resumable or at least fail cleanly, with no
  half-written files left in place (download to a temp name, then rename).
- Add `.dev/` to `.gitignore`. Never commit binaries or models.
- In development mode the app uses `.dev/bin/` and `.dev/models/` (through `config/`, e.g. a
  `DOM_DEV=1` environment variable). The integration test uses them too.
- Download only the binary and these two model files, from those two sources. Ask me before
  fetching anything else.

## Platforms

It must work on **Windows x64, Windows ARM64 (Snapdragon X) and Linux**. Use `pathlib` everywhere,
and don't hard-code `.exe` or path separators. If a dependency doesn't support Windows ARM64,
**stop and tell me** before working around it.

## Out of scope (don't build these yet)

In-app downloading of models or binaries (the dev script above is the only exception), hardware
detection, automatic parameter tuning, a settings
screen, renaming/deleting/searching conversations, multimodal input, tools/function calling,
packaging/installers, theming beyond a clean default light/dark.

## How to work

1. **Plan first.** Before writing code, give me a short plan: the file layout, the SQLite schema,
   the exact llama-server command line you'll use (after checking `--help`) and the order you'll
   build things in. Wait for my OK.
2. Then build in this order, getting each step to pass the checks before moving on:
   1. Project skeleton + tooling config (`ruff format --check`, `ruff check`, `pyright` all clean
      on an empty package), then `scripts/fetch_test_assets.py`. Run it, and confirm
      `.dev/bin/llama-server --version` works and both model files are in place.
   2. `domain/` + `storage/` with tests.
   3. `backend/` with tests against a fake HTTP transport (`httpx.MockTransport`).
   4. `ui/` wired to fakes first, then to the real backend in `app.py`.
3. `git init` at the start and commit after each step with a clear message.
4. If something in this prompt turns out to be wrong or impossible (for example a flag that
   doesn't exist in the current llama.cpp, or a Flet API that changed), **tell me** instead of
   quietly working around it.

## Acceptance criteria (definition of done)

Automated, and all must pass:

- [ ] `uv run ruff format --check .` and `uv run ruff check .` → clean.
- [ ] `uv run pyright` → 0 errors, 0 warnings, strict mode.
- [ ] `uv run pytest` → all pass, covering at least:
  - storage: create conversation → add messages → **close and reopen the database** → the same
    conversations and messages come back in the same order.
  - backend client: parses the model list and its statuses; load request + polling until
    `loaded`; streamed chat chunks are put together into the full reply; cancellation stops the
    stream; server errors become typed exceptions rather than crashes.
- [ ] An **integration test** marked `@pytest.mark.integration` (skipped by default, and skipped if
  the test assets haven't been fetched) that starts the real `.dev/bin/llama-server` with
  `.dev/models/`, checks both Qwen3.5-0.8B quants are listed, loads the Q4_K_M, sends one message
  and gets a non-empty reply, then switches to the Q8_0 (the first one is unloaded) and gets a
  reply from it too. You must actually run it and confirm it passes.

Manual, which I'll check myself (write these steps into `README.md`):

- [ ] Both test models (Q4_K_M and Q8_0) appear in the model picker.
- [ ] Picking one shows loading → loaded. Switching to another unloads the first.
- [ ] Sending a message streams back a reply; Stop cancels it.
- [ ] Closing and reopening the app shows the same conversation(s) and messages in the sidebar.
- [ ] With `llama-server` missing, the app shows the friendly error instead of crashing.
- [ ] After closing the app, no `llama-server` process is left running.

When everything passes, give me a short summary of what was built, anything you weren't sure
about, and suggestions for milestone 2.
