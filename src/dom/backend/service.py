"""``InferenceService`` backed by a local llama-server router, with automatic tuning."""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from dom.backend.client import LlamaClient
from dom.backend.errors import ModelTooLargeError, ServerUnavailableError
from dom.domain import LoadedModel, ModelInfo, ModelStatus
from dom.tuning import BenchmarkCache, Machine, Placement, TooLarge, choose_faster, plan_model

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Callable, Sequence

    from dom.backend.presets import PresetFile
    from dom.backend.process import LlamaServerProcess
    from dom.config import GenerationSettings, InferenceSettings
    from dom.domain import ChatDelta, ChatMessage, ModelMemory
    from dom.hardware import HardwareProfile, MemoryStatus
    from dom.tuning import LlamaBench, ModelPlan

logger = logging.getLogger(__name__)

MEASUREMENT_WAIT_S = 120.0
MEASUREMENT_POLL_S = 0.5

type Progress = Callable[[str], None]


def _ignore_progress(_message: str) -> None:
    """Default progress callback."""


@dataclass(frozen=True, slots=True)
class AutoTuning:
    """Everything the service needs to tune models for this computer."""

    hardware: HardwareProfile
    presets: PresetFile
    defaults: InferenceSettings
    memory_reader: Callable[[], MemoryStatus]
    bench: LlamaBench | None
    bench_cache: BenchmarkCache
    build_version: str


class LlamaInferenceService:
    """Owns one ``LlamaServerProcess``, the ``LlamaClient`` that talks to it, and tuning."""

    def __init__(
        self,
        process: LlamaServerProcess,
        tuning: AutoTuning,
    ) -> None:
        """Wrap ``process``, tuning models with ``tuning``."""
        self._process = process
        self._tuning = tuning
        self._client: LlamaClient | None = None
        self._stopping = False
        # First (unpinned) measurement per model: once a model's context is pinned in the
        # presets file, the router only reports memory at that one size.
        self._memories: dict[str, ModelMemory] = {}
        self._plans: dict[str, ModelPlan] = {}

    # ---- lifecycle -----------------------------------------------------------------------

    async def start(self) -> None:
        """Start (or restart) the server and connect to it."""
        await self._close_client()
        self._stopping = False
        self._memories.clear()
        self._plans.clear()
        endpoint = await self._process.start()
        self._client = LlamaClient(endpoint.base_url, endpoint.api_key)

    async def stop(self) -> None:
        """Stop the server and all model processes."""
        self._stopping = True
        await self._close_client()
        await self._process.stop()

    def stop_now(self) -> None:
        """Stop the server synchronously (for interpreter/app shutdown)."""
        self._stopping = True
        self._client = None
        self._process.stop_now()

    async def wait_for_crash(self) -> int | None:
        """Wait until the server exits; ``None`` means it was stopped on purpose."""
        code = await self._process.wait_for_exit()
        if self._stopping:
            return None
        logger.error("llama-server exited unexpectedly with code %s", code)
        logger.error("Last server log lines:\n%s", self._process.log_tail())
        # The router is gone, but a model process it started may not be: clean up the tree.
        await self._close_client()
        await self._process.stop()
        return code

    def hardware_profile(self) -> HardwareProfile | None:
        """What was detected about this computer."""
        return self._tuning.hardware

    async def reload_models(self) -> list[ModelInfo]:
        """Re-scan the models folder, then list the models."""
        await self._require_client().list_models(reload=True)
        return await self.list_models()

    # ---- models --------------------------------------------------------------------------

    async def list_models(self) -> list[ModelInfo]:
        """The local models, each marked if it is too large for the memory free right now."""
        models = await self._require_client().list_models()
        self._remember_memories(models)
        ram = await asyncio.to_thread(self._tuning.memory_reader)
        # Switching unloads the current model first, so its memory counts as free for the others.
        in_use = sum(
            plan.required_bytes or 0
            for model in models
            if model.status == ModelStatus.LOADED and (plan := self._plans.get(model.id))
        )
        ram = dataclasses.replace(ram, available_bytes=ram.available_bytes + in_use)
        return [self._annotate(m, ram) for m in models]

    async def select_model(self, model_id: str, on_progress: Progress | None = None) -> LoadedModel:
        """Measure, place, tune and load ``model_id`` (unloading any other model first).

        Raises:
            ModelTooLargeError: if it cannot fit even with the smallest context.
            BackendError: for any other loading problem.
        """
        progress = on_progress or _ignore_progress
        client = self._require_client()
        info = await client.unload_others(model_id)
        previous = self._plans.get(model_id)
        if info.status == ModelStatus.LOADED and previous is not None:
            return self._loaded(info, previous)
        memory = await self._wait_for_memory(client, info, progress)
        placement = await self._choose_placement(info, progress)
        ram = await asyncio.to_thread(self._tuning.memory_reader)
        plan = plan_model(
            info, memory, Machine(ram, self._tuning.hardware.gpu), placement, self._tuning.defaults
        )
        if isinstance(plan, TooLarge):
            raise ModelTooLargeError(plan.message())
        logger.info("Plan for %s: %s", model_id, plan)
        changed = await asyncio.to_thread(self._tuning.presets.set_model, model_id, plan.settings)
        if changed:
            if info.status != ModelStatus.UNLOADED:
                await client.unload_model(model_id)
                await client.wait_for_status(model_id, ModelStatus.UNLOADED)
            await client.list_models(reload=True)
        progress("Loading the model…")
        loaded = await client.load_and_wait(model_id)
        self._plans[model_id] = plan
        return self._loaded(loaded, plan)

    def stream_chat(
        self,
        model_id: str,
        messages: Sequence[ChatMessage],
        settings: GenerationSettings | None = None,
    ) -> AsyncGenerator[ChatDelta]:
        """Stream a reply; stop iterating or cancel to stop generation."""
        return self._require_client().stream_chat(model_id, messages, settings)

    # ---- tuning steps --------------------------------------------------------------------

    def _remember_memories(self, models: list[ModelInfo]) -> None:
        for model in models:
            if model.memory is not None and model.memory.can_extrapolate:
                self._memories.setdefault(model.id, model.memory)

    def _annotate(self, model: ModelInfo, ram: MemoryStatus) -> ModelInfo:
        memory = self._memories.get(model.id)
        if memory is None or model.status != ModelStatus.UNLOADED:
            return model
        gpu = self._tuning.hardware.gpu
        placement = Placement.GPU if gpu is not None else Placement.CPU
        plan = plan_model(model, memory, Machine(ram, gpu), placement, self._tuning.defaults)
        if isinstance(plan, TooLarge):
            return dataclasses.replace(model, unavailable_reason=plan.message())
        return model

    async def _wait_for_memory(
        self, client: LlamaClient, info: ModelInfo, progress: Progress
    ) -> ModelMemory | None:
        if info.id in self._memories:
            return self._memories[info.id]
        progress("Checking how much memory the model needs…")
        with contextlib.suppress(TimeoutError):
            async with asyncio.timeout(MEASUREMENT_WAIT_S):
                while info.id not in self._memories:
                    models = await client.list_models()
                    self._remember_memories(models)
                    current = next((m for m in models if m.id == info.id), None)
                    if info.id in self._memories or current is None:
                        break
                    if current.memory_state in ("measured", "failed"):
                        break  # measured but pinned, or failed: no better measurement is coming
                    await asyncio.sleep(MEASUREMENT_POLL_S)
        memory = self._memories.get(info.id)
        if memory is None:
            logger.warning("No memory measurement for %s; using default settings", info.id)
        return memory

    async def _choose_placement(self, info: ModelInfo, progress: Progress) -> Placement:
        gpu = self._tuning.hardware.gpu
        if gpu is None:
            return Placement.CPU
        bench = self._tuning.bench
        if not gpu.integrated or bench is None or not bench.available or info.path is None:
            return Placement.GPU
        model_path = Path(info.path)
        key = BenchmarkCache.key(model_path, gpu.name, self._tuning.build_version)
        cache = self._tuning.bench_cache
        cached = await asyncio.to_thread(cache.get, key)
        if cached is not None:
            return cached
        progress("Testing whether the CPU or GPU is faster on this computer (one time only)…")
        gpu_result = await bench.run(model_path, Placement.GPU)
        cpu_result = await bench.run(model_path, Placement.CPU)
        placement = choose_faster(gpu_result, cpu_result)
        await asyncio.to_thread(cache.put, key, placement, gpu_result, cpu_result)
        logger.info("Placement for %s: %s", info.id, placement)
        return placement

    def _loaded(self, info: ModelInfo, plan: ModelPlan) -> LoadedModel:
        return LoadedModel(
            info, plan.ctx_size, self._device_label(plan), plan.required_bytes is not None
        )

    def _device_label(self, plan: ModelPlan) -> str:
        gpu = self._tuning.hardware.gpu
        if plan.placement == Placement.GPU and gpu is not None:
            return f"{gpu.name} (GPU)"
        return "CPU"

    # ---- helpers -------------------------------------------------------------------------

    def _require_client(self) -> LlamaClient:
        if self._client is None:
            raise ServerUnavailableError("The AI server is not running.")
        return self._client

    async def _close_client(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
