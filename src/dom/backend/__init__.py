"""Backend layer: the llama-server process, its HTTP client, and the InferenceService."""

from dom.backend.client import LlamaClient
from dom.backend.errors import (
    ApiError,
    BackendError,
    ChatStreamError,
    EngineMissingError,
    ModelLoadError,
    ModelNotFoundError,
    ModelTooLargeError,
    ServerStartError,
    ServerUnavailableError,
)
from dom.backend.process import LlamaServerProcess, ServerConfig, ServerEndpoint, build_command
from dom.backend.protocols import InferenceService
from dom.backend.service import AutoTuning, LlamaInferenceService

__all__ = [
    "ApiError",
    "AutoTuning",
    "BackendError",
    "ChatStreamError",
    "EngineMissingError",
    "InferenceService",
    "LlamaClient",
    "LlamaInferenceService",
    "LlamaServerProcess",
    "ModelLoadError",
    "ModelNotFoundError",
    "ModelTooLargeError",
    "ServerConfig",
    "ServerEndpoint",
    "ServerStartError",
    "ServerUnavailableError",
    "build_command",
]
