"""Backend layer: the llama-server process, its HTTP client, and the InferenceService."""

from dom.backend.client import LlamaClient
from dom.backend.errors import (
    ApiError,
    BackendError,
    ChatStreamError,
    ModelLoadError,
    ModelNotFoundError,
    ServerCrashedError,
    ServerStartError,
    ServerUnavailableError,
)
from dom.backend.process import LlamaServerProcess, ServerConfig, ServerEndpoint, build_command
from dom.backend.protocols import InferenceService
from dom.backend.service import LlamaInferenceService

__all__ = [
    "ApiError",
    "BackendError",
    "ChatStreamError",
    "InferenceService",
    "LlamaClient",
    "LlamaInferenceService",
    "LlamaServerProcess",
    "ModelLoadError",
    "ModelNotFoundError",
    "ServerConfig",
    "ServerCrashedError",
    "ServerEndpoint",
    "ServerStartError",
    "ServerUnavailableError",
    "build_command",
]
