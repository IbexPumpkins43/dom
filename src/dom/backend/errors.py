"""Backend-layer exceptions. Every message is plain English, safe to show in the UI."""

from __future__ import annotations


class BackendError(Exception):
    """Base class for inference backend failures."""


class ServerStartError(BackendError):
    """llama-server could not be started or never became healthy."""


class ServerCrashedError(BackendError):
    """llama-server stopped unexpectedly."""


class ServerUnavailableError(BackendError):
    """llama-server could not be reached over HTTP."""


class ApiError(BackendError):
    """llama-server answered a request with an error."""

    def __init__(self, status_code: int, message: str, error_type: str | None = None) -> None:
        self.status_code = status_code
        self.server_message = message
        self.error_type = error_type
        super().__init__(f"The AI server reported an error ({status_code}): {message}")


class ModelNotFoundError(ApiError):
    """The requested model is not in the models folder."""


class ModelLoadError(BackendError):
    """A model failed to load."""


class ChatStreamError(BackendError):
    """A streamed reply ended with an error or could not be parsed."""


class ModelTooLargeError(BackendError):
    """The model does not fit in this computer's free memory."""
