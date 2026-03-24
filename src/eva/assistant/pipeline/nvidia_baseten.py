import riva.client
from pipecat.frames.frames import CancelFrame, EndFrame
from pipecat.services.nvidia.stt import NvidiaSTTService
from pipecat.services.nvidia.tts import NvidiaTTSService

from eva.utils.logging import get_logger

logger = get_logger(__name__)


def _close_grpc_channel(auth, service_name: str):
    """Explicitly close a Riva gRPC channel to release resources immediately.

    gRPC channels don't close automatically when dereferenced - they persist
    until garbage collected, consuming connection slots on Baseten.
    """
    if auth is not None and hasattr(auth, "channel") and auth.channel is not None:
        try:
            auth.channel.close()
            logger.debug(f"Closed Baseten {service_name} gRPC channel")
        except Exception as e:
            logger.warning(f"Error closing Baseten {service_name} gRPC channel: {e}")


class BasetenSTTService(NvidiaSTTService):
    """NvidiaSTTService that authenticates against a Baseten-hosted Riva deployment."""

    def __init__(self, *, api_key: str, base_url: str, **kwargs):
        # Extract "model-{id}" from "model-{id}.grpc.api.baseten.co:443"
        model_id_header = base_url.split(".")[0]
        super().__init__(
            api_key=api_key,
            server=base_url,
            model_function_map={"function_id": model_id_header, "model_name": model_id_header},
            use_ssl=True,
            **kwargs,
        )
        self._auth = None

    def _initialize_client(self):
        metadata = [
            ("baseten-authorization", f"Api-Key {self._api_key}"),
            ("baseten-model-id", self._function_id),
        ]
        self._auth = riva.client.Auth(None, self._use_ssl, self._server, metadata)
        self._asr_service = riva.client.ASRService(self._auth)

    def _cleanup(self):
        _close_grpc_channel(self._auth, "STT")
        self._auth = None
        self._asr_service = None

    async def stop(self, frame: EndFrame):
        """Stop the STT service and close the gRPC channel."""
        await super().stop(frame)
        self._cleanup()

    async def cancel(self, frame: CancelFrame):
        """Cancel the STT service and close the gRPC channel."""
        await super().cancel(frame)
        self._cleanup()


class BasetenTTSService(NvidiaTTSService):
    """NvidiaTTSService that authenticates against a Baseten-hosted Riva deployment."""

    def __init__(self, *, api_key: str, base_url: str, **kwargs):
        # Extract "model-{id}" from "model-{id}.grpc.api.baseten.co:443"
        model_id_header = base_url.split(".")[0]
        super().__init__(
            api_key=api_key,
            server=base_url,
            model_function_map={"function_id": model_id_header, "model_name": model_id_header},
            use_ssl=True,
            **kwargs,
        )
        self._auth = None

    def _initialize_client(self):
        if self._service is not None:
            return
        metadata = [
            ("baseten-authorization", f"Api-Key {self._api_key}"),
            ("baseten-model-id", self._function_id),
        ]
        self._auth = riva.client.Auth(None, self._use_ssl, self._server, metadata)
        self._service = riva.client.SpeechSynthesisService(self._auth)

    def _cleanup(self):
        _close_grpc_channel(self._auth, "TTS")
        self._auth = None
        self._service = None

    async def stop(self, frame: EndFrame):
        """Stop the TTS service and close the gRPC channel."""
        await super().stop(frame)
        self._cleanup()

    async def cancel(self, frame: CancelFrame):
        """Cancel the TTS service and close the gRPC channel."""
        await super().cancel(frame)
        self._cleanup()
