from dataclasses import dataclass
from enum import IntEnum
from typing import NoReturn, Tuple, Type

from rtp_llm.config.exceptions import ExceptionType, FtRuntimeException


def _error_scalar(value) -> bool:
    # Do not retain request objects, tensors or exceptions through custom
    # attributes/args. Runtime error-code enums are immutable singletons.
    if value is None or type(value) in (str, int, float, bool):
        return True
    if isinstance(value, IntEnum):
        return True
    return type(value) is tuple and all(_error_scalar(item) for item in value)


@dataclass(frozen=True)
class MMErrorSnapshot:
    """Failure metadata with no producer traceback or request references.

    Every consumer gets a new exception: clearing a shared exception's stack
    once is insufficient because each subsequent raise attaches new frames.
    """

    error_type: Type[Exception]
    args: tuple
    attributes: Tuple[Tuple[str, object], ...]

    @classmethod
    def capture(cls, error: Exception) -> "MMErrorSnapshot":
        args = error.args if _error_scalar(error.args) else (str(error),)
        attributes = {
            key: value for key, value in vars(error).items() if _error_scalar(value)
        }
        # OSError keeps these fields in native slots, not __dict__.
        if isinstance(error, OSError):
            for key in ("errno", "strerror", "filename", "filename2"):
                value = getattr(error, key, None)
                if value is not None and _error_scalar(value):
                    attributes[key] = value
        return cls(type(error), args, tuple(attributes.items()))

    def exception(self) -> Exception:
        try:
            # Runtime/scheduler constructors need more arguments than .args.
            # Restore scalar state without invoking those constructors or
            # copying the original exception's arbitrary object graph.
            error = self.error_type.__new__(self.error_type, *self.args)
            Exception.__init__(error, *self.args)
            for key, value in self.attributes:
                setattr(error, key, value)
            return error
        except Exception:
            return RuntimeError(*self.args)


class MMErr:
    URL_INVALID = (
        "The provided URL does not appear to be valid. "
        "Ensure it is correctly formatted."
    )
    DL_TIMEOUT = "Download multimodal file timed out"
    DL_FAILED = "Failed to download multimodal content"
    FILE_TOO_LARGE = "Multimodal file size is too large"
    MISS_CONTENT_LEN = "Missing Content-Length of multimodal url"
    IMG_TOO_SMALL = (
        "Input image is too small after resizing based on max_pixels. "
        "Consider increasing max_pixels."
    )
    IMG_OPEN = "The image format is illegal and cannot be opened"
    IMG_HW = "The image length and width do not meet the model restrictions. [{}]"
    IMAGE_REQ = "The image modality input does not meet the requirements because: {}"
    VIDEO_INVALID = "Invalid video file."
    VIDEO_REQ = "The video modality input does not meet the requirements because: {}"


def format_mm_rpc_error(error: FtRuntimeException) -> str:
    """Encode an application error for the C++ multimodal RPC client."""
    return f"[{error.exception_type.name}] {error.message}"


def raise_mm(
    message: str,
    code: ExceptionType = ExceptionType.MM_WRONG_FORMAT_ERROR,
) -> NoReturn:
    raise FtRuntimeException(code, message)
