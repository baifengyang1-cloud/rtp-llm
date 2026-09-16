import logging

from rtp_llm.config.py_config_modules import (
    MM_TRANSPORT_MODE_AUTO,
    MM_TRANSPORT_MODE_GRPC,
    MM_TRANSPORT_MODE_RDMA,
    MM_TRANSPORT_MODES,
    MMTransportConfig,
)
from rtp_llm.multimodal.transport.base import MMOutputTransport
from rtp_llm.multimodal.transport.grpc.backend import GrpcInlineOutputBackend


def create_mm_output_transport(
    transport_config=None, local_device_id: int = 0
) -> MMOutputTransport:
    transport_config = (
        transport_config if transport_config is not None else MMTransportConfig()
    )
    fallback = None
    if transport_config.mode == MM_TRANSPORT_MODE_GRPC:
        backend = GrpcInlineOutputBackend()
    elif transport_config.mode in (MM_TRANSPORT_MODE_AUTO, MM_TRANSPORT_MODE_RDMA):
        from rtp_llm.multimodal.transport.rdma.backend import RdmaOutputBackend

        try:
            backend = RdmaOutputBackend.create(transport_config.rdma, local_device_id)
        except Exception:
            if transport_config.mode == MM_TRANSPORT_MODE_RDMA:
                raise
            logging.warning(
                "[VIT] RDMA initialization failed; using inline gRPC", exc_info=True
            )
            backend = GrpcInlineOutputBackend()
        else:
            if transport_config.mode == MM_TRANSPORT_MODE_AUTO:
                fallback = GrpcInlineOutputBackend()
    else:
        raise ValueError(
            f"invalid mm_transport_mode: {transport_config.mode!r}; "
            f"expected one of {MM_TRANSPORT_MODES}"
        )

    return MMOutputTransport(backend, fallback)
