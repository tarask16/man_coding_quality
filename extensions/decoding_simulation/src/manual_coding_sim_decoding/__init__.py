"""Изолированное расширение моделирования декодирования сообщений."""

from manual_coding_sim_decoding.base_adapter import (
    BaseContractCheckResult,
    ManualCodingSimAdapterContract,
)
from manual_coding_sim_decoding.config import (
    BaseContractConfig,
    DecodingExtensionConfig,
    ExtensionMetadata,
    ExtensionPathConfig,
    MaterialEncodingConfig,
    load_decoding_extension_config,
)
from manual_coding_sim_decoding.encoded_message import (
    EncodedElement,
    EncodedMessage,
    EncodedMessageBuilder,
    EncodingProtocol,
    EncodingTraceStep,
    MaterialEncodingResult,
    summarize_material_encoding,
)
from manual_coding_sim_decoding.paths import DecodingExtensionPaths

__all__ = [
    "BaseContractCheckResult",
    "BaseContractConfig",
    "DecodingExtensionConfig",
    "DecodingExtensionPaths",
    "EncodedElement",
    "EncodedMessage",
    "EncodedMessageBuilder",
    "EncodingProtocol",
    "EncodingTraceStep",
    "ExtensionMetadata",
    "ExtensionPathConfig",
    "ManualCodingSimAdapterContract",
    "MaterialEncodingConfig",
    "MaterialEncodingResult",
    "load_decoding_extension_config",
    "summarize_material_encoding",
]

__version__ = "0.1.0"
