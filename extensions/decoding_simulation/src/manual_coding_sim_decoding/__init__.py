"""Изолированное расширение моделирования декодирования сообщений."""

from manual_coding_sim_decoding.base_adapter import (
    BaseContractCheckResult,
    ManualCodingSimAdapterContract,
)
from manual_coding_sim_decoding.config import (
    BaseContractConfig,
    DecodingExtensionConfig,
    DecodingOperationRule,
    ExtensionMetadata,
    ExtensionPathConfig,
    FormalDecodingConfig,
    MaterialEncodingConfig,
    load_decoding_extension_config,
)
from manual_coding_sim_decoding.decoding_procedure import (
    DecodingPlan,
    DecodingProcedureModel,
    DecodingStep,
    TokenParseResult,
    summarize_decoding_plan,
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
    "DecodingOperationRule",
    "DecodingPlan",
    "DecodingProcedureModel",
    "DecodingStep",
    "EncodedElement",
    "EncodedMessage",
    "EncodedMessageBuilder",
    "EncodingProtocol",
    "EncodingTraceStep",
    "ExtensionMetadata",
    "ExtensionPathConfig",
    "FormalDecodingConfig",
    "ManualCodingSimAdapterContract",
    "MaterialEncodingConfig",
    "MaterialEncodingResult",
    "TokenParseResult",
    "load_decoding_extension_config",
    "summarize_decoding_plan",
    "summarize_material_encoding",
]

__version__ = "0.1.0"
