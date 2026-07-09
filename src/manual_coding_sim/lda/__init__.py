"""Модуль главы 4 для LDA-анализа латентных факторов качества.

Пакет содержит программные компоненты, которые используют табличные
артефакты главы 3 как входные данные и формируют воспроизводимые артефакты
для главы 4: корпус токенов, LDA-модель, латентные профили и отчеты.

На первом этапе в пакете фиксируются только конфигурации, пути и защита от
утечки фактических признаков в априорную модель ``LDA_prior``.
"""

from manual_coding_sim.lda.config import (
    Chapter4LdaConfig,
    LdaInputPaths,
    LdaModelConfig,
    LdaOutputPaths,
    LdaTokenizationConfig,
)
from manual_coding_sim.lda.leakage_guard import (
    LeakageCheckResult,
    LeakageGuard,
    LdaLeakageError,
)
from manual_coding_sim.lda.paths import (
    DEFAULT_DIAGNOSTIC_FEATURES_PATH,
    DEFAULT_FACT_FEATURES_PATH,
    DEFAULT_LDA_DATA_DIR,
    DEFAULT_LDA_MODELS_DIR,
    DEFAULT_LDA_REPORTS_DIR,
    DEFAULT_PRIOR_FEATURES_PATH,
    DEFAULT_PROTOCOLS_PATH,
    DEFAULT_QUALITY_TARGETS_PATH,
)

__all__ = [
    "Chapter4LdaConfig",
    "DEFAULT_DIAGNOSTIC_FEATURES_PATH",
    "DEFAULT_FACT_FEATURES_PATH",
    "DEFAULT_LDA_DATA_DIR",
    "DEFAULT_LDA_MODELS_DIR",
    "DEFAULT_LDA_REPORTS_DIR",
    "DEFAULT_PRIOR_FEATURES_PATH",
    "DEFAULT_PROTOCOLS_PATH",
    "DEFAULT_QUALITY_TARGETS_PATH",
    "LdaInputPaths",
    "LdaLeakageError",
    "LdaModelConfig",
    "LdaOutputPaths",
    "LdaTokenizationConfig",
    "LeakageCheckResult",
    "LeakageGuard",
]
