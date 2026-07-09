"""Модуль главы 4 для LDA-анализа латентных факторов качества.

Пакет содержит программные компоненты, которые используют табличные
артефакты главы 3 как входные данные и формируют воспроизводимые артефакты
для главы 4: корпус токенов, LDA-модель, латентные профили и отчеты.

Пакет постепенно расширяется: сначала фиксируются конфигурации, пути и
защита от утечки, затем добавляются токенизация и построение корпуса для
априорной модели ``LDA_prior``.
"""


from manual_coding_sim.lda.chapter4_report import (
    Chapter4LdaReportBuilder,
    Chapter4LdaReportConfig,
    Chapter4LdaReportResult,
)
from manual_coding_sim.lda.chapter4_runner import (
    Chapter4LdaRunner,
    Chapter4RunResult,
    Chapter4RunnerConfig,
)

from manual_coding_sim.lda.config import (
    Chapter4LdaConfig,
    LdaInputPaths,
    LdaModelConfig,
    LdaOutputPaths,
    LdaTokenizationConfig,
)
from manual_coding_sim.lda.corpus_builder import (
    LdaCorpusBuilder,
    LdaCorpusBuilderConfig,
    LdaCorpusBuildResult,
)


from manual_coding_sim.lda.k_selection import (
    LdaKSelectionCandidate,
    LdaKSelectionConfig,
    LdaKSelectionResult,
    LdaKSelector,
)


from manual_coding_sim.lda.lda_diagnostic_model import (
    LdaDiagnosticModel,
    LdaDiagnosticModelConfig,
    LdaDiagnosticTrainingResult,
)

from manual_coding_sim.lda.lda_prior_model import (
    LdaPriorModel,
    LdaPriorModelConfig,
    LdaPriorTrainingResult,
)
from manual_coding_sim.lda.matrix_builder import (
    LdaDocumentMetadata,
    LdaMatrixBuilder,
    LdaMatrixBuildResult,
    LdaVocabularyItem,
)


from manual_coding_sim.lda.topic_stability import (
    LdaTopicStabilityAnalyzer,
    LdaTopicStabilityConfig,
    LdaTopicStabilityResult,
    LdaTopicStabilityRun,
)


from manual_coding_sim.lda.topic_interpreter import (
    LdaTopicInterpretation,
    LdaTopicInterpretationResult,
    LdaTopicInterpreter,
    LdaTopicInterpreterConfig,
)

from manual_coding_sim.lda.topic_metrics import (
    LdaTopicMetricsConfig,
    LdaTopicMetricsEvaluator,
    LdaTopicMetricsResult,
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
from manual_coding_sim.lda.tokenization import (
    FeatureTokenizationRule,
    FeatureTokenizer,
    TokenizedFeature,
)

__all__ = [
    "Chapter4LdaReportBuilder",
    "Chapter4LdaReportConfig",
    "Chapter4LdaReportResult",
    "Chapter4LdaRunner",
    "Chapter4RunResult",
    "Chapter4RunnerConfig",
    "Chapter4LdaConfig",
    "DEFAULT_DIAGNOSTIC_FEATURES_PATH",
    "DEFAULT_FACT_FEATURES_PATH",
    "DEFAULT_LDA_DATA_DIR",
    "DEFAULT_LDA_MODELS_DIR",
    "DEFAULT_LDA_REPORTS_DIR",
    "DEFAULT_PRIOR_FEATURES_PATH",
    "DEFAULT_PROTOCOLS_PATH",
    "DEFAULT_QUALITY_TARGETS_PATH",
    "FeatureTokenizationRule",
    "FeatureTokenizer",
    "LdaCorpusBuilder",
    "LdaCorpusBuilderConfig",
    "LdaCorpusBuildResult",
    "LdaDiagnosticModel",
    "LdaDiagnosticModelConfig",
    "LdaDiagnosticTrainingResult",
    "LdaDocumentMetadata",
    "LdaKSelectionCandidate",
    "LdaKSelectionConfig",
    "LdaKSelectionResult",
    "LdaKSelector",
    "LdaMatrixBuilder",
    "LdaMatrixBuildResult",
    "LdaPriorModel",
    "LdaPriorModelConfig",
    "LdaPriorTrainingResult",
    "LdaTopicStabilityAnalyzer",
    "LdaTopicStabilityConfig",
    "LdaTopicStabilityResult",
    "LdaTopicStabilityRun",
    "LdaTopicInterpretation",
    "LdaTopicInterpretationResult",
    "LdaTopicInterpreter",
    "LdaTopicInterpreterConfig",
    "LdaTopicMetricsConfig",
    "LdaTopicMetricsEvaluator",
    "LdaTopicMetricsResult",
    "LdaVocabularyItem",
    "TokenizedFeature",
    "LdaInputPaths",
    "LdaLeakageError",
    "LdaModelConfig",
    "LdaOutputPaths",
    "LdaTokenizationConfig",
    "LeakageCheckResult",
    "LeakageGuard",
]
