"""Программный блок главы 5 для априорного прогнозирования качества.

Пакет предназначен для построения ``Q_pred(A)`` по априорным признакам,
латентному профилю ``theta_prior(A)`` и параметрам неопределенности. Этап 2
фиксирует конфигурационный слой: веса, направления факторов, пороги решений и
параметры неопределенности.
"""

from manual_coding_sim.prediction.chapter5_config import (
    Chapter5ConfigFilePaths,
    Chapter5ConfigLoader,
    Chapter5DecisionThresholds,
    Chapter5FactorDirections,
    Chapter5FeatureCriterionWeights,
    Chapter5FeatureWeights,
    Chapter5InputPaths,
    Chapter5OutputPaths,
    Chapter5PredictionConfig,
    Chapter5PriorFeatureDictionary,
    Chapter5QualityWeights,
    Chapter5UncertaintyConfig,
    load_chapter5_prediction_config,
)
from manual_coding_sim.prediction.chapter5_data_loader import (
    Chapter5DataLoadError,
    Chapter5DataLoader,
    Chapter5InputContract,
    Chapter5InputValidationReport,
    Chapter5LoadedInputs,
)
from manual_coding_sim.prediction.chapter5_leakage_guard import (
    Chapter5LeakageCheckResult,
    Chapter5LeakageError,
    Chapter5LeakageGuard,
)
from manual_coding_sim.prediction.chapter5_report_builder import Chapter5ReportBuilder
from manual_coding_sim.prediction.chapter5_pipeline import (
    Chapter5PipelineRunReport,
    Chapter5PipelineRunReporter,
)
from manual_coding_sim.prediction.integral_quality_predictor import (
    IntegralQualityCriterionReport,
    IntegralQualityPredictionError,
    IntegralQualityPredictionReport,
    IntegralQualityPredictionResult,
    IntegralQualityPredictor,
)
from manual_coding_sim.prediction.latent_quality_component import (
    LatentQualityComponentCalculator,
    LatentQualityComponentError,
    LatentQualityComponentReport,
    LatentQualityComponentResult,
)
from manual_coding_sim.prediction.partial_quality_predictor import (
    PartialQualityCriterionReport,
    PartialQualityPredictionError,
    PartialQualityPredictionReport,
    PartialQualityPredictionResult,
    PartialQualityPredictor,
)
from manual_coding_sim.prediction.prediction_uncertainty import (
    PredictionUncertaintyError,
    PredictionUncertaintyEstimator,
    PredictionUncertaintyReport,
    PredictionUncertaintyResult,
)
from manual_coding_sim.prediction.prior_feature_normalizer import (
    PriorFeatureNormalizationError,
    PriorFeatureNormalizationItem,
    PriorFeatureNormalizationReport,
    PriorFeatureNormalizationResult,
    PriorFeatureNormalizer,
)

__all__ = [
    "Chapter5ConfigFilePaths",
    "Chapter5ConfigLoader",
    "Chapter5DataLoadError",
    "Chapter5DataLoader",
    "Chapter5DecisionThresholds",
    "Chapter5FactorDirections",
    "Chapter5FeatureCriterionWeights",
    "Chapter5FeatureWeights",
    "Chapter5InputContract",
    "Chapter5InputValidationReport",
    "Chapter5LoadedInputs",
    "Chapter5InputPaths",
    "Chapter5LeakageCheckResult",
    "Chapter5LeakageError",
    "Chapter5LeakageGuard",
    "Chapter5OutputPaths",
    "Chapter5PredictionConfig",
    "Chapter5PipelineRunReport",
    "Chapter5PipelineRunReporter",
    "Chapter5PriorFeatureDictionary",
    "Chapter5QualityWeights",
    "Chapter5ReportBuilder",
    "Chapter5UncertaintyConfig",
    "IntegralQualityCriterionReport",
    "IntegralQualityPredictionError",
    "IntegralQualityPredictionReport",
    "IntegralQualityPredictionResult",
    "IntegralQualityPredictor",
    "LatentQualityComponentCalculator",
    "LatentQualityComponentError",
    "LatentQualityComponentReport",
    "LatentQualityComponentResult",
    "PartialQualityCriterionReport",
    "PartialQualityPredictionError",
    "PartialQualityPredictionReport",
    "PartialQualityPredictionResult",
    "PartialQualityPredictor",
    "PredictionUncertaintyError",
    "PredictionUncertaintyEstimator",
    "PredictionUncertaintyReport",
    "PredictionUncertaintyResult",
    "PriorFeatureNormalizationError",
    "PriorFeatureNormalizationItem",
    "PriorFeatureNormalizationReport",
    "PriorFeatureNormalizationResult",
    "PriorFeatureNormalizer",
    "load_chapter5_prediction_config",
]
