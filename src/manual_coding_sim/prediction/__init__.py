"""Программный блок главы 5 для априорного прогнозирования качества.

Пакет предназначен для построения ``Q_pred(A)`` по априорным признакам,
латентному профилю ``theta_prior(A)`` и параметрам неопределенности. На этапе 1
зафиксирован только каркас пакета; расчетные компоненты будут закрываться
последовательно отдельными этапами.
"""

from manual_coding_sim.prediction.chapter5_config import (
    Chapter5InputPaths,
    Chapter5OutputPaths,
    Chapter5PredictionConfig,
)
from manual_coding_sim.prediction.chapter5_data_loader import (
    Chapter5DataLoader,
    Chapter5InputContract,
)
from manual_coding_sim.prediction.chapter5_leakage_guard import (
    Chapter5LeakageCheckResult,
    Chapter5LeakageError,
    Chapter5LeakageGuard,
)
from manual_coding_sim.prediction.chapter5_report_builder import Chapter5ReportBuilder
from manual_coding_sim.prediction.integral_quality_predictor import (
    IntegralQualityPredictor,
)
from manual_coding_sim.prediction.latent_quality_component import (
    LatentQualityComponentCalculator,
)
from manual_coding_sim.prediction.partial_quality_predictor import PartialQualityPredictor
from manual_coding_sim.prediction.prediction_uncertainty import (
    PredictionUncertaintyEstimator,
)
from manual_coding_sim.prediction.prior_feature_normalizer import PriorFeatureNormalizer

__all__ = [
    "Chapter5DataLoader",
    "Chapter5InputContract",
    "Chapter5InputPaths",
    "Chapter5LeakageCheckResult",
    "Chapter5LeakageError",
    "Chapter5LeakageGuard",
    "Chapter5OutputPaths",
    "Chapter5PredictionConfig",
    "Chapter5ReportBuilder",
    "IntegralQualityPredictor",
    "LatentQualityComponentCalculator",
    "PartialQualityPredictor",
    "PredictionUncertaintyEstimator",
    "PriorFeatureNormalizer",
]
