"""Сравнение модели главы 5 с базовыми априорными схемами.

Модуль реализует этап 9 программного контура главы 6. Он формирует
четыре набора прогнозов для одного и того же проверочного корпуса:

- среднее фактическое качество, рассчитанное вне текущего fold;
- оценку только по априорным компонентам признаков без LDA;
- оценку только по латентной компоненте ``q_latent``;
- неизменный интегральный прогноз ``q_pred`` главы 5.

Фактическое качество используется для обучения только mean baseline и
только внутри обучающей части каждого fold. Для остальных моделей цели
не участвуют в формировании прогнозов.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

from manual_coding_sim.validation.chapter6_config import Chapter6ValidationConfig


CLASS_LABELS: tuple[str, ...] = ("low", "medium", "high")
MODEL_ORDER: tuple[str, ...] = (
    "mean_baseline",
    "prior_only_baseline",
    "theta_only_baseline",
    "chapter5_model",
)
PRIOR_COMPONENT_COLUMNS: tuple[str, ...] = (
    "q_acc_feature_component",
    "q_time_feature_component",
    "q_effort_feature_component",
    "q_res_feature_component",
    "q_rep_feature_component",
    "q_fit_feature_component",
)
QUALITY_WEIGHT_KEYS: tuple[str, ...] = (
    "q_acc",
    "q_time",
    "q_effort",
    "q_res",
    "q_rep",
    "q_fit",
)
DEFAULT_QUALITY_WEIGHTS: dict[str, float] = {
    "q_acc": 0.1666666667,
    "q_time": 0.1666666667,
    "q_effort": 0.1666666667,
    "q_res": 0.1666666667,
    "q_rep": 0.1666666666,
    "q_fit": 0.1666666666,
}
DEFAULT_FOLD_COUNT = 5
UNIT_INTERVAL_TOLERANCE = 1e-12
WEIGHT_SUM_TOLERANCE = 1e-9


class BaselineComparisonError(ValueError):
    """Ошибка построения или сравнения baseline-моделей."""


@dataclass(frozen=True)
class BaselineComparisonResult:
    """Результат построения прогнозов и таблицы сравнения."""

    predictions: pd.DataFrame
    comparison: pd.DataFrame
    report: dict[str, Any]
    predictions_path: Path | None
    comparison_path: Path | None
    json_path: Path | None
    markdown_path: Path | None

    @property
    def passed(self) -> bool:
        """Вернуть итоговый статус этапа."""

        return bool(self.report["passed"])


class BaselineModelsValidator:
    """Построить baseline-прогнозы и сравнить их с моделью главы 5."""

    def __init__(
        self,
        config: Chapter6ValidationConfig,
        project_root: str | Path = ".",
        fold_count: int = DEFAULT_FOLD_COUNT,
    ) -> None:
        """Сохранить конфигурацию, корень проекта и число fold."""

        self.config = config
        self.project_root = Path(project_root)
        self.fold_count = int(fold_count)
        self.config.validate()
        if self.fold_count < 2:
            raise BaselineComparisonError(
                "Число fold для mean baseline должно быть не меньше двух."
            )

    def validate(
        self,
        dataset: pd.DataFrame | None = None,
    ) -> BaselineComparisonResult:
        """Построить baseline-прогнозы и рассчитать единый набор метрик."""

        source = dataset.copy() if dataset is not None else self._load_dataset()
        self._validate_source(source)

        keys = list(self._join_keys())
        numeric_columns = [
            "q_fact",
            "q_pred",
            "q_latent",
            *PRIOR_COMPONENT_COLUMNS,
        ]
        numeric = source[numeric_columns].apply(pd.to_numeric, errors="raise")
        q_fact = numeric["q_fact"].to_numpy(dtype=float)
        q_pred = numeric["q_pred"].to_numpy(dtype=float)
        q_latent = numeric["q_latent"].to_numpy(dtype=float)

        quality_weights, weight_source = self._load_quality_weights()
        weight_vector = np.asarray(
            [quality_weights[key] for key in QUALITY_WEIGHT_KEYS],
            dtype=float,
        )
        prior_matrix = numeric[list(PRIOR_COMPONENT_COLUMNS)].to_numpy(dtype=float)
        prior_prediction = prior_matrix @ weight_vector

        fold_ids, fold_training_means, fold_training_counts = (
            self._build_oof_mean_predictions(q_fact)
        )

        predictions = source[keys].copy()
        predictions["q_fact"] = q_fact
        predictions["q_fact_class"] = [self._quality_class(value) for value in q_fact]
        predictions["oof_fold"] = fold_ids
        predictions["mean_baseline_training_count"] = fold_training_counts

        model_values: dict[str, np.ndarray] = {
            "mean_baseline": fold_training_means,
            "prior_only_baseline": prior_prediction,
            "theta_only_baseline": q_latent,
            "chapter5_model": q_pred,
        }
        for model_name in MODEL_ORDER:
            values = model_values[model_name]
            self._validate_prediction_vector(model_name, values)
            predictions[model_name] = values
            predictions[f"{model_name}_class"] = [
                self._quality_class(value) for value in values
            ]
            predictions[f"{model_name}_error"] = values - q_fact
            predictions[f"{model_name}_absolute_error"] = np.abs(values - q_fact)

        comparison_rows = [
            self._calculate_model_metrics(
                model_name=model_name,
                predicted=model_values[model_name],
                factual=q_fact,
            )
            for model_name in MODEL_ORDER
        ]
        comparison = pd.DataFrame(comparison_rows)

        fold_summary = self._build_fold_summary(
            fold_ids=fold_ids,
            training_means=fold_training_means,
            training_counts=fold_training_counts,
        )
        finite_metrics = self._comparison_is_finite(comparison)
        unchanged_chapter5 = bool(np.array_equal(model_values["chapter5_model"], q_pred))
        passed = (
            len(predictions) == self._expected_row_count()
            and finite_metrics
            and unchanged_chapter5
            and int(predictions["mean_baseline_training_count"].min()) > 0
        )

        report = {
            "stage": 9,
            "report_type": "baseline_model_comparison",
            "passed": passed,
            "row_count": int(len(predictions)),
            "expected_row_count": self._expected_row_count(),
            "model_order": list(MODEL_ORDER),
            "quality_weights": quality_weights,
            "quality_weights_source": weight_source,
            "mean_baseline": {
                "strategy": "deterministic_shuffled_out_of_fold_mean",
                "requested_fold_count": self.fold_count,
                "actual_fold_count": len(fold_summary),
                "random_seed": self._random_seed(),
                "folds": fold_summary,
            },
            "methodology": {
                "mean_baseline": (
                    "Для каждой строки используется среднее q_fact только по "
                    "обучающим fold; цели текущего fold исключены."
                ),
                "prior_only_baseline": (
                    "Шесть q_*_feature_component агрегируются фиксированными "
                    "весами качества главы 5 без использования q_latent."
                ),
                "theta_only_baseline": (
                    "Используется неизменная латентная компонента q_latent."
                ),
                "chapter5_model": (
                    "Используется неизменный интегральный прогноз q_pred главы 5."
                ),
            },
            "leakage_checks": {
                "mean_baseline_is_out_of_fold": True,
                "test_fold_targets_excluded_from_training_mean": True,
                "prior_only_uses_q_fact": False,
                "theta_only_uses_q_fact": False,
                "chapter5_prediction_unchanged": unchanged_chapter5,
            },
            "metrics": comparison_rows,
            "best_models": self._find_best_models(comparison),
            "chapter5_differences": self._chapter5_differences(comparison),
            "interpretation": {
                "mae_rmse": (
                    "Меньшие значения характеризуют более точное воспроизведение "
                    "абсолютной шкалы качества."
                ),
                "rank_metrics": (
                    "Большие Spearman и Kendall характеризуют лучшее сохранение "
                    "порядка сценариев по качеству."
                ),
                "classification": (
                    "Accuracy следует интерпретировать совместно с Balanced "
                    "Accuracy и Macro F1 из-за дисбаланса классов."
                ),
            },
        }

        return BaselineComparisonResult(
            predictions=predictions,
            comparison=comparison,
            report=report,
            predictions_path=None,
            comparison_path=None,
            json_path=None,
            markdown_path=None,
        )

    def validate_and_save(
        self,
        dataset: pd.DataFrame | None = None,
    ) -> BaselineComparisonResult:
        """Выполнить сравнение и сохранить CSV-, JSON- и Markdown-артефакты."""

        result = self.validate(dataset=dataset)
        predictions_path = self._resolve_output_path(
            ("baseline_predictions_path", "baseline_predictions"),
            "reports/chapter6/baseline_predictions.csv",
        )
        comparison_path = self._resolve_output_path(
            ("baseline_comparison_path", "baseline_comparison"),
            "reports/chapter6/baseline_comparison.csv",
        )
        json_path = self._resolve_output_path(
            ("baseline_comparison_report_json_path", "baseline_report_json"),
            "reports/chapter6/baseline_comparison_report.json",
        )
        markdown_path = self._resolve_output_path(
            ("baseline_comparison_report_md_path", "baseline_report_md"),
            "reports/chapter6/baseline_comparison_report.md",
        )

        for path in (
            predictions_path,
            comparison_path,
            json_path,
            markdown_path,
        ):
            path.parent.mkdir(parents=True, exist_ok=True)

        result.predictions.to_csv(predictions_path, index=False, encoding="utf-8")
        result.comparison.to_csv(comparison_path, index=False, encoding="utf-8")
        json_path.write_text(
            json.dumps(result.report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        markdown_path.write_text(
            self._build_markdown_report(result.report),
            encoding="utf-8",
        )

        return BaselineComparisonResult(
            predictions=result.predictions,
            comparison=result.comparison,
            report=result.report,
            predictions_path=predictions_path,
            comparison_path=comparison_path,
            json_path=json_path,
            markdown_path=markdown_path,
        )

    def _load_dataset(self) -> pd.DataFrame:
        """Загрузить проверочный датасет, сформированный на этапе 3."""

        path = self._resolve_output_path(
            ("validation_dataset_path", "validation_dataset"),
            "reports/chapter6/validation_dataset.csv",
        )
        if not path.exists():
            raise FileNotFoundError(
                "Проверочный датасет этапа 3 не найден: "
                f"{path}. Сначала выполните --build-validation-dataset."
            )
        try:
            return pd.read_csv(path)
        except (OSError, pd.errors.ParserError) as error:
            raise BaselineComparisonError(
                f"Не удалось прочитать проверочный датасет: {path}"
            ) from error

    def _validate_source(self, dataset: pd.DataFrame) -> None:
        """Проверить структуру, ключи и числовые значения датасета."""

        required = [
            *self._join_keys(),
            "q_fact",
            "q_pred",
            "q_latent",
            *PRIOR_COMPONENT_COLUMNS,
        ]
        missing = [column for column in required if column not in dataset.columns]
        if missing:
            raise BaselineComparisonError(
                "В validation_dataset.csv отсутствуют обязательные колонки: "
                + ", ".join(missing)
            )

        if len(dataset) != self._expected_row_count():
            raise BaselineComparisonError(
                "Некорректное число строк проверочного датасета: "
                f"ожидалось {self._expected_row_count()}, получено {len(dataset)}."
            )
        if len(dataset) < 3:
            raise BaselineComparisonError(
                "Для out-of-fold baseline требуется не менее трех сценариев."
            )

        keys = list(self._join_keys())
        if dataset[keys].isna().any().any():
            raise BaselineComparisonError(
                "Составной ключ содержит пропущенные значения."
            )
        if dataset.duplicated(subset=keys).any():
            raise BaselineComparisonError(
                "Составной ключ scenario_id, protocol_id не является уникальным."
            )

        numeric_columns = [
            "q_fact",
            "q_pred",
            "q_latent",
            *PRIOR_COMPONENT_COLUMNS,
        ]
        try:
            numeric = dataset[numeric_columns].apply(pd.to_numeric, errors="raise")
        except (TypeError, ValueError) as error:
            raise BaselineComparisonError(
                "Baseline-входы должны содержать только числовые значения."
            ) from error

        values = numeric.to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise BaselineComparisonError(
                "Baseline-входы содержат NaN, inf или -inf."
            )
        if (
            (values < -UNIT_INTERVAL_TOLERANCE)
            | (values > 1.0 + UNIT_INTERVAL_TOLERANCE)
        ).any():
            raise BaselineComparisonError(
                "Показатели качества и компоненты должны находиться в [0; 1]."
            )
        if numeric["q_fact"].nunique(dropna=False) < 2:
            raise BaselineComparisonError(
                "q_fact должен содержать не менее двух различных значений."
            )

    def _load_quality_weights(self) -> tuple[dict[str, float], str]:
        """Загрузить веса качества главы 5 или использовать фиксированный набор."""

        path = self.project_root / "configs/chapter5_quality_weights.yaml"
        if not path.exists():
            return dict(DEFAULT_QUALITY_WEIGHTS), "default_stage5_weights"

        try:
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as error:
            raise BaselineComparisonError(
                f"Не удалось прочитать веса качества главы 5: {path}"
            ) from error
        if not isinstance(payload, Mapping):
            raise BaselineComparisonError(
                "Файл весов главы 5 должен содержать YAML-словарь."
            )
        raw_weights = payload.get("quality_weights")
        if not isinstance(raw_weights, Mapping):
            raise BaselineComparisonError(
                "В файле отсутствует словарь quality_weights."
            )

        missing = [key for key in QUALITY_WEIGHT_KEYS if key not in raw_weights]
        extra = sorted(set(raw_weights) - set(QUALITY_WEIGHT_KEYS))
        if missing or extra:
            details: list[str] = []
            if missing:
                details.append("отсутствуют: " + ", ".join(missing))
            if extra:
                details.append("лишние: " + ", ".join(extra))
            raise BaselineComparisonError(
                "Некорректный состав quality_weights (" + "; ".join(details) + ")."
            )

        try:
            weights = {key: float(raw_weights[key]) for key in QUALITY_WEIGHT_KEYS}
        except (TypeError, ValueError) as error:
            raise BaselineComparisonError(
                "Все quality_weights должны быть числовыми."
            ) from error
        if not all(math.isfinite(value) and value >= 0.0 for value in weights.values()):
            raise BaselineComparisonError(
                "Веса качества должны быть конечными и неотрицательными."
            )
        if not math.isclose(
            sum(weights.values()),
            1.0,
            rel_tol=0.0,
            abs_tol=WEIGHT_SUM_TOLERANCE,
        ):
            raise BaselineComparisonError(
                "Сумма quality_weights должна быть равна единице."
            )
        return weights, str(path.relative_to(self.project_root))

    def _build_oof_mean_predictions(
        self,
        factual: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Построить детерминированные out-of-fold прогнозы средним."""

        row_count = len(factual)
        actual_fold_count = min(self.fold_count, row_count)
        random_generator = np.random.default_rng(self._random_seed())
        shuffled_indices = random_generator.permutation(row_count)
        fold_ids = np.empty(row_count, dtype=int)
        for fold_id, test_indices in enumerate(
            np.array_split(shuffled_indices, actual_fold_count)
        ):
            fold_ids[test_indices] = fold_id

        predictions = np.empty(row_count, dtype=float)
        training_counts = np.empty(row_count, dtype=int)
        for fold_id in range(actual_fold_count):
            test_mask = fold_ids == fold_id
            train_mask = ~test_mask
            if not test_mask.any() or not train_mask.any():
                raise BaselineComparisonError(
                    "Некорректное разбиение fold для mean baseline."
                )
            training_mean = float(np.mean(factual[train_mask]))
            predictions[test_mask] = training_mean
            training_counts[test_mask] = int(train_mask.sum())
        return fold_ids, predictions, training_counts

    def _calculate_model_metrics(
        self,
        model_name: str,
        predicted: np.ndarray,
        factual: np.ndarray,
    ) -> dict[str, Any]:
        """Рассчитать регрессионные, ранговые и классовые метрики модели."""

        errors = predicted - factual
        absolute_errors = np.abs(errors)
        factual_classes = np.asarray(
            [self._quality_class(value) for value in factual],
            dtype=object,
        )
        predicted_classes = np.asarray(
            [self._quality_class(value) for value in predicted],
            dtype=object,
        )
        classification = self._classification_metrics(
            factual_classes=factual_classes,
            predicted_classes=predicted_classes,
        )
        return {
            "model": model_name,
            "description": self._model_description(model_name),
            "mae": float(np.mean(absolute_errors)),
            "rmse": float(math.sqrt(float(np.mean(np.square(errors))))),
            "bias": float(np.mean(errors)),
            "median_absolute_error": float(np.median(absolute_errors)),
            "max_absolute_error": float(np.max(absolute_errors)),
            "pearson": self._pearson(predicted, factual),
            "spearman": self._spearman(predicted, factual),
            "kendall": self._kendall_tau_b(predicted, factual),
            "r2": self._r2_score(predicted, factual),
            "accuracy": classification["accuracy"],
            "balanced_accuracy": classification["balanced_accuracy"],
            "macro_f1": classification["macro_f1"],
            "weighted_f1": classification["weighted_f1"],
            "prediction_mean": float(np.mean(predicted)),
            "prediction_std": float(np.std(predicted, ddof=1)),
            "predicted_low_count": int((predicted_classes == "low").sum()),
            "predicted_medium_count": int((predicted_classes == "medium").sum()),
            "predicted_high_count": int((predicted_classes == "high").sum()),
        }

    def _classification_metrics(
        self,
        factual_classes: np.ndarray,
        predicted_classes: np.ndarray,
    ) -> dict[str, float]:
        """Рассчитать общие классовые метрики по трем фиксированным классам."""

        total = float(len(factual_classes))
        accuracy = float(np.mean(factual_classes == predicted_classes))
        recalls: list[float] = []
        f1_values: list[float] = []
        weighted_f1_numerator = 0.0
        for label in CLASS_LABELS:
            factual_mask = factual_classes == label
            predicted_mask = predicted_classes == label
            true_positive = float(np.sum(factual_mask & predicted_mask))
            support = float(np.sum(factual_mask))
            predicted_count = float(np.sum(predicted_mask))
            precision = self._safe_divide(true_positive, predicted_count)
            recall = self._safe_divide(true_positive, support)
            f1 = self._safe_divide(2.0 * precision * recall, precision + recall)
            recalls.append(recall)
            f1_values.append(f1)
            weighted_f1_numerator += f1 * support
        return {
            "accuracy": accuracy,
            "balanced_accuracy": float(np.mean(recalls)),
            "macro_f1": float(np.mean(f1_values)),
            "weighted_f1": self._safe_divide(weighted_f1_numerator, total),
        }

    def _build_fold_summary(
        self,
        fold_ids: np.ndarray,
        training_means: np.ndarray,
        training_counts: np.ndarray,
    ) -> list[dict[str, Any]]:
        """Сформировать сведения о каждом fold mean baseline."""

        summary: list[dict[str, Any]] = []
        for fold_id in sorted(np.unique(fold_ids).tolist()):
            mask = fold_ids == fold_id
            summary.append(
                {
                    "fold_id": int(fold_id),
                    "test_count": int(mask.sum()),
                    "training_count": int(training_counts[mask][0]),
                    "training_mean": float(training_means[mask][0]),
                }
            )
        return summary

    def _find_best_models(self, comparison: pd.DataFrame) -> dict[str, dict[str, Any]]:
        """Определить лучшие модели по основным метрикам без статистических выводов."""

        directions = {
            "mae": "min",
            "rmse": "min",
            "absolute_bias": "min",
            "spearman": "max",
            "kendall": "max",
            "accuracy": "max",
            "balanced_accuracy": "max",
            "macro_f1": "max",
        }
        working = comparison.copy()
        working["absolute_bias"] = working["bias"].abs()
        result: dict[str, dict[str, Any]] = {}
        for metric, direction in directions.items():
            index = (
                working[metric].idxmin()
                if direction == "min"
                else working[metric].idxmax()
            )
            result[metric] = {
                "model": str(working.loc[index, "model"]),
                "value": float(working.loc[index, metric]),
                "direction": direction,
            }
        return result

    def _chapter5_differences(
        self,
        comparison: pd.DataFrame,
    ) -> list[dict[str, Any]]:
        """Рассчитать точечные разности метрик главы 5 и baseline."""

        indexed = comparison.set_index("model")
        chapter5 = indexed.loc["chapter5_model"]
        rows: list[dict[str, Any]] = []
        for baseline in MODEL_ORDER[:-1]:
            baseline_row = indexed.loc[baseline]
            rows.append(
                {
                    "baseline": baseline,
                    "delta_mae_chapter5_minus_baseline": float(
                        chapter5["mae"] - baseline_row["mae"]
                    ),
                    "delta_rmse_chapter5_minus_baseline": float(
                        chapter5["rmse"] - baseline_row["rmse"]
                    ),
                    "delta_spearman_chapter5_minus_baseline": float(
                        chapter5["spearman"] - baseline_row["spearman"]
                    ),
                    "delta_kendall_chapter5_minus_baseline": float(
                        chapter5["kendall"] - baseline_row["kendall"]
                    ),
                    "delta_accuracy_chapter5_minus_baseline": float(
                        chapter5["accuracy"] - baseline_row["accuracy"]
                    ),
                    "delta_macro_f1_chapter5_minus_baseline": float(
                        chapter5["macro_f1"] - baseline_row["macro_f1"]
                    ),
                }
            )
        return rows

    def _comparison_is_finite(self, comparison: pd.DataFrame) -> bool:
        """Проверить конечность всех числовых метрик сравнения."""

        numeric = comparison.select_dtypes(include=[np.number])
        return bool(np.isfinite(numeric.to_numpy(dtype=float)).all())

    def _validate_prediction_vector(
        self,
        model_name: str,
        values: np.ndarray,
    ) -> None:
        """Проверить размер, конечность и диапазон прогнозов модели."""

        if len(values) != self._expected_row_count():
            raise BaselineComparisonError(
                f"Модель {model_name} сформировала некорректное число прогнозов."
            )
        if not np.isfinite(values).all():
            raise BaselineComparisonError(
                f"Прогнозы модели {model_name} содержат NaN, inf или -inf."
            )
        if (
            (values < -UNIT_INTERVAL_TOLERANCE)
            | (values > 1.0 + UNIT_INTERVAL_TOLERANCE)
        ).any():
            raise BaselineComparisonError(
                f"Прогнозы модели {model_name} выходят за диапазон [0; 1]."
            )

    def _quality_class(self, value: float) -> str:
        """Преобразовать непрерывную оценку в класс low, medium или high."""

        numeric = float(value)
        if numeric < self.config.decision_thresholds.low_max:
            return "low"
        if numeric < self.config.decision_thresholds.high_min:
            return "medium"
        return "high"

    def _pearson(self, left: np.ndarray, right: np.ndarray) -> float:
        """Рассчитать линейную корреляцию Пирсона."""

        left_centered = left - float(np.mean(left))
        right_centered = right - float(np.mean(right))
        denominator = math.sqrt(
            float(np.sum(np.square(left_centered)))
            * float(np.sum(np.square(right_centered)))
        )
        if denominator == 0.0:
            return 0.0
        return float(np.sum(left_centered * right_centered) / denominator)

    def _spearman(self, left: np.ndarray, right: np.ndarray) -> float:
        """Рассчитать корреляцию Спирмена по средним рангам."""

        left_ranks = pd.Series(left).rank(method="average").to_numpy(dtype=float)
        right_ranks = pd.Series(right).rank(method="average").to_numpy(dtype=float)
        return self._pearson(left_ranks, right_ranks)

    def _kendall_tau_b(self, left: np.ndarray, right: np.ndarray) -> float:
        """Рассчитать коэффициент Kendall tau-b с учетом совпадающих рангов."""

        concordant = 0
        discordant = 0
        ties_left = 0
        ties_right = 0
        for first in range(len(left) - 1):
            left_diff = left[first + 1 :] - left[first]
            right_diff = right[first + 1 :] - right[first]
            left_sign = np.sign(left_diff)
            right_sign = np.sign(right_diff)
            products = left_sign * right_sign
            concordant += int(np.sum(products > 0))
            discordant += int(np.sum(products < 0))
            ties_left += int(np.sum((left_sign == 0) & (right_sign != 0)))
            ties_right += int(np.sum((right_sign == 0) & (left_sign != 0)))

        denominator = math.sqrt(
            float(concordant + discordant + ties_left)
            * float(concordant + discordant + ties_right)
        )
        if denominator == 0.0:
            return 0.0
        return float((concordant - discordant) / denominator)

    def _r2_score(self, predicted: np.ndarray, factual: np.ndarray) -> float:
        """Рассчитать R² относительно среднего фактического качества."""

        residual_sum = float(np.sum(np.square(predicted - factual)))
        total_sum = float(np.sum(np.square(factual - float(np.mean(factual)))))
        if total_sum == 0.0:
            raise BaselineComparisonError(
                "R² не определен при нулевой дисперсии q_fact."
            )
        return float(1.0 - residual_sum / total_sum)

    def _safe_divide(self, numerator: float, denominator: float) -> float:
        """Выполнить деление с нулевым результатом при нулевом знаменателе."""

        if denominator == 0.0:
            return 0.0
        return float(numerator / denominator)

    def _model_description(self, model_name: str) -> str:
        """Вернуть русскоязычное описание baseline-модели."""

        descriptions = {
            "mean_baseline": "Out-of-fold среднее фактического качества",
            "prior_only_baseline": "Только априорные feature-компоненты без LDA",
            "theta_only_baseline": "Только латентная компонента q_latent",
            "chapter5_model": "Неизменный интегральный прогноз главы 5",
        }
        return descriptions[model_name]

    def _build_markdown_report(self, report: Mapping[str, Any]) -> str:
        """Сформировать человекочитаемый отчет этапа 9."""

        status = "выполнен" if report["passed"] else "не выполнен"
        metric_rows = report["metrics"]
        lines = [
            "# Сравнение с базовыми моделями без LDA",
            "",
            "## Итоговый статус",
            "",
            f"Расчетный этап **{status}**.",
            "",
            f"- этап: {report['stage']};",
            f"- сценариев: {report['row_count']};",
            (
                "- mean baseline: "
                f"{report['mean_baseline']['actual_fold_count']}-fold OOF, "
                f"random_seed={report['mean_baseline']['random_seed']};"
            ),
            "- прогноз главы 5 не изменяется и не перекалибровывается.",
            "",
            "## Сравнение моделей",
            "",
            (
                "| Модель | MAE | RMSE | Bias | Spearman | Kendall | "
                "Accuracy | Balanced Accuracy | Macro F1 |"
            ),
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for row in metric_rows:
            lines.append(
                "| {model} | {mae:.10f} | {rmse:.10f} | {bias:.10f} | "
                "{spearman:.10f} | {kendall:.10f} | {accuracy:.10f} | "
                "{balanced_accuracy:.10f} | {macro_f1:.10f} |".format(**row)
            )

        best = report["best_models"]
        lines.extend(
            [
                "",
                "## Лучшие точечные результаты",
                "",
                (
                    f"- минимальный MAE: `{best['mae']['model']}` = "
                    f"{best['mae']['value']:.10f};"
                ),
                (
                    f"- минимальный RMSE: `{best['rmse']['model']}` = "
                    f"{best['rmse']['value']:.10f};"
                ),
                (
                    f"- максимальный Spearman: `{best['spearman']['model']}` = "
                    f"{best['spearman']['value']:.10f};"
                ),
                (
                    f"- максимальный Kendall: `{best['kendall']['model']}` = "
                    f"{best['kendall']['value']:.10f};"
                ),
                (
                    f"- максимальный Macro F1: `{best['macro_f1']['model']}` = "
                    f"{best['macro_f1']['value']:.10f}."
                ),
                "",
                "## Методическая защита от утечки",
                "",
                "- mean baseline строится out-of-fold;",
                "- цели проверочного fold не участвуют в обучающем среднем;",
                "- prior-only baseline не использует `q_fact`;",
                "- theta-only baseline не использует `q_fact`;",
                "- модель главы 5 использует исходный `q_pred` без изменений.",
                "",
                "## Ограничение интерпретации",
                "",
                (
                    "Точечные различия между моделями не являются статистическим "
                    "доказательством превосходства. Доверительные интервалы для "
                    "разностей метрик должны быть рассчитаны на этапе 10."
                ),
                "",
            ]
        )
        return "\n".join(lines)

    def _resolve_output_path(
        self,
        candidate_names: Sequence[str],
        default_relative_path: str,
    ) -> Path:
        """Разрешить путь, сохраняя совместимость с конфигурацией этапа 1."""

        outputs = self.config.outputs
        for name in candidate_names:
            value = getattr(outputs, name, None)
            if value is not None:
                path = Path(value)
                return path if path.is_absolute() else self.project_root / path
        return self.project_root / default_relative_path

    def _join_keys(self) -> tuple[str, ...]:
        """Вернуть ключи объединения через совместимый API."""

        keys = getattr(self.config.merge, "key_columns", None)
        if keys is None:
            keys = getattr(self.config.merge, "keys", None)
        if keys is None:
            raise BaselineComparisonError(
                "Конфигурация не содержит ключи объединения."
            )
        return tuple(str(value) for value in keys)

    def _expected_row_count(self) -> int:
        """Вернуть ожидаемое число строк корпуса."""

        return int(self.config.merge.expected_row_count)

    def _random_seed(self) -> int:
        """Вернуть random_seed из зафиксированной bootstrap-конфигурации."""

        return int(self.config.bootstrap.random_seed)
