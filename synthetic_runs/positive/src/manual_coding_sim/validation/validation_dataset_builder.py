"""Формирование единого проверочного датасета главы 6.

Модуль реализует этап 3 программного контура главы 6. Он объединяет
проверенные артефакты глав 3–5 по составному ключу ``scenario_id`` и
``protocol_id`` в режиме ``one_to_one``. Значения прогноза главы 5 не
пересчитываются и не корректируются по фактическим данным.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from manual_coding_sim.validation.chapter6_config import Chapter6ValidationConfig
from manual_coding_sim.validation.chapter6_data_loader import (
    FACTUAL_CRITERIA,
    PREDICTED_CRITERIA,
    THETA_COLUMNS,
    Chapter6DataLoader,
    Chapter6LoadedInputs,
)


class ValidationDatasetBuildError(ValueError):
    """Ошибка формирования проверочного датасета главы 6."""


@dataclass(frozen=True)
class MergeStepResult:
    """Результат одного проверяемого объединения таблиц."""

    source_name: str
    added_columns: tuple[str, ...]
    row_count_before: int
    row_count_after: int


@dataclass(frozen=True)
class ValidationDatasetBuildResult:
    """Результат построения и сохранения проверочного датасета."""

    dataset: pd.DataFrame
    output_path: Path | None
    merge_steps: tuple[MergeStepResult, ...]
    predicted_class_counts: dict[str, int]
    factual_class_counts: dict[str, int]


class ValidationDatasetBuilder:
    """Построитель единого проверочного датасета этапа 3."""

    def __init__(
        self,
        config: Chapter6ValidationConfig,
        project_root: str | Path = ".",
    ) -> None:
        """Сохранить конфигурацию и корень проекта."""

        self.config = config
        self.project_root = Path(project_root)
        self.config.validate()

    def build(
        self,
        loaded_inputs: Chapter6LoadedInputs | None = None,
    ) -> ValidationDatasetBuildResult:
        """Построить датасет из ранее проверенных входных таблиц."""

        loaded = loaded_inputs or Chapter6DataLoader(
            config=self.config,
            project_root=self.project_root,
        ).load()

        keys = list(self._join_keys())
        base_columns = keys + self._available_columns(
            loaded.q_pred,
            (
                "run_id",
                "alternative_id",
                "q_latent",
                *PREDICTED_CRITERIA,
                "q_pred",
            ),
        )
        self._require_columns(
            loaded.q_pred,
            keys + list(PREDICTED_CRITERIA) + ["q_pred"],
            "q_pred",
        )
        dataset = loaded.q_pred[base_columns].copy()
        self._validate_intermediate_dataset(dataset, "q_pred")

        steps: list[MergeStepResult] = []

        component_columns = self._component_columns(loaded.q_pred_components)
        dataset = self._merge_new_columns(
            dataset,
            loaded.q_pred_components,
            "q_pred_components",
            component_columns,
            steps,
        )

        uncertainty_columns = self._available_columns(
            loaded.prediction_uncertainty,
            (
                "theta_entropy",
                "lda_instability",
                "input_missing_share",
                "uncertainty_score",
                "interval_radius",
                "q_pred_lower",
                "q_pred_upper",
            ),
        )
        self._require_columns(
            loaded.prediction_uncertainty,
            ["uncertainty_score", "q_pred_lower", "q_pred_upper"],
            "prediction_uncertainty",
        )
        dataset = self._merge_new_columns(
            dataset,
            loaded.prediction_uncertainty,
            "prediction_uncertainty",
            uncertainty_columns,
            steps,
        )

        normalized_columns = sorted(
            column
            for column in loaded.normalized_prior_features.columns
            if column.endswith("_norm")
        )
        if not normalized_columns:
            raise ValidationDatasetBuildError(
                "В normalized_prior_features отсутствуют колонки *_norm."
            )
        dataset = self._merge_new_columns(
            dataset,
            loaded.normalized_prior_features,
            "normalized_prior_features",
            normalized_columns,
            steps,
        )

        theta_columns = list(THETA_COLUMNS) + self._available_columns(
            loaded.theta_prior,
            (
                "document_index",
                "selected_k",
                "random_state",
            ),
        )
        self._require_columns(
            loaded.theta_prior,
            list(THETA_COLUMNS),
            "theta_prior",
        )
        dataset = self._merge_new_columns(
            dataset,
            loaded.theta_prior,
            "theta_prior",
            theta_columns,
            steps,
        )

        latent_columns = self._available_columns(
            loaded.latent_quality_component,
            (
                "q_latent",
                "latent_direction_score",
                "theta_dominant_topic",
                "theta_dominant_value",
            ),
        )
        dataset = self._merge_new_columns(
            dataset,
            loaded.latent_quality_component,
            "latent_quality_component",
            latent_columns,
            steps,
        )

        factual_columns = list(FACTUAL_CRITERIA) + ["integral_quality"]
        self._require_columns(
            loaded.quality_targets,
            factual_columns,
            "quality_targets",
        )
        dataset = self._merge_new_columns(
            dataset,
            loaded.quality_targets,
            "quality_targets",
            factual_columns,
            steps,
        )

        fact_feature_columns = sorted(
            column
            for column in loaded.fact_features.columns
            if column.startswith("fact_")
        )
        if not fact_feature_columns:
            raise ValidationDatasetBuildError(
                "В fact_features отсутствуют диагностические колонки fact_*."
            )
        dataset = self._merge_new_columns(
            dataset,
            loaded.fact_features,
            "fact_features",
            fact_feature_columns,
            steps,
        )

        dataset["q_fact"] = pd.to_numeric(
            dataset["integral_quality"],
            errors="raise",
        )
        dataset["q_pred_class"] = self._assign_quality_class(dataset["q_pred"])
        dataset["q_fact_class"] = self._assign_quality_class(dataset["q_fact"])

        dataset = self._order_columns(dataset)
        self._validate_final_dataset(dataset)

        return ValidationDatasetBuildResult(
            dataset=dataset,
            output_path=None,
            merge_steps=tuple(steps),
            predicted_class_counts=self._class_counts(dataset["q_pred_class"]),
            factual_class_counts=self._class_counts(dataset["q_fact_class"]),
        )

    def build_and_save(
        self,
        loaded_inputs: Chapter6LoadedInputs | None = None,
    ) -> ValidationDatasetBuildResult:
        """Построить датасет и сохранить его в CSV."""

        result = self.build(loaded_inputs=loaded_inputs)
        output_path = self._resolve_output_path()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.dataset.to_csv(output_path, index=False, encoding="utf-8")
        return ValidationDatasetBuildResult(
            dataset=result.dataset,
            output_path=output_path,
            merge_steps=result.merge_steps,
            predicted_class_counts=result.predicted_class_counts,
            factual_class_counts=result.factual_class_counts,
        )

    def _merge_new_columns(
        self,
        left: pd.DataFrame,
        right: pd.DataFrame,
        source_name: str,
        requested_columns: Sequence[str],
        steps: list[MergeStepResult],
    ) -> pd.DataFrame:
        """Присоединить только новые колонки и проверить число строк."""

        keys = list(self._join_keys())
        new_columns = tuple(
            column
            for column in requested_columns
            if column in right.columns and column not in left.columns
        )
        before = len(left)
        if new_columns:
            right_part = right[keys + list(new_columns)].copy()
            merged = left.merge(
                right_part,
                on=keys,
                how="left",
                validate=self._merge_validation(),
                sort=False,
            )
        else:
            merged = left.copy()

        after = len(merged)
        expected = self._expected_row_count()
        if before != expected or after != expected:
            raise ValidationDatasetBuildError(
                "После объединения с таблицей "
                f"{source_name} число строк изменилось: "
                f"до {before}, после {after}, ожидалось {expected}."
            )
        if merged[keys].isna().any().any():
            raise ValidationDatasetBuildError(
                f"После объединения с {source_name} появились пропуски в ключах."
            )
        if new_columns and merged[list(new_columns)].isna().any().any():
            missing = [
                column
                for column in new_columns
                if merged[column].isna().any()
            ]
            raise ValidationDatasetBuildError(
                "После объединения с таблицей "
                f"{source_name} появились пропуски в колонках: "
                + ", ".join(missing)
            )

        steps.append(
            MergeStepResult(
                source_name=source_name,
                added_columns=new_columns,
                row_count_before=before,
                row_count_after=after,
            )
        )
        return merged

    def _validate_intermediate_dataset(
        self,
        dataset: pd.DataFrame,
        source_name: str,
    ) -> None:
        """Проверить базовый или промежуточный набор строк."""

        keys = list(self._join_keys())
        expected = self._expected_row_count()
        if len(dataset) != expected:
            raise ValidationDatasetBuildError(
                f"Таблица {source_name} содержит {len(dataset)} строк, "
                f"ожидалось {expected}."
            )
        if dataset.duplicated(keys).any():
            raise ValidationDatasetBuildError(
                f"Таблица {source_name} содержит дубли составного ключа."
            )

    def _validate_final_dataset(self, dataset: pd.DataFrame) -> None:
        """Проверить обязательный контракт итогового CSV."""

        required = list(self._join_keys()) + [
            "q_pred",
            "q_fact",
            "integral_quality",
            *PREDICTED_CRITERIA,
            *FACTUAL_CRITERIA,
            *THETA_COLUMNS,
            "uncertainty_score",
            "q_pred_lower",
            "q_pred_upper",
            "q_pred_class",
            "q_fact_class",
        ]
        self._require_columns(dataset, required, "validation_dataset")
        self._validate_intermediate_dataset(dataset, "validation_dataset")

        if dataset.columns.duplicated().any():
            duplicates = dataset.columns[dataset.columns.duplicated()].tolist()
            raise ValidationDatasetBuildError(
                "В итоговом датасете обнаружены повторяющиеся колонки: "
                + ", ".join(duplicates)
            )
        if dataset[required].isna().any().any():
            missing = [
                column for column in required if dataset[column].isna().any()
            ]
            raise ValidationDatasetBuildError(
                "В обязательных колонках итогового датасета обнаружены пропуски: "
                + ", ".join(missing)
            )

        numeric_columns = [
            "q_pred",
            "q_fact",
            "integral_quality",
            *PREDICTED_CRITERIA,
            *FACTUAL_CRITERIA,
            *THETA_COLUMNS,
            "uncertainty_score",
            "q_pred_lower",
            "q_pred_upper",
        ]
        numeric = dataset[numeric_columns].apply(pd.to_numeric, errors="coerce")
        if numeric.isna().any().any() or not np.isfinite(
            numeric.to_numpy(dtype=float)
        ).all():
            raise ValidationDatasetBuildError(
                "Итоговый датасет содержит нечисловые или бесконечные значения."
            )

        if not np.allclose(
            dataset["q_fact"].to_numpy(dtype=float),
            dataset["integral_quality"].to_numpy(dtype=float),
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValidationDatasetBuildError(
                "Колонка q_fact должна точно соответствовать integral_quality."
            )

        allowed = {"low", "medium", "high"}
        for column in ("q_pred_class", "q_fact_class"):
            actual = set(dataset[column].astype(str).unique())
            if not actual.issubset(allowed):
                raise ValidationDatasetBuildError(
                    f"Колонка {column} содержит неизвестные классы: "
                    + ", ".join(sorted(actual - allowed))
                )

    def _assign_quality_class(self, values: pd.Series) -> pd.Series:
        """Преобразовать непрерывную оценку в класс low, medium или high."""

        numeric = pd.to_numeric(values, errors="raise")
        low_max = float(self.config.decision_thresholds.low_max)
        high_min = float(self.config.decision_thresholds.high_min)
        classes = np.select(
            [numeric < low_max, numeric < high_min],
            ["low", "medium"],
            default="high",
        )
        return pd.Series(classes, index=values.index, dtype="string")

    def _order_columns(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """Установить стабильный порядок ключевых и аналитических колонок."""

        preferred = [
            *self._join_keys(),
            "run_id",
            "alternative_id",
            "q_pred",
            "q_fact",
            "integral_quality",
            "q_pred_class",
            "q_fact_class",
            *PREDICTED_CRITERIA,
            *FACTUAL_CRITERIA,
            "q_latent",
            *THETA_COLUMNS,
            "theta_dominant_topic",
            "theta_dominant_value",
            "theta_entropy",
            "lda_instability",
            "input_missing_share",
            "uncertainty_score",
            "interval_radius",
            "q_pred_lower",
            "q_pred_upper",
        ]
        first = [column for column in preferred if column in dataset.columns]
        remaining = [column for column in dataset.columns if column not in first]
        return dataset[first + remaining]

    def _resolve_output_path(self) -> Path:
        """Разрешить путь validation_dataset.csv относительно корня проекта."""

        candidates = (
            "validation_dataset_path",
            "validation_dataset",
        )
        configured: Any | None = None
        for name in candidates:
            if hasattr(self.config.outputs, name):
                configured = getattr(self.config.outputs, name)
                break
        path = Path(configured) if configured is not None else Path(
            "reports/chapter6/validation_dataset.csv"
        )
        return path if path.is_absolute() else self.project_root / path

    def _join_keys(self) -> tuple[str, ...]:
        """Вернуть составной ключ объединения."""

        keys = getattr(self.config.merge, "key_columns", None)
        if keys is None:
            keys = getattr(self.config.merge, "keys", None)
        if keys is None:
            raise ValidationDatasetBuildError(
                "В конфигурации отсутствуют ключи объединения."
            )
        return tuple(keys)

    def _merge_validation(self) -> str:
        """Вернуть режим проверки pandas.merge."""

        value = getattr(self.config.merge, "validation", None)
        if value is None:
            value = getattr(self.config.merge, "validate_mode", None)
        if value is None:
            raise ValidationDatasetBuildError(
                "В конфигурации отсутствует режим проверки объединения."
            )
        return str(value)

    def _expected_row_count(self) -> int:
        """Вернуть ожидаемое число сценариев."""

        return int(self.config.merge.expected_row_count)

    @staticmethod
    def _available_columns(
        table: pd.DataFrame,
        columns: Iterable[str],
    ) -> list[str]:
        """Вернуть колонки из списка, реально присутствующие в таблице."""

        return [column for column in columns if column in table.columns]

    @staticmethod
    def _component_columns(table: pd.DataFrame) -> list[str]:
        """Выбрать разложение частных прогнозов для последующих baseline."""

        suffixes = (
            "_feature_component",
            "_latent_component",
            "_observed_weight",
            "_latent_weight",
        )
        columns = [
            column
            for column in table.columns
            if column.endswith(suffixes)
        ]
        if "q_latent" in table.columns:
            columns.insert(0, "q_latent")
        return columns

    @staticmethod
    def _require_columns(
        table: pd.DataFrame,
        required: Sequence[str],
        source_name: str,
    ) -> None:
        """Проверить наличие обязательных колонок."""

        missing = [column for column in required if column not in table.columns]
        if missing:
            raise ValidationDatasetBuildError(
                f"В таблице {source_name} отсутствуют обязательные колонки: "
                + ", ".join(missing)
            )

    @staticmethod
    def _class_counts(values: pd.Series) -> dict[str, int]:
        """Вернуть счетчики классов в фиксированном порядке."""

        counts = values.value_counts(dropna=False)
        return {
            label: int(counts.get(label, 0))
            for label in ("low", "medium", "high")
        }


__all__ = [
    "MergeStepResult",
    "ValidationDatasetBuildError",
    "ValidationDatasetBuilder",
    "ValidationDatasetBuildResult",
]
