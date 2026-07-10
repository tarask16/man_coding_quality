"""Расчет латентной компоненты качества главы 5.

Модуль преобразует латентный профиль ``theta_prior(A)`` из главы 4 в
нормированную компоненту качества ``Q_lat``. Компоненты theta интерпретируются
через заранее заданные направления влияния факторов: отрицательные факторы
снижают прогноз качества, положительные — повышают.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd

THETA_COLUMNS = ("theta_0", "theta_1", "theta_2")
SERVICE_COLUMNS = ("scenario_id", "protocol_id", "run_id", "alternative_id")


@dataclass(frozen=True)
class LatentQualityComponentReport:
    """Сводный отчет расчета латентной компоненты качества."""

    row_count: int
    theta_columns: tuple[str, ...]
    factor_directions: dict[str, float]
    q_latent_min: float
    q_latent_max: float
    q_latent_mean: float
    dominant_topic_counts: dict[str, int]
    output_columns: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        """Преобразовать отчет в JSON-совместимый словарь."""

        return asdict(self)


@dataclass(frozen=True)
class LatentQualityComponentResult:
    """Результат расчета латентной компоненты качества."""

    latent_quality: pd.DataFrame
    report: LatentQualityComponentReport


class LatentQualityComponentError(ValueError):
    """Ошибка расчета латентной компоненты качества главы 5."""


class LatentQualityComponentCalculator:
    """Рассчитывает ``Q_lat`` по латентному профилю ``theta_prior``."""

    def __init__(
        self,
        factor_directions: Mapping[str, float] | object,
        *,
        theta_columns: Sequence[str] = THETA_COLUMNS,
        service_columns: Sequence[str] = SERVICE_COLUMNS,
        tolerance: float = 1e-6,
    ) -> None:
        """Создать калькулятор латентной компоненты.

        ``factor_directions`` может быть обычным словарем или объектом
        конфигурации ``Chapter5FactorDirections`` с полем ``directions``.
        """

        raw_directions = getattr(factor_directions, "directions", factor_directions)
        self.factor_directions = {
            str(name): float(value) for name, value in dict(raw_directions).items()
        }
        self.theta_columns = tuple(theta_columns)
        self.service_columns = tuple(service_columns)
        self.tolerance = float(tolerance)
        self._validate_directions()

    def calculate(self, theta_prior: pd.DataFrame) -> LatentQualityComponentResult:
        """Рассчитать таблицу ``latent_quality_component.csv``."""

        if theta_prior.empty:
            msg = "Таблица theta_prior пуста, расчет латентной компоненты невозможен."
            raise LatentQualityComponentError(msg)
        self._require_columns(theta_prior)
        self._validate_theta_values(theta_prior)

        result = theta_prior[
            [column for column in self.service_columns if column in theta_prior.columns]
        ].copy()
        for column in self.theta_columns:
            result[column] = pd.to_numeric(theta_prior[column], errors="raise")

        direction_score = sum(
            result[column] * self.factor_directions[column] for column in self.theta_columns
        )
        result["latent_direction_score"] = direction_score
        result["q_latent"] = ((direction_score + 1.0) / 2.0).clip(0.0, 1.0)
        result["theta_dominant_topic"] = result[list(self.theta_columns)].idxmax(axis=1)
        result["theta_dominant_value"] = result[list(self.theta_columns)].max(axis=1)

        self._require_result_range(result)
        report = self._build_report(result)
        return LatentQualityComponentResult(latent_quality=result, report=report)

    def save_outputs(
        self,
        result: LatentQualityComponentResult,
        *,
        latent_component_path: Path,
        report_path: Path | None = None,
    ) -> None:
        """Сохранить таблицу латентной компоненты и необязательный JSON-отчет."""

        latent_component_path.parent.mkdir(parents=True, exist_ok=True)
        result.latent_quality.to_csv(latent_component_path, index=False)
        if report_path is not None:
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(
                json.dumps(result.report.to_dict(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    def _validate_directions(self) -> None:
        """Проверить направления факторов."""

        missing = [column for column in self.theta_columns if column not in self.factor_directions]
        if missing:
            joined = ", ".join(missing)
            msg = f"Не заданы направления латентных факторов: {joined}."
            raise LatentQualityComponentError(msg)
        for column in self.theta_columns:
            value = self.factor_directions[column]
            if not -1.0 <= value <= 1.0 or value == 0.0:
                msg = (
                    "Направление латентного фактора должно быть ненулевым "
                    f"числом в диапазоне [-1, 1]: {column}={value}."
                )
                raise LatentQualityComponentError(msg)

    def _require_columns(self, theta_prior: pd.DataFrame) -> None:
        """Проверить наличие обязательных колонок theta."""

        missing = [column for column in self.theta_columns if column not in theta_prior.columns]
        if missing:
            joined = ", ".join(missing)
            msg = f"В theta_prior отсутствуют обязательные колонки: {joined}."
            raise LatentQualityComponentError(msg)
        if "scenario_id" not in theta_prior.columns:
            msg = "В theta_prior отсутствует обязательная колонка scenario_id."
            raise LatentQualityComponentError(msg)

    def _validate_theta_values(self, theta_prior: pd.DataFrame) -> None:
        """Проверить неотрицательность и нормировку theta-профиля."""

        theta_values = theta_prior[list(self.theta_columns)].apply(pd.to_numeric, errors="coerce")
        if theta_values.isna().any().any():
            msg = "В theta_prior найдены нечисловые или пропущенные theta-компоненты."
            raise LatentQualityComponentError(msg)
        if (theta_values < 0.0).any().any():
            msg = "В theta_prior найдены отрицательные theta-компоненты."
            raise LatentQualityComponentError(msg)
        max_sum_error = float((theta_values.sum(axis=1) - 1.0).abs().max())
        if max_sum_error > self.tolerance:
            msg = (
                "Сумма theta_0 + theta_1 + theta_2 должна быть равна 1. "
                f"Максимальное отклонение: {max_sum_error:.12f}."
            )
            raise LatentQualityComponentError(msg)

    @staticmethod
    def _require_result_range(result: pd.DataFrame) -> None:
        """Проверить, что ``q_latent`` находится в диапазоне [0, 1]."""

        if ((result["q_latent"] < 0.0) | (result["q_latent"] > 1.0)).any():
            msg = "Латентная компонента q_latent вышла за диапазон [0, 1]."
            raise LatentQualityComponentError(msg)

    def _build_report(self, result: pd.DataFrame) -> LatentQualityComponentReport:
        """Сформировать сводный отчет расчета латентной компоненты."""

        counts = {
            str(topic): int(count)
            for topic, count in result["theta_dominant_topic"].value_counts().sort_index().items()
        }
        return LatentQualityComponentReport(
            row_count=int(result.shape[0]),
            theta_columns=self.theta_columns,
            factor_directions=dict(self.factor_directions),
            q_latent_min=float(result["q_latent"].min()),
            q_latent_max=float(result["q_latent"].max()),
            q_latent_mean=float(result["q_latent"].mean()),
            dominant_topic_counts=counts,
            output_columns=tuple(str(column) for column in result.columns),
        )
