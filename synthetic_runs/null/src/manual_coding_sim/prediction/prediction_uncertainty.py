"""Расчет неопределенности интегрального прогноза главы 5.

Модуль формирует интервальную оценку для ``Q_pred`` без обращения к
фактическим значениям качества. Источниками неопределенности считаются:

* энтропия априорного латентного профиля ``theta_prior``;
* устойчивость LDA-модели, зафиксированная в конфигурации главы 5;
* техническое качество входных нормированных признаков, выраженное долей
  пропусков в колонках ``*_norm``.

Итоговый безразмерный показатель ``uncertainty_score`` ограничивается
диапазоном ``[0, 1]`` и используется для построения интервала
``[q_pred_lower, q_pred_upper]``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd

from manual_coding_sim.prediction.chapter5_config import Chapter5UncertaintyConfig

MERGE_COLUMNS = ("scenario_id", "protocol_id")
SERVICE_COLUMNS = ("scenario_id", "protocol_id", "run_id", "alternative_id")
THETA_COLUMNS = ("theta_0", "theta_1", "theta_2")
Q_PRED_COLUMN = "q_pred"


@dataclass(frozen=True)
class PredictionUncertaintyReport:
    """Сводный отчет расчета неопределенности прогноза."""

    row_count: int
    merge_columns: tuple[str, ...]
    theta_columns: tuple[str, ...]
    weights: dict[str, float]
    weight_sum: float
    delta: float
    mean_stability: float
    min_stability: float
    lda_instability: float
    input_missing_share: float
    theta_entropy_min: float
    theta_entropy_max: float
    theta_entropy_mean: float
    uncertainty_score_min: float
    uncertainty_score_max: float
    uncertainty_score_mean: float
    interval_radius_min: float
    interval_radius_max: float
    interval_radius_mean: float
    q_pred_lower_min: float
    q_pred_upper_max: float
    output_columns: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        """Преобразовать отчет в JSON-совместимый словарь."""

        return asdict(self)


@dataclass(frozen=True)
class PredictionUncertaintyResult:
    """Результат интервального оценивания ``Q_pred``."""

    uncertainty: pd.DataFrame
    report: PredictionUncertaintyReport


class PredictionUncertaintyError(ValueError):
    """Ошибка расчета неопределенности прогноза главы 5."""


class PredictionUncertaintyEstimator:
    """Строит показатель неопределенности и интервал для ``Q_pred``."""

    def __init__(
        self,
        uncertainty_config: Chapter5UncertaintyConfig | Mapping[str, object],
        *,
        merge_columns: Sequence[str] = MERGE_COLUMNS,
        service_columns: Sequence[str] = SERVICE_COLUMNS,
        theta_columns: Sequence[str] = THETA_COLUMNS,
        q_pred_column: str = Q_PRED_COLUMN,
    ) -> None:
        """Создать оцениватель неопределенности.

        ``uncertainty_config`` может быть объектом ``Chapter5UncertaintyConfig``
        или словарем совместимой структуры. Сумма весов источников
        неопределенности должна быть равна единице.
        """

        self.merge_columns = tuple(merge_columns)
        self.service_columns = tuple(service_columns)
        self.theta_columns = tuple(theta_columns)
        self.q_pred_column = q_pred_column
        self.delta, self.mean_stability, self.min_stability, self.weights = (
            self._coerce_config(uncertainty_config)
        )
        self._validate_config()

    def estimate(
        self,
        q_pred: pd.DataFrame,
        theta_prior: pd.DataFrame,
        normalized_prior_features: pd.DataFrame | None = None,
    ) -> PredictionUncertaintyResult:
        """Рассчитать неопределенность и интервалы для ``Q_pred``."""

        if q_pred.empty:
            msg = "Таблица интегрального прогноза q_pred пуста."
            raise PredictionUncertaintyError(msg)
        if theta_prior.empty:
            msg = "Таблица theta_prior пуста."
            raise PredictionUncertaintyError(msg)

        self._require_columns(q_pred, (*self.merge_columns, self.q_pred_column), "таблице q_pred")
        self._require_columns(theta_prior, (*self.merge_columns, *self.theta_columns), "theta_prior")
        self._reject_duplicate_keys(q_pred, "таблице q_pred")
        self._reject_duplicate_keys(theta_prior, "theta_prior")

        result = self._merge_q_pred_and_theta(q_pred, theta_prior)
        result[self.q_pred_column] = pd.to_numeric(result[self.q_pred_column], errors="coerce")
        if result[self.q_pred_column].isna().any():
            msg = "В колонке q_pred найдены пропуски."
            raise PredictionUncertaintyError(msg)
        if not result[self.q_pred_column].between(0.0, 1.0).all():
            msg = "Значения q_pred должны находиться в диапазоне [0, 1]."
            raise PredictionUncertaintyError(msg)

        theta_entropy = self._calculate_theta_entropy(result)
        input_missing_share = self._calculate_input_missing_share(normalized_prior_features)
        lda_instability = 1.0 - self.mean_stability
        uncertainty_score = (
            theta_entropy * self.weights["theta_entropy"]
            + lda_instability * self.weights["lda_stability"]
            + input_missing_share * self.weights["input_quality"]
        ).clip(0.0, 1.0)
        interval_radius = (self.delta * uncertainty_score).clip(0.0, 1.0)

        result["theta_entropy"] = theta_entropy
        result["lda_instability"] = lda_instability
        result["input_missing_share"] = input_missing_share
        result["uncertainty_score"] = uncertainty_score
        result["interval_radius"] = interval_radius
        result["q_pred_lower"] = (result[self.q_pred_column] - interval_radius).clip(0.0, 1.0)
        result["q_pred_upper"] = (result[self.q_pred_column] + interval_radius).clip(0.0, 1.0)

        self._require_output_ranges(result)
        report = self._build_report(result, input_missing_share, lda_instability)
        return PredictionUncertaintyResult(uncertainty=result, report=report)

    def save_outputs(
        self,
        result: PredictionUncertaintyResult,
        *,
        uncertainty_path: Path,
        report_path: Path | None = None,
    ) -> None:
        """Сохранить таблицу неопределенности и необязательный JSON-отчет."""

        uncertainty_path.parent.mkdir(parents=True, exist_ok=True)
        result.uncertainty.to_csv(uncertainty_path, index=False)
        if report_path is not None:
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(
                json.dumps(result.report.to_dict(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    def _merge_q_pred_and_theta(self, q_pred: pd.DataFrame, theta_prior: pd.DataFrame) -> pd.DataFrame:
        """Объединить ``q_pred`` и ``theta_prior`` по ключам главы 5."""

        base_columns = [column for column in self.service_columns if column in q_pred.columns]
        q_pred_columns = [*base_columns, self.q_pred_column]
        theta_columns = [*self.merge_columns, *self.theta_columns]
        merged = q_pred[q_pred_columns].merge(
            theta_prior[theta_columns],
            on=list(self.merge_columns),
            how="inner",
            validate="one_to_one",
        )
        if merged.shape[0] != q_pred.shape[0]:
            msg = (
                "Не все строки q_pred удалось сопоставить с theta_prior по ключам "
                f"{self.merge_columns}."
            )
            raise PredictionUncertaintyError(msg)
        return merged

    def _calculate_theta_entropy(self, data_frame: pd.DataFrame) -> pd.Series:
        """Рассчитать нормированную энтропию латентного профиля."""

        theta = data_frame[list(self.theta_columns)].apply(pd.to_numeric, errors="coerce")
        if theta.isna().any().any():
            msg = "В theta_prior найдены нечисловые или пропущенные значения."
            raise PredictionUncertaintyError(msg)
        if not theta.ge(0.0).all().all():
            msg = "Компоненты theta_prior не могут быть отрицательными."
            raise PredictionUncertaintyError(msg)
        row_sums = theta.sum(axis=1)
        if not row_sums.between(1.0 - 1e-6, 1.0 + 1e-6).all():
            msg = "Компоненты theta_prior должны суммироваться в единицу."
            raise PredictionUncertaintyError(msg)

        safe_theta = theta.where(theta > 0.0, 1.0)
        entropy = -(theta * safe_theta.map(math.log)).sum(axis=1)
        return (entropy / math.log(len(self.theta_columns))).clip(0.0, 1.0)

    def _calculate_input_missing_share(
        self,
        normalized_prior_features: pd.DataFrame | None,
    ) -> float:
        """Оценить долю пропусков в нормированных априорных признаках."""

        if normalized_prior_features is None or normalized_prior_features.empty:
            return 0.0
        normalized_columns = [
            column for column in normalized_prior_features.columns if str(column).endswith("_norm")
        ]
        if not normalized_columns:
            return 0.0
        missing_count = int(normalized_prior_features[normalized_columns].isna().sum().sum())
        total_count = int(len(normalized_columns) * normalized_prior_features.shape[0])
        if total_count == 0:
            return 0.0
        return float(missing_count / total_count)

    def _build_report(
        self,
        result: pd.DataFrame,
        input_missing_share: float,
        lda_instability: float,
    ) -> PredictionUncertaintyReport:
        """Собрать JSON-совместимый отчет неопределенности."""

        return PredictionUncertaintyReport(
            row_count=int(result.shape[0]),
            merge_columns=self.merge_columns,
            theta_columns=self.theta_columns,
            weights=dict(self.weights),
            weight_sum=float(sum(self.weights.values())),
            delta=float(self.delta),
            mean_stability=float(self.mean_stability),
            min_stability=float(self.min_stability),
            lda_instability=float(lda_instability),
            input_missing_share=float(input_missing_share),
            theta_entropy_min=float(result["theta_entropy"].min()),
            theta_entropy_max=float(result["theta_entropy"].max()),
            theta_entropy_mean=float(result["theta_entropy"].mean()),
            uncertainty_score_min=float(result["uncertainty_score"].min()),
            uncertainty_score_max=float(result["uncertainty_score"].max()),
            uncertainty_score_mean=float(result["uncertainty_score"].mean()),
            interval_radius_min=float(result["interval_radius"].min()),
            interval_radius_max=float(result["interval_radius"].max()),
            interval_radius_mean=float(result["interval_radius"].mean()),
            q_pred_lower_min=float(result["q_pred_lower"].min()),
            q_pred_upper_max=float(result["q_pred_upper"].max()),
            output_columns=tuple(str(column) for column in result.columns),
        )

    def _validate_config(self) -> None:
        """Проверить параметры расчета неопределенности."""

        if self.delta < 0.0:
            msg = "Коэффициент delta не может быть отрицательным."
            raise PredictionUncertaintyError(msg)
        if not 0.0 <= self.mean_stability <= 1.0:
            msg = "mean_stability должен находиться в диапазоне [0, 1]."
            raise PredictionUncertaintyError(msg)
        if not 0.0 <= self.min_stability <= 1.0:
            msg = "min_stability должен находиться в диапазоне [0, 1]."
            raise PredictionUncertaintyError(msg)
        required_weights = ("theta_entropy", "lda_stability", "input_quality")
        missing = [name for name in required_weights if name not in self.weights]
        if missing:
            msg = f"В весах неопределенности отсутствуют ключи: {missing}."
            raise PredictionUncertaintyError(msg)
        unknown = [name for name in self.weights if name not in required_weights]
        if unknown:
            msg = f"В весах неопределенности найдены лишние ключи: {unknown}."
            raise PredictionUncertaintyError(msg)
        negative = [name for name, value in self.weights.items() if value < 0.0]
        if negative:
            msg = f"Веса неопределенности не могут быть отрицательными: {negative}."
            raise PredictionUncertaintyError(msg)
        weight_sum = float(sum(self.weights.values()))
        if abs(weight_sum - 1.0) > 1e-6:
            msg = f"Сумма весов неопределенности должна быть равна 1: {weight_sum}."
            raise PredictionUncertaintyError(msg)

    def _require_output_ranges(self, result: pd.DataFrame) -> None:
        """Проверить диапазоны выходных колонок неопределенности."""

        columns = (
            "theta_entropy",
            "lda_instability",
            "input_missing_share",
            "uncertainty_score",
            "interval_radius",
            "q_pred_lower",
            "q_pred_upper",
        )
        for column in columns:
            if result[column].isna().any():
                msg = f"В колонке {column} найдены пропуски."
                raise PredictionUncertaintyError(msg)
            if not result[column].between(0.0, 1.0).all():
                msg = f"Значения {column} должны находиться в диапазоне [0, 1]."
                raise PredictionUncertaintyError(msg)
        if not (result["q_pred_lower"] <= result[self.q_pred_column]).all():
            msg = "Нижняя граница интервала не может быть выше q_pred."
            raise PredictionUncertaintyError(msg)
        if not (result[self.q_pred_column] <= result["q_pred_upper"]).all():
            msg = "Верхняя граница интервала не может быть ниже q_pred."
            raise PredictionUncertaintyError(msg)

    def _require_columns(
        self,
        data_frame: pd.DataFrame,
        columns: Sequence[str],
        source_name: str,
    ) -> None:
        """Проверить наличие обязательных колонок."""

        missing = [column for column in columns if column not in data_frame.columns]
        if missing:
            msg = f"В {source_name} отсутствуют обязательные колонки: {missing}."
            raise PredictionUncertaintyError(msg)

    def _reject_duplicate_keys(self, data_frame: pd.DataFrame, source_name: str) -> None:
        """Запретить дубли ключей объединения."""

        duplicated = data_frame.duplicated(list(self.merge_columns))
        if duplicated.any():
            examples = data_frame.loc[duplicated, list(self.merge_columns)].head(5)
            msg = (
                f"В {source_name} найдены дубли ключей главы 5: "
                f"{examples.to_dict(orient='records')}."
            )
            raise PredictionUncertaintyError(msg)

    def _coerce_config(
        self,
        config: Chapter5UncertaintyConfig | Mapping[str, object],
    ) -> tuple[float, float, float, dict[str, float]]:
        """Преобразовать конфигурацию неопределенности к простым типам."""

        if isinstance(config, Chapter5UncertaintyConfig):
            return (
                float(config.delta),
                float(config.mean_stability),
                float(config.min_stability),
                {str(key): float(value) for key, value in config.weights.items()},
            )
        delta = float(config.get("delta", 0.15))
        mean_stability = float(config.get("mean_stability", 0.84885))
        min_stability = float(config.get("min_stability", 0.808418))
        raw_weights = config.get("weights", config.get("uncertainty_weights", {}))
        if not isinstance(raw_weights, Mapping):
            msg = "Веса неопределенности должны быть словарем."
            raise PredictionUncertaintyError(msg)
        weights = {str(key): float(value) for key, value in raw_weights.items()}
        return delta, mean_stability, min_stability, weights
