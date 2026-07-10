"""Извлечение априорных, фактических и диагностических признаков.

Модуль относится к главе 3 диссертации и преобразует результат
интегрального моделирования ручного кодирования в три непересекающиеся
группы признаков:

* X_prior — априорные признаки сценария A = {S, O, U, G, K};
* X_fact — фактические признаки, полученные после моделируемого
  возникновения ошибок и применения контрольных процедур;
* X_diag — диагностические признаки для проверки и анализа модели.

Такое разделение нужно для исключения утечки фактического результата
в процедуру априорной оценки качества. На данном этапе итоговый вектор
качества q(A) еще не рассчитывается.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from manual_coding_sim.protocol_simulator import SimulationResult
from manual_coding_sim.types import FeatureGroup


@dataclass(frozen=True)
class FeatureExtractorConfig:
    """Конфигурация извлечения признаков из протокола моделирования.

    Параметр round_digits задает число знаков после запятой для
    числовых признаков. Это повышает стабильность отчетных таблиц и
    облегчает сравнение повторных запусков вычислительного эксперимента.
    """

    round_digits: int = 6
    min_denominator: float = 1e-12

    def validate(self) -> None:
        """Проверяет корректность конфигурации извлечения признаков."""
        if self.round_digits < 0:
            raise ValueError("round_digits не должен быть отрицательным.")

        if self.min_denominator <= 0:
            raise ValueError("min_denominator должен быть положительным.")


class FeatureExtractor:
    """Извлекатель признаков X_prior, X_fact и X_diag.

    Извлекатель работает с уже полученным SimulationResult. Он не
    изменяет исходный протокол, не генерирует новые ошибки и не
    рассчитывает итоговое качество q(A). Его задача — подготовить
    признаки для следующих глав: LDA-модели, метода априорной оценки
    и вычислительного эксперимента.
    """

    def __init__(self, config: FeatureExtractorConfig | None = None) -> None:
        """Инициализирует извлекатель признаков."""
        self.config = config or FeatureExtractorConfig()
        self.config.validate()

    def extract(self, result: SimulationResult) -> FeatureGroup:
        """Извлекает группы признаков из одного результата моделирования."""
        self._validate_result(result)

        prior_features = self.extract_prior_features(result)
        fact_features = self.extract_fact_features(result)
        diagnostic_features = self.extract_diagnostic_features(result)

        feature_group = FeatureGroup(
            scenario_id=result.scenario.scenario_id,
            prior_features=prior_features,
            fact_features=fact_features,
            diagnostic_features=diagnostic_features,
        )
        validate_feature_group(feature_group)
        return feature_group

    def extract_batch(
        self,
        results: Iterable[SimulationResult],
    ) -> tuple[FeatureGroup, ...]:
        """Извлекает группы признаков для серии протоколов."""
        return tuple(self.extract(result) for result in results)

    def extract_prior_features(
        self,
        result: SimulationResult,
    ) -> dict[str, float]:
        """Извлекает априорные признаки X_prior.

        Эти признаки известны до фактического выполнения процедуры.
        Они включают характеристики сообщения M, нормативного плана S,
        профиля оператора O и условий применения U. В эту группу не
        включаются числа фактически возникших, обнаруженных или
        остаточных ошибок.
        """
        message_elements = result.message.elements
        procedure_metadata = result.procedure_plan.metadata
        operator_metadata = result.operator_estimate.metadata
        condition_metadata = result.condition_estimate.metadata

        message_length = len(message_elements)
        mean_criticality = sum(
            element.criticality for element in message_elements
        ) / message_length

        element_type_counts = self._count_element_types(result)
        reference_ratio = self._safe_ratio(
            float(procedure_metadata["reference_step_count"]),
            float(procedure_metadata["step_count"]),
        )
        control_marker_ratio = self._safe_ratio(
            float(procedure_metadata["control_marker_step_count"]),
            float(procedure_metadata["step_count"]),
        )

        return self._round_features(
            {
                "prior_message_length": float(message_length),
                "prior_mean_message_criticality": mean_criticality,
                "prior_symbol_ratio": self._safe_ratio(
                    element_type_counts.get("symbol", 0.0),
                    message_length,
                ),
                "prior_digit_ratio": self._safe_ratio(
                    element_type_counts.get("digit", 0.0),
                    message_length,
                ),
                "prior_service_ratio": self._safe_ratio(
                    element_type_counts.get("service", 0.0),
                    message_length,
                ),
                "prior_step_count": float(procedure_metadata["step_count"]),
                "prior_total_nominal_time": float(
                    procedure_metadata["total_nominal_time"],
                ),
                "prior_mean_complexity": float(
                    procedure_metadata["mean_complexity"],
                ),
                "prior_reference_ratio": reference_ratio,
                "prior_control_marker_ratio": control_marker_ratio,
                "prior_operator_total_estimated_time": float(
                    operator_metadata["total_estimated_time"],
                ),
                "prior_operator_total_effort": float(
                    operator_metadata["total_effort"],
                ),
                "prior_operator_mean_attention": float(
                    operator_metadata["mean_attention"],
                ),
                "prior_operator_final_fatigue": float(
                    operator_metadata["final_fatigue"],
                ),
                "prior_operator_preparation_level": float(
                    operator_metadata["preparation_level"],
                ),
                "prior_operator_experience_level": float(
                    operator_metadata["experience_level"],
                ),
                "prior_operator_control_skill": float(
                    operator_metadata["control_skill"],
                ),
                "prior_condition_total_adjusted_time": float(
                    condition_metadata["total_adjusted_time"],
                ),
                "prior_condition_mean_adjusted_attention": float(
                    condition_metadata["mean_adjusted_attention"],
                ),
                "prior_condition_mean_environmental_load": float(
                    condition_metadata["mean_environmental_load"],
                ),
                "prior_condition_mean_stability_index": float(
                    condition_metadata["mean_stability_index"],
                ),
                "prior_condition_time_pressure": float(
                    condition_metadata["time_pressure"],
                ),
                "prior_condition_noise_level": float(
                    condition_metadata["noise_level"],
                ),
                "prior_condition_workload_level": float(
                    condition_metadata["workload_level"],
                ),
                "prior_condition_instruction_access": float(
                    condition_metadata["instruction_access"],
                ),
            },
        )

    def extract_fact_features(
        self,
        result: SimulationResult,
    ) -> dict[str, float]:
        """Извлекает фактические признаки X_fact.

        Фактические признаки становятся известны только после
        моделирования возникновения ошибок и применения контрольных
        процедур K. Эти признаки не должны использоваться как вход
        априорного прогноза качества.
        """
        error_metadata = result.error_protocol.metadata
        control_metadata = result.control_protocol.metadata

        return self._round_features(
            {
                "fact_error_count": float(error_metadata["error_count"]),
                "fact_error_rate": float(error_metadata["error_rate"]),
                "fact_weighted_error_sum": float(
                    error_metadata["weighted_error_sum"],
                ),
                "fact_detected_error_count": float(
                    control_metadata["detected_error_count"],
                ),
                "fact_corrected_error_count": float(
                    control_metadata["corrected_error_count"],
                ),
                "fact_residual_error_count": float(
                    control_metadata["residual_error_count"],
                ),
                "fact_detection_rate": float(
                    control_metadata["detection_rate"],
                ),
                "fact_correction_rate": float(
                    control_metadata["correction_rate"],
                ),
                "fact_residual_error_rate": float(
                    control_metadata["residual_error_rate"],
                ),
                "fact_total_control_effort": float(
                    control_metadata["total_control_effort"],
                ),
            },
        )

    def extract_diagnostic_features(
        self,
        result: SimulationResult,
    ) -> dict[str, float]:
        """Извлекает диагностические признаки X_diag.

        Диагностические признаки предназначены для контроля
        воспроизводимости и анализа работы моделей. Они не являются
        основной группой априорных признаков и отделены от X_prior.
        """
        error_metadata = result.error_protocol.metadata
        control_metadata = result.control_protocol.metadata
        simulation_metadata = result.metadata

        return self._round_features(
            {
                "diag_mean_error_probability": float(
                    error_metadata["mean_error_probability"],
                ),
                "diag_mean_detection_probability": float(
                    control_metadata["mean_detection_probability"],
                ),
                "diag_message_random_seed": float(
                    simulation_metadata["message_random_seed"],
                ),
                "diag_error_random_seed": float(
                    simulation_metadata["error_random_seed"],
                ),
                "diag_control_random_seed": float(
                    simulation_metadata["control_random_seed"],
                ),
            },
        )

    def _validate_result(self, result: SimulationResult) -> None:
        """Проверяет пригодность результата моделирования для извлечения признаков."""
        if not result.message.elements:
            raise ValueError("Сообщение M не содержит элементов.")

        step_count = len(result.message.elements)
        related_lengths = (
            len(result.procedure_plan.steps),
            len(result.operator_estimate.step_estimates),
            len(result.condition_estimate.step_estimates),
            len(result.error_protocol.step_outcomes),
            len(result.control_protocol.step_outcomes),
        )

        if any(length != step_count for length in related_lengths):
            raise ValueError(
                "Нарушена согласованность числа шагов в протоколе моделирования.",
            )

    def _count_element_types(
        self,
        result: SimulationResult,
    ) -> dict[str, float]:
        """Подсчитывает типы элементов исходного сообщения M."""
        counts: dict[str, float] = {}
        for element in result.message.elements:
            counts[element.element_type] = counts.get(element.element_type, 0.0) + 1.0
        return counts

    def _safe_ratio(self, numerator: float, denominator: float) -> float:
        """Рассчитывает отношение с защитой от нулевого знаменателя."""
        if abs(denominator) < self.config.min_denominator:
            return 0.0
        return numerator / denominator

    def _round_features(self, features: dict[str, float]) -> dict[str, float]:
        """Округляет числовые признаки для стабильности отчетов."""
        return {
            name: round(float(value), self.config.round_digits)
            for name, value in features.items()
        }


def validate_feature_group(feature_group: FeatureGroup) -> None:
    """Проверяет разделение X_prior, X_fact и X_diag.

    Проверка фиксирует базовое методическое требование: априорные,
    фактические и диагностические признаки должны храниться раздельно и
    не иметь одинаковых имен.
    """
    if not feature_group.scenario_id:
        raise ValueError("Идентификатор сценария A не задан.")

    groups = (
        feature_group.prior_features,
        feature_group.fact_features,
        feature_group.diagnostic_features,
    )
    if any(not group for group in groups):
        raise ValueError("Каждая группа признаков должна быть непустой.")

    prior_keys = set(feature_group.prior_features)
    fact_keys = set(feature_group.fact_features)
    diagnostic_keys = set(feature_group.diagnostic_features)

    if prior_keys & fact_keys:
        raise ValueError("X_prior и X_fact содержат пересекающиеся имена признаков.")

    if prior_keys & diagnostic_keys:
        raise ValueError("X_prior и X_diag содержат пересекающиеся имена признаков.")

    if fact_keys & diagnostic_keys:
        raise ValueError("X_fact и X_diag содержат пересекающиеся имена признаков.")

    if any(name.startswith("fact_") for name in prior_keys):
        raise ValueError("В X_prior обнаружен фактический признак.")


def summarize_feature_group(
    feature_group: FeatureGroup,
) -> dict[str, int | str]:
    """Возвращает сводку по группе признаков одного сценария."""
    validate_feature_group(feature_group)
    return {
        "scenario_id": feature_group.scenario_id,
        "prior_feature_count": len(feature_group.prior_features),
        "fact_feature_count": len(feature_group.fact_features),
        "diagnostic_feature_count": len(feature_group.diagnostic_features),
        "total_feature_count": (
            len(feature_group.prior_features)
            + len(feature_group.fact_features)
            + len(feature_group.diagnostic_features)
        ),
    }


def feature_group_to_flat_row(
    feature_group: FeatureGroup,
    include_fact: bool = True,
    include_diagnostic: bool = True,
) -> dict[str, float | str]:
    """Преобразует FeatureGroup в одну строку плоской таблицы."""
    validate_feature_group(feature_group)
    row: dict[str, float | str] = {"scenario_id": feature_group.scenario_id}
    row.update(feature_group.prior_features)

    if include_fact:
        row.update(feature_group.fact_features)

    if include_diagnostic:
        row.update(feature_group.diagnostic_features)

    return row


def feature_groups_to_rows(
    feature_groups: Iterable[FeatureGroup],
    include_fact: bool = True,
    include_diagnostic: bool = True,
) -> list[dict[str, float | str]]:
    """Преобразует набор FeatureGroup в строки таблицы."""
    return [
        feature_group_to_flat_row(
            feature_group,
            include_fact=include_fact,
            include_diagnostic=include_diagnostic,
        )
        for feature_group in feature_groups
    ]
