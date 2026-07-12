"""Расчет частных показателей качества ручного кодирования.

Модуль относится к главе 3 диссертации и преобразует раздельные
признаки X_prior, X_fact и X_diag в вектор частных показателей
качества q(A):

* q_acc — показатель точности;
* q_time — показатель временной эффективности;
* q_effort — показатель трудоемкости;
* q_res — показатель устойчивости к ошибкам;
* q_rep — показатель повторяемости условий выполнения;
* q_fit — показатель пригодности сценария применения.

На этом этапе расчет выполняется по уже извлеченным признакам. Модуль
не формирует датасет для LDA, не обучает LDA-модель и не выполняет
проверку прогнозной достоверности метода.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from manual_coding_sim.types import FeatureGroup, QualityVector


def _clip_unit(value: float) -> float:
    """Ограничивает показатель качества диапазоном [0; 1]."""
    return min(1.0, max(0.0, float(value)))


@dataclass(frozen=True)
class QualityCalculatorConfig:
    """Конфигурация расчета частных показателей качества.

    Весовые коэффициенты используются только для агрегирования
    частных показателей в служебный integral_quality. Сам вектор
    q(A) сохраняет все частные показатели раздельно, что соответствует
    логике диссертационного метода.
    """

    max_effort_per_step: float = 3.0
    min_denominator: float = 1e-12
    round_digits: int = 6
    indicator_weights: dict[str, float] = field(
        default_factory=lambda: {
            "q_acc": 1.0,
            "q_time": 1.0,
            "q_effort": 1.0,
            "q_res": 1.0,
            "q_rep": 1.0,
            "q_fit": 1.0,
        },
    )

    def validate(self) -> None:
        """Проверяет корректность конфигурации расчета качества."""
        if self.max_effort_per_step <= 0:
            raise ValueError("max_effort_per_step должен быть положительным.")

        if self.min_denominator <= 0:
            raise ValueError("min_denominator должен быть положительным.")

        if self.round_digits < 0:
            raise ValueError("round_digits не должен быть отрицательным.")

        required_indicators = {
            "q_acc",
            "q_time",
            "q_effort",
            "q_res",
            "q_rep",
            "q_fit",
        }
        if set(self.indicator_weights) != required_indicators:
            raise ValueError(
                "indicator_weights должен содержать веса всех частных показателей q(A).",
            )

        if any(weight < 0 for weight in self.indicator_weights.values()):
            raise ValueError("Весовые коэффициенты не должны быть отрицательными.")

        if sum(self.indicator_weights.values()) <= 0:
            raise ValueError("Сумма весовых коэффициентов должна быть положительной.")


@dataclass(frozen=True)
class QualityAssessment:
    """Результат расчета качества для одного сценария A.

    quality_vector хранит частные показатели q(A), а integral_quality
    используется как служебная агрегированная оценка для отчетных
    таблиц и последующих вычислительных экспериментов.
    """

    scenario_id: str
    quality_vector: QualityVector
    integral_quality: float
    metadata: dict[str, int | float | str]


class QualityCalculator:
    """Калькулятор частных показателей качества q(A).

    Калькулятор использует FeatureGroup, где априорные признаки
    X_prior отделены от фактических X_fact и диагностических X_diag.
    Показатель q(A) на данном этапе является фактической оценкой
    результата моделирования, а не априорным прогнозом.
    """

    def __init__(self, config: QualityCalculatorConfig | None = None) -> None:
        """Инициализирует калькулятор качества."""
        self.config = config or QualityCalculatorConfig()
        self.config.validate()

    def calculate(self, feature_group: FeatureGroup) -> QualityAssessment:
        """Рассчитывает вектор частных показателей качества q(A)."""
        self._validate_feature_group(feature_group)

        prior = feature_group.prior_features
        fact = feature_group.fact_features

        q_acc = self._calculate_accuracy(fact)
        q_time = self._calculate_time_efficiency(prior)
        q_effort = self._calculate_effort_efficiency(prior)
        q_res = self._calculate_resilience(fact)
        q_rep = self._calculate_repeatability(prior)
        q_fit = self._calculate_scenario_fit(
            q_acc=q_acc,
            q_time=q_time,
            q_effort=q_effort,
            q_res=q_res,
            q_rep=q_rep,
        )

        quality_vector = QualityVector(
            q_acc=self._round(q_acc),
            q_time=self._round(q_time),
            q_effort=self._round(q_effort),
            q_res=self._round(q_res),
            q_rep=self._round(q_rep),
            q_fit=self._round(q_fit),
        )
        integral_quality = self._calculate_integral_quality(quality_vector)

        return QualityAssessment(
            scenario_id=feature_group.scenario_id,
            quality_vector=quality_vector,
            integral_quality=integral_quality,
            metadata={
                "scenario_id": feature_group.scenario_id,
                "prior_feature_count": len(feature_group.prior_features),
                "fact_feature_count": len(feature_group.fact_features),
                "diagnostic_feature_count": len(feature_group.diagnostic_features),
                "step_count": int(prior["prior_step_count"]),
                "residual_error_count": int(fact["fact_residual_error_count"]),
                "residual_error_rate": float(fact["fact_residual_error_rate"]),
                "total_adjusted_time": float(
                    prior["prior_condition_total_adjusted_time"],
                ),
            },
        )

    def calculate_batch(
        self,
        feature_groups: Iterable[FeatureGroup],
    ) -> tuple[QualityAssessment, ...]:
        """Рассчитывает q(A) для набора групп признаков."""
        return tuple(self.calculate(group) for group in feature_groups)

    def _calculate_accuracy(self, fact: dict[str, float]) -> float:
        """Рассчитывает q_acc по доле остаточных ошибок."""
        residual_error_rate = float(fact["fact_residual_error_rate"])
        return _clip_unit(1.0 - residual_error_rate)

    def _calculate_time_efficiency(self, prior: dict[str, float]) -> float:
        """Рассчитывает q_time по отношению нормативного и расчетного времени."""
        nominal_time = float(prior["prior_total_nominal_time"])
        adjusted_time = float(prior["prior_condition_total_adjusted_time"])
        return _clip_unit(self._safe_ratio(nominal_time, adjusted_time))

    def _calculate_effort_efficiency(self, prior: dict[str, float]) -> float:
        """Рассчитывает q_effort по средней трудоемкости одного шага."""
        total_effort = float(prior["prior_operator_total_effort"])
        step_count = float(prior["prior_step_count"])
        effort_per_step = self._safe_ratio(total_effort, step_count)
        normalized_effort = self._safe_ratio(
            effort_per_step,
            self.config.max_effort_per_step,
        )
        return _clip_unit(1.0 - normalized_effort)

    def _calculate_resilience(self, fact: dict[str, float]) -> float:
        """Рассчитывает q_res по остаточным ошибкам и эффективности контроля K."""
        residual_component = 1.0 - float(fact["fact_residual_error_rate"])
        detection_component = float(fact["fact_detection_rate"])
        correction_component = float(fact["fact_correction_rate"])
        return _clip_unit(
            0.50 * residual_component
            + 0.25 * detection_component
            + 0.25 * correction_component,
        )

    def _calculate_repeatability(self, prior: dict[str, float]) -> float:
        """Рассчитывает q_rep по устойчивости условий и вниманию оператора."""
        stability = float(prior["prior_condition_mean_stability_index"])
        attention = float(prior["prior_condition_mean_adjusted_attention"])
        time_pressure = float(prior["prior_condition_time_pressure"])
        return _clip_unit(
            0.40 * stability
            + 0.30 * attention
            + 0.30 * (1.0 - time_pressure),
        )

    def _calculate_scenario_fit(
        self,
        q_acc: float,
        q_time: float,
        q_effort: float,
        q_res: float,
        q_rep: float,
    ) -> float:
        """Рассчитывает q_fit как пригодность сценария применения A."""
        return _clip_unit(
            0.30 * q_acc
            + 0.20 * q_time
            + 0.15 * q_effort
            + 0.20 * q_res
            + 0.15 * q_rep,
        )

    def _calculate_integral_quality(self, quality_vector: QualityVector) -> float:
        """Рассчитывает служебную агрегированную оценку качества."""
        values = quality_vector_to_dict(quality_vector)
        weighted_sum = sum(
            values[name] * weight
            for name, weight in self.config.indicator_weights.items()
        )
        weight_sum = sum(self.config.indicator_weights.values())
        return self._round(weighted_sum / weight_sum)

    def _validate_feature_group(self, feature_group: FeatureGroup) -> None:
        """Проверяет наличие признаков, необходимых для расчета q(A)."""
        if not feature_group.scenario_id:
            raise ValueError("Идентификатор сценария A не задан.")

        required_prior = {
            "prior_step_count",
            "prior_total_nominal_time",
            "prior_operator_total_effort",
            "prior_condition_total_adjusted_time",
            "prior_condition_mean_adjusted_attention",
            "prior_condition_mean_stability_index",
            "prior_condition_time_pressure",
        }
        required_fact = {
            "fact_residual_error_count",
            "fact_residual_error_rate",
            "fact_detection_rate",
            "fact_correction_rate",
        }

        missing_prior = required_prior - set(feature_group.prior_features)
        missing_fact = required_fact - set(feature_group.fact_features)
        if missing_prior:
            raise ValueError(f"В X_prior отсутствуют признаки: {sorted(missing_prior)}")

        if missing_fact:
            raise ValueError(f"В X_fact отсутствуют признаки: {sorted(missing_fact)}")

        if float(feature_group.prior_features["prior_step_count"]) <= 0:
            raise ValueError("prior_step_count должен быть положительным.")

    def _safe_ratio(self, numerator: float, denominator: float) -> float:
        """Рассчитывает отношение с защитой от нулевого знаменателя."""
        if abs(denominator) < self.config.min_denominator:
            return 0.0
        return numerator / denominator

    def _round(self, value: float) -> float:
        """Округляет показатель для устойчивого табличного представления."""
        return round(float(value), self.config.round_digits)


def quality_vector_to_dict(quality_vector: QualityVector) -> dict[str, float]:
    """Преобразует QualityVector в словарь частных показателей."""
    return {
        "q_acc": quality_vector.q_acc,
        "q_time": quality_vector.q_time,
        "q_effort": quality_vector.q_effort,
        "q_res": quality_vector.q_res,
        "q_rep": quality_vector.q_rep,
        "q_fit": quality_vector.q_fit,
    }


def summarize_quality_assessment(
    assessment: QualityAssessment,
) -> dict[str, int | float | str]:
    """Возвращает отчетную сводку по расчету качества q(A)."""
    row: dict[str, int | float | str] = {
        "scenario_id": assessment.scenario_id,
        "integral_quality": assessment.integral_quality,
    }
    row.update(quality_vector_to_dict(assessment.quality_vector))
    row.update(assessment.metadata)
    return row


def quality_assessments_to_rows(
    assessments: Iterable[QualityAssessment],
) -> list[dict[str, int | float | str]]:
    """Преобразует набор оценок качества в строки таблицы."""
    return [summarize_quality_assessment(item) for item in assessments]
