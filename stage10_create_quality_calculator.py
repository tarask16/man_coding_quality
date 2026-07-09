"""
Создание модуля расчета частных показателей качества QualityCalculator.

Скрипт относится к этапу 10 программной реализации главы 3 диссертации.
Он создает модуль, который преобразует раздельные признаки X_prior,
X_fact и X_diag в вектор частных показателей качества q(A).

На этом этапе не формируется итоговый датасет для LDA и не выполняется
обучение тематической модели. Реализуется только расчет q_acc, q_time,
q_effort, q_res, q_rep и q_fit по уже полученной группе признаков.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
import textwrap


ROOT = Path.cwd()
SRC_DIR = ROOT / "src" / "manual_coding_sim"
TESTS_DIR = ROOT / "tests"
REPORTS_DIR = ROOT / "reports" / "chapter3"


def write_text_file(path: Path, content: str) -> None:
    """Записывает текстовый файл в кодировке UTF-8 без лишних отступов."""
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized_content = textwrap.dedent(content).lstrip("\n")
    path.write_text(normalized_content, encoding="utf-8")


def append_text_once(path: Path, marker: str, content: str) -> None:
    """Добавляет текстовый блок в файл только один раз."""
    current_text = path.read_text(encoding="utf-8") if path.exists() else ""
    if marker in current_text:
        return
    normalized_content = textwrap.dedent(content).strip("\n")
    path.write_text(current_text.rstrip() + "\n\n" + normalized_content + "\n", encoding="utf-8")


def check_python_syntax(path: Path) -> dict[str, str]:
    """Проверяет синтаксическую корректность Python-файла."""
    try:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        return {"status": "OK", "message": "Синтаксис корректен"}
    except SyntaxError as error:
        return {"status": "ERROR", "message": str(error)}


def main() -> None:
    """Создает модуль QualityCalculator, тесты и отчет этапа 10."""
    SRC_DIR.mkdir(parents=True, exist_ok=True)
    TESTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    write_text_file(
        SRC_DIR / "quality_calculator.py",
        '''
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
        ''',
    )

    append_text_once(
        SRC_DIR / "__init__.py",
        "from manual_coding_sim.quality_calculator import",
        '''
        # Экспорт этапа 10: расчет частных показателей качества q(A).
        from manual_coding_sim.quality_calculator import (
            QualityAssessment,
            QualityCalculator,
            QualityCalculatorConfig,
            quality_assessments_to_rows,
            quality_vector_to_dict,
            summarize_quality_assessment,
        )
        ''',
    )

    write_text_file(
        TESTS_DIR / "test_stage10_quality_calculator.py",
        '''
        """Тесты этапа 10: расчет частных показателей качества q(A)."""

        from __future__ import annotations

        import pytest

        from manual_coding_sim import FeatureExtractor, ProtocolSimulator
        from manual_coding_sim.quality_calculator import (
            QualityAssessment,
            QualityCalculator,
            QualityCalculatorConfig,
            quality_assessments_to_rows,
            quality_vector_to_dict,
            summarize_quality_assessment,
        )
        from manual_coding_sim.types import FeatureGroup, QualityVector


        def _make_feature_group() -> FeatureGroup:
            """Формирует FeatureGroup из полного результата моделирования."""
            result = ProtocolSimulator().simulate_once(message_id="M_QUALITY")
            return FeatureExtractor().extract(result)


        def _manual_feature_group(residual_error_rate: float) -> FeatureGroup:
            """Создает управляемую группу признаков для проверки чувствительности q(A)."""
            return FeatureGroup(
                scenario_id="A_MANUAL",
                prior_features={
                    "prior_step_count": 10.0,
                    "prior_total_nominal_time": 10.0,
                    "prior_operator_total_effort": 10.0,
                    "prior_condition_total_adjusted_time": 12.0,
                    "prior_condition_mean_adjusted_attention": 0.80,
                    "prior_condition_mean_stability_index": 0.75,
                    "prior_condition_time_pressure": 0.10,
                },
                fact_features={
                    "fact_residual_error_count": residual_error_rate * 10.0,
                    "fact_residual_error_rate": residual_error_rate,
                    "fact_detection_rate": 0.80,
                    "fact_correction_rate": 0.70,
                },
                diagnostic_features={"diag_mean_error_probability": 0.10},
            )


        def test_quality_calculator_imports() -> None:
            """Проверяет импортируемость калькулятора качества."""
            calculator = QualityCalculator()

            assert isinstance(calculator.config, QualityCalculatorConfig)


        def test_calculate_returns_quality_assessment() -> None:
            """Проверяет получение QualityAssessment для FeatureGroup."""
            assessment = QualityCalculator().calculate(_make_feature_group())

            assert isinstance(assessment, QualityAssessment)
            assert assessment.scenario_id == "A_001"
            assert isinstance(assessment.quality_vector, QualityVector)


        def test_quality_indicators_are_unit_interval() -> None:
            """Проверяет диапазон всех частных показателей q(A)."""
            assessment = QualityCalculator().calculate(_make_feature_group())
            values = quality_vector_to_dict(assessment.quality_vector)

            assert all(0.0 <= value <= 1.0 for value in values.values())
            assert 0.0 <= assessment.integral_quality <= 1.0


        def test_quality_vector_to_dict_contains_all_indicators() -> None:
            """Проверяет преобразование QualityVector в словарь."""
            assessment = QualityCalculator().calculate(_make_feature_group())
            row = quality_vector_to_dict(assessment.quality_vector)

            assert set(row) == {"q_acc", "q_time", "q_effort", "q_res", "q_rep", "q_fit"}


        def test_summary_contains_quality_and_metadata() -> None:
            """Проверяет отчетную сводку по q(A)."""
            assessment = QualityCalculator().calculate(_make_feature_group())
            summary = summarize_quality_assessment(assessment)

            assert summary["scenario_id"] == "A_001"
            assert "q_acc" in summary
            assert "integral_quality" in summary
            assert summary["step_count"] > 0


        def test_batch_calculation_returns_requested_count() -> None:
            """Проверяет пакетный расчет q(A)."""
            results = ProtocolSimulator().simulate_batch(3)
            feature_groups = FeatureExtractor().extract_batch(results)
            assessments = QualityCalculator().calculate_batch(feature_groups)

            assert len(assessments) == 3
            assert all(item.scenario_id == "A_001" for item in assessments)


        def test_quality_assessments_to_rows() -> None:
            """Проверяет преобразование набора оценок качества в строки таблицы."""
            feature_groups = FeatureExtractor().extract_batch(
                ProtocolSimulator().simulate_batch(2),
            )
            assessments = QualityCalculator().calculate_batch(feature_groups)
            rows = quality_assessments_to_rows(assessments)

            assert len(rows) == 2
            assert all("q_fit" in row for row in rows)
            assert all("integral_quality" in row for row in rows)


        def test_accuracy_decreases_when_residual_errors_increase() -> None:
            """Проверяет чувствительность q_acc к остаточным ошибкам."""
            calculator = QualityCalculator()
            low_error = calculator.calculate(_manual_feature_group(0.10))
            high_error = calculator.calculate(_manual_feature_group(0.60))

            assert low_error.quality_vector.q_acc > high_error.quality_vector.q_acc
            assert low_error.integral_quality > high_error.integral_quality


        def test_invalid_config_is_rejected() -> None:
            """Проверяет отклонение некорректной конфигурации качества."""
            with pytest.raises(ValueError):
                QualityCalculatorConfig(max_effort_per_step=0.0).validate()

            with pytest.raises(ValueError):
                QualityCalculatorConfig(
                    indicator_weights={
                        "q_acc": 1.0,
                        "q_time": 1.0,
                        "q_effort": 1.0,
                        "q_res": 1.0,
                        "q_rep": 1.0,
                        "q_fit": -1.0,
                    },
                ).validate()


        def test_missing_required_features_are_rejected() -> None:
            """Проверяет отклонение FeatureGroup без необходимых признаков."""
            broken_group = FeatureGroup(
                scenario_id="A_BAD",
                prior_features={"prior_step_count": 10.0},
                fact_features={"fact_residual_error_rate": 0.1},
                diagnostic_features={"diag_value": 1.0},
            )

            with pytest.raises(ValueError):
                QualityCalculator().calculate(broken_group)
        ''',
    )

    created_files = [
        SRC_DIR / "quality_calculator.py",
        SRC_DIR / "__init__.py",
        TESTS_DIR / "test_stage10_quality_calculator.py",
    ]
    syntax_report = {
        str(path.relative_to(ROOT)): check_python_syntax(path)
        for path in created_files
    }

    report = {
        "stage": 10,
        "title": "Расчет частных показателей качества QualityCalculator",
        "created_files": [str(path.relative_to(ROOT)) for path in created_files],
        "syntax_report": syntax_report,
        "scientific_scope": (
            "Расчет вектора q(A) = (q_acc, q_time, q_effort, q_res, q_rep, q_fit) "
            "по группам признаков X_prior, X_fact и X_diag."
        ),
        "not_implemented_yet": [
            "формирование итогового датасета",
            "сохранение таблиц признаков и качества в CSV",
            "токенизация признаков для LDA",
            "обучение LDA-модели",
            "проверка достоверности прогноза качества",
        ],
    }

    report_path = REPORTS_DIR / "stage10_quality_calculator_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("ЭТАП 10. РАСЧЕТ ЧАСТНЫХ ПОКАЗАТЕЛЕЙ КАЧЕСТВА q(A)")
    print("=" * 70)
    for path in created_files:
        rel_path = path.relative_to(ROOT)
        status = syntax_report[str(rel_path)]["status"]
        print(f"[{status}] {rel_path}")
    print(f"[OK] Отчет: {report_path}")
    print()
    print("Теперь выполните команду:")
    print("python -m pytest")


if __name__ == "__main__":
    main()
