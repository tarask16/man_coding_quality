"""
Создание модуля извлечения признаков FeatureExtractor.

Скрипт относится к этапу 9 программной реализации главы 3 диссертации.
Он создает модуль, который преобразует результат интегрального моделирования
в три раздельные группы признаков: X_prior, X_fact и X_diag.

Разделение признаков принципиально важно для диссертации: априорная оценка
качества не должна использовать фактические признаки, которые становятся
известны только после моделируемого выполнения процедуры ручного кодирования.
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


def check_python_syntax(path: Path) -> dict[str, str]:
    """Проверяет синтаксическую корректность Python-файла."""
    try:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        return {"status": "OK", "message": "Синтаксис корректен"}
    except SyntaxError as error:
        return {"status": "ERROR", "message": str(error)}


def main() -> None:
    """Создает модуль FeatureExtractor, тесты и отчет этапа 9."""
    SRC_DIR.mkdir(parents=True, exist_ok=True)
    TESTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    write_text_file(
        SRC_DIR / "feature_extractor.py",
        '''
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
        ''',
    )

    write_text_file(
        SRC_DIR / "__init__.py",
        '''
        """
        Базовый пакет исследовательского симулятора ручного кодирования.

        Пакет предназначен для программной реализации главы 3 диссертации:
        компьютерного моделирования процессов ручного кодирования и декодирования
        при априорной оценке качества ручных средств кодирования информации.
        """

        from manual_coding_sim.condition_model import (
            ConditionModel,
            ConditionModelConfig,
            ConditionPlanEstimate,
            ConditionProfile,
            ConditionStepEstimate,
            condition_estimates_to_rows,
            summarize_condition_estimate,
        )
        from manual_coding_sim.config import load_experiment_config
        from manual_coding_sim.control_model import (
            ControlModel,
            ControlModelConfig,
            ControlProfile,
            ControlProtocol,
            ControlStepOutcome,
            control_protocols_to_rows,
            summarize_control_protocol,
        )
        from manual_coding_sim.error_model import (
            ErrorModel,
            ErrorModelConfig,
            ErrorProtocol,
            ErrorStepOutcome,
            error_protocols_to_rows,
            summarize_error_protocol,
        )
        from manual_coding_sim.feature_extractor import (
            FeatureExtractor,
            FeatureExtractorConfig,
            feature_group_to_flat_row,
            feature_groups_to_rows,
            summarize_feature_group,
            validate_feature_group,
        )
        from manual_coding_sim.message_model import (
            MessageGenerationConfig,
            MessageModel,
            messages_to_rows,
            summarize_message,
        )
        from manual_coding_sim.operator_model import (
            OperatorModel,
            OperatorModelConfig,
            OperatorPlanEstimate,
            OperatorProfile,
            OperatorState,
            OperatorStepEstimate,
            operator_estimates_to_rows,
            summarize_operator_estimate,
        )
        from manual_coding_sim.procedure_model import (
            CodingOperationRule,
            ProcedureModel,
            ProcedureModelConfig,
            ProcedurePlan,
            ProcedureStep,
            procedure_plans_to_rows,
            summarize_procedure_plan,
        )
        from manual_coding_sim.protocol_simulator import (
            ProtocolSimulator,
            ProtocolSimulatorConfig,
            SimulationResult,
            simulation_results_to_rows,
            summarize_simulation_result,
        )
        from manual_coding_sim.types import (
            FeatureGroup,
            GeneratedMessage,
            MessageElement,
            QualityVector,
            ScenarioParameters,
        )

        __version__ = "0.1.0"

        __all__ = [
            "CodingOperationRule",
            "ConditionModel",
            "ConditionModelConfig",
            "ConditionPlanEstimate",
            "ConditionProfile",
            "ConditionStepEstimate",
            "ControlModel",
            "ControlModelConfig",
            "ControlProfile",
            "ControlProtocol",
            "ControlStepOutcome",
            "ErrorModel",
            "ErrorModelConfig",
            "ErrorProtocol",
            "ErrorStepOutcome",
            "FeatureExtractor",
            "FeatureExtractorConfig",
            "FeatureGroup",
            "GeneratedMessage",
            "MessageElement",
            "MessageGenerationConfig",
            "MessageModel",
            "OperatorModel",
            "OperatorModelConfig",
            "OperatorPlanEstimate",
            "OperatorProfile",
            "OperatorState",
            "OperatorStepEstimate",
            "ProcedureModel",
            "ProcedureModelConfig",
            "ProcedurePlan",
            "ProcedureStep",
            "ProtocolSimulator",
            "ProtocolSimulatorConfig",
            "QualityVector",
            "ScenarioParameters",
            "SimulationResult",
            "condition_estimates_to_rows",
            "control_protocols_to_rows",
            "error_protocols_to_rows",
            "feature_group_to_flat_row",
            "feature_groups_to_rows",
            "load_experiment_config",
            "messages_to_rows",
            "operator_estimates_to_rows",
            "procedure_plans_to_rows",
            "simulation_results_to_rows",
            "summarize_condition_estimate",
            "summarize_control_protocol",
            "summarize_error_protocol",
            "summarize_feature_group",
            "summarize_message",
            "summarize_operator_estimate",
            "summarize_procedure_plan",
            "summarize_simulation_result",
            "validate_feature_group",
        ]
        ''',
    )

    write_text_file(
        TESTS_DIR / "test_stage9_feature_extractor.py",
        '''
        """Тесты этапа 9: извлечение признаков X_prior, X_fact и X_diag."""

        from __future__ import annotations

        import pytest

        from manual_coding_sim import FeatureExtractor, ProtocolSimulator
        from manual_coding_sim.feature_extractor import (
            FeatureExtractorConfig,
            feature_group_to_flat_row,
            feature_groups_to_rows,
            summarize_feature_group,
            validate_feature_group,
        )
        from manual_coding_sim.types import FeatureGroup


        def _make_feature_group() -> FeatureGroup:
            """Формирует FeatureGroup для тестов этапа 9."""
            result = ProtocolSimulator().simulate_once(message_id="M_FEATURE")
            return FeatureExtractor().extract(result)


        def test_feature_extractor_imports() -> None:
            """Проверяет импортируемость извлекателя признаков."""
            extractor = FeatureExtractor()

            assert isinstance(extractor.config, FeatureExtractorConfig)


        def test_extract_returns_feature_group() -> None:
            """Проверяет получение FeatureGroup из результата моделирования."""
            feature_group = _make_feature_group()

            assert isinstance(feature_group, FeatureGroup)
            assert feature_group.scenario_id == "A_001"
            assert feature_group.prior_features
            assert feature_group.fact_features
            assert feature_group.diagnostic_features


        def test_prior_features_do_not_contain_fact_prefix() -> None:
            """Проверяет отсутствие фактических признаков в X_prior."""
            feature_group = _make_feature_group()

            assert all(
                not name.startswith("fact_")
                for name in feature_group.prior_features
            )
            assert all(name.startswith("prior_") for name in feature_group.prior_features)


        def test_feature_groups_have_disjoint_names() -> None:
            """Проверяет непересечение имен X_prior, X_fact и X_diag."""
            feature_group = _make_feature_group()
            prior_keys = set(feature_group.prior_features)
            fact_keys = set(feature_group.fact_features)
            diagnostic_keys = set(feature_group.diagnostic_features)

            assert not prior_keys & fact_keys
            assert not prior_keys & diagnostic_keys
            assert not fact_keys & diagnostic_keys


        def test_prior_feature_values_are_consistent() -> None:
            """Проверяет отдельные априорные признаки сценария."""
            result = ProtocolSimulator().simulate_once(message_id="M_PRIOR")
            feature_group = FeatureExtractor().extract(result)

            assert feature_group.prior_features["prior_message_length"] == len(
                result.message.elements,
            )
            assert feature_group.prior_features["prior_step_count"] == len(
                result.procedure_plan.steps,
            )
            assert 0.0 <= feature_group.prior_features[
                "prior_condition_mean_environmental_load"
            ] <= 1.0


        def test_fact_features_match_protocol_metadata() -> None:
            """Проверяет соответствие X_fact фактическому протоколу."""
            result = ProtocolSimulator().simulate_once(message_id="M_FACT")
            feature_group = FeatureExtractor().extract(result)

            assert feature_group.fact_features["fact_error_count"] == float(
                result.error_protocol.metadata["error_count"],
            )
            assert feature_group.fact_features["fact_residual_error_count"] == float(
                result.control_protocol.metadata["residual_error_count"],
            )
            assert feature_group.fact_features[
                "fact_residual_error_count"
            ] <= feature_group.fact_features["fact_error_count"]


        def test_diagnostic_features_include_random_seeds() -> None:
            """Проверяет диагностические признаки воспроизводимости."""
            feature_group = _make_feature_group()

            assert feature_group.diagnostic_features["diag_message_random_seed"] == 42.0
            assert feature_group.diagnostic_features["diag_error_random_seed"] == 1042.0
            assert feature_group.diagnostic_features["diag_control_random_seed"] == 2042.0


        def test_extract_batch_returns_requested_count() -> None:
            """Проверяет пакетное извлечение признаков."""
            results = ProtocolSimulator().simulate_batch(3)
            feature_groups = FeatureExtractor().extract_batch(results)

            assert len(feature_groups) == 3
            assert all(group.scenario_id == "A_001" for group in feature_groups)


        def test_flat_row_can_exclude_fact_and_diagnostic_features() -> None:
            """Проверяет формирование строки только с X_prior."""
            feature_group = _make_feature_group()
            row = feature_group_to_flat_row(
                feature_group,
                include_fact=False,
                include_diagnostic=False,
            )

            assert "scenario_id" in row
            assert "prior_message_length" in row
            assert all(not name.startswith("fact_") for name in row)
            assert all(not name.startswith("diag_") for name in row)


        def test_feature_groups_to_rows() -> None:
            """Проверяет преобразование набора FeatureGroup в строки таблицы."""
            results = ProtocolSimulator().simulate_batch(2)
            feature_groups = FeatureExtractor().extract_batch(results)
            rows = feature_groups_to_rows(feature_groups)

            assert len(rows) == 2
            assert all("prior_message_length" in row for row in rows)
            assert all("fact_error_count" in row for row in rows)
            assert all("diag_mean_error_probability" in row for row in rows)


        def test_summarize_feature_group() -> None:
            """Проверяет сводку по группам признаков."""
            feature_group = _make_feature_group()
            summary = summarize_feature_group(feature_group)

            assert summary["scenario_id"] == "A_001"
            assert summary["prior_feature_count"] == len(feature_group.prior_features)
            assert summary["fact_feature_count"] == len(feature_group.fact_features)
            assert summary["diagnostic_feature_count"] == len(
                feature_group.diagnostic_features,
            )


        def test_validate_feature_group_rejects_overlap() -> None:
            """Проверяет отклонение пересекающихся имен признаков."""
            broken_group = FeatureGroup(
                scenario_id="A_BAD",
                prior_features={"same_name": 1.0},
                fact_features={"same_name": 2.0},
                diagnostic_features={"diag_value": 3.0},
            )

            with pytest.raises(ValueError):
                validate_feature_group(broken_group)


        def test_invalid_config_is_rejected() -> None:
            """Проверяет отклонение некорректной конфигурации."""
            with pytest.raises(ValueError):
                FeatureExtractorConfig(round_digits=-1).validate()
        ''',
    )

    created_files = [
        SRC_DIR / "feature_extractor.py",
        SRC_DIR / "__init__.py",
        TESTS_DIR / "test_stage9_feature_extractor.py",
    ]
    syntax_report = {
        str(path.relative_to(ROOT)): check_python_syntax(path)
        for path in created_files
    }

    report = {
        "stage": 9,
        "title": "Извлечение признаков FeatureExtractor",
        "created_files": [str(path.relative_to(ROOT)) for path in created_files],
        "syntax_report": syntax_report,
        "scientific_scope": (
            "Раздельное формирование X_prior, X_fact и X_diag по результату "
            "интегрального моделирования ручного кодирования."
        ),
        "not_implemented_yet": [
            "расчет итогового вектора качества q(A)",
            "формирование итогового датасета для LDA",
            "обучение LDA-модели",
            "вычислительный эксперимент главы 6",
        ],
    }

    report_path = REPORTS_DIR / "stage9_feature_extractor_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("ЭТАП 9. ИЗВЛЕЧЕНИЕ ПРИЗНАКОВ X_PRIOR, X_FACT, X_DIAG")
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
