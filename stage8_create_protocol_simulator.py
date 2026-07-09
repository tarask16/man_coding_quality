"""
Создание интегрального симулятора протоколов ProtocolSimulator.

Скрипт относится к этапу 8 программной реализации главы 3 диссертации.
Он создает модуль ProtocolSimulator, который объединяет ранее созданные
модели G, S, O, U, ErrorModel и K в воспроизводимую цепочку получения
протокола моделирования ручного кодирования. На данном этапе не выполняются
декодирование D_h, восстановление сообщения M' и расчет итогового вектора
качества q(A).
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
    """Записывает текстовый файл в кодировке UTF-8 без лишних начальных отступов."""
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
    """Создает модуль ProtocolSimulator, тесты и отчет этапа 8."""
    SRC_DIR.mkdir(parents=True, exist_ok=True)
    TESTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    write_text_file(
        SRC_DIR / "protocol_simulator.py",
        '''
        """Интегральный симулятор протоколов ручного кодирования.

        Модуль относится к главе 3 диссертации и объединяет ранее созданные
        модели класса сообщений G, средства ручного кодирования S, оператора O,
        условий применения U, вероятностной модели ошибок и контрольных
        процедур K.

        Симулятор формирует воспроизводимый протокол моделирования цепочки:

            S, O, U, G, K -> M -> план S -> оценка O -> оценка U -> ошибки -> контроль

        На данном этапе не выполняются декодирование D_h, восстановление M'
        и итоговый расчет вектора частных показателей качества q(A). Эти
        операции будут добавлены на следующих этапах после появления моделей
        извлечения признаков и расчета качества.
        """

        from __future__ import annotations

        from dataclasses import dataclass, field
        from typing import Iterable

        from manual_coding_sim.condition_model import (
            ConditionModel,
            ConditionModelConfig,
            ConditionPlanEstimate,
        )
        from manual_coding_sim.control_model import (
            ControlModel,
            ControlModelConfig,
            ControlProtocol,
        )
        from manual_coding_sim.error_model import (
            ErrorModel,
            ErrorModelConfig,
            ErrorProtocol,
        )
        from manual_coding_sim.message_model import (
            MessageGenerationConfig,
            MessageModel,
        )
        from manual_coding_sim.operator_model import (
            OperatorModel,
            OperatorModelConfig,
            OperatorPlanEstimate,
        )
        from manual_coding_sim.procedure_model import (
            ProcedureModel,
            ProcedureModelConfig,
            ProcedurePlan,
        )
        from manual_coding_sim.types import GeneratedMessage, ScenarioParameters


        @dataclass(frozen=True)
        class ProtocolSimulatorConfig:
            """Конфигурация интегрального симулятора протоколов.

            Конфигурация задает параметры сценария A = {S, O, U, G, K} и
            фиксированные зерна генераторов случайных чисел. Разделение зерен
            позволяет независимо воспроизводить генерацию сообщений M,
            возникновение ошибок и работу контрольных процедур K.
            """

            scenario_id: str = "A_001"
            message_config: MessageGenerationConfig = field(
                default_factory=MessageGenerationConfig,
            )
            procedure_config: ProcedureModelConfig = field(
                default_factory=ProcedureModelConfig,
            )
            operator_config: OperatorModelConfig = field(
                default_factory=OperatorModelConfig,
            )
            condition_config: ConditionModelConfig = field(
                default_factory=ConditionModelConfig,
            )
            error_config: ErrorModelConfig = field(default_factory=ErrorModelConfig)
            control_config: ControlModelConfig = field(
                default_factory=ControlModelConfig,
            )
            message_random_seed: int = 42
            error_random_seed: int = 1042
            control_random_seed: int = 2042

            def validate(self) -> None:
                """Проверяет корректность конфигурации симулятора."""
                if not self.scenario_id:
                    raise ValueError("Идентификатор сценария A не задан.")

                if self.message_random_seed < 0:
                    raise ValueError("message_random_seed не должен быть отрицательным.")

                if self.error_random_seed < 0:
                    raise ValueError("error_random_seed не должен быть отрицательным.")

                if self.control_random_seed < 0:
                    raise ValueError("control_random_seed не должен быть отрицательным.")

                self.message_config.validate()
                self.procedure_config.validate()
                self.operator_config.validate()
                self.condition_config.validate()
                self.error_config.validate()
                self.control_config.validate()


        @dataclass(frozen=True)
        class SimulationResult:
            """Результат одного полного прогона симулятора протоколов.

            Объект сохраняет все промежуточные артефакты моделирования, чтобы
            обеспечить трассируемость перехода от априорного сценария
            A = {S, O, U, G, K} к фактическому протоколу ошибок и контроля.
            """

            scenario: ScenarioParameters
            message: GeneratedMessage
            procedure_plan: ProcedurePlan
            operator_estimate: OperatorPlanEstimate
            condition_estimate: ConditionPlanEstimate
            error_protocol: ErrorProtocol
            control_protocol: ControlProtocol
            metadata: dict[str, int | float | str]


        class ProtocolSimulator:
            """Интегральный симулятор протоколов ручного кодирования.

            Симулятор не раскрывает конкретные прикладные способы скрытой связи.
            Он оперирует абстрактными сообщениями, нормативными операциями,
            априорными профилями оператора, условий и контроля, а также
            вероятностным протоколом ошибок.
            """

            def __init__(self, config: ProtocolSimulatorConfig | None = None) -> None:
                """Инициализирует все модели, входящие в сценарий A."""
                self.config = config or ProtocolSimulatorConfig()
                self.config.validate()

                self.message_model = MessageModel(
                    config=self.config.message_config,
                    random_seed=self.config.message_random_seed,
                )
                self.procedure_model = ProcedureModel(self.config.procedure_config)
                self.operator_model = OperatorModel(self.config.operator_config)
                self.condition_model = ConditionModel(self.config.condition_config)
                self.error_model = ErrorModel(
                    config=self.config.error_config,
                    random_seed=self.config.error_random_seed,
                )
                self.control_model = ControlModel(
                    config=self.config.control_config,
                    random_seed=self.config.control_random_seed,
                )

            def simulate_once(self, message_id: str | None = None) -> SimulationResult:
                """Выполняет один воспроизводимый прогон моделирования."""
                message = self.message_model.generate_message(message_id=message_id)
                procedure_plan = self.procedure_model.build_plan(message)
                operator_estimate = self.operator_model.estimate_plan(procedure_plan)
                condition_estimate = self.condition_model.estimate_plan(operator_estimate)
                error_protocol = self.error_model.generate_protocol(condition_estimate)
                control_protocol = self.control_model.generate_protocol(error_protocol)

                scenario = self._build_scenario()
                metadata = self._build_metadata(
                    message=message,
                    procedure_plan=procedure_plan,
                    condition_estimate=condition_estimate,
                    error_protocol=error_protocol,
                    control_protocol=control_protocol,
                )

                return SimulationResult(
                    scenario=scenario,
                    message=message,
                    procedure_plan=procedure_plan,
                    operator_estimate=operator_estimate,
                    condition_estimate=condition_estimate,
                    error_protocol=error_protocol,
                    control_protocol=control_protocol,
                    metadata=metadata,
                )

            def simulate_batch(self, count: int) -> tuple[SimulationResult, ...]:
                """Выполняет пакет прогонов для серии сообщений M."""
                if count <= 0:
                    raise ValueError("Число прогонов моделирования должно быть положительным.")

                return tuple(self.simulate_once() for _ in range(count))

            def reset(self) -> None:
                """Возвращает псевдослучайные модели к начальному состоянию."""
                self.message_model.reset()
                self.error_model.reset()
                self.control_model.reset()

            def _build_scenario(self) -> ScenarioParameters:
                """Формирует описание сценария A = {S, O, U, G, K}."""
                return ScenarioParameters(
                    scenario_id=self.config.scenario_id,
                    coding_tool_id=self.config.procedure_config.procedure_id,
                    operator_id=self.config.operator_config.profile.operator_id,
                    condition_id=self.config.condition_config.profile.condition_id,
                    message_class_id=self.config.message_config.message_class_id,
                    control_procedure_id=self.config.control_config.profile.control_id,
                )

            def _build_metadata(
                self,
                message: GeneratedMessage,
                procedure_plan: ProcedurePlan,
                condition_estimate: ConditionPlanEstimate,
                error_protocol: ErrorProtocol,
                control_protocol: ControlProtocol,
            ) -> dict[str, int | float | str]:
                """Формирует контрольную метаинформацию одного прогона."""
                step_count = int(procedure_plan.metadata["step_count"])
                error_count = int(error_protocol.metadata["error_count"])
                residual_error_count = int(
                    control_protocol.metadata["residual_error_count"],
                )
                adjusted_time = sum(
                    step.adjusted_time for step in condition_estimate.step_estimates
                )

                return {
                    "scenario_id": self.config.scenario_id,
                    "message_id": message.message_id,
                    "step_count": step_count,
                    "message_length": len(message.elements),
                    "original_error_count": error_count,
                    "residual_error_count": residual_error_count,
                    "nominal_time": float(
                        procedure_plan.metadata["total_nominal_time"],
                    ),
                    "condition_adjusted_time": round(adjusted_time, 6),
                    "message_random_seed": self.config.message_random_seed,
                    "error_random_seed": self.config.error_random_seed,
                    "control_random_seed": self.config.control_random_seed,
                }


        def summarize_simulation_result(
            result: SimulationResult,
        ) -> dict[str, int | float | str]:
            """Возвращает контрольную сводку по одному прогону симулятора."""
            if not result.procedure_plan.steps:
                raise ValueError("Результат моделирования не содержит шагов процедуры.")

            adjusted_time = sum(
                step.adjusted_time for step in result.condition_estimate.step_estimates
            )
            total_control_effort = float(
                result.control_protocol.metadata["total_control_effort"],
            )

            return {
                "scenario_id": result.scenario.scenario_id,
                "message_id": result.message.message_id,
                "message_length": len(result.message.elements),
                "step_count": len(result.procedure_plan.steps),
                "nominal_time": float(
                    result.procedure_plan.metadata["total_nominal_time"],
                ),
                "condition_adjusted_time": round(adjusted_time, 6),
                "original_error_count": int(
                    result.control_protocol.metadata["original_error_count"],
                ),
                "detected_error_count": int(
                    result.control_protocol.metadata["detected_error_count"],
                ),
                "corrected_error_count": int(
                    result.control_protocol.metadata["corrected_error_count"],
                ),
                "residual_error_count": int(
                    result.control_protocol.metadata["residual_error_count"],
                ),
                "total_control_effort": round(total_control_effort, 6),
            }


        def simulation_results_to_rows(
            results: Iterable[SimulationResult],
        ) -> list[dict[str, int | float | str]]:
            """Преобразует результаты моделирования в строки контрольной таблицы."""
            return [summarize_simulation_result(result) for result in results]
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
            "load_experiment_config",
            "messages_to_rows",
            "operator_estimates_to_rows",
            "procedure_plans_to_rows",
            "simulation_results_to_rows",
            "summarize_condition_estimate",
            "summarize_control_protocol",
            "summarize_error_protocol",
            "summarize_message",
            "summarize_operator_estimate",
            "summarize_procedure_plan",
            "summarize_simulation_result",
        ]
        ''',
    )

    write_text_file(
        TESTS_DIR / "test_stage8_protocol_simulator.py",
        '''
        """Тесты этапа 8: интегральный симулятор протоколов."""

        from __future__ import annotations

        import pytest

        from manual_coding_sim import ProtocolSimulator
        from manual_coding_sim.protocol_simulator import (
            ProtocolSimulatorConfig,
            SimulationResult,
            simulation_results_to_rows,
            summarize_simulation_result,
        )


        def test_protocol_simulator_imports() -> None:
            """Проверяет импортируемость интегрального симулятора."""
            simulator = ProtocolSimulator()

            assert isinstance(simulator.config, ProtocolSimulatorConfig)


        def test_simulate_once_returns_complete_result() -> None:
            """Проверяет полный прогон от M до протокола контроля K."""
            simulator = ProtocolSimulator()
            result = simulator.simulate_once(message_id="M_TEST")

            assert isinstance(result, SimulationResult)
            assert result.message.message_id == "M_TEST"
            assert result.procedure_plan.message_id == "M_TEST"
            assert result.operator_estimate.message_id == "M_TEST"
            assert result.condition_estimate.message_id == "M_TEST"
            assert result.error_protocol.message_id == "M_TEST"
            assert result.control_protocol.message_id == "M_TEST"


        def test_step_counts_are_consistent() -> None:
            """Проверяет согласованность числа шагов во всех артефактах."""
            result = ProtocolSimulator().simulate_once()
            step_count = len(result.message.elements)

            assert len(result.procedure_plan.steps) == step_count
            assert len(result.operator_estimate.step_estimates) == step_count
            assert len(result.condition_estimate.step_estimates) == step_count
            assert len(result.error_protocol.step_outcomes) == step_count
            assert len(result.control_protocol.step_outcomes) == step_count


        def test_scenario_identifiers_are_preserved() -> None:
            """Проверяет сохранение компонентов сценария A = {S, O, U, G, K}."""
            config = ProtocolSimulatorConfig(scenario_id="A_TEST")
            result = ProtocolSimulator(config).simulate_once()

            assert result.scenario.scenario_id == "A_TEST"
            assert result.scenario.coding_tool_id == "S_001"
            assert result.scenario.operator_id == "O_001"
            assert result.scenario.condition_id == "U_001"
            assert result.scenario.message_class_id == "G_001"
            assert result.scenario.control_procedure_id == "K_001"


        def test_reset_reproduces_first_result() -> None:
            """Проверяет воспроизводимость интегрального прогона после reset()."""
            simulator = ProtocolSimulator()
            first = summarize_simulation_result(simulator.simulate_once())
            simulator.reset()
            second = summarize_simulation_result(simulator.simulate_once())

            assert first == second


        def test_equal_seeds_reproduce_protocols() -> None:
            """Проверяет воспроизводимость при одинаковой конфигурации."""
            config = ProtocolSimulatorConfig(
                message_random_seed=11,
                error_random_seed=22,
                control_random_seed=33,
            )
            first = summarize_simulation_result(ProtocolSimulator(config).simulate_once())
            second = summarize_simulation_result(ProtocolSimulator(config).simulate_once())

            assert first == second


        def test_simulate_batch_returns_requested_count() -> None:
            """Проверяет пакетный запуск симулятора."""
            simulator = ProtocolSimulator()
            results = simulator.simulate_batch(3)

            assert len(results) == 3
            assert [result.message.message_id for result in results] == [
                "M_000001",
                "M_000002",
                "M_000003",
            ]


        def test_simulate_batch_rejects_non_positive_count() -> None:
            """Проверяет отклонение некорректного числа прогонов."""
            simulator = ProtocolSimulator()

            with pytest.raises(ValueError):
                simulator.simulate_batch(0)


        def test_summary_contains_expected_fields() -> None:
            """Проверяет контрольную сводку результата моделирования."""
            result = ProtocolSimulator().simulate_once(message_id="M_SUM")
            summary = summarize_simulation_result(result)

            assert summary["scenario_id"] == "A_001"
            assert summary["message_id"] == "M_SUM"
            assert summary["message_length"] == len(result.message.elements)
            assert summary["step_count"] == len(result.procedure_plan.steps)
            assert summary["condition_adjusted_time"] >= summary["nominal_time"]
            assert summary["residual_error_count"] <= summary["original_error_count"]


        def test_result_metadata_matches_summary() -> None:
            """Проверяет согласованность метаданных результата и сводки."""
            result = ProtocolSimulator().simulate_once()
            summary = summarize_simulation_result(result)

            assert result.metadata["message_id"] == summary["message_id"]
            assert result.metadata["step_count"] == summary["step_count"]
            assert result.metadata["condition_adjusted_time"] == summary[
                "condition_adjusted_time"
            ]


        def test_results_to_rows() -> None:
            """Проверяет преобразование результатов моделирования в таблицу."""
            results = ProtocolSimulator().simulate_batch(2)
            rows = simulation_results_to_rows(results)

            assert len(rows) == 2
            assert rows[0]["message_id"] == "M_000001"
            assert rows[1]["message_id"] == "M_000002"
            assert all("residual_error_count" in row for row in rows)
        ''',
    )

    created_files = [
        SRC_DIR / "protocol_simulator.py",
        SRC_DIR / "__init__.py",
        TESTS_DIR / "test_stage8_protocol_simulator.py",
    ]
    syntax_report = {
        str(path.relative_to(ROOT)): check_python_syntax(path)
        for path in created_files
    }

    report = {
        "stage": 8,
        "title": "Интегральный симулятор протоколов ProtocolSimulator",
        "created_files": [str(path.relative_to(ROOT)) for path in created_files],
        "syntax_report": syntax_report,
        "scientific_scope": (
            "Объединение моделей G, S, O, U, ErrorModel и K в воспроизводимую "
            "цепочку получения протокола моделирования ручного кодирования."
        ),
        "not_implemented_yet": [
            "декодирование D_h",
            "восстановленное сообщение M'",
            "извлечение признаков X_prior, X_fact, X_diag",
            "расчет итогового вектора качества q(A)",
        ],
    }

    report_path = REPORTS_DIR / "stage8_protocol_simulator_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("ЭТАП 8. ИНТЕГРАЛЬНЫЙ СИМУЛЯТОР ПРОТОКОЛОВ")
    print("=" * 60)
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
