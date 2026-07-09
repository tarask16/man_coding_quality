"""
Создание базовой структуры Python-пакета исследовательского симулятора главы 3.

Скрипт выполняется из корня проекта manual_coding_quality и создает минимальный
каркас пакета manual_coding_sim без реализации предметных моделей. На данном
этапе фиксируются только базовые типы данных, загрузка конфигурации и проверка
импортируемости пакета.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from textwrap import dedent


@dataclass(frozen=True)
class CreatedFile:
    """Описание файла, созданного или проверенного скриптом."""

    path: Path
    status: str


PROJECT_DIR = Path.cwd()
PACKAGE_DIR = PROJECT_DIR / "src" / "manual_coding_sim"
TESTS_DIR = PROJECT_DIR / "tests"
REPORTS_DIR = PROJECT_DIR / "reports" / "chapter3"
CONFIGS_DIR = PROJECT_DIR / "configs"

FILES: dict[Path, str] = {
    PACKAGE_DIR / "__init__.py": dedent(
        '''\
        """
        Исследовательский симулятор процессов ручного кодирования.

        Пакет предназначен для программной реализации главы 3 диссертации:
        компьютерного моделирования процессов ручного кодирования и декодирования
        с последующим формированием априорных, фактических и диагностических
        признаков для оценки качества ручных средств кодирования информации.
        """

        from manual_coding_sim.config import ExperimentConfig, load_experiment_config
        from manual_coding_sim.types import (
            FeatureGroup,
            GeneratedMessage,
            MessageElement,
            QualityVector,
            ScenarioParameters,
        )

        __all__ = [
            "ExperimentConfig",
            "FeatureGroup",
            "GeneratedMessage",
            "MessageElement",
            "QualityVector",
            "ScenarioParameters",
            "load_experiment_config",
        ]

        __version__ = "0.1.0"
        '''
    ),
    PACKAGE_DIR / "config.py": dedent(
        '''\
        """
        Загрузка и проверка конфигурации вычислительного эксперимента.

        В терминах диссертации конфигурация задает параметры воспроизводимого
        моделирования сценариев A = {S, O, U, G, K}, где S — средство ручного
        кодирования, O — оператор, U — условия применения, G — класс сообщений,
        K — контрольные процедуры.
        """

        from __future__ import annotations

        from dataclasses import dataclass
        from pathlib import Path
        from typing import Any

        import yaml


        @dataclass(frozen=True)
        class ExperimentConfig:
            """Минимальная конфигурация вычислительного эксперимента главы 3."""

            name: str
            random_seed: int


        def load_experiment_config(config_path: str | Path) -> ExperimentConfig:
            """
            Загрузить базовую конфигурацию эксперимента из YAML-файла.

            Параметр random_seed фиксирует начальное состояние генераторов
            случайных чисел и используется для воспроизводимого формирования
            протоколов ручного кодирования.
            """

            path = Path(config_path)
            if not path.exists():
                raise FileNotFoundError(f"Файл конфигурации не найден: {path}")

            with path.open("r", encoding="utf-8") as file:
                raw_config: dict[str, Any] = yaml.safe_load(file) or {}

            experiment_section = raw_config.get("experiment")
            if not isinstance(experiment_section, dict):
                raise ValueError("В конфигурации отсутствует раздел 'experiment'.")

            name = experiment_section.get("name")
            random_seed = experiment_section.get("random_seed")

            if not isinstance(name, str) or not name.strip():
                raise ValueError("Поле 'experiment.name' должно быть непустой строкой.")

            if not isinstance(random_seed, int):
                raise ValueError("Поле 'experiment.random_seed' должно быть целым числом.")

            return ExperimentConfig(name=name, random_seed=random_seed)
        '''
    ),
    PACKAGE_DIR / "types.py": dedent(
        '''\
        """
        Базовые типы данных для исследовательского симулятора.

        В этом модуле задаются только структуры данных. Алгоритмы генерации
        сообщений, моделирования ошибок, контроля и расчета показателей качества
        будут реализованы в последующих задачах.
        """

        from __future__ import annotations

        from dataclasses import dataclass
        from enum import StrEnum


        class FeatureGroup(StrEnum):
            """Группы признаков, используемые в главах 3–6 диссертации."""

            PRIOR = "prior"
            FACT = "fact"
            DIAGNOSTIC = "diagnostic"


        @dataclass(frozen=True)
        class ScenarioParameters:
            """Параметры сценария A = {S, O, U, G, K}."""

            scenario_id: str
            coding_tool_id: str
            operator_id: str
            condition_id: str
            message_class_id: str
            control_procedure_id: str


        @dataclass(frozen=True)
        class MessageElement:
            """Элемент исходного сообщения M, подлежащий ручному кодированию."""

            value: str
            element_type: str
            position: int
            criticality: float


        @dataclass(frozen=True)
        class GeneratedMessage:
            """Сгенерированное исходное сообщение M."""

            message_id: str
            message_class_id: str
            elements: tuple[MessageElement, ...]


        @dataclass(frozen=True)
        class QualityVector:
            """Вектор частных показателей качества ручного кодирования."""

            q_acc: float
            q_time: float
            q_effort: float
            q_res: float
            q_rep: float
            q_fit: float
        '''
    ),
    TESTS_DIR / "test_stage1_package_structure.py": dedent(
        '''\
        """
        Проверки базовой структуры пакета исследовательского симулятора.

        Тесты подтверждают, что каркас пакета импортируется, конфигурация
        эксперимента загружается, а базовые типы данных создаются без ошибок.
        """

        from pathlib import Path

        from manual_coding_sim import (
            FeatureGroup,
            GeneratedMessage,
            MessageElement,
            QualityVector,
            ScenarioParameters,
            load_experiment_config,
        )


        def test_load_experiment_config(tmp_path: Path) -> None:
            """Проверить загрузку базовой конфигурации эксперимента."""

            config_path = tmp_path / "base_experiment.yaml"
            config_path.write_text(
                "experiment:\n"
                "  name: chapter3_base_environment_check\n"
                "  random_seed: 42\n",
                encoding="utf-8",
            )

            config = load_experiment_config(config_path)

            assert config.name == "chapter3_base_environment_check"
            assert config.random_seed == 42


        def test_scenario_parameters_describe_full_scenario() -> None:
            """Проверить структуру сценария A = {S, O, U, G, K}."""

            scenario = ScenarioParameters(
                scenario_id="scenario_001",
                coding_tool_id="S_basic",
                operator_id="O_regular",
                condition_id="U_normal",
                message_class_id="G_short",
                control_procedure_id="K_double_check",
            )

            assert scenario.scenario_id == "scenario_001"
            assert scenario.coding_tool_id.startswith("S_")
            assert scenario.operator_id.startswith("O_")
            assert scenario.condition_id.startswith("U_")
            assert scenario.message_class_id.startswith("G_")
            assert scenario.control_procedure_id.startswith("K_")


        def test_generated_message_contains_elements() -> None:
            """Проверить представление исходного сообщения M."""

            element = MessageElement(
                value="A1",
                element_type="symbol",
                position=0,
                criticality=0.75,
            )
            message = GeneratedMessage(
                message_id="M_001",
                message_class_id="G_short",
                elements=(element,),
            )

            assert message.elements[0].value == "A1"
            assert message.elements[0].criticality == 0.75


        def test_quality_vector_contains_partial_quality_indicators() -> None:
            """Проверить вектор частных показателей качества."""

            quality = QualityVector(
                q_acc=0.95,
                q_time=0.80,
                q_effort=0.70,
                q_res=0.90,
                q_rep=0.85,
                q_fit=0.88,
            )

            assert quality.q_acc > quality.q_effort
            assert FeatureGroup.PRIOR == "prior"
        '''
    ),
}


BASE_CONFIG_TEXT = dedent(
    """\
    experiment:
      name: chapter3_base_environment_check
      random_seed: 42
    """
)


def write_file_if_needed(path: Path, content: str) -> CreatedFile:
    """Создать файл, если он отсутствует, не перезаписывая ручные изменения."""

    if path.exists():
        return CreatedFile(path=path, status="exists")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="\n")
    return CreatedFile(path=path, status="created")


def validate_stage1() -> dict[str, bool]:
    """Проверить наличие файлов базовой структуры пакета."""

    required_paths = [
        PACKAGE_DIR / "__init__.py",
        PACKAGE_DIR / "config.py",
        PACKAGE_DIR / "types.py",
        TESTS_DIR / "test_stage1_package_structure.py",
        CONFIGS_DIR / "base_experiment.yaml",
        PROJECT_DIR / "pyproject.toml",
    ]
    return {str(path.relative_to(PROJECT_DIR)): path.exists() for path in required_paths}


def main() -> None:
    """Создать каркас пакета и сформировать отчет по этапу 1."""

    created_files = []

    for directory in (PACKAGE_DIR, TESTS_DIR, REPORTS_DIR, CONFIGS_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    for path, content in FILES.items():
        created_files.append(write_file_if_needed(path, content))

    created_files.append(
        write_file_if_needed(CONFIGS_DIR / "base_experiment.yaml", BASE_CONFIG_TEXT)
    )

    validation = validate_stage1()
    report = {
        "stage": "Этап 1",
        "title": "Базовая структура Python-пакета исследовательского симулятора",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "project_dir": str(PROJECT_DIR),
        "files": [
            {
                "path": str(file.path.relative_to(PROJECT_DIR)),
                "status": file.status,
            }
            for file in created_files
        ],
        "validation": validation,
        "is_ready_for_tests": all(validation.values()),
    }

    report_path = REPORTS_DIR / "stage1_package_structure_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("Этап 1: базовая структура Python-пакета")
    print("=" * 56)
    for relative_path, is_ok in validation.items():
        status = "OK" if is_ok else "ERROR"
        print(f"[{status}] {relative_path}")

    if all(validation.values()):
        print("\nСтруктура пакета создана. Теперь выполните: python -m pytest")
    else:
        print("\nОбнаружены отсутствующие файлы. Проверьте отчет:")
        print(report_path)


if __name__ == "__main__":
    main()
