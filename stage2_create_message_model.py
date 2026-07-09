"""
Создание модели исходного сообщения для исследовательского симулятора.

Скрипт относится к этапу 2 программной реализации главы 3 диссертации.
Он создает модуль MessageModel, который формирует исходное сообщение M
как последовательность элементов до выполнения ручного кодирования E_h.
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
    """Создает модуль MessageModel, тесты и отчет этапа 2."""
    SRC_DIR.mkdir(parents=True, exist_ok=True)
    TESTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    write_text_file(
        SRC_DIR / "message_model.py",
        '''
        """Модель генерации исходных сообщений M.

        Модуль относится к главе 3 диссертации и реализует начальный блок
        вычислительного моделирования: формирование исходного сообщения M
        до применения ручного средства кодирования E_h.
        """

        from __future__ import annotations

        from dataclasses import dataclass, field
        import random
        from typing import Iterable

        from manual_coding_sim.types import GeneratedMessage, MessageElement


        @dataclass(frozen=True)
        class MessageGenerationConfig:
            """
            Конфигурация генерации исходного сообщения M.

            Параметры класса задают класс сообщений G: допустимую длину,
            алфавит элементов, типы элементов и базовую критичность каждого
            типа элемента. Эти параметры являются априорными: они известны
            до выполнения ручного кодирования и не используют фактический
            результат декодирования M'.
            """

            message_class_id: str = "G_001"
            min_length: int = 5
            max_length: int = 12
            alphabet: tuple[str, ...] = (
                "A",
                "B",
                "C",
                "D",
                "E",
                "F",
                "G",
                "H",
                "I",
                "J",
            )
            element_types: tuple[str, ...] = ("symbol", "digit", "service")
            criticality_by_type: dict[str, float] = field(
                default_factory=lambda: {
                    "symbol": 0.60,
                    "digit": 0.75,
                    "service": 0.40,
                }
            )

            def validate(self) -> None:
                """Проверяет корректность параметров класса сообщений G."""
                if not self.message_class_id:
                    raise ValueError("Идентификатор класса сообщений G не задан.")

                if self.min_length <= 0:
                    raise ValueError("Минимальная длина сообщения должна быть положительной.")

                if self.max_length < self.min_length:
                    raise ValueError(
                        "Максимальная длина сообщения не может быть меньше минимальной."
                    )

                if not self.alphabet:
                    raise ValueError("Алфавит элементов сообщения не должен быть пустым.")

                if not self.element_types:
                    raise ValueError("Список типов элементов сообщения не должен быть пустым.")

                for element_type in self.element_types:
                    if element_type not in self.criticality_by_type:
                        raise ValueError(
                            "Для каждого типа элемента должна быть задана базовая "
                            f"критичность: {element_type}."
                        )

                for element_type, criticality in self.criticality_by_type.items():
                    if not 0.0 <= criticality <= 1.0:
                        raise ValueError(
                            "Критичность элемента должна находиться в диапазоне [0; 1]: "
                            f"{element_type}={criticality}."
                        )


        class MessageModel:
            """
            Генератор исходных сообщений M.

            Модель формирует сообщение M как упорядоченную последовательность
            элементов m_j. Каждый элемент содержит значение, тип, позицию
            и критичность. На данном этапе не моделируются ручное кодирование,
            ошибки оператора, контрольные процедуры и восстановленное сообщение M'.
            """

            def __init__(
                self,
                config: MessageGenerationConfig | None = None,
                random_seed: int = 42,
            ) -> None:
                """Инициализирует модель сообщений с фиксированным random_seed."""
                self.config = config or MessageGenerationConfig()
                self.config.validate()
                self.random_seed = random_seed
                self._rng = random.Random(random_seed)
                self._generated_count = 0

            def generate_message(self, message_id: str | None = None) -> GeneratedMessage:
                """
                Формирует одно исходное сообщение M.

                Если идентификатор сообщения не передан, он создается автоматически
                в виде M_000001, M_000002 и так далее. Автоматическая нумерация
                нужна для трассируемости протоколов моделирования в следующих этапах.
                """
                self._generated_count += 1
                final_message_id = message_id or self._make_message_id(self._generated_count)
                message_length = self._rng.randint(
                    self.config.min_length,
                    self.config.max_length,
                )

                elements = tuple(
                    self._generate_element(position=position)
                    for position in range(message_length)
                )

                return GeneratedMessage(
                    message_id=final_message_id,
                    elements=elements,
                    metadata={
                        "message_class_id": self.config.message_class_id,
                        "message_length": message_length,
                        "random_seed": self.random_seed,
                    },
                )

            def generate_batch(self, count: int) -> tuple[GeneratedMessage, ...]:
                """Формирует пакет исходных сообщений M для серии сценариев."""
                if count <= 0:
                    raise ValueError("Число сообщений в пакете должно быть положительным.")

                return tuple(self.generate_message() for _ in range(count))

            def reset(self) -> None:
                """
                Возвращает генератор к начальному состоянию.

                После сброса при том же random_seed модель снова формирует ту же
                последовательность сообщений. Это требуется для воспроизводимости
                вычислительного эксперимента главы 3.
                """
                self._rng = random.Random(self.random_seed)
                self._generated_count = 0

            def _generate_element(self, position: int) -> MessageElement:
                """Формирует один элемент m_j исходного сообщения M."""
                element_type = self._rng.choice(self.config.element_types)
                value = self._generate_value(element_type)
                criticality = self._generate_criticality(element_type)

                return MessageElement(
                    value=value,
                    element_type=element_type,
                    position=position,
                    criticality=criticality,
                )

            def _generate_value(self, element_type: str) -> str:
                """Формирует значение элемента с учетом его типа."""
                if element_type == "digit":
                    return str(self._rng.randint(0, 9))

                if element_type == "service":
                    service_values = ("SEP", "CTRL", "END")
                    return self._rng.choice(service_values)

                return self._rng.choice(self.config.alphabet)

            def _generate_criticality(self, element_type: str) -> float:
                """Формирует критичность элемента в диапазоне [0; 1]."""
                base_criticality = self.config.criticality_by_type[element_type]
                variation = self._rng.uniform(-0.05, 0.05)
                criticality = min(1.0, max(0.0, base_criticality + variation))
                return round(criticality, 4)

            @staticmethod
            def _make_message_id(index: int) -> str:
                """Создает стабильный идентификатор сообщения."""
                return f"M_{index:06d}"


        def summarize_message(message: GeneratedMessage) -> dict[str, int | float | str]:
            """
            Возвращает краткое описание сообщения M.

            Сводка используется только для контроля корректности генерации и не
            заменяет дальнейшее извлечение априорных, фактических и диагностических
            признаков.
            """
            if not message.elements:
                raise ValueError("Сообщение не содержит элементов.")

            mean_criticality = sum(
                element.criticality for element in message.elements
            ) / len(message.elements)

            return {
                "message_id": message.message_id,
                "message_length": len(message.elements),
                "mean_criticality": round(mean_criticality, 4),
                "message_class_id": str(message.metadata.get("message_class_id", "")),
            }


        def messages_to_rows(
            messages: Iterable[GeneratedMessage],
        ) -> list[dict[str, int | float | str]]:
            """Преобразует сообщения M в строки для контрольной таблицы."""
            return [summarize_message(message) for message in messages]
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

        from manual_coding_sim.config import load_experiment_config
        from manual_coding_sim.message_model import (
            MessageGenerationConfig,
            MessageModel,
            messages_to_rows,
            summarize_message,
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
            "FeatureGroup",
            "GeneratedMessage",
            "MessageElement",
            "MessageGenerationConfig",
            "MessageModel",
            "QualityVector",
            "ScenarioParameters",
            "load_experiment_config",
            "messages_to_rows",
            "summarize_message",
        ]
        ''',
    )

    write_text_file(
        TESTS_DIR / "test_stage2_message_model.py",
        '''
        """Тесты модели генерации исходного сообщения M."""

        import pytest

        from manual_coding_sim.message_model import (
            MessageGenerationConfig,
            MessageModel,
            messages_to_rows,
            summarize_message,
        )


        def test_message_model_generates_message_with_valid_structure() -> None:
            """Модель должна формировать сообщение M с корректными элементами."""
            config = MessageGenerationConfig(min_length=4, max_length=4)
            model = MessageModel(config=config, random_seed=42)

            message = model.generate_message(message_id="M_TEST")

            assert message.message_id == "M_TEST"
            assert len(message.elements) == 4
            assert message.metadata["message_class_id"] == "G_001"
            assert message.metadata["message_length"] == 4

            for expected_position, element in enumerate(message.elements):
                assert element.position == expected_position
                assert element.element_type in config.element_types
                assert 0.0 <= element.criticality <= 1.0
                assert element.value


        def test_message_model_is_reproducible_with_same_seed() -> None:
            """Одинаковый random_seed должен давать одинаковое сообщение M."""
            config = MessageGenerationConfig(min_length=6, max_length=6)
            first_model = MessageModel(config=config, random_seed=123)
            second_model = MessageModel(config=config, random_seed=123)

            first_message = first_model.generate_message()
            second_message = second_model.generate_message()

            assert first_message == second_message


        def test_message_model_reset_restores_initial_sequence() -> None:
            """Сброс модели должен восстанавливать начальную последовательность M."""
            model = MessageModel(random_seed=777)

            first_message = model.generate_message()
            model.generate_message()
            model.reset()
            repeated_first_message = model.generate_message()

            assert first_message == repeated_first_message


        def test_generate_batch_creates_unique_message_ids() -> None:
            """Пакет сообщений должен иметь стабильные уникальные идентификаторы."""
            model = MessageModel(random_seed=42)

            messages = model.generate_batch(3)
            message_ids = [message.message_id for message in messages]

            assert message_ids == ["M_000001", "M_000002", "M_000003"]


        def test_generate_batch_rejects_non_positive_count() -> None:
            """Число сообщений в пакете должно быть положительным."""
            model = MessageModel(random_seed=42)

            with pytest.raises(ValueError, match="положительным"):
                model.generate_batch(0)


        def test_message_summary_contains_control_fields() -> None:
            """Сводка сообщения должна содержать контрольные поля."""
            config = MessageGenerationConfig(min_length=5, max_length=5)
            model = MessageModel(config=config, random_seed=42)
            message = model.generate_message(message_id="M_SUMMARY")

            summary = summarize_message(message)
            rows = messages_to_rows((message,))

            assert summary["message_id"] == "M_SUMMARY"
            assert summary["message_length"] == 5
            assert 0.0 <= summary["mean_criticality"] <= 1.0
            assert rows == [summary]


        def test_message_generation_config_validation() -> None:
            """Конфигурация класса сообщений G должна проверять некорректные значения."""
            invalid_config = MessageGenerationConfig(min_length=10, max_length=5)

            with pytest.raises(ValueError, match="Максимальная длина"):
                MessageModel(config=invalid_config, random_seed=42)
        ''',
    )

    python_files = [
        SRC_DIR / "__init__.py",
        SRC_DIR / "message_model.py",
        TESTS_DIR / "test_stage2_message_model.py",
    ]
    syntax_report = {str(path.relative_to(ROOT)): check_python_syntax(path) for path in python_files}

    report = {
        "stage": "stage_2_message_model",
        "status": "OK" if all(item["status"] == "OK" for item in syntax_report.values()) else "ERROR",
        "created_or_rewritten_files": [str(path.relative_to(ROOT)) for path in python_files],
        "scientific_scope": (
            "Реализована только модель генерации исходного сообщения M. "
            "Средство ручного кодирования S, оператор O, условия U, контроль K, "
            "ручное кодирование E_h и декодирование D_h на этом этапе не реализуются."
        ),
        "syntax_report": syntax_report,
        "next_command": "python -m pytest",
    }

    report_path = REPORTS_DIR / "stage2_message_model_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("ЭТАП 2. МОДЕЛЬ ИСХОДНОГО СООБЩЕНИЯ M")
    print("=" * 56)
    for file_path in report["created_or_rewritten_files"]:
        print(f"[OK] {file_path}")
    print(f"[OK] Отчет: {report_path}")
    print("\nТеперь выполните команду:")
    print("python -m pytest")


if __name__ == "__main__":
    main()
