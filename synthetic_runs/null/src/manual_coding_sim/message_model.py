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
