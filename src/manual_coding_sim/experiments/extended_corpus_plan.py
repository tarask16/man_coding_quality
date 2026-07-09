"""Планирование расширенного корпуса сценариев главы 3.

Модуль формирует не копии существующих строк, а воспроизводимый план разных
сценариев применения ручных средств кодирования. План охватывает вариации
сложности сообщения, критичности, временных ограничений, состояния оператора,
условий выполнения и контроля. Сформированные параметры затем преобразуются в
стандартные CSV-артефакты главы 3.
"""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Sequence


@dataclass(frozen=True)
class ExtendedCorpusPlanConfig:
    """Параметры генерации расширенного плана сценариев.

    ``document_count`` задает число документов/протоколов. Для финального
    корпуса главы 4 рекомендуется использовать не менее 100 документов.
    """

    document_count: int = 150
    random_seed: int = 20260709
    protocols_per_scenario: int = 1

    def validate(self) -> None:
        """Проверить корректность параметров плана."""

        if self.document_count < 1:
            msg = "document_count должен быть положительным целым числом."
            raise ValueError(msg)
        if self.protocols_per_scenario < 1:
            msg = "protocols_per_scenario должен быть положительным целым числом."
            raise ValueError(msg)
        if self.document_count < self.protocols_per_scenario:
            msg = "document_count не должен быть меньше protocols_per_scenario."
            raise ValueError(msg)


@dataclass(frozen=True)
class ExtendedScenario:
    """Один сценарий применения ручного средства кодирования."""

    run_id: str
    scenario_id: str
    protocol_id: str
    alternative_id: str
    message_length: int
    message_complexity: int
    message_criticality: int
    digit_ratio: float
    control_marker_ratio: float
    procedure_steps: int
    operator_skill: int
    operator_fatigue: int
    operator_attention: int
    noise_level: int
    time_pressure: int
    control_intensity: int
    coding_tool_type: str

    @property
    def condition_profile(self) -> str:
        """Вернуть категорию условий выполнения сценария."""

        if self.noise_level >= 3 or self.time_pressure >= 3:
            return "hard"
        if self.noise_level >= 2 or self.time_pressure >= 2:
            return "moderate"
        return "normal"


class ExtendedCorpusPlanBuilder:
    """Строит воспроизводимый план расширенного корпуса сценариев."""

    _MESSAGE_LENGTHS = (16, 24, 32, 48, 64, 96, 128)
    _RATIO_DIGITS = (0.05, 0.10, 0.18, 0.25, 0.33, 0.42, 0.55, 0.70)
    _RATIO_CONTROLS = (0.00, 0.03, 0.06, 0.10, 0.15, 0.22)
    _TOOL_TYPES = ("codebook", "table", "mixed", "mnemonic")

    def __init__(self, config: ExtendedCorpusPlanConfig | None = None) -> None:
        """Создать построитель расширенного плана."""

        self.config = config or ExtendedCorpusPlanConfig()
        self.config.validate()

    def build(self) -> list[ExtendedScenario]:
        """Сформировать список сценариев без искусственного дублирования строк."""

        rng = random.Random(self.config.random_seed)
        scenarios: list[ExtendedScenario] = []
        for index in range(self.config.document_count):
            scenario_number = index // self.config.protocols_per_scenario
            protocol_number = index % self.config.protocols_per_scenario
            scenarios.append(
                self._build_scenario(
                    index=index,
                    scenario_number=scenario_number,
                    protocol_number=protocol_number,
                    rng=rng,
                )
            )
        return scenarios

    def _build_scenario(
        self,
        index: int,
        scenario_number: int,
        protocol_number: int,
        rng: random.Random,
    ) -> ExtendedScenario:
        """Сформировать один сценарий с детерминированной вариативностью."""

        complexity = 1 + index % 5
        criticality = 1 + (index // 5) % 5
        fatigue = (index // 7) % 5
        noise = (index // 11) % 4
        time_pressure = (index // 13) % 4
        operator_skill = 1 + (index // 17) % 5
        control_intensity = (index // 19) % 4
        message_length = self._MESSAGE_LENGTHS[index % len(self._MESSAGE_LENGTHS)]
        digit_ratio = self._jitter_ratio(
            self._RATIO_DIGITS[index % len(self._RATIO_DIGITS)],
            rng,
        )
        control_marker_ratio = self._jitter_ratio(
            self._RATIO_CONTROLS[(index // 3) % len(self._RATIO_CONTROLS)],
            rng,
        )
        procedure_steps = 3 + ((complexity + criticality + index) % 8)
        operator_attention = max(1, min(5, operator_skill + 1 - fatigue // 2))
        tool_type = self._TOOL_TYPES[(index // 23) % len(self._TOOL_TYPES)]

        return ExtendedScenario(
            run_id="extended_run_001",
            scenario_id=f"scn_{scenario_number:04d}",
            protocol_id=f"prt_{scenario_number:04d}_{protocol_number:02d}",
            alternative_id=f"alt_{scenario_number % 12:03d}",
            message_length=message_length,
            message_complexity=complexity,
            message_criticality=criticality,
            digit_ratio=digit_ratio,
            control_marker_ratio=control_marker_ratio,
            procedure_steps=procedure_steps,
            operator_skill=operator_skill,
            operator_fatigue=fatigue,
            operator_attention=operator_attention,
            noise_level=noise,
            time_pressure=time_pressure,
            control_intensity=control_intensity,
            coding_tool_type=tool_type,
        )

    def _jitter_ratio(self, base_value: float, rng: random.Random) -> float:
        """Добавить малую воспроизводимую вариацию к долевому параметру."""

        value = base_value + rng.uniform(-0.01, 0.01)
        return round(max(0.0, min(0.95, value)), 4)

    @staticmethod
    def unique_scenario_count(scenarios: Sequence[ExtendedScenario]) -> int:
        """Посчитать число уникальных идентификаторов сценариев."""

        return len({scenario.scenario_id for scenario in scenarios})

    @staticmethod
    def unique_protocol_count(scenarios: Sequence[ExtendedScenario]) -> int:
        """Посчитать число уникальных идентификаторов протоколов."""

        return len({scenario.protocol_id for scenario in scenarios})
