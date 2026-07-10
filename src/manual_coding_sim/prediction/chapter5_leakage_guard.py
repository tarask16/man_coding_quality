"""Каркас защиты главы 5 от утечки фактических и целевых признаков.

На этапе 1 модуль предоставляет минимальную проверку имен колонок. Расширенный
отчет об утечке и интеграция с runner-ом будут выполнены отдельным этапом.
"""

from __future__ import annotations

from dataclasses import dataclass
import fnmatch
from typing import Iterable


DEFAULT_FORBIDDEN_COLUMN_PATTERNS = (
    "fact_*",
    "actual_*",
    "target_*",
    "q_acc",
    "q_time",
    "q_effort",
    "q_res",
    "q_rep",
    "q_fit",
    "integral_quality",
    "fact_duration_sec",
    "fact_error_count",
    "fact_reject_count",
    "fact_success",
)


@dataclass(frozen=True)
class Chapter5LeakageCheckResult:
    """Результат первичной проверки колонок на методическую утечку."""

    is_safe: bool
    forbidden_columns: tuple[str, ...]


class Chapter5LeakageError(ValueError):
    """Ошибка методической утечки в априорном расчете главы 5."""


class Chapter5LeakageGuard:
    """Проверяет, что в априорный расчет не попали фактические признаки."""

    def __init__(
        self,
        forbidden_patterns: Iterable[str] = DEFAULT_FORBIDDEN_COLUMN_PATTERNS,
    ) -> None:
        """Создать объект проверки с заданным перечнем запрещенных шаблонов."""

        self.forbidden_patterns = tuple(forbidden_patterns)

    def check_columns(self, columns: Iterable[str]) -> Chapter5LeakageCheckResult:
        """Проверить имена колонок на совпадение с запрещенными шаблонами."""

        forbidden_columns = tuple(
            column
            for column in columns
            if any(fnmatch.fnmatch(column, pattern) for pattern in self.forbidden_patterns)
        )
        return Chapter5LeakageCheckResult(
            is_safe=not forbidden_columns,
            forbidden_columns=forbidden_columns,
        )

    def require_safe_columns(self, columns: Iterable[str]) -> None:
        """Прервать выполнение, если найдены фактические или целевые признаки."""

        result = self.check_columns(columns)
        if not result.is_safe:
            joined_columns = ", ".join(result.forbidden_columns)
            msg = (
                "Обнаружена методическая утечка: в априорный расчет попали "
                f"фактические или целевые признаки: {joined_columns}."
            )
            raise Chapter5LeakageError(msg)
