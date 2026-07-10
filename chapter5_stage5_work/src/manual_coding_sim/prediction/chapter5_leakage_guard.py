"""Защита главы 5 от утечки фактических и целевых признаков.

Модуль выполняет методический контроль входов априорного прогноза. В главе 5
допустимы только идентификаторы сценария, служебные идентификаторы запуска и
признаки, известные до фактического выполнения процедуры. Фактические признаки
и целевые показатели качества должны использоваться только в главе 6.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import fnmatch
import json
from pathlib import Path
from typing import Iterable, Mapping

import pandas as pd


DEFAULT_FORBIDDEN_COLUMN_PATTERNS = (
    "fact_*",
    "actual_*",
    "target_*",
    "targets_*",
    "*_fact",
    "*_actual",
    "*_target",
    "q_acc",
    "q_time",
    "q_effort",
    "q_res",
    "q_rep",
    "q_fit",
    "integral_quality",
    "quality_class",
    "fact_duration_sec",
    "fact_error_count",
    "fact_reject_count",
    "fact_success",
)

DEFAULT_ALLOWED_SERVICE_COLUMNS = (
    "scenario_id",
    "protocol_id",
    "run_id",
    "alternative_id",
    "document_index",
    "random_seed",
)


@dataclass(frozen=True)
class Chapter5LeakageCheckResult:
    """Результат проверки колонок на методическую утечку."""

    is_safe: bool
    forbidden_columns: tuple[str, ...]
    checked_column_count: int = 0
    prior_columns: tuple[str, ...] = ()
    service_columns: tuple[str, ...] = ()
    non_prior_columns: tuple[str, ...] = ()
    forbidden_patterns: tuple[str, ...] = ()
    source_name: str = "input"
    matched_patterns_by_column: Mapping[str, tuple[str, ...]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        """Преобразовать результат проверки в JSON-совместимый словарь."""

        payload = asdict(self)
        payload["matched_patterns_by_column"] = {
            column: list(patterns)
            for column, patterns in self.matched_patterns_by_column.items()
        }
        return payload


class Chapter5LeakageError(ValueError):
    """Ошибка методической утечки в априорном расчете главы 5."""


class Chapter5LeakageGuard:
    """Проверяет, что в априорный расчет не попали фактические признаки."""

    def __init__(
        self,
        forbidden_patterns: Iterable[str] = DEFAULT_FORBIDDEN_COLUMN_PATTERNS,
        allowed_service_columns: Iterable[str] = DEFAULT_ALLOWED_SERVICE_COLUMNS,
    ) -> None:
        """Создать объект проверки с заданными списками правил.

        Шаблоны проверяются без учета регистра, чтобы заблокировать варианты вида
        ``Fact_Error_Count`` или ``TARGET_quality``.
        """

        self.forbidden_patterns = tuple(pattern.lower() for pattern in forbidden_patterns)
        self.allowed_service_columns = tuple(column.lower() for column in allowed_service_columns)

    def check_columns(
        self,
        columns: Iterable[str],
        *,
        source_name: str = "input",
    ) -> Chapter5LeakageCheckResult:
        """Проверить имена колонок на совпадение с запрещенными шаблонами."""

        normalized_columns = tuple(str(column) for column in columns)
        matched_patterns_by_column: dict[str, tuple[str, ...]] = {}
        forbidden_columns: list[str] = []
        prior_columns: list[str] = []
        service_columns: list[str] = []
        non_prior_columns: list[str] = []

        for column in normalized_columns:
            column_lower = column.lower()
            matched_patterns = self._matched_forbidden_patterns(column_lower)
            if matched_patterns:
                forbidden_columns.append(column)
                matched_patterns_by_column[column] = matched_patterns
            elif column_lower.startswith("prior_"):
                prior_columns.append(column)
            elif column_lower in self.allowed_service_columns:
                service_columns.append(column)
            else:
                non_prior_columns.append(column)

        return Chapter5LeakageCheckResult(
            is_safe=not forbidden_columns,
            forbidden_columns=tuple(forbidden_columns),
            checked_column_count=len(normalized_columns),
            prior_columns=tuple(prior_columns),
            service_columns=tuple(service_columns),
            non_prior_columns=tuple(non_prior_columns),
            forbidden_patterns=self.forbidden_patterns,
            source_name=source_name,
            matched_patterns_by_column=matched_patterns_by_column,
        )

    def check_dataframe(
        self,
        df: pd.DataFrame,
        *,
        source_name: str = "input",
    ) -> Chapter5LeakageCheckResult:
        """Проверить таблицу pandas по набору ее колонок."""

        return self.check_columns(df.columns, source_name=source_name)

    def require_safe_columns(
        self,
        columns: Iterable[str],
        *,
        source_name: str = "input",
    ) -> Chapter5LeakageCheckResult:
        """Прервать выполнение, если найдены фактические или целевые признаки."""

        result = self.check_columns(columns, source_name=source_name)
        self._raise_if_unsafe(result)
        return result

    def require_safe_dataframe(
        self,
        df: pd.DataFrame,
        *,
        source_name: str = "input",
    ) -> Chapter5LeakageCheckResult:
        """Проверить таблицу pandas и прервать выполнение при утечке."""

        result = self.check_dataframe(df, source_name=source_name)
        self._raise_if_unsafe(result)
        return result

    @staticmethod
    def save_json_report(path: Path, result: Chapter5LeakageCheckResult) -> None:
        """Сохранить отчет проверки утечки в JSON-файл."""

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(result.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _matched_forbidden_patterns(self, column_lower: str) -> tuple[str, ...]:
        """Вернуть запрещенные шаблоны, которым соответствует колонка."""

        return tuple(
            pattern
            for pattern in self.forbidden_patterns
            if fnmatch.fnmatch(column_lower, pattern)
        )

    @staticmethod
    def _raise_if_unsafe(result: Chapter5LeakageCheckResult) -> None:
        """Сформировать русскоязычную ошибку при обнаруженной утечке."""

        if result.is_safe:
            return
        joined_columns = ", ".join(result.forbidden_columns)
        msg = (
            "Обнаружена методическая утечка: в априорный расчет попали "
            f"фактические или целевые признаки: {joined_columns}. "
            f"Источник: {result.source_name}."
        )
        raise Chapter5LeakageError(msg)
