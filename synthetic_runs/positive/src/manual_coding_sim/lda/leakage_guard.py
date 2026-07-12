"""Защита априорной LDA-модели от утечки фактических данных.

``LDA_prior`` должна обучаться только на априорных признаках ``X_prior``.
Фактические признаки, диагностические результаты качества и интегральные
целевые показатели запрещены как вход основной модели. Этот модуль содержит
жесткие проверки, которые должны выполняться до построения LDA-корпуса.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


class LdaLeakageError(ValueError):
    """Ошибка нарушения априорности входных данных для ``LDA_prior``."""


@dataclass(frozen=True)
class LeakageCheckResult:
    """Результат проверки входов ``LDA_prior`` на утечку данных."""

    is_safe: bool
    checked_sources: tuple[str, ...]
    checked_columns: tuple[str, ...]
    forbidden_sources: tuple[str, ...]
    forbidden_columns: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        """Преобразовать результат проверки в JSON-совместимый словарь."""

        return {
            "is_safe": self.is_safe,
            "checked_sources": list(self.checked_sources),
            "checked_columns": list(self.checked_columns),
            "forbidden_sources": list(self.forbidden_sources),
            "forbidden_columns": list(self.forbidden_columns),
        }


class LeakageGuard:
    """Проверяет, что ``LDA_prior`` не использует фактические данные.

    Проверяются как имена файлов-источников, так и имена колонок. Это нужно,
    чтобы ошибка не могла быть скрыта переименованием пайплайна или случайным
    объединением таблиц перед обучением модели.
    """

    FORBIDDEN_SOURCE_NAMES = frozenset(
        {
            "fact_features.csv",
            "quality_targets.csv",
            "all_features.csv",
        }
    )
    FORBIDDEN_COLUMN_NAMES = frozenset(
        {
            "q_acc",
            "q_time",
            "q_effort",
            "q_res",
            "q_rep",
            "q_fit",
            "integral_quality",
            "quality_integral",
            "quality_score",
            "target",
        }
    )
    FORBIDDEN_COLUMN_PREFIXES = (
        "fact_",
        "target_",
        "quality_",
    )

    def check_prior_input(
        self,
        source_paths: Sequence[str | Path],
        columns: Iterable[str],
    ) -> LeakageCheckResult:
        """Проверить источники и колонки для ``LDA_prior`` без исключения.

        Метод возвращает полный результат проверки. Для строгого режима следует
        использовать ``validate_prior_input``.
        """

        checked_sources = self._normalize_source_names(source_paths)
        checked_columns = self._normalize_columns(columns)
        forbidden_sources = self._find_forbidden_sources(checked_sources)
        forbidden_columns = self._find_forbidden_columns(checked_columns)

        return LeakageCheckResult(
            is_safe=not forbidden_sources and not forbidden_columns,
            checked_sources=checked_sources,
            checked_columns=checked_columns,
            forbidden_sources=forbidden_sources,
            forbidden_columns=forbidden_columns,
        )

    def validate_prior_input(
        self,
        source_paths: Sequence[str | Path],
        columns: Iterable[str],
    ) -> LeakageCheckResult:
        """Проверить входы ``LDA_prior`` и прервать выполнение при утечке."""

        result = self.check_prior_input(source_paths=source_paths, columns=columns)
        if not result.is_safe:
            parts: list[str] = []
            if result.forbidden_sources:
                parts.append(
                    "запрещенные источники: "
                    + ", ".join(result.forbidden_sources)
                )
            if result.forbidden_columns:
                parts.append(
                    "запрещенные колонки: "
                    + ", ".join(result.forbidden_columns)
                )
            msg = "Обнаружена утечка данных в LDA_prior: " + "; ".join(parts)
            raise LdaLeakageError(msg)
        return result

    def validate_prior_sources(
        self,
        source_paths: Sequence[str | Path],
    ) -> tuple[str, ...]:
        """Проверить только имена файлов-источников ``LDA_prior``."""

        checked_sources = self._normalize_source_names(source_paths)
        forbidden_sources = self._find_forbidden_sources(checked_sources)
        if forbidden_sources:
            msg = (
                "В LDA_prior переданы запрещенные источники: "
                + ", ".join(forbidden_sources)
            )
            raise LdaLeakageError(msg)
        return checked_sources

    def validate_prior_columns(self, columns: Iterable[str]) -> tuple[str, ...]:
        """Проверить только имена колонок ``LDA_prior``."""

        checked_columns = self._normalize_columns(columns)
        forbidden_columns = self._find_forbidden_columns(checked_columns)
        if forbidden_columns:
            msg = (
                "В LDA_prior переданы запрещенные колонки: "
                + ", ".join(forbidden_columns)
            )
            raise LdaLeakageError(msg)
        return checked_columns

    def _normalize_source_names(
        self,
        source_paths: Sequence[str | Path],
    ) -> tuple[str, ...]:
        """Получить нормализованные имена файлов-источников."""

        return tuple(Path(source_path).name for source_path in source_paths)

    def _normalize_columns(self, columns: Iterable[str]) -> tuple[str, ...]:
        """Получить нормализованный кортеж имен колонок."""

        return tuple(str(column).strip() for column in columns)

    def _find_forbidden_sources(
        self,
        checked_sources: Iterable[str],
    ) -> tuple[str, ...]:
        """Найти запрещенные источники среди переданных файлов."""

        forbidden = sorted(
            source
            for source in checked_sources
            if source in self.FORBIDDEN_SOURCE_NAMES
        )
        return tuple(forbidden)

    def _find_forbidden_columns(
        self,
        checked_columns: Iterable[str],
    ) -> tuple[str, ...]:
        """Найти запрещенные колонки среди признаков корпуса."""

        forbidden: list[str] = []
        for column in checked_columns:
            if column in self.FORBIDDEN_COLUMN_NAMES:
                forbidden.append(column)
                continue
            if column.startswith(self.FORBIDDEN_COLUMN_PREFIXES):
                forbidden.append(column)
        return tuple(sorted(set(forbidden)))
