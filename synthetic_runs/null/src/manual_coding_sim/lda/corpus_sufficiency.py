"""Проверка достаточности корпуса для финальных выводов главы 4.

Модуль не генерирует искусственные протоколы и не размножает строки датасета.
Его задача — проверить, можно ли использовать текущий корпус априорных
признаков как основной диссертационный корпус для выбора ``K``, интерпретации
латентных факторов и последующей проверки метода в главах 5–6.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


SERVICE_COLUMNS = {
    "run_id",
    "protocol_id",
    "scenario_id",
    "alternative_id",
    "sample_id",
    "document_id",
}

DEFAULT_REPORT_JSON_NAME = "corpus_sufficiency_report.json"
DEFAULT_REPORT_MD_NAME = "corpus_sufficiency_report.md"


@dataclass(frozen=True)
class CorpusCoverageRule:
    """Правило покрытия значений одного признака расширенного корпуса."""

    column: str
    min_unique_values: int = 2

    def validate(self) -> None:
        """Проверить корректность правила покрытия."""

        if not self.column:
            msg = "Имя колонки покрытия не должно быть пустым."
            raise ValueError(msg)
        if self.min_unique_values < 1:
            msg = "min_unique_values должен быть положительным целым числом."
            raise ValueError(msg)


@dataclass(frozen=True)
class CorpusSufficiencyConfig:
    """Пороговые требования к расширенному корпусу главы 4."""

    min_documents: int = 50
    min_unique_scenarios: int = 50
    min_unique_protocols: int = 50
    min_vocabulary_tokens: int = 30
    coverage_rules: tuple[CorpusCoverageRule, ...] = field(default_factory=tuple)
    overwrite: bool = True

    def validate(self) -> None:
        """Проверить корректность пороговых значений."""

        if self.min_documents < 1:
            msg = "min_documents должен быть положительным целым числом."
            raise ValueError(msg)
        if self.min_unique_scenarios < 1:
            msg = "min_unique_scenarios должен быть положительным целым числом."
            raise ValueError(msg)
        if self.min_unique_protocols < 1:
            msg = "min_unique_protocols должен быть положительным целым числом."
            raise ValueError(msg)
        if self.min_vocabulary_tokens < 1:
            msg = "min_vocabulary_tokens должен быть положительным целым числом."
            raise ValueError(msg)
        for rule in self.coverage_rules:
            rule.validate()


@dataclass(frozen=True)
class CorpusCoverageResult:
    """Результат проверки покрытия по одному признаку."""

    column: str
    exists: bool
    unique_count: int
    required_min_unique_values: int
    passed: bool
    sample_values: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        """Преобразовать результат покрытия в JSON-совместимый словарь."""

        return {
            "column": self.column,
            "exists": self.exists,
            "unique_count": self.unique_count,
            "required_min_unique_values": self.required_min_unique_values,
            "passed": self.passed,
            "sample_values": list(self.sample_values),
        }


@dataclass(frozen=True)
class CorpusSufficiencyResult:
    """Итог проверки достаточности корпуса."""

    passed: bool
    document_count: int
    min_documents: int
    unique_scenario_count: int
    min_unique_scenarios: int
    unique_protocol_count: int
    min_unique_protocols: int
    vocabulary_token_count: int
    min_vocabulary_tokens: int
    coverage_results: tuple[CorpusCoverageResult, ...]
    recommendations: tuple[str, ...]
    report_json_path: Path
    report_md_path: Path

    def to_dict(self) -> dict[str, object]:
        """Преобразовать итог проверки в JSON-совместимый словарь."""

        return {
            "passed": self.passed,
            "document_count": self.document_count,
            "min_documents": self.min_documents,
            "unique_scenario_count": self.unique_scenario_count,
            "min_unique_scenarios": self.min_unique_scenarios,
            "unique_protocol_count": self.unique_protocol_count,
            "min_unique_protocols": self.min_unique_protocols,
            "vocabulary_token_count": self.vocabulary_token_count,
            "min_vocabulary_tokens": self.min_vocabulary_tokens,
            "coverage_results": [item.to_dict() for item in self.coverage_results],
            "recommendations": list(self.recommendations),
            "report_json_path": str(self.report_json_path),
            "report_md_path": str(self.report_md_path),
        }


class CorpusSufficiencyAnalyzer:
    """Анализирует, достаточен ли корпус для финальных выводов главы 4."""

    def __init__(self, config: CorpusSufficiencyConfig | None = None) -> None:
        """Создать анализатор достаточности корпуса."""

        self.config = config or CorpusSufficiencyConfig()
        self.config.validate()

    def analyze_from_files(
        self,
        prior_features_path: str | Path,
        reports_dir: str | Path,
        dictionary_path: str | Path | None = None,
    ) -> CorpusSufficiencyResult:
        """Проверить корпус по ``prior_features.csv`` и необязательному словарю LDA."""

        prior_path = Path(prior_features_path)
        report_dir = Path(reports_dir)
        rows = self._read_csv(prior_path)
        if not rows:
            msg = f"Файл {prior_path} не содержит строк априорных признаков."
            raise ValueError(msg)

        dictionary = self._read_dictionary(dictionary_path)
        vocabulary_token_count = (
            len(dictionary) if dictionary is not None else self._estimate_raw_feature_count(rows[0])
        )

        coverage_results = self._check_coverage(rows)
        recommendations = self._build_recommendations(
            rows=rows,
            vocabulary_token_count=vocabulary_token_count,
            coverage_results=coverage_results,
        )
        passed = not recommendations

        report_json_path = report_dir / DEFAULT_REPORT_JSON_NAME
        report_md_path = report_dir / DEFAULT_REPORT_MD_NAME
        self._ensure_can_write([report_json_path, report_md_path])

        result = CorpusSufficiencyResult(
            passed=passed,
            document_count=len(rows),
            min_documents=self.config.min_documents,
            unique_scenario_count=self._unique_count(rows, "scenario_id"),
            min_unique_scenarios=self.config.min_unique_scenarios,
            unique_protocol_count=self._unique_count(rows, "protocol_id"),
            min_unique_protocols=self.config.min_unique_protocols,
            vocabulary_token_count=vocabulary_token_count,
            min_vocabulary_tokens=self.config.min_vocabulary_tokens,
            coverage_results=tuple(coverage_results),
            recommendations=tuple(recommendations),
            report_json_path=report_json_path,
            report_md_path=report_md_path,
        )
        self._write_json(report_json_path, result.to_dict())
        self._write_markdown(report_md_path, result)
        return result

    def _read_csv(self, path: Path) -> list[dict[str, str]]:
        """Прочитать CSV-файл априорных признаков."""

        if not path.exists():
            msg = f"Файл априорных признаков не найден: {path}"
            raise FileNotFoundError(msg)
        with path.open("r", encoding="utf-8", newline="") as file_obj:
            reader = csv.DictReader(file_obj)
            if reader.fieldnames is None:
                msg = f"Файл {path} не содержит заголовок CSV."
                raise ValueError(msg)
            return [dict(row) for row in reader]

    def _read_dictionary(self, path: str | Path | None) -> dict[str, object] | None:
        """Прочитать словарь LDA, если он уже сформирован."""

        if path is None:
            return None
        dictionary_path = Path(path)
        if not dictionary_path.exists():
            msg = f"Файл словаря LDA не найден: {dictionary_path}"
            raise FileNotFoundError(msg)
        with dictionary_path.open("r", encoding="utf-8") as file_obj:
            payload = json.load(file_obj)
        if not isinstance(payload, dict):
            msg = "Словарь LDA должен быть JSON-объектом."
            raise ValueError(msg)
        if isinstance(payload.get("token_count"), int):
            return {
                f"__token_{index}": {}
                for index in range(int(payload["token_count"]))
            }
        tokens = payload.get("tokens")
        if isinstance(tokens, list):
            return {
                str(item.get("token", f"__token_{index}")): item
                for index, item in enumerate(tokens)
                if isinstance(item, dict)
            }
        return payload

    def _estimate_raw_feature_count(self, header_row: Mapping[str, str]) -> int:
        """Оценить число признаков, если итоговый LDA-словарь еще не построен."""

        return sum(1 for column in header_row if column not in SERVICE_COLUMNS)

    def _check_coverage(self, rows: Sequence[Mapping[str, str]]) -> list[CorpusCoverageResult]:
        """Проверить покрытие корпуса по пользовательским правилам."""

        results: list[CorpusCoverageResult] = []
        for rule in self.config.coverage_rules:
            values = sorted(
                {
                    str(row.get(rule.column, "")).strip()
                    for row in rows
                    if str(row.get(rule.column, "")).strip()
                }
            )
            exists = rule.column in rows[0]
            unique_count = len(values) if exists else 0
            results.append(
                CorpusCoverageResult(
                    column=rule.column,
                    exists=exists,
                    unique_count=unique_count,
                    required_min_unique_values=rule.min_unique_values,
                    passed=exists and unique_count >= rule.min_unique_values,
                    sample_values=tuple(values[:10]),
                )
            )
        return results

    def _build_recommendations(
        self,
        rows: Sequence[Mapping[str, str]],
        vocabulary_token_count: int,
        coverage_results: Sequence[CorpusCoverageResult],
    ) -> list[str]:
        """Сформировать рекомендации по доведению корпуса до требуемого уровня."""

        recommendations: list[str] = []
        document_count = len(rows)
        unique_scenarios = self._unique_count(rows, "scenario_id")
        unique_protocols = self._unique_count(rows, "protocol_id")

        if document_count < self.config.min_documents:
            recommendations.append(
                "Увеличить число документов корпуса минимум до "
                f"{self.config.min_documents}; сейчас {document_count}."
            )
        if unique_scenarios < self.config.min_unique_scenarios:
            recommendations.append(
                "Увеличить число уникальных scenario_id минимум до "
                f"{self.config.min_unique_scenarios}; сейчас {unique_scenarios}."
            )
        if unique_protocols < self.config.min_unique_protocols:
            recommendations.append(
                "Увеличить число уникальных protocol_id минимум до "
                f"{self.config.min_unique_protocols}; сейчас {unique_protocols}."
            )
        if vocabulary_token_count < self.config.min_vocabulary_tokens:
            recommendations.append(
                "Расширить разнообразие априорных признаков и токенов минимум до "
                f"{self.config.min_vocabulary_tokens}; сейчас {vocabulary_token_count}."
            )
        for coverage in coverage_results:
            if not coverage.exists:
                recommendations.append(
                    f"Добавить колонку покрытия '{coverage.column}' "
                    "или убрать ее из требований конфигурации."
                )
            elif not coverage.passed:
                recommendations.append(
                    "Расширить покрытие признака "
                    f"'{coverage.column}' минимум до "
                    f"{coverage.required_min_unique_values} уникальных значений; "
                    f"сейчас {coverage.unique_count}."
                )
        return recommendations

    def _unique_count(self, rows: Sequence[Mapping[str, str]], column: str) -> int:
        """Посчитать число уникальных непустых значений колонки."""

        if column not in rows[0]:
            return len(rows)
        return len(
            {
                str(row.get(column, "")).strip()
                for row in rows
                if str(row.get(column, "")).strip()
            }
        )

    def _ensure_can_write(self, paths: Sequence[Path]) -> None:
        """Проверить возможность записи отчетов."""

        for path in paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists() and not self.config.overwrite:
                msg = f"Файл уже существует и overwrite=False: {path}"
                raise FileExistsError(msg)

    def _write_json(self, path: Path, payload: Mapping[str, object]) -> None:
        """Записать JSON-отчет."""

        with path.open("w", encoding="utf-8") as file_obj:
            json.dump(payload, file_obj, ensure_ascii=False, indent=2)

    def _write_markdown(self, path: Path, result: CorpusSufficiencyResult) -> None:
        """Записать Markdown-отчет о достаточности корпуса."""

        lines = [
            "# Проверка достаточности расширенного корпуса",
            "",
            f"Статус: {'passed' if result.passed else 'failed'}.",
            "",
            "## Основные показатели",
            "",
            "| Показатель | Факт | Минимум | Статус |",
            "|---|---:|---:|---|",
            _status_row(
                "Документы",
                result.document_count,
                result.min_documents,
                result.document_count >= result.min_documents,
            ),
            _status_row(
                "Уникальные сценарии",
                result.unique_scenario_count,
                result.min_unique_scenarios,
                result.unique_scenario_count >= result.min_unique_scenarios,
            ),
            _status_row(
                "Уникальные протоколы",
                result.unique_protocol_count,
                result.min_unique_protocols,
                result.unique_protocol_count >= result.min_unique_protocols,
            ),
            _status_row(
                "Токены словаря",
                result.vocabulary_token_count,
                result.min_vocabulary_tokens,
                result.vocabulary_token_count >= result.min_vocabulary_tokens,
            ),
            "",
        ]
        if result.coverage_results:
            lines.extend(
                [
                    "## Покрытие признаков",
                    "",
                    "| Колонка | Найдена | Уникальных значений | Минимум | Статус |",
                    "|---|---:|---:|---:|---|",
                ]
            )
            for item in result.coverage_results:
                lines.append(
                    "| "
                    f"{item.column} | {item.exists} | {item.unique_count} | "
                    f"{item.required_min_unique_values} | {_status(item.passed)} |"
                )
            lines.append("")
        lines.extend(["## Рекомендации", ""])
        if result.recommendations:
            lines.extend(f"- {recommendation}" for recommendation in result.recommendations)
        else:
            lines.append("- Корпус проходит заданные пороги достаточности.")
        lines.append("")
        path.write_text("\n".join(lines), encoding="utf-8")


class CorpusSufficiencyConfigLoader:
    """Загружает секцию ``corpus`` из YAML-конфига расширенного запуска."""

    def load(
        self,
        config_path: str | Path,
        overwrite_override: bool | None = None,
    ) -> CorpusSufficiencyConfig:
        """Прочитать YAML и вернуть конфигурацию достаточности корпуса."""

        path = Path(config_path)
        if not path.exists():
            msg = f"Файл конфигурации не найден: {path}"
            raise FileNotFoundError(msg)
        with path.open("r", encoding="utf-8") as file_obj:
            payload = yaml.safe_load(file_obj) or {}
        if not isinstance(payload, dict):
            msg = "Корневой элемент YAML-конфига должен быть словарем."
            raise ValueError(msg)

        corpus_section = payload.get("corpus", {})
        if corpus_section is None:
            corpus_section = {}
        if not isinstance(corpus_section, dict):
            msg = "Секция 'corpus' должна быть словарем."
            raise ValueError(msg)

        coverage_rules = tuple(
            CorpusCoverageRule(
                column=str(item.get("column", "")),
                min_unique_values=_positive_int(
                    item.get("min_unique_values", 2),
                    "min_unique_values",
                ),
            )
            for item in _coverage_rule_items(corpus_section.get("coverage_rules", []))
        )
        overwrite = _bool(corpus_section.get("overwrite", True))
        if overwrite_override is not None:
            overwrite = overwrite_override

        config = CorpusSufficiencyConfig(
            min_documents=_positive_int(corpus_section.get("min_documents", 50), "min_documents"),
            min_unique_scenarios=_positive_int(
                corpus_section.get("min_unique_scenarios", 50),
                "min_unique_scenarios",
            ),
            min_unique_protocols=_positive_int(
                corpus_section.get("min_unique_protocols", 50),
                "min_unique_protocols",
            ),
            min_vocabulary_tokens=_positive_int(
                corpus_section.get("min_vocabulary_tokens", 30),
                "min_vocabulary_tokens",
            ),
            coverage_rules=coverage_rules,
            overwrite=overwrite,
        )
        config.validate()
        return config


def _coverage_rule_items(value: Any) -> list[dict[str, object]]:
    """Проверить и вернуть элементы правил покрытия."""

    if value is None:
        return []
    if not isinstance(value, list):
        msg = "corpus.coverage_rules должен быть списком."
        raise ValueError(msg)
    result: list[dict[str, object]] = []
    for item in value:
        if not isinstance(item, dict):
            msg = "Каждое правило покрытия должно быть словарем."
            raise ValueError(msg)
        result.append(item)
    return result


def _positive_int(value: object, name: str) -> int:
    """Преобразовать значение в положительное целое число."""

    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        msg = f"{name} должен быть целым числом."
        raise ValueError(msg) from exc
    if result < 1:
        msg = f"{name} должен быть положительным целым числом."
        raise ValueError(msg)
    return result


def _bool(value: object) -> bool:
    """Преобразовать значение YAML в bool."""

    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y", "да"}:
            return True
        if normalized in {"false", "0", "no", "n", "нет"}:
            return False
    return bool(value)


def _status(passed: bool) -> str:
    """Вернуть текстовый статус для Markdown-таблицы."""

    return "OK" if passed else "FAIL"


def _status_row(name: str, actual: int, required: int, passed: bool) -> str:
    """Сформировать строку Markdown-таблицы со статусом проверки."""

    return f"| {name} | {actual} | {required} | {_status(passed)} |"
