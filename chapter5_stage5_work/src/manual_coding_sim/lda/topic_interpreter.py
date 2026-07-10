"""Интерпретация латентных факторов качества по top-токенам LDA.

Модуль преобразует машинный артефакт ``topic_word.csv`` в набор
человекочитаемых отчетов. Интерпретатор не переобучает LDA-модель и не
использует фактические признаки качества. Его задача — сгруппировать
наиболее весомые токены каждой темы, восстановить признаки-источники и
сформировать предварительные названия латентных факторов для главы 4.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping, Sequence


_REQUIRED_TOPIC_WORD_COLUMNS = frozenset(
    {
        "topic_id",
        "token_id",
        "token",
        "document_frequency",
        "weight",
    }
)

_SOURCE_PREFIX_LABELS = {
    "diag": "диагностический признак",
    "fact": "фактический признак",
}

_DESCRIPTOR_LABELS = {
    "present": "наличие признака",
    "absent": "отсутствие признака",
    "missing": "пропуск значения",
}


@dataclass(frozen=True)
class LdaTopicInterpreterConfig:
    """Параметры интерпретации латентных факторов качества."""

    top_n: int = 10
    model_name: str = "LDA_prior"
    encoding: str = "utf-8"
    overwrite: bool = True

    def validate(self) -> None:
        """Проверить корректность параметров интерпретации."""

        if self.top_n < 1:
            msg = "top_n должен быть положительным целым числом."
            raise ValueError(msg)
        if not self.model_name.strip():
            msg = "model_name не должен быть пустой строкой."
            raise ValueError(msg)


@dataclass(frozen=True)
class LdaTopicInterpretation:
    """Человекочитаемое описание одного латентного фактора."""

    topic_id: int
    suggested_factor_name: str
    top_tokens: tuple[str, ...]
    source_features: tuple[str, ...]
    dominant_token: str
    max_weight: float
    top_weight_sum: float
    interpretation_comment: str


@dataclass(frozen=True)
class LdaTopicInterpretationResult:
    """Пути к отчетам интерпретации тем LDA."""

    interpretation_csv_path: Path
    interpretation_json_path: Path
    interpretation_md_path: Path
    topic_count: int
    top_n: int
    model_name: str


@dataclass(frozen=True)
class _TopicWordRow:
    """Одна строка распределения ``φ_k`` из ``topic_word.csv``."""

    topic_id: int
    token_id: int
    token: str
    document_frequency: int
    weight: float


class LdaTopicInterpreter:
    """Формирует интерпретационные отчеты по ``topic_word.csv``."""

    def __init__(self, config: LdaTopicInterpreterConfig | None = None) -> None:
        """Создать интерпретатор латентных факторов качества."""

        self.config = config or LdaTopicInterpreterConfig()
        self.config.validate()

    def interpret_from_topic_word(
        self,
        topic_word_path: str | Path,
        reports_dir: str | Path,
    ) -> LdaTopicInterpretationResult:
        """Построить интерпретацию тем по сохраненному ``topic_word.csv``."""

        source_path = Path(topic_word_path)
        rows = self._read_topic_word_csv(source_path)
        grouped_rows = self._group_rows_by_topic(rows)
        interpretations = [
            self._interpret_topic(topic_id=topic_id, rows=topic_rows)
            for topic_id, topic_rows in sorted(grouped_rows.items())
        ]
        if not interpretations:
            msg = "Не найдено ни одной темы для интерпретации."
            raise ValueError(msg)

        reports_path = Path(reports_dir)
        reports_path.mkdir(parents=True, exist_ok=True)
        csv_path = reports_path / "topic_interpretation.csv"
        json_path = reports_path / "topic_interpretation.json"
        md_path = reports_path / "topic_interpretation.md"
        self._ensure_can_write([csv_path, json_path, md_path])

        payload = self._build_json_payload(
            topic_word_path=source_path,
            interpretations=interpretations,
        )
        self._write_csv(csv_path, interpretations)
        self._write_json(json_path, payload)
        self._write_markdown(md_path, payload)

        return LdaTopicInterpretationResult(
            interpretation_csv_path=csv_path,
            interpretation_json_path=json_path,
            interpretation_md_path=md_path,
            topic_count=len(interpretations),
            top_n=self.config.top_n,
            model_name=self.config.model_name,
        )

    def _read_topic_word_csv(self, path: Path) -> list[_TopicWordRow]:
        """Прочитать и проверить CSV-файл распределений ``φ_k``."""

        if not path.exists():
            msg = f"Файл topic_word.csv не найден: {path}"
            raise FileNotFoundError(msg)
        with path.open("r", encoding=self.config.encoding, newline="") as file_obj:
            reader = csv.DictReader(file_obj)
            if reader.fieldnames is None:
                msg = f"Файл {path} не содержит заголовок CSV."
                raise ValueError(msg)
            missing_columns = _REQUIRED_TOPIC_WORD_COLUMNS - set(reader.fieldnames)
            if missing_columns:
                msg = "В topic_word.csv отсутствуют обязательные колонки: "
                msg += ", ".join(sorted(missing_columns))
                raise ValueError(msg)
            rows = [self._parse_topic_word_row(row) for row in reader]
        if not rows:
            msg = "Файл topic_word.csv не содержит строк для интерпретации."
            raise ValueError(msg)
        return rows

    def _parse_topic_word_row(self, row: Mapping[str, str]) -> _TopicWordRow:
        """Преобразовать строку CSV в типизированную запись."""

        try:
            topic_id = int(row["topic_id"])
            token_id = int(row["token_id"])
            document_frequency = int(row["document_frequency"])
            weight = float(row["weight"])
        except ValueError as exc:
            msg = f"Некорректная числовая строка topic_word.csv: {row}"
            raise ValueError(msg) from exc
        token = row["token"].strip()
        if not token:
            msg = "Токен в topic_word.csv не должен быть пустым."
            raise ValueError(msg)
        if topic_id < 0 or token_id < 0:
            msg = "topic_id и token_id должны быть неотрицательными."
            raise ValueError(msg)
        if document_frequency < 0:
            msg = "document_frequency не должен быть отрицательным."
            raise ValueError(msg)
        if weight < 0:
            msg = "weight не должен быть отрицательным."
            raise ValueError(msg)
        return _TopicWordRow(
            topic_id=topic_id,
            token_id=token_id,
            token=token,
            document_frequency=document_frequency,
            weight=weight,
        )

    def _group_rows_by_topic(
        self,
        rows: Sequence[_TopicWordRow],
    ) -> dict[int, list[_TopicWordRow]]:
        """Сгруппировать строки ``topic_word.csv`` по идентификатору темы."""

        grouped_rows: dict[int, list[_TopicWordRow]] = {}
        for row in rows:
            grouped_rows.setdefault(row.topic_id, []).append(row)
        return grouped_rows

    def _interpret_topic(
        self,
        topic_id: int,
        rows: Sequence[_TopicWordRow],
    ) -> LdaTopicInterpretation:
        """Сформировать текстовую интерпретацию одной темы."""

        ordered_rows = sorted(
            rows,
            key=lambda row: (-row.weight, row.token_id, row.token),
        )
        top_rows = ordered_rows[: self.config.top_n]
        if not top_rows:
            msg = f"Тема {topic_id} не содержит токенов."
            raise ValueError(msg)

        top_tokens = tuple(row.token for row in top_rows)
        source_features = self._unique_preserve_order(
            self._extract_source_feature(row.token) for row in top_rows
        )
        dominant_token = top_rows[0].token
        max_weight = float(top_rows[0].weight)
        top_weight_sum = float(sum(row.weight for row in top_rows))
        suggested_factor_name = self._suggest_factor_name(
            topic_id=topic_id,
            source_features=source_features,
        )
        interpretation_comment = self._build_interpretation_comment(
            topic_id=topic_id,
            top_tokens=top_tokens,
            source_features=source_features,
            dominant_token=dominant_token,
        )
        return LdaTopicInterpretation(
            topic_id=topic_id,
            suggested_factor_name=suggested_factor_name,
            top_tokens=top_tokens,
            source_features=source_features,
            dominant_token=dominant_token,
            max_weight=max_weight,
            top_weight_sum=top_weight_sum,
            interpretation_comment=interpretation_comment,
        )

    def _extract_source_feature(self, token: str) -> str:
        """Получить имя исходного признака из строкового LDA-токена."""

        feature_part = token.split("__", maxsplit=1)[0]
        if feature_part.startswith("diag_"):
            return feature_part.removeprefix("diag_")
        if feature_part.startswith("fact_"):
            return feature_part.removeprefix("fact_")
        return feature_part

    def _suggest_factor_name(
        self,
        topic_id: int,
        source_features: Sequence[str],
    ) -> str:
        """Сформировать предварительное название латентного фактора."""

        label = f"Латентный фактор качества {topic_id}"
        if not source_features:
            return label
        visible_features = ", ".join(source_features[:3])
        return f"{label}: {visible_features}"

    def _build_interpretation_comment(
        self,
        topic_id: int,
        top_tokens: Sequence[str],
        source_features: Sequence[str],
        dominant_token: str,
    ) -> str:
        """Сформировать пояснение для таблицы интерпретации факторов."""

        token_descriptions = [self._describe_token(token) for token in top_tokens[:3]]
        feature_text = ", ".join(source_features[:5]) or "не выделены"
        token_text = "; ".join(token_descriptions)
        return (
            f"Тема {topic_id} интерпретируется как фактор, связанный "
            f"с признаками: {feature_text}. Доминирующий токен: "
            f"{dominant_token}. Наиболее характерные состояния: {token_text}."
        )

    def _describe_token(self, token: str) -> str:
        """Преобразовать машинный токен в короткое пояснение."""

        parts = token.split("__", maxsplit=1)
        feature = parts[0]
        descriptor = parts[1] if len(parts) == 2 else ""
        source_label = self._source_label(feature)
        clean_feature = self._extract_source_feature(token)
        if descriptor.startswith("level_"):
            value_text = f"уровень {descriptor.removeprefix('level_')}"
        elif descriptor.startswith("value_"):
            value_text = f"значение {descriptor.removeprefix('value_')}"
        else:
            value_text = _DESCRIPTOR_LABELS.get(descriptor, descriptor or "состояние")
        return f"{source_label} {clean_feature}: {value_text}"

    def _source_label(self, feature: str) -> str:
        """Вернуть текстовую метку источника признака."""

        prefix = feature.split("_", maxsplit=1)[0]
        return _SOURCE_PREFIX_LABELS.get(prefix, "априорный признак")

    def _unique_preserve_order(self, values: Sequence[str]) -> tuple[str, ...]:
        """Удалить повторы без изменения исходного порядка."""

        seen_values: set[str] = set()
        result: list[str] = []
        for value in values:
            if value not in seen_values:
                result.append(value)
                seen_values.add(value)
        return tuple(result)

    def _build_json_payload(
        self,
        topic_word_path: Path,
        interpretations: Sequence[LdaTopicInterpretation],
    ) -> dict[str, object]:
        """Сформировать JSON-представление интерпретации тем."""

        return {
            "model_name": self.config.model_name,
            "topic_word_path": str(topic_word_path),
            "top_n": self.config.top_n,
            "topic_count": len(interpretations),
            "allowed_for_apriori_forecast": self.config.model_name == "LDA_prior",
            "topics": [self._interpretation_to_dict(item) for item in interpretations],
        }

    def _interpretation_to_dict(
        self,
        interpretation: LdaTopicInterpretation,
    ) -> dict[str, object]:
        """Преобразовать интерпретацию темы в JSON-совместимый словарь."""

        return {
            "topic_id": interpretation.topic_id,
            "suggested_factor_name": interpretation.suggested_factor_name,
            "top_tokens": list(interpretation.top_tokens),
            "source_features": list(interpretation.source_features),
            "dominant_token": interpretation.dominant_token,
            "max_weight": interpretation.max_weight,
            "top_weight_sum": interpretation.top_weight_sum,
            "interpretation_comment": interpretation.interpretation_comment,
        }

    def _write_csv(
        self,
        path: Path,
        interpretations: Sequence[LdaTopicInterpretation],
    ) -> None:
        """Сохранить табличный отчет интерпретации тем."""

        fieldnames = [
            "topic_id",
            "suggested_factor_name",
            "top_tokens",
            "source_features",
            "dominant_token",
            "max_weight",
            "top_weight_sum",
            "interpretation_comment",
        ]
        with path.open("w", encoding=self.config.encoding, newline="") as file_obj:
            writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
            writer.writeheader()
            for interpretation in interpretations:
                writer.writerow(
                    {
                        "topic_id": interpretation.topic_id,
                        "suggested_factor_name": interpretation.suggested_factor_name,
                        "top_tokens": "; ".join(interpretation.top_tokens),
                        "source_features": "; ".join(
                            interpretation.source_features
                        ),
                        "dominant_token": interpretation.dominant_token,
                        "max_weight": f"{interpretation.max_weight:.12g}",
                        "top_weight_sum": f"{interpretation.top_weight_sum:.12g}",
                        "interpretation_comment": (
                            interpretation.interpretation_comment
                        ),
                    }
                )

    def _write_json(self, path: Path, payload: Mapping[str, object]) -> None:
        """Сохранить JSON-отчет интерпретации тем."""

        with path.open("w", encoding=self.config.encoding) as file_obj:
            json.dump(payload, file_obj, ensure_ascii=False, indent=2, sort_keys=True)
            file_obj.write("\n")

    def _write_markdown(self, path: Path, payload: Mapping[str, object]) -> None:
        """Сохранить Markdown-отчет для текстовой части главы 4."""

        topics = payload["topics"]
        if not isinstance(topics, list):
            msg = "Некорректная структура JSON-отчета интерпретации тем."
            raise TypeError(msg)

        lines = [
            "# Интерпретация латентных факторов качества",
            "",
            f"Модель: `{payload['model_name']}`.",
            f"Число интерпретируемых тем: {payload['topic_count']}.",
            f"Число top-токенов на тему: {payload['top_n']}.",
            "",
        ]
        for topic in topics:
            if not isinstance(topic, dict):
                msg = "Описание темы в JSON-отчете должно быть словарем."
                raise TypeError(msg)
            lines.extend(
                [
                    f"## Тема {topic['topic_id']}",
                    "",
                    f"**Предварительное название:** {topic['suggested_factor_name']}",
                    "",
                    f"**Исходные признаки:** {', '.join(topic['source_features'])}",
                    "",
                    "**Top-токены:**",
                    "",
                ]
            )
            for token in topic["top_tokens"]:
                lines.append(f"- `{token}`")
            lines.extend(
                [
                    "",
                    f"**Комментарий:** {topic['interpretation_comment']}",
                    "",
                ]
            )
        path.write_text("\n".join(lines), encoding=self.config.encoding)

    def _ensure_can_write(self, paths: Sequence[Path]) -> None:
        """Проверить возможность записи выходных отчетов."""

        if self.config.overwrite:
            return
        existing_paths = [path for path in paths if path.exists()]
        if existing_paths:
            joined_paths = ", ".join(str(path) for path in existing_paths)
            msg = f"Файлы интерпретации уже существуют: {joined_paths}"
            raise FileExistsError(msg)
