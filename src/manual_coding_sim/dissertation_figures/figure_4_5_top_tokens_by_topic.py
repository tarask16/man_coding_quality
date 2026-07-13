"""Генерация рисунка 4.5 с наиболее весомыми токенами латентных факторов."""

from __future__ import annotations

import argparse
import csv
import json
import math
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

from manual_coding_sim.dissertation_figures.common import (
    FigureExportResult,
    configure_dissertation_style,
    export_figure,
)

OUTPUT_DIR = Path("reports/chapter4/figures")
FILE_STEM = "figure_4_5_top_tokens_by_topic"
DEFAULT_TOPIC_WORD_PATH = Path("reports/chapter4/topic_word.csv")
DEFAULT_INTERPRETATION_PATH = Path("reports/chapter4/topic_interpretation.json")
DEFAULT_TOP_N = 10

TOPIC_NAMES = {
    0: "Тема 0 — процедурная трудоёмкость",
    1: "Тема 1 — операционный риск",
    2: "Тема 2 — благоприятные условия",
}

FEATURE_LABELS = {
    "prior_total_nominal_time": "Номинальное время",
    "prior_memory_load_index": "Нагрузка памяти",
    "prior_manual_operation_count": "Ручные операции",
    "prior_operator_total_estimated_time": "Расчётное время оператора",
    "prior_condition_total_adjusted_time": "Скорректированное время",
    "prior_message_length": "Длина сообщения",
    "prior_symbol_group_count": "Группы символов",
    "prior_operator_attention": "Внимание оператора",
    "prior_procedure_branch_count": "Ветвления процедуры",
    "prior_operator_error_susceptibility": "Ошибкоопасность оператора",
    "prior_condition_mean_adjusted_attention": "Скорректированное внимание",
    "prior_attention_deficit": "Дефицит внимания",
    "prior_control_intensity": "Интенсивность контроля",
    "prior_expected_error_probability": "Ожидаемая ошибкоопасность",
    "prior_condition_time_pressure": "Давление времени",
    "prior_time_pressure_index": "Индекс давления времени",
    "prior_verification_required": "Обязательная проверка",
    "prior_operator_skill": "Квалификация оператора",
    "prior_repetition_expected_count": "Ожидаемые повторения",
    "prior_operator_fatigue": "Утомление оператора",
    "prior_condition_noise_adjustment": "Поправка на шум",
    "prior_condition_noise_level": "Уровень шума",
    "prior_condition_profile": "Профиль условий",
    "prior_coding_tool_type": "Тип средства кодирования",
}

STATE_LABELS = {
    "level_high": "высокий уровень",
    "level_mid": "средний уровень",
    "level_low": "низкий уровень",
    "present": "присутствует",
    "absent": "отсутствует",
    "value_hard": "сложные условия",
    "value_normal": "обычные условия",
    "value_easy": "благоприятные условия",
    "value_codebook": "кодовая книга",
    "value_mixed": "смешанный тип",
    "value_mnemonic": "мнемонический тип",
    "value_table": "табличный тип",
}


@dataclass(frozen=True, slots=True)
class TopicToken:
    """Вес одного токена в одном латентном факторе."""

    topic_id: int
    token_id: int
    token: str
    document_frequency: int
    weight: float


@dataclass(frozen=True, slots=True)
class TopicInterpretation:
    """Краткая интерпретация латентного фактора."""

    topic_id: int
    factor_name: str
    dominant_token: str
    top_weight_sum: float


def load_topic_tokens(path: str | Path) -> tuple[TopicToken, ...]:
    """Загрузить веса токенов из отчёта ``topic_word.csv``."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Не найден отчёт весов токенов: {source}")

    required = {"topic_id", "token_id", "token", "document_frequency", "weight"}
    records: list[TopicToken] = []
    with source.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError("CSV-отчёт не содержит строки заголовка.")
        missing = required.difference(reader.fieldnames)
        if missing:
            raise ValueError(
                "В отчёте отсутствуют обязательные колонки: "
                + ", ".join(sorted(missing))
            )
        for row_number, row in enumerate(reader, start=2):
            try:
                record = TopicToken(
                    topic_id=int(row["topic_id"]),
                    token_id=int(row["token_id"]),
                    token=row["token"].strip(),
                    document_frequency=int(row["document_frequency"]),
                    weight=float(row["weight"]),
                )
            except (TypeError, ValueError, KeyError) as error:
                raise ValueError(
                    f"Некорректная строка {row_number} отчёта весов токенов: {error}."
                ) from error
            records.append(record)

    result = tuple(records)
    validate_topic_tokens(result)
    return result


def validate_topic_tokens(records: Sequence[TopicToken]) -> None:
    """Проверить полноту и согласованность весов токенов."""

    if not records:
        raise ValueError("Отчёт весов токенов не должен быть пустым.")

    topic_ids = sorted({record.topic_id for record in records})
    if topic_ids != list(range(len(topic_ids))):
        raise ValueError("Идентификаторы тем должны образовывать последовательность от нуля.")
    if len(topic_ids) != 3:
        raise ValueError("Для рисунка 4.5 ожидаются три латентных фактора.")

    tokens_by_topic: dict[int, set[str]] = {}
    for record in records:
        if not record.token:
            raise ValueError("Имя токена не должно быть пустым.")
        if record.document_frequency <= 0:
            raise ValueError("Документная частота токена должна быть положительной.")
        if not math.isfinite(record.weight) or record.weight < 0.0:
            raise ValueError("Вес токена должен быть конечным и неотрицательным.")
        tokens_by_topic.setdefault(record.topic_id, set()).add(record.token)

    token_sets = list(tokens_by_topic.values())
    if any(token_set != token_sets[0] for token_set in token_sets[1:]):
        raise ValueError("Все темы должны быть заданы на одном словаре токенов.")

    for topic_id in topic_ids:
        topic_records = [record for record in records if record.topic_id == topic_id]
        if len(topic_records) != len({record.token_id for record in topic_records}):
            raise ValueError("Идентификаторы токенов внутри темы должны быть уникальными.")
        total_weight = sum(record.weight for record in topic_records)
        if not math.isclose(total_weight, 1.0, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError("Сумма весов токенов каждой темы должна быть равна единице.")


def load_topic_interpretations(path: str | Path) -> tuple[TopicInterpretation, ...]:
    """Загрузить краткие интерпретации тем из JSON-отчёта."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Не найден отчёт интерпретации тем: {source}")
    with source.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)

    topics = payload.get("topics")
    if not isinstance(topics, list) or not topics:
        raise ValueError("JSON-отчёт не содержит непустой список topics.")

    interpretations: list[TopicInterpretation] = []
    for item in topics:
        interpretations.append(
            TopicInterpretation(
                topic_id=int(item["topic_id"]),
                factor_name=str(item["suggested_factor_name"]),
                dominant_token=str(item["dominant_token"]),
                top_weight_sum=float(item["top_weight_sum"]),
            )
        )
    result = tuple(sorted(interpretations, key=lambda item: item.topic_id))
    if [item.topic_id for item in result] != [0, 1, 2]:
        raise ValueError("Интерпретации должны быть заданы для тем 0, 1 и 2.")
    return result


def select_top_tokens(
    records: Sequence[TopicToken], *, top_n: int = DEFAULT_TOP_N
) -> dict[int, tuple[TopicToken, ...]]:
    """Выбрать ``top_n`` наиболее весомых токенов для каждой темы."""

    validate_topic_tokens(records)
    if top_n < 3:
        raise ValueError("Для содержательной диаграммы требуется не менее трёх токенов.")

    result: dict[int, tuple[TopicToken, ...]] = {}
    for topic_id in sorted({record.topic_id for record in records}):
        topic_records = [record for record in records if record.topic_id == topic_id]
        if top_n > len(topic_records):
            raise ValueError("Число отображаемых токенов превышает размер словаря темы.")
        selected = sorted(
            topic_records,
            key=lambda item: (-item.weight, item.token_id),
        )[:top_n]
        result[topic_id] = tuple(selected)
    return result


def token_to_russian_label(token: str) -> str:
    """Преобразовать техническое имя токена в краткую русскую подпись."""

    if "__" not in token:
        return token
    feature, state = token.split("__", maxsplit=1)
    feature_label = FEATURE_LABELS.get(
        feature,
        feature.removeprefix("prior_").replace("_", " "),
    )
    state_label = STATE_LABELS.get(state, state.replace("_", " "))
    return f"{feature_label} — {state_label}"


def _wrapped_label(token: str) -> str:
    """Перенести длинную русскую подпись на две строки."""

    label = token_to_russian_label(token)
    return "\n".join(textwrap.wrap(label, width=29, break_long_words=False))


def create_figure(
    records: Sequence[TopicToken],
    interpretations: Sequence[TopicInterpretation],
    *,
    top_n: int = DEFAULT_TOP_N,
) -> plt.Figure:
    """Построить три горизонтальные диаграммы на единой шкале весов."""

    configure_dissertation_style()
    top_tokens = select_top_tokens(records, top_n=top_n)
    interpretation_map = {item.topic_id: item for item in interpretations}
    if set(interpretation_map) != set(top_tokens):
        raise ValueError("Набор интерпретаций не совпадает с набором тем.")

    max_weight = max(item.weight for values in top_tokens.values() for item in values)
    x_limit = max_weight * 1.18

    figure, axes = plt.subplots(1, 3, figsize=(21.0, 9.7), sharex=True)
    figure.suptitle(
        "Рисунок 4.5 — Наиболее весомые токены латентных факторов LDA_prior",
        fontsize=17,
        fontweight="bold",
        y=0.975,
    )
    figure.text(
        0.5,
        0.938,
        "Для всех трёх факторов используется единая шкала веса; показаны первые "
        f"{top_n} токенов из итогового словаря.",
        ha="center",
        fontsize=11.5,
    )

    for axis, topic_id in zip(axes, sorted(top_tokens), strict=True):
        selected = list(reversed(top_tokens[topic_id]))
        y_positions = np.arange(len(selected))
        weights = np.array([item.weight for item in selected], dtype=float)
        bars = axis.barh(y_positions, weights, height=0.68)
        axis.set_yticks(y_positions, [_wrapped_label(item.token) for item in selected])
        axis.set_xlim(0.0, x_limit)
        axis.grid(axis="x", alpha=0.28, linewidth=0.8)
        axis.set_axisbelow(True)
        axis.set_xlabel("Вес токена в распределении темы")
        axis.tick_params(axis="y", labelsize=8.4, pad=4)
        axis.tick_params(axis="x", labelsize=9)

        concise_name = TOPIC_NAMES.get(topic_id, f"Тема {topic_id}")
        axis.set_title(concise_name, fontsize=13, fontweight="bold", pad=13)

        for bar, record in zip(bars, selected, strict=True):
            axis.text(
                record.weight + x_limit * 0.012,
                bar.get_y() + bar.get_height() / 2,
                f"{record.weight:.4f}",
                va="center",
                ha="left",
                fontsize=8.3,
            )

        top_sum = sum(item.weight for item in top_tokens[topic_id])
        interpretation = interpretation_map[topic_id]
        dominant_label = token_to_russian_label(interpretation.dominant_token)
        axis.text(
            0.02,
            -0.135,
            f"Сумма весов top-{top_n}: {top_sum:.4f}\n"
            f"Доминирующий признак: {dominant_label}",
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=8.8,
            bbox={
                "boxstyle": "round,pad=0.35",
                "facecolor": "white",
                "edgecolor": "0.72",
                "linewidth": 0.8,
            },
        )

    figure.text(
        0.5,
        0.035,
        "Интерпретация основана на относительных весах токенов внутри каждой темы. "
        "Близкие веса не означают причинную связь признака с фактическим качеством; "
        "темы используются как априорные латентные факторы.",
        ha="center",
        va="bottom",
        fontsize=10.2,
        wrap=True,
    )
    figure.subplots_adjust(left=0.06, right=0.99, top=0.875, bottom=0.185, wspace=0.72)
    return figure


def generate(
    *,
    project_root: str | Path,
    topic_word_path: str | Path | None = None,
    interpretation_path: str | Path | None = None,
    top_n: int = DEFAULT_TOP_N,
    dpi: int = 300,
) -> FigureExportResult:
    """Сформировать рисунок 4.5 в PNG и SVG."""

    root = Path(project_root).resolve()
    topic_word = (
        Path(topic_word_path)
        if topic_word_path is not None
        else root / DEFAULT_TOPIC_WORD_PATH
    )
    interpretation = (
        Path(interpretation_path)
        if interpretation_path is not None
        else root / DEFAULT_INTERPRETATION_PATH
    )
    records = load_topic_tokens(topic_word)
    interpretations = load_topic_interpretations(interpretation)
    figure = create_figure(records, interpretations, top_n=top_n)
    return export_figure(
        figure,
        project_root=root,
        relative_output_dir=OUTPUT_DIR,
        file_stem=FILE_STEM,
        dpi=dpi,
    )


def build_argument_parser() -> argparse.ArgumentParser:
    """Создать парсер аргументов командной строки."""

    parser = argparse.ArgumentParser(
        description="Сформировать рисунок 4.5 с наиболее весомыми токенами тем."
    )
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--input", type=Path, default=None, help="Путь к topic_word.csv.")
    parser.add_argument(
        "--interpretation",
        type=Path,
        default=None,
        help="Путь к topic_interpretation.json.",
    )
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    parser.add_argument("--dpi", type=int, default=300)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Выполнить генерацию рисунка из командной строки."""

    arguments = build_argument_parser().parse_args(argv)
    result = generate(
        project_root=arguments.project_root,
        topic_word_path=arguments.input,
        interpretation_path=arguments.interpretation,
        top_n=arguments.top_n,
        dpi=arguments.dpi,
    )
    print("Рисунок 4.5 успешно сформирован.")
    print(f"PNG: {result.png_path}")
    print(f"SVG: {result.svg_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
