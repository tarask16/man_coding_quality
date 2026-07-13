"""Генерация рисунка 4.2 с частотами токенов итогового словаря LDA_prior."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

from manual_coding_sim.dissertation_figures.common import (
    FigureExportResult,
    configure_dissertation_style,
    export_figure,
)

OUTPUT_DIR = Path("reports/chapter4/figures")
FILE_STEM = "figure_4_2_token_frequency_and_filtering"
DEFAULT_DICTIONARY_PATH = Path("data/processed/lda/dictionary.json")
DEFAULT_METADATA_PATH = Path("data/processed/lda/corpus_metadata.json")


@dataclass(frozen=True, slots=True)
class TokenFrequency:
    """Запись итогового словаря с документной частотой токена."""

    token_id: int
    token: str
    document_frequency: int


@dataclass(frozen=True, slots=True)
class TokenFilteringSummary:
    """Сводные параметры частотной фильтрации словаря."""

    document_count: int
    df_min: int
    df_max_ratio: float
    dictionary_token_count: int
    min_document_frequency: int
    max_document_frequency: int
    mean_document_frequency: float
    median_document_frequency: float

    @property
    def upper_document_frequency_limit(self) -> float:
        """Вернуть верхнюю границу документной частоты в документах."""

        return self.document_count * self.df_max_ratio


RUSSIAN_TOKEN_LABELS: Mapping[str, str] = {
    "prior_verification_required__absent": "проверка не требуется",
    "prior_condition_noise_adjustment__level_low": "низкая поправка на шум",
    "prior_condition_noise_level__level_low": "низкий уровень шума",
    "prior_repetition_expected_count__level_low": "мало ожидаемых повторов",
    "prior_condition_time_pressure__level_low": "низкое давление времени",
    "prior_time_pressure_index__level_low": "низкий индекс дефицита времени",
    "prior_coding_tool_type__value_mnemonic": "мнемоническое средство",
    "prior_operator_attention__level_high": "высокое внимание оператора",
    "prior_operator_fatigue__level_high": "высокая утомлённость",
}


def _read_json(path: Path) -> dict[str, object]:
    """Прочитать JSON-файл и проверить, что его корень является объектом."""

    if not path.is_file():
        raise FileNotFoundError(f"Не найден входной файл: {path}")
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError(f"Корень JSON-файла должен быть объектом: {path}")
    return payload


def load_token_frequencies(dictionary_path: str | Path) -> tuple[TokenFrequency, ...]:
    """Загрузить документные частоты токенов из ``dictionary.json``."""

    payload = _read_json(Path(dictionary_path))
    raw_tokens = payload.get("tokens")
    if not isinstance(raw_tokens, list):
        raise ValueError("В dictionary.json отсутствует список tokens.")

    records: list[TokenFrequency] = []
    for index, item in enumerate(raw_tokens):
        if not isinstance(item, dict):
            raise ValueError(f"Элемент tokens[{index}] должен быть объектом.")
        try:
            token_id = int(item["token_id"])
            token = str(item["token"])
            document_frequency = int(item["document_frequency"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Некорректная запись tokens[{index}].") from error
        records.append(
            TokenFrequency(
                token_id=token_id,
                token=token,
                document_frequency=document_frequency,
            )
        )

    declared_count = payload.get("token_count")
    if declared_count is not None and int(declared_count) != len(records):
        raise ValueError("token_count не совпадает с числом записей словаря.")
    return tuple(records)


def load_filtering_parameters(metadata_path: str | Path) -> tuple[int, int, float]:
    """Загрузить число документов и параметры частотного фильтра."""

    payload = _read_json(Path(metadata_path))
    try:
        document_count = int(payload["document_count"])
        df_min = int(payload["df_min"])
        df_max_ratio = float(payload["df_max_ratio"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "В corpus_metadata.json отсутствуют document_count, df_min или df_max_ratio."
        ) from error
    return document_count, df_min, df_max_ratio


def validate_token_frequency_data(
    records: Sequence[TokenFrequency],
    *,
    document_count: int,
    df_min: int,
    df_max_ratio: float,
) -> None:
    """Проверить целостность словаря и соблюдение границ фильтрации."""

    if not records:
        raise ValueError("Итоговый словарь не должен быть пустым.")
    if document_count <= 0:
        raise ValueError("Число документов должно быть положительным.")
    if df_min < 1:
        raise ValueError("df_min должен быть не меньше единицы.")
    if not 0.0 < df_max_ratio <= 1.0:
        raise ValueError("df_max_ratio должен принадлежать интервалу (0; 1].")
    if df_min > document_count * df_max_ratio:
        raise ValueError("Нижняя граница df_min превышает верхнюю границу df_max_ratio.")

    token_ids = [record.token_id for record in records]
    tokens = [record.token for record in records]
    if len(set(token_ids)) != len(token_ids):
        raise ValueError("Идентификаторы token_id должны быть уникальными.")
    if len(set(tokens)) != len(tokens):
        raise ValueError("Имена токенов должны быть уникальными.")

    upper_limit = document_count * df_max_ratio
    for record in records:
        if not record.token.strip():
            raise ValueError("Имя токена не должно быть пустым.")
        if record.document_frequency <= 0:
            raise ValueError("Документная частота должна быть положительной.")
        if record.document_frequency > document_count:
            raise ValueError("Документная частота не может превышать число документов.")
        if record.document_frequency < df_min:
            raise ValueError("В итоговом словаре обнаружен токен ниже границы df_min.")
        if record.document_frequency > upper_limit:
            raise ValueError("В итоговом словаре обнаружен токен выше границы df_max_ratio.")


def summarize_token_frequencies(
    records: Sequence[TokenFrequency],
    *,
    document_count: int,
    df_min: int,
    df_max_ratio: float,
) -> TokenFilteringSummary:
    """Рассчитать сводную статистику итогового словаря."""

    validate_token_frequency_data(
        records,
        document_count=document_count,
        df_min=df_min,
        df_max_ratio=df_max_ratio,
    )
    values = np.asarray([record.document_frequency for record in records], dtype=float)
    return TokenFilteringSummary(
        document_count=document_count,
        df_min=df_min,
        df_max_ratio=df_max_ratio,
        dictionary_token_count=len(records),
        min_document_frequency=int(values.min()),
        max_document_frequency=int(values.max()),
        mean_document_frequency=float(values.mean()),
        median_document_frequency=float(np.median(values)),
    )


def _short_label(token: str) -> str:
    """Вернуть компактную русскую подпись для выделенного токена."""

    if token in RUSSIAN_TOKEN_LABELS:
        return RUSSIAN_TOKEN_LABELS[token]
    label = token.removeprefix("prior_")
    label = label.replace("__level_", ": ")
    label = label.replace("__value_", ": ")
    label = label.replace("__", ": ")
    return label.replace("_", " ")


def _add_summary_card(
    axis: plt.Axes,
    *,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    value: str,
    note: str,
    edgecolor: str,
    facecolor: str,
) -> None:
    """Добавить компактную карточку со сводным показателем."""

    box = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=1.2,
        edgecolor=edgecolor,
        facecolor=facecolor,
        transform=axis.transAxes,
        clip_on=False,
    )
    axis.add_patch(box)
    axis.text(
        x + width / 2,
        y + height * 0.76,
        title,
        ha="center",
        va="center",
        fontsize=8.3,
        fontweight="bold",
        color="#263946",
        transform=axis.transAxes,
    )
    axis.text(
        x + width / 2,
        y + height * 0.47,
        value,
        ha="center",
        va="center",
        fontsize=14.5,
        fontweight="bold",
        color=edgecolor,
        transform=axis.transAxes,
    )
    axis.text(
        x + width / 2,
        y + height * 0.20,
        note,
        ha="center",
        va="center",
        fontsize=7.1,
        color="#536976",
        transform=axis.transAxes,
    )


def build_figure(
    records: Sequence[TokenFrequency],
    *,
    document_count: int,
    df_min: int,
    df_max_ratio: float,
) -> plt.Figure:
    """Построить составной график частот токенов и границ фильтрации."""

    configure_dissertation_style()
    summary = summarize_token_frequencies(
        records,
        document_count=document_count,
        df_min=df_min,
        df_max_ratio=df_max_ratio,
    )

    ranked = sorted(records, key=lambda item: (-item.document_frequency, item.token_id))
    frequencies = np.asarray([item.document_frequency for item in ranked], dtype=float)
    ranks = np.arange(1, len(ranked) + 1)
    upper_limit = summary.upper_document_frequency_limit

    figure = plt.figure(figsize=(15.6, 9.0), constrained_layout=False)
    grid = figure.add_gridspec(
        2,
        2,
        left=0.075,
        right=0.965,
        top=0.835,
        bottom=0.145,
        width_ratios=(1.62, 1.0),
        height_ratios=(1.52, 1.0),
        hspace=0.36,
        wspace=0.27,
    )
    ranked_axis = figure.add_subplot(grid[:, 0])
    histogram_axis = figure.add_subplot(grid[0, 1])
    summary_axis = figure.add_subplot(grid[1, 1])

    figure.suptitle(
        "Рисунок 4.2 — Документная частота токенов и границы фильтрации словаря LDA_prior",
        fontsize=14.2,
        fontweight="bold",
        y=0.955,
        color="#1D2B35",
    )
    figure.text(
        0.5,
        0.910,
        (
            f"Расширенный корпус: N = {document_count} документов; "
            f"итоговый словарь: V = {len(records)} токенов"
        ),
        ha="center",
        va="center",
        fontsize=9.3,
        color="#536976",
    )

    ranked_axis.axhspan(
        df_min,
        upper_limit,
        facecolor="#EAF5EE",
        alpha=0.88,
        zorder=0,
        label="допустимая область",
    )
    ranked_axis.plot(
        ranks,
        frequencies,
        linewidth=1.75,
        color="#2D6F8B",
        zorder=3,
    )
    ranked_axis.scatter(
        ranks,
        frequencies,
        s=22,
        color="#2D6F8B",
        edgecolors="white",
        linewidths=0.45,
        zorder=4,
    )
    ranked_axis.axhline(
        df_min,
        color="#A64646",
        linewidth=1.6,
        linestyle="--",
        zorder=5,
    )
    ranked_axis.axhline(
        upper_limit,
        color="#A64646",
        linewidth=1.6,
        linestyle="--",
        zorder=5,
    )
    ranked_axis.text(
        len(records) * 0.985,
        df_min + 3.0,
        f"df_min = {df_min}",
        ha="right",
        va="bottom",
        fontsize=8.3,
        fontweight="bold",
        color="#963D3D",
    )
    ranked_axis.text(
        len(records) * 0.985,
        upper_limit - 3.0,
        f"df_max_ratio = {df_max_ratio:.2f} → {upper_limit:.1f}",
        ha="right",
        va="top",
        fontsize=8.3,
        fontweight="bold",
        color="#963D3D",
    )

    annotation_indices = (0, 1, 2, len(ranked) - 3, len(ranked) - 2, len(ranked) - 1)
    annotation_offsets = ((22, 16), (28, -5), (28, -23), (-28, 18), (-28, 0), (-28, -20))
    for index, offset in zip(annotation_indices, annotation_offsets, strict=True):
        item = ranked[index]
        ranked_axis.annotate(
            f"{_short_label(item.token)}\ndf = {item.document_frequency}",
            xy=(index + 1, item.document_frequency),
            xytext=offset,
            textcoords="offset points",
            ha="left" if offset[0] > 0 else "right",
            va="center",
            fontsize=7.0,
            color="#334A5A",
            arrowprops={
                "arrowstyle": "-",
                "color": "#8499A4",
                "linewidth": 0.8,
            },
            bbox={
                "boxstyle": "round,pad=0.25",
                "facecolor": "white",
                "edgecolor": "#CBD5DA",
                "linewidth": 0.65,
                "alpha": 0.95,
            },
            zorder=6,
        )

    ranked_axis.set_title(
        "а) Ранжированное распределение document frequency",
        fontsize=10.2,
        fontweight="bold",
        pad=10,
    )
    ranked_axis.set_xlabel("Ранг токена по убыванию документной частоты")
    ranked_axis.set_ylabel("Число документов, содержащих токен")
    ranked_axis.set_xlim(0, len(records) + 2)
    ranked_axis.set_ylim(0, document_count * 1.04)
    ranked_axis.set_xticks(np.arange(0, len(records) + 1, 12))
    ranked_axis.grid(axis="y", linewidth=0.7, alpha=0.28)
    ranked_axis.spines[["top", "right"]].set_visible(False)

    percentage_axis = ranked_axis.secondary_yaxis(
        "right",
        functions=(
            lambda values: values / document_count * 100.0,
            lambda percentages: percentages / 100.0 * document_count,
        ),
    )
    percentage_axis.set_ylabel("Доля документов, %")
    percentage_axis.set_yticks([0, 20, 40, 60, 80, 100])

    bins = np.arange(
        np.floor(frequencies.min() / 5.0) * 5.0,
        np.ceil(frequencies.max() / 5.0) * 5.0 + 5.1,
        5.0,
    )
    histogram_axis.hist(
        frequencies,
        bins=bins,
        edgecolor="white",
        linewidth=0.9,
        color="#6E9CAF",
        alpha=0.95,
    )
    histogram_axis.axvline(
        summary.mean_document_frequency,
        color="#2F7A50",
        linewidth=1.7,
        linestyle="-",
        label=f"среднее = {summary.mean_document_frequency:.1f}",
    )
    histogram_axis.axvline(
        summary.median_document_frequency,
        color="#B77A2B",
        linewidth=1.6,
        linestyle="--",
        label=f"медиана = {summary.median_document_frequency:.1f}",
    )
    histogram_axis.set_title(
        "б) Распределение частот итогового словаря",
        fontsize=10.2,
        fontweight="bold",
        pad=10,
    )
    histogram_axis.set_xlabel("Document frequency")
    histogram_axis.set_ylabel("Число токенов")
    histogram_axis.grid(axis="y", linewidth=0.7, alpha=0.28)
    histogram_axis.spines[["top", "right"]].set_visible(False)
    histogram_axis.legend(loc="upper right", frameon=True, fontsize=7.7)

    summary_axis.axis("off")
    summary_axis.set_title(
        "в) Итог частотной фильтрации",
        fontsize=10.2,
        fontweight="bold",
        pad=7,
    )
    _add_summary_card(
        summary_axis,
        x=0.00,
        y=0.52,
        width=0.47,
        height=0.34,
        title="Нижняя граница",
        value=f"df ≥ {df_min}",
        note=f"не менее {df_min / document_count * 100:.1f}% корпуса",
        edgecolor="#A64646",
        facecolor="#FFF5F5",
    )
    _add_summary_card(
        summary_axis,
        x=0.53,
        y=0.52,
        width=0.47,
        height=0.34,
        title="Верхняя граница",
        value=f"df ≤ {upper_limit:.1f}",
        note=f"не более {df_max_ratio * 100:.0f}% корпуса",
        edgecolor="#A64646",
        facecolor="#FFF5F5",
    )
    _add_summary_card(
        summary_axis,
        x=0.00,
        y=0.06,
        width=0.47,
        height=0.34,
        title="Наблюдаемый диапазон",
        value=f"{summary.min_document_frequency}–{summary.max_document_frequency}",
        note=(
            f"{summary.min_document_frequency / document_count * 100:.1f}–"
            f"{summary.max_document_frequency / document_count * 100:.1f}% документов"
        ),
        edgecolor="#2D6F8B",
        facecolor="#F2F8FA",
    )
    _add_summary_card(
        summary_axis,
        x=0.53,
        y=0.06,
        width=0.47,
        height=0.34,
        title="Итоговый словарь",
        value=f"V = {summary.dictionary_token_count}",
        note="все токены внутри границ",
        edgecolor="#2F7A50",
        facecolor="#F3FAF5",
    )

    figure.text(
        0.5,
        0.075,
        (
            "Примечание. Диаграмма описывает уже сформированный итоговый словарь. "
            "В dictionary.json отсутствуют сведения о кандидатах до фильтрации; отброшенные токены и их число по итоговому словарю не определяются. "
            "поэтому рисунок не используется для оценки их числа."
        ),
        ha="center",
        va="center",
        fontsize=8.2,
        color="#536976",
    )
    figure.text(
        0.5,
        0.035,
        (
            f"Итог: document frequency всех {len(records)} токенов находится в диапазоне "
            f"{summary.min_document_frequency}–{summary.max_document_frequency} документов "
            f"при заданных границах [{df_min}; {upper_limit:.1f}]."
        ),
        ha="center",
        va="center",
        fontsize=8.4,
        fontweight="bold",
        color="#314A57",
    )
    return figure


def generate(
    project_root: str | Path = ".",
    *,
    dictionary_path: str | Path | None = None,
    metadata_path: str | Path | None = None,
    dpi: int = 300,
) -> FigureExportResult:
    """Сформировать рисунок 4.2 в форматах PNG и SVG."""

    root = Path(project_root).resolve()
    resolved_dictionary = (
        Path(dictionary_path).resolve()
        if dictionary_path is not None
        else root / DEFAULT_DICTIONARY_PATH
    )
    resolved_metadata = (
        Path(metadata_path).resolve()
        if metadata_path is not None
        else root / DEFAULT_METADATA_PATH
    )
    records = load_token_frequencies(resolved_dictionary)
    document_count, df_min, df_max_ratio = load_filtering_parameters(resolved_metadata)
    figure = build_figure(
        records,
        document_count=document_count,
        df_min=df_min,
        df_max_ratio=df_max_ratio,
    )
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
        description="Генерация рисунка 4.2 с частотами токенов и границами фильтрации.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("."),
        help="Корень проекта с данными и каталогом reports.",
    )
    parser.add_argument(
        "--dictionary",
        type=Path,
        default=None,
        help="Путь к dictionary.json. По умолчанию используется data/processed/lda/dictionary.json.",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help="Путь к corpus_metadata.json с параметрами df_min и df_max_ratio.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Разрешение PNG. Рекомендуемое значение — 300 dpi.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Выполнить CLI-генерацию рисунка 4.2."""

    args = build_argument_parser().parse_args(argv)
    result = generate(
        project_root=args.project_root,
        dictionary_path=args.dictionary,
        metadata_path=args.metadata,
        dpi=args.dpi,
    )
    print("Рисунок 4.2 успешно сформирован.")
    print(f"PNG: {result.png_path}")
    print(f"SVG: {result.svg_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
