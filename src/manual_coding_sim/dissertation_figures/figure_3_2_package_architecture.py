"""Генерация рисунка 3.2 с компонентной архитектурой программного пакета."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from manual_coding_sim.dissertation_figures.common import (
    FigureExportResult,
    configure_dissertation_style,
    export_figure,
)

OUTPUT_DIR = Path("reports/chapter3/figures")
FILE_STEM = "figure_3_2_package_architecture"


@dataclass(frozen=True, slots=True)
class PackageComponent:
    """Компонент программного пакета и его роль в вычислительном контуре."""

    code: str
    module_name: str
    title: str
    responsibility: tuple[str, ...]
    output: str
    layer: str


@dataclass(frozen=True, slots=True)
class ComponentConnection:
    """Направленная связь между компонентами архитектуры."""

    source: str
    target: str
    label: str


COMPONENTS: tuple[PackageComponent, ...] = (
    PackageComponent(
        code="runner",
        module_name="runner",
        title="Управляющий запуск",
        responsibility=(
            "загрузка конфигурации",
            "фиксация random_seed",
            "серийный запуск сценариев",
        ),
        output="параметры эксперимента",
        layer="orchestration",
    ),
    PackageComponent(
        code="message",
        module_name="message",
        title="Модель сообщения",
        responsibility=("генерация M", "длина и структура", "критичность элементов"),
        output="M, G",
        layer="domain",
    ),
    PackageComponent(
        code="procedure",
        module_name="procedure",
        title="Модель процедуры",
        responsibility=("правила и таблицы", "операции и ветвления", "номинальное время"),
        output="S",
        layer="domain",
    ),
    PackageComponent(
        code="operator",
        module_name="operator",
        title="Модель оператора",
        responsibility=("подготовка", "внимание и утомление", "склонность к ошибкам"),
        output="O",
        layer="domain",
    ),
    PackageComponent(
        code="condition",
        module_name="condition",
        title="Модель условий",
        responsibility=("давление времени", "помехи", "доступность инструкции"),
        output="U",
        layer="domain",
    ),
    PackageComponent(
        code="error",
        module_name="error",
        title="Модель ошибок",
        responsibility=("вероятности по типам", "генерация событий", "зависимость от сценария"),
        output="E",
        layer="domain",
    ),
    PackageComponent(
        code="control",
        module_name="control",
        title="Модель контроля",
        responsibility=("обнаружение", "исправление", "контрольные затраты"),
        output="K",
        layer="domain",
    ),
    PackageComponent(
        code="protocol",
        module_name="protocol",
        title="Симулятор протокола",
        responsibility=("выполнение A_i", "кодирование и декодирование", "журнал событий"),
        output="P_i, M′, T, E_i",
        layer="simulation",
    ),
    PackageComponent(
        code="features",
        module_name="features",
        title="Извлечение признаков",
        responsibility=("априорные признаки", "фактические признаки", "диагностика"),
        output="X_prior, X_fact",
        layer="analytics",
    ),
    PackageComponent(
        code="quality",
        module_name="quality",
        title="Расчёт качества",
        responsibility=("шесть частных критериев", "интегральное качество", "класс качества"),
        output="Y_fact, Q_fact",
        layer="analytics",
    ),
    PackageComponent(
        code="dataset",
        module_name="dataset",
        title="Сборка датасета",
        responsibility=("объединение по идентификаторам", "контроль схемы", "экспорт CSV / JSON"),
        output="исследовательские артефакты",
        layer="data",
    ),
)

EXPECTED_COMPONENT_CODES: tuple[str, ...] = (
    "runner",
    "message",
    "procedure",
    "operator",
    "condition",
    "error",
    "control",
    "protocol",
    "features",
    "quality",
    "dataset",
)

CONNECTIONS: tuple[ComponentConnection, ...] = (
    ComponentConnection("runner", "message", "параметры"),
    ComponentConnection("runner", "procedure", "параметры"),
    ComponentConnection("runner", "operator", "параметры"),
    ComponentConnection("runner", "condition", "параметры"),
    ComponentConnection("runner", "error", "параметры"),
    ComponentConnection("runner", "control", "параметры"),
    ComponentConnection("message", "protocol", "M, G"),
    ComponentConnection("procedure", "protocol", "S"),
    ComponentConnection("operator", "protocol", "O"),
    ComponentConnection("condition", "protocol", "U"),
    ComponentConnection("error", "protocol", "события ошибок"),
    ComponentConnection("control", "protocol", "события контроля"),
    ComponentConnection("protocol", "features", "P_i"),
    ComponentConnection("protocol", "quality", "M, M′, T, E_i"),
    ComponentConnection("features", "dataset", "X_prior, X_fact"),
    ComponentConnection("quality", "dataset", "Y_fact, Q_fact"),
)

REQUIRED_OUTPUT_GROUPS: tuple[str, ...] = (
    "prior_features.csv",
    "fact_features.csv",
    "diagnostic_features.csv",
    "quality_targets.csv",
    "protocols.csv",
)


def validate_package_architecture() -> None:
    """Проверить полноту компонентной архитектуры и обязательные связи."""

    codes = tuple(component.code for component in COMPONENTS)
    if codes != EXPECTED_COMPONENT_CODES:
        raise ValueError("Архитектура должна содержать 11 компонентов в заданном порядке.")

    connection_pairs = {(item.source, item.target) for item in CONNECTIONS}
    for domain_code in ("message", "procedure", "operator", "condition", "error", "control"):
        if ("runner", domain_code) not in connection_pairs:
            raise ValueError(f"Runner должен управлять компонентом {domain_code}.")
        if (domain_code, "protocol") not in connection_pairs:
            raise ValueError(f"Компонент {domain_code} должен передавать данные в protocol.")

    for pair in (
        ("protocol", "features"),
        ("protocol", "quality"),
        ("features", "dataset"),
        ("quality", "dataset"),
    ):
        if pair not in connection_pairs:
            raise ValueError(f"Отсутствует обязательная связь {pair[0]} → {pair[1]}.")

    for component in COMPONENTS:
        if len(component.responsibility) != 3:
            raise ValueError(
                f"Компонент {component.module_name} должен иметь три функции."
            )
        if not component.output.strip():
            raise ValueError(f"Для компонента {component.module_name} не указан выход.")


def _add_component_box(
    axis: plt.Axes,
    *,
    component: PackageComponent,
    x: float,
    y: float,
    width: float,
    height: float,
    facecolor: str,
    edgecolor: str,
    title_size: float = 9.2,
    body_size: float = 7.1,
    output_size: float = 7.1,
) -> None:
    """Добавить блок компонента программного пакета."""

    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.010,rounding_size=0.014",
        linewidth=1.35,
        edgecolor=edgecolor,
        facecolor=facecolor,
        zorder=4,
    )
    axis.add_patch(patch)
    axis.text(
        x + width / 2,
        y + height * 0.83,
        component.module_name,
        ha="center",
        va="center",
        fontsize=title_size,
        fontweight="bold",
        color="#17212B",
        family="DejaVu Sans Mono",
        zorder=5,
    )
    axis.text(
        x + width / 2,
        y + height * 0.65,
        component.title,
        ha="center",
        va="center",
        fontsize=body_size + 0.4,
        fontweight="bold",
        color="#314452",
        zorder=5,
    )
    axis.text(
        x + width / 2,
        y + height * 0.36,
        "\n".join(f"• {item}" for item in component.responsibility),
        ha="center",
        va="center",
        fontsize=body_size,
        color="#405464",
        linespacing=1.18,
        zorder=5,
    )
    axis.text(
        x + width / 2,
        y + height * 0.09,
        f"Выход: {component.output}",
        ha="center",
        va="center",
        fontsize=output_size,
        fontweight="bold",
        color=edgecolor,
        zorder=5,
    )


def _add_arrow(
    axis: plt.Axes,
    *,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str = "#607D8B",
    linewidth: float = 1.35,
    label: str | None = None,
    label_offset: tuple[float, float] = (0.0, 0.0),
    connectionstyle: str = "arc3,rad=0.0",
    zorder: int = 2,
) -> None:
    """Добавить направленную связь и необязательную подпись."""

    axis.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=linewidth,
            color=color,
            shrinkA=3,
            shrinkB=3,
            connectionstyle=connectionstyle,
            zorder=zorder,
        )
    )
    if label:
        axis.text(
            (start[0] + end[0]) / 2 + label_offset[0],
            (start[1] + end[1]) / 2 + label_offset[1],
            label,
            ha="center",
            va="center",
            fontsize=6.4,
            color=color,
            bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "none", "alpha": 0.9},
            zorder=zorder + 2,
        )


def _component_by_code(code: str) -> PackageComponent:
    """Вернуть компонент по его служебному коду."""

    return next(component for component in COMPONENTS if component.code == code)


def build_figure() -> plt.Figure:
    """Построить компонентную диаграмму программного пакета главы 3."""

    validate_package_architecture()
    configure_dissertation_style()

    figure, axis = plt.subplots(figsize=(15.4, 8.8))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")

    figure.suptitle(
        "Компонентная архитектура программного пакета компьютерного моделирования",
        fontsize=15.0,
        fontweight="bold",
        y=0.985,
        color="#17212B",
    )

    layer_specs = (
        (0.025, 0.805, 0.95, 0.145, "Оркестрация эксперимента", "#F5F7FA", "#78909C"),
        (0.025, 0.535, 0.95, 0.245, "Предметные модели сценария A_i", "#F7FAFC", "#5C7C8A"),
        (0.025, 0.355, 0.95, 0.155, "Имитационное выполнение", "#FBF7F0", "#9B6B30"),
        (0.025, 0.115, 0.95, 0.215, "Формирование исследовательских данных", "#F5FAF7", "#4F7D63"),
    )
    for x, y, width, height, title, facecolor, edgecolor in layer_specs:
        axis.add_patch(
            FancyBboxPatch(
                (x, y),
                width,
                height,
                boxstyle="round,pad=0.008,rounding_size=0.012",
                linewidth=1.0,
                edgecolor=edgecolor,
                facecolor=facecolor,
                alpha=0.72,
                zorder=0,
            )
        )
        axis.text(
            x + 0.012,
            y + height - 0.017,
            title,
            ha="left",
            va="top",
            fontsize=8.1,
            fontweight="bold",
            color=edgecolor,
            zorder=1,
        )

    axis.text(
        0.12,
        0.865,
        "Конфигурация YAML\n+ random_seed\n+ число сценариев",
        ha="center",
        va="center",
        fontsize=8.2,
        fontweight="bold",
        color="#455A64",
        bbox={
            "boxstyle": "round,pad=0.45",
            "fc": "#FFFFFF",
            "ec": "#78909C",
            "lw": 1.25,
        },
        zorder=4,
    )

    _add_component_box(
        axis,
        component=_component_by_code("runner"),
        x=0.37,
        y=0.825,
        width=0.26,
        height=0.105,
        facecolor="#E8EEF3",
        edgecolor="#4F6F82",
        title_size=9.0,
        body_size=6.7,
        output_size=6.8,
    )
    axis.text(
        0.86,
        0.865,
        "Журнал запуска\nверсии и параметры\nконтрольные суммы",
        ha="center",
        va="center",
        fontsize=8.0,
        fontweight="bold",
        color="#455A64",
        bbox={
            "boxstyle": "round,pad=0.45",
            "fc": "#FFFFFF",
            "ec": "#78909C",
            "lw": 1.25,
        },
        zorder=4,
    )
    _add_arrow(axis, start=(0.21, 0.865), end=(0.365, 0.865), label="вход")
    _add_arrow(axis, start=(0.635, 0.865), end=(0.78, 0.865), label="протоколирование")

    domain_codes = ("message", "procedure", "operator", "condition", "error", "control")
    domain_x = (0.045, 0.202, 0.359, 0.516, 0.673, 0.830)
    domain_positions: dict[str, tuple[float, float, float, float]] = {}
    for code, x in zip(domain_codes, domain_x, strict=True):
        component = _component_by_code(code)
        domain_positions[code] = (x, 0.565, 0.125, 0.18)
        _add_component_box(
            axis,
            component=component,
            x=x,
            y=0.565,
            width=0.125,
            height=0.18,
            facecolor="#EAF2F6" if code not in {"error", "control"} else "#F7EEE6",
            edgecolor="#4F7183" if code not in {"error", "control"} else "#9A6531",
            title_size=8.3,
            body_size=6.35,
            output_size=6.5,
        )
        _add_arrow(
            axis,
            start=(0.50, 0.823),
            end=(x + 0.0625, 0.748),
            color="#78909C",
            linewidth=0.95,
            connectionstyle=f"arc3,rad={(x - 0.5) * 0.18}",
        )

    _add_component_box(
        axis,
        component=_component_by_code("protocol"),
        x=0.355,
        y=0.385,
        width=0.29,
        height=0.105,
        facecolor="#F4E8D8",
        edgecolor="#9A6531",
        title_size=9.4,
        body_size=6.9,
        output_size=7.0,
    )

    for code in domain_codes:
        x, y, width, _height = domain_positions[code]
        label = next(item.label for item in CONNECTIONS if item.source == code and item.target == "protocol")
        _add_arrow(
            axis,
            start=(x + width / 2, y - 0.003),
            end=(0.50, 0.493),
            color="#607D8B" if code not in {"error", "control"} else "#9A6531",
            linewidth=1.0,
            label=label,
            label_offset=(0.0, 0.005),
            connectionstyle=f"arc3,rad={(x - 0.5) * -0.20}",
        )

    _add_component_box(
        axis,
        component=_component_by_code("features"),
        x=0.105,
        y=0.145,
        width=0.235,
        height=0.145,
        facecolor="#E5F1EA",
        edgecolor="#4F7D63",
        title_size=9.0,
        body_size=6.8,
        output_size=7.0,
    )
    _add_component_box(
        axis,
        component=_component_by_code("quality"),
        x=0.660,
        y=0.145,
        width=0.235,
        height=0.145,
        facecolor="#E5F1EA",
        edgecolor="#4F7D63",
        title_size=9.0,
        body_size=6.8,
        output_size=7.0,
    )
    _add_component_box(
        axis,
        component=_component_by_code("dataset"),
        x=0.385,
        y=0.145,
        width=0.23,
        height=0.145,
        facecolor="#DCECE3",
        edgecolor="#35684E",
        title_size=9.0,
        body_size=6.8,
        output_size=7.0,
    )

    _add_arrow(
        axis,
        start=(0.43, 0.383),
        end=(0.235, 0.293),
        color="#4F7D63",
        label="P_i",
        connectionstyle="arc3,rad=0.10",
    )
    _add_arrow(
        axis,
        start=(0.57, 0.383),
        end=(0.775, 0.293),
        color="#4F7D63",
        label="M, M′, T, E_i",
        connectionstyle="arc3,rad=-0.10",
    )
    _add_arrow(
        axis,
        start=(0.342, 0.217),
        end=(0.382, 0.217),
        color="#35684E",
        label="X_prior / X_fact",
        label_offset=(0.0, 0.025),
    )
    _add_arrow(
        axis,
        start=(0.658, 0.217),
        end=(0.618, 0.217),
        color="#35684E",
        label="Y_fact / Q_fact",
        label_offset=(0.0, 0.025),
    )

    output_text = (
        "Выходные артефакты:  prior_features.csv  |  fact_features.csv  |  "
        "diagnostic_features.csv  |  quality_targets.csv  |  protocols.csv"
    )
    axis.text(
        0.5,
        0.078,
        output_text,
        ha="center",
        va="center",
        fontsize=7.9,
        fontweight="bold",
        color="#2F5E47",
        bbox={
            "boxstyle": "round,pad=0.42",
            "fc": "#F4FAF6",
            "ec": "#4F7D63",
            "lw": 1.15,
        },
        zorder=4,
    )
    _add_arrow(axis, start=(0.50, 0.143), end=(0.50, 0.101), color="#35684E")

    axis.text(
        0.5,
        0.026,
        "Архитектура повторяет формальную модель A_i и обеспечивает раздельное формирование "
        "X_prior, X_fact и Y_fact без смешения априорного и фактического контуров.",
        ha="center",
        va="center",
        fontsize=7.6,
        color="#536773",
        style="italic",
    )

    figure.subplots_adjust(left=0.015, right=0.985, top=0.955, bottom=0.025)
    return figure


def generate(*, project_root: Path, dpi: int = 300) -> FigureExportResult:
    """Сформировать рисунок 3.2 в PNG и SVG."""

    return export_figure(
        build_figure(),
        project_root=project_root,
        relative_output_dir=OUTPUT_DIR,
        file_stem=FILE_STEM,
        dpi=dpi,
    )


def build_argument_parser() -> argparse.ArgumentParser:
    """Создать парсер аргументов командной строки."""

    parser = argparse.ArgumentParser(
        description="Сформировать компонентную диаграмму программного пакета главы 3."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Корень проекта, относительно которого сохраняются отчёты.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Разрешение PNG; значение должно быть не ниже 150 dpi.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Выполнить CLI-генерацию рисунка 3.2."""

    args = build_argument_parser().parse_args(argv)
    result = generate(project_root=args.project_root, dpi=args.dpi)
    print("Рисунок 3.2 успешно сформирован.")
    print(f"PNG: {result.png_path}")
    print(f"SVG: {result.svg_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
