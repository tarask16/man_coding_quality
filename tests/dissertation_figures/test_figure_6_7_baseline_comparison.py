"""Тесты генератора рисунка 6.7 со сравнением baseline-моделей."""

from __future__ import annotations

import csv
import struct
from pathlib import Path

import pytest

from manual_coding_sim.dissertation_figures.figure_6_7_baseline_comparison import (
    DEFAULT_INPUT_PATH,
    FILE_STEM,
    MODEL_ORDER,
    BaselineMetrics,
    generate,
    load_baseline_metrics,
    main,
    summarise_baselines,
)


def _write_baseline_report(
    project_root: Path,
    *,
    rows: tuple[tuple[str, float, float], ...] | None = None,
    model_column: str = "model",
) -> Path:
    """Создать минимальный CSV-отчёт baseline-моделей."""

    output_path = project_root / DEFAULT_INPUT_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    source_rows = rows or (
        ("mean", 0.097227, 0.112903),
        ("prior_only", 0.111822, 0.132113),
        ("full", 0.159244, 0.194403),
        ("theta_only", 0.304125, 0.360544),
    )
    with output_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=[model_column, "mae", "rmse"])
        writer.writeheader()
        for model, mae, rmse in source_rows:
            writer.writerow({model_column: model, "mae": mae, "rmse": rmse})
    return output_path


def _read_png_dimensions(path: Path) -> tuple[int, int]:
    """Прочитать ширину и высоту PNG без внешних библиотек."""

    with path.open("rb") as stream:
        signature = stream.read(24)
    if signature[:8] != b"\x89PNG\r\n\x1a\n":
        raise AssertionError("Файл не является PNG.")
    return struct.unpack(">II", signature[16:24])


def test_constants_fix_expected_output_contract() -> None:
    """Константы должны фиксировать состав и путь рисунка 6.7."""

    assert FILE_STEM == "baseline_comparison"
    assert DEFAULT_INPUT_PATH == Path("reports/chapter6/baseline_comparison.csv")
    assert MODEL_ORDER == ("mean", "prior_only", "full", "theta_only")


def test_load_baseline_metrics_preserves_required_order(tmp_path: Path) -> None:
    """Загрузчик должен возвращать четыре модели в методическом порядке."""

    source = _write_baseline_report(
        tmp_path,
        rows=(
            ("theta_only", 0.30, 0.36),
            ("full", 0.16, 0.19),
            ("mean", 0.10, 0.11),
            ("prior_only", 0.12, 0.13),
        ),
    )
    metrics = load_baseline_metrics(source)

    assert tuple(item.model for item in metrics) == MODEL_ORDER
    assert metrics[0].mae == pytest.approx(0.10)
    assert metrics[3].rmse == pytest.approx(0.36)


def test_load_baseline_metrics_accepts_report_aliases(tmp_path: Path) -> None:
    """Допустимые имена моделей и колонки должны нормализоваться."""

    source = _write_baseline_report(
        tmp_path,
        model_column="model_name",
        rows=(
            ("Mean baseline", 0.10, 0.11),
            ("Prior-only baseline", 0.12, 0.13),
            ("chapter5_model", 0.16, 0.19),
            ("Theta-only baseline", 0.30, 0.36),
        ),
    )

    metrics = load_baseline_metrics(source)
    assert tuple(item.model for item in metrics) == MODEL_ORDER


def test_load_baseline_metrics_rejects_missing_model(tmp_path: Path) -> None:
    """Отчёт без одной обязательной модели должен отклоняться."""

    source = _write_baseline_report(
        tmp_path,
        rows=(
            ("mean", 0.10, 0.11),
            ("prior_only", 0.12, 0.13),
            ("full", 0.16, 0.19),
        ),
    )

    with pytest.raises(ValueError, match="отсутствуют обязательные модели"):
        load_baseline_metrics(source)


def test_load_baseline_metrics_rejects_duplicate_model(tmp_path: Path) -> None:
    """Повтор одной модели должен считаться ошибкой отчёта."""

    source = _write_baseline_report(
        tmp_path,
        rows=(
            ("mean", 0.10, 0.11),
            ("mean_baseline", 0.10, 0.11),
            ("prior_only", 0.12, 0.13),
            ("full", 0.16, 0.19),
            ("theta_only", 0.30, 0.36),
        ),
    )

    with pytest.raises(ValueError, match="более одного раза"):
        load_baseline_metrics(source)


def test_load_baseline_metrics_rejects_negative_error(tmp_path: Path) -> None:
    """MAE и RMSE не могут быть отрицательными."""

    source = _write_baseline_report(
        tmp_path,
        rows=(
            ("mean", -0.10, 0.11),
            ("prior_only", 0.12, 0.13),
            ("full", 0.16, 0.19),
            ("theta_only", 0.30, 0.36),
        ),
    )

    with pytest.raises(ValueError, match="не может быть отрицательным"):
        load_baseline_metrics(source)


def test_summary_identifies_actual_ranking() -> None:
    """Сводка должна фиксировать место полной модели между prior-only и theta-only."""

    metrics = (
        BaselineMetrics("mean", 0.097227, 0.112903),
        BaselineMetrics("prior_only", 0.111822, 0.132113),
        BaselineMetrics("full", 0.159244, 0.194403),
        BaselineMetrics("theta_only", 0.304125, 0.360544),
    )
    summary = summarise_baselines(metrics)

    assert summary.best_mae_model == "mean"
    assert summary.best_rmse_model == "mean"
    assert summary.full_mae_rank == 3
    assert summary.full_rmse_rank == 3
    assert summary.full_better_than_theta is True
    assert summary.full_better_than_mean is False
    assert summary.full_better_than_prior is False


def test_generate_creates_expected_output_names(tmp_path: Path) -> None:
    """Генератор должен создавать PNG и SVG с установленным именем."""

    _write_baseline_report(tmp_path)
    result = generate(project_root=tmp_path, dpi=180)

    assert result.png_path.name == f"{FILE_STEM}.png"
    assert result.svg_path.name == f"{FILE_STEM}.svg"
    assert result.png_path.is_file()
    assert result.svg_path.is_file()


def test_png_and_svg_are_dissertation_ready(tmp_path: Path) -> None:
    """PNG должен быть крупным, а SVG — содержать редактируемые подписи."""

    _write_baseline_report(tmp_path)
    result = generate(project_root=tmp_path, dpi=300)
    width, height = _read_png_dimensions(result.png_path)
    svg = result.svg_path.read_text(encoding="utf-8")

    assert width >= 4300
    assert height >= 2200
    for text in (
        "Сравнение полной модели с базовыми прогнозами",
        "MAE и RMSE четырёх вариантов прогноза",
        "Mean baseline",
        "Prior-only",
        "Полная модель",
        "Theta-only",
        "меньше — лучше",
        "Полная модель лучше theta-only",
        "не заменяет анализ ранговой согласованности",
    ):
        assert text in svg


def test_cli_main_generates_files(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """CLI должен формировать оба формата и печатать их пути."""

    source = _write_baseline_report(tmp_path)
    exit_code = main(
        [
            "--project-root",
            str(tmp_path),
            "--input",
            str(source),
            "--dpi",
            "180",
        ]
    )
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "Рисунок 6.7 успешно сформирован." in output
    assert (tmp_path / "reports/chapter6/figures/baseline_comparison.png").is_file()
    assert (tmp_path / "reports/chapter6/figures/baseline_comparison.svg").is_file()
