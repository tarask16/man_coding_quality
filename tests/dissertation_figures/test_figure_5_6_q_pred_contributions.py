"""Локальные тесты генератора рисунка 5.6."""

from __future__ import annotations

import csv
import struct
from pathlib import Path

import pytest

from manual_coding_sim.dissertation_figures.figure_5_6_q_pred_contributions import (
    CRITERIA,
    DEFAULT_COMPONENTS_PATH,
    DEFAULT_Q_PRED_PATH,
    FILE_STEM,
    ComponentRow,
    QPredRow,
    calculate_summary,
    generate,
    load_components,
    load_q_pred,
    validate_component_rows,
    validate_q_pred_rows,
)


def _read_png_dimensions(path: Path) -> tuple[int, int]:
    """Прочитать размеры PNG из заголовка IHDR."""

    with path.open("rb") as stream:
        stream.read(16)
        return struct.unpack(">II", stream.read(8))


def _reference_rows() -> tuple[tuple[ComponentRow, ...], tuple[QPredRow, ...]]:
    """Сформировать согласованный набор данных для двух сценариев."""

    criteria = [spec.key for spec in CRITERIA]
    observed = {
        "q_acc": 0.65,
        "q_time": 0.70,
        "q_effort": 0.70,
        "q_res": 0.60,
        "q_rep": 0.60,
        "q_fit": 0.65,
    }
    component_rows: list[ComponentRow] = []
    q_pred_rows: list[QPredRow] = []
    for index, q_latent in enumerate((0.30, 0.70), start=1):
        feature = {
            criterion: 0.35 + 0.04 * position + 0.02 * index
            for position, criterion in enumerate(criteria)
        }
        latent = {criterion: q_latent for criterion in criteria}
        latent_weights = {criterion: 1.0 - observed[criterion] for criterion in criteria}
        predictions = {
            criterion: observed[criterion] * feature[criterion]
            + latent_weights[criterion] * q_latent
            for criterion in criteria
        }
        weights = {criterion: 1.0 / 6.0 for criterion in criteria}
        contributions = {
            criterion: weights[criterion] * predictions[criterion]
            for criterion in criteria
        }
        component_rows.append(
            ComponentRow(
                scenario_id=f"scn_{index:04d}",
                protocol_id=f"prt_{index:04d}",
                q_latent=q_latent,
                feature_components=feature,
                latent_components=latent,
                observed_weights=observed,
                latent_weights=latent_weights,
                predictions=predictions,
            )
        )
        q_pred_rows.append(
            QPredRow(
                scenario_id=f"scn_{index:04d}",
                protocol_id=f"prt_{index:04d}",
                criterion_weights=weights,
                criterion_contributions=contributions,
                q_pred=sum(contributions.values()),
            )
        )
    return tuple(component_rows), tuple(q_pred_rows)


def _write_reference_inputs(project_root: Path) -> tuple[Path, Path]:
    """Записать тестовые q_pred_components.csv и q_pred.csv."""

    component_rows, q_pred_rows = _reference_rows()
    components_path = project_root / DEFAULT_COMPONENTS_PATH
    components_path.parent.mkdir(parents=True, exist_ok=True)
    component_fields = ["scenario_id", "protocol_id", "q_latent"]
    for spec in CRITERIA:
        component_fields.extend(
            [
                f"{spec.key}_feature_component",
                f"{spec.key}_latent_component",
                f"{spec.key}_observed_weight",
                f"{spec.key}_latent_weight",
                f"{spec.key}_pred",
            ]
        )
    with components_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=component_fields)
        writer.writeheader()
        for row in component_rows:
            payload: dict[str, object] = {
                "scenario_id": row.scenario_id,
                "protocol_id": row.protocol_id,
                "q_latent": row.q_latent,
            }
            for spec in CRITERIA:
                key = spec.key
                payload[f"{key}_feature_component"] = row.feature_components[key]
                payload[f"{key}_latent_component"] = row.latent_components[key]
                payload[f"{key}_observed_weight"] = row.observed_weights[key]
                payload[f"{key}_latent_weight"] = row.latent_weights[key]
                payload[f"{key}_pred"] = row.predictions[key]
            writer.writerow(payload)

    q_pred_path = project_root / DEFAULT_Q_PRED_PATH
    q_pred_fields = ["scenario_id", "protocol_id"]
    for spec in CRITERIA:
        q_pred_fields.extend([f"{spec.key}_weight", f"{spec.key}_contribution"])
    q_pred_fields.append("q_pred")
    with q_pred_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=q_pred_fields)
        writer.writeheader()
        for row in q_pred_rows:
            payload = {
                "scenario_id": row.scenario_id,
                "protocol_id": row.protocol_id,
                "q_pred": row.q_pred,
            }
            for spec in CRITERIA:
                key = spec.key
                payload[f"{key}_weight"] = row.criterion_weights[key]
                payload[f"{key}_contribution"] = row.criterion_contributions[key]
            writer.writerow(payload)
    return components_path, q_pred_path


def test_criteria_match_six_chapter5_components() -> None:
    """Рисунок должен учитывать шесть частных критериев главы 5."""

    assert [spec.key for spec in CRITERIA] == [
        "q_acc",
        "q_time",
        "q_effort",
        "q_res",
        "q_rep",
        "q_fit",
    ]


def test_loaders_read_reference_inputs(tmp_path: Path) -> None:
    """Загрузчики должны прочитать согласованные CSV-файлы."""

    components_path, q_pred_path = _write_reference_inputs(tmp_path)
    assert len(load_components(components_path)) == 2
    assert len(load_q_pred(q_pred_path)) == 2


def test_component_validation_rejects_duplicate_key() -> None:
    """Повтор пары ключей в компонентах должен отклоняться."""

    component_rows, _ = _reference_rows()
    with pytest.raises(ValueError, match="должны быть уникальными"):
        validate_component_rows((component_rows[0], component_rows[0]))


def test_component_validation_rejects_non_normalized_inner_weights() -> None:
    """Веса признаковой и латентной частей должны давать единицу."""

    component_rows, _ = _reference_rows()
    source = component_rows[0]
    broken_observed = dict(source.observed_weights)
    broken_observed["q_acc"] = 0.80
    broken = ComponentRow(
        scenario_id=source.scenario_id,
        protocol_id=source.protocol_id,
        q_latent=source.q_latent,
        feature_components=source.feature_components,
        latent_components=source.latent_components,
        observed_weights=broken_observed,
        latent_weights=source.latent_weights,
        predictions=source.predictions,
    )
    with pytest.raises(ValueError, match="должны давать единицу"):
        validate_component_rows((broken,))


def test_q_pred_validation_rejects_non_normalized_criterion_weights() -> None:
    """Сумма весов критериев должна быть равна единице."""

    _, q_pred_rows = _reference_rows()
    source = q_pred_rows[0]
    weights = dict(source.criterion_weights)
    weights["q_acc"] = 0.30
    broken = QPredRow(
        scenario_id=source.scenario_id,
        protocol_id=source.protocol_id,
        criterion_weights=weights,
        criterion_contributions=source.criterion_contributions,
        q_pred=source.q_pred,
    )
    with pytest.raises(ValueError, match="должна быть равна единице"):
        validate_q_pred_rows((broken,))


def test_summary_rejects_mismatched_key_sets() -> None:
    """Два входных файла должны описывать один набор сценариев."""

    component_rows, q_pred_rows = _reference_rows()
    altered = QPredRow(
        scenario_id="other",
        protocol_id="other",
        criterion_weights=q_pred_rows[0].criterion_weights,
        criterion_contributions=q_pred_rows[0].criterion_contributions,
        q_pred=q_pred_rows[0].q_pred,
    )
    with pytest.raises(ValueError, match="Наборы ключей"):
        calculate_summary(component_rows, (altered, q_pred_rows[1]))


def test_summary_exactly_reconstructs_mean_q_pred() -> None:
    """Сумма средних вкладов должна восстанавливать средний Q_pred."""

    component_rows, q_pred_rows = _reference_rows()
    summary = calculate_summary(component_rows, q_pred_rows)
    assert summary.count == 2
    assert summary.feature_total + summary.latent_total == pytest.approx(
        summary.q_pred_mean
    )
    assert sum(item.total_contribution for item in summary.criteria) == pytest.approx(
        summary.q_pred_mean
    )
    assert summary.maximum_reconstruction_error < 1e-12


def test_source_shares_sum_to_one() -> None:
    """Доли признакового и латентного источников должны давать единицу."""

    component_rows, q_pred_rows = _reference_rows()
    summary = calculate_summary(component_rows, q_pred_rows)
    assert summary.feature_share + summary.latent_share == pytest.approx(1.0)
    assert 0.0 < summary.feature_share < 1.0
    assert 0.0 < summary.latent_share < 1.0


def test_png_and_svg_are_dissertation_ready(tmp_path: Path) -> None:
    """PNG должен быть крупным, а SVG — содержать редактируемые подписи."""

    _write_reference_inputs(tmp_path)
    result = generate(project_root=tmp_path, dpi=300)
    width, height = _read_png_dimensions(result.png_path)
    svg = result.svg_path.read_text(encoding="utf-8")

    assert result.png_path.name == f"{FILE_STEM}.png"
    assert result.svg_path.name == f"{FILE_STEM}.svg"
    assert width >= 4300
    assert height >= 2200
    for text in (
        "Вклад частных критериев и латентной компоненты в Q_pred",
        "Средний вклад частных критериев",
        "Состав среднего Q_pred по критериям",
        "Источники среднего Q_pred",
        "априорные признаки",
        "латентная компонента",
        "Точная декомпозиция",
        "не являются причинными эффектами",
    ):
        assert text in svg
    assert "<text" in svg
    assert "<path" in svg


def test_cli_main_generates_files(tmp_path: Path) -> None:
    """CLI должен создавать PNG и SVG в стандартном каталоге."""

    from manual_coding_sim.dissertation_figures.figure_5_6_q_pred_contributions import (
        main,
    )

    _write_reference_inputs(tmp_path)
    exit_code = main(["--project-root", str(tmp_path), "--dpi", "300"])
    assert exit_code == 0
    output_dir = tmp_path / "reports" / "chapter5" / "figures"
    assert (output_dir / f"{FILE_STEM}.png").is_file()
    assert (output_dir / f"{FILE_STEM}.svg").is_file()
