"""Локальные тесты генератора рисунка 5.2."""

from __future__ import annotations

import csv
import struct
from pathlib import Path

import numpy as np
import pytest

from manual_coding_sim.dissertation_figures.figure_5_2_latent_quality_component import (
    DEFAULT_INPUT_PATH,
    DIRECTION_WEIGHTS,
    FILE_STEM,
    ThetaProfile,
    calculate_latent_quality,
    calculate_latent_values,
    calculate_summary,
    generate,
    load_theta_profiles,
    validate_theta_profiles,
)


def _read_png_dimensions(path: Path) -> tuple[int, int]:
    """Прочитать размеры PNG из заголовка IHDR."""

    with path.open("rb") as stream:
        stream.read(16)
        return struct.unpack(">II", stream.read(8))



def _reference_profiles() -> tuple[ThetaProfile, ...]:
    """Сформировать небольшой корректный набор латентных профилей."""

    return (
        ThetaProfile("scn_0001", "prt_0001", 0.10, 0.20, 0.70, 3),
        ThetaProfile("scn_0002", "prt_0002", 0.55, 0.35, 0.10, 3),
        ThetaProfile("scn_0003", "prt_0003", 0.15, 0.65, 0.20, 3),
        ThetaProfile("scn_0004", "prt_0004", 0.05, 0.05, 0.90, 3),
        ThetaProfile("scn_0005", "prt_0005", 0.25, 0.25, 0.50, 3),
        ThetaProfile("scn_0006", "prt_0006", 0.33, 0.33, 0.34, 3),
    )



def _write_reference_input(project_root: Path) -> Path:
    """Записать тестовый theta_prior.csv в стандартный каталог проекта."""

    path = project_root / DEFAULT_INPUT_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "scenario_id",
                "protocol_id",
                "theta_0",
                "theta_1",
                "theta_2",
                "selected_k",
            )
        )
        for profile in _reference_profiles():
            writer.writerow(
                (
                    profile.scenario_id,
                    profile.protocol_id,
                    profile.theta_0,
                    profile.theta_1,
                    profile.theta_2,
                    profile.selected_k,
                )
            )
    return path



def test_direction_weights_match_chapter5_method() -> None:
    """Направления факторов должны соответствовать формуле главы 5."""

    assert DIRECTION_WEIGHTS == (-1.0, -1.0, 1.0)



def test_latent_quality_equals_theta_2_for_normalized_profile() -> None:
    """При сумме theta, равной единице, q_latent должен совпадать с theta_2."""

    for profile in _reference_profiles():
        assert calculate_latent_quality(profile) == pytest.approx(profile.theta_2)



def test_calculate_latent_values_preserves_sample_size() -> None:
    """Расчёт должен возвращать по одному значению на сценарий."""

    values = calculate_latent_values(_reference_profiles())
    assert values.shape == (6,)
    assert np.all((0.0 <= values) & (values <= 1.0))



def test_summary_contains_range_and_zero_identity_error() -> None:
    """Сводка должна содержать диапазон и подтверждать точное тождество."""

    summary = calculate_summary(_reference_profiles())
    assert summary.sample_size == 6
    assert summary.minimum == pytest.approx(0.10)
    assert summary.maximum == pytest.approx(0.90)
    assert summary.max_identity_error <= 1e-12
    assert len(summary.mean_signed_terms) == 3



def test_validate_rejects_non_normalized_profile() -> None:
    """Профиль с суммой компонент, отличной от единицы, должен отклоняться."""

    invalid = (ThetaProfile("scn", "prt", 0.2, 0.2, 0.2, 3),)
    with pytest.raises(ValueError, match="равна единице"):
        validate_theta_profiles(invalid)



def test_validate_rejects_wrong_topic_count() -> None:
    """Профиль с selected_k, отличным от трёх, должен отклоняться."""

    invalid = (ThetaProfile("scn", "prt", 0.2, 0.3, 0.5, 4),)
    with pytest.raises(ValueError, match="selected_k = 3"):
        validate_theta_profiles(invalid)



def test_load_theta_profiles_reads_reference_csv(tmp_path: Path) -> None:
    """Загрузчик должен читать корректный тестовый CSV."""

    source = _write_reference_input(tmp_path)
    profiles = load_theta_profiles(source)
    assert len(profiles) == 6
    assert profiles[0].scenario_id == "scn_0001"
    assert profiles[-1].theta_2 == pytest.approx(0.34)



def test_load_theta_profiles_rejects_missing_column(tmp_path: Path) -> None:
    """CSV без обязательной колонки должен отклоняться."""

    path = tmp_path / "theta_prior.csv"
    path.write_text("scenario_id,protocol_id,theta_0,theta_1,selected_k\n", encoding="utf-8")
    with pytest.raises(ValueError, match="отсутствуют обязательные колонки"):
        load_theta_profiles(path)



def test_png_and_svg_are_dissertation_ready(tmp_path: Path) -> None:
    """PNG должен быть крупным, а SVG — содержать редактируемые подписи."""

    _write_reference_input(tmp_path)
    result = generate(project_root=tmp_path, dpi=300)
    width, height = _read_png_dimensions(result.png_path)
    svg = result.svg_path.read_text(encoding="utf-8")

    assert result.png_path.name == f"{FILE_STEM}.png"
    assert result.svg_path.name == f"{FILE_STEM}.svg"
    assert width >= 4200
    assert height >= 2200
    for text in (
        "Распределение латентной компоненты качества",
        "Частотное распределение латентной компоненты",
        "Диапазон и квартильная структура",
        "Направления латентных компонентов",
        "q_latent = θ₂",
        "процедурная трудоёмкость",
        "операционный риск",
        "благоприятные условия",
        "не является наблюдаемой фактической оценкой качества",
    ):
        assert text in svg
    assert "<text" in svg
    assert "<path" in svg



def test_cli_main_generates_files(tmp_path: Path) -> None:
    """CLI должен завершаться успешно и создавать оба формата."""

    from manual_coding_sim.dissertation_figures.figure_5_2_latent_quality_component import main

    _write_reference_input(tmp_path)
    exit_code = main(["--project-root", str(tmp_path), "--dpi", "300"])
    assert exit_code == 0
    output_dir = tmp_path / "reports" / "chapter5" / "figures"
    assert (output_dir / f"{FILE_STEM}.png").is_file()
    assert (output_dir / f"{FILE_STEM}.svg").is_file()
