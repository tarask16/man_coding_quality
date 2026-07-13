"""Локальные тесты генератора рисунка 4.6."""

from __future__ import annotations

import csv
import struct
from pathlib import Path

import pytest

from manual_coding_sim.dissertation_figures.figure_4_6_theta_profile_distribution import (
    FILE_STEM,
    ThetaProfile,
    calculate_topic_summaries,
    generate,
    load_theta_profiles,
    validate_theta_profiles,
)


def _write_reference_input(root: Path) -> Path:
    """Создать согласованный theta_prior.csv для локальных тестов."""

    report_dir = root / "reports" / "chapter4"
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / "theta_prior.csv"
    rows = [
        ("scn_000", "prt_000", 0.80, 0.10, 0.10),
        ("scn_001", "prt_001", 0.70, 0.20, 0.10),
        ("scn_002", "prt_002", 0.10, 0.80, 0.10),
        ("scn_003", "prt_003", 0.10, 0.70, 0.20),
        ("scn_004", "prt_004", 0.10, 0.10, 0.80),
        ("scn_005", "prt_005", 0.20, 0.10, 0.70),
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "scenario_id",
                "protocol_id",
                "theta_0",
                "theta_1",
                "theta_2",
                "selected_k",
            ],
        )
        writer.writeheader()
        for scenario_id, protocol_id, theta_0, theta_1, theta_2 in rows:
            writer.writerow(
                {
                    "scenario_id": scenario_id,
                    "protocol_id": protocol_id,
                    "theta_0": theta_0,
                    "theta_1": theta_1,
                    "theta_2": theta_2,
                    "selected_k": 3,
                }
            )
    return path


def _read_png_dimensions(path: Path) -> tuple[int, int]:
    """Прочитать размеры PNG из заголовка IHDR."""

    with path.open("rb") as stream:
        stream.read(16)
        return struct.unpack(">II", stream.read(8))


def test_load_theta_profiles_reads_expected_rows(tmp_path: Path) -> None:
    """Загрузчик должен прочитать все профили и три компоненты."""

    profiles = load_theta_profiles(_write_reference_input(tmp_path))
    assert len(profiles) == 6
    assert profiles[0].values == pytest.approx((0.8, 0.1, 0.1))


def test_validate_rejects_empty_input() -> None:
    """Пустой набор профилей должен отклоняться."""

    with pytest.raises(ValueError, match="не должен быть пустым"):
        validate_theta_profiles(())


def test_validate_rejects_non_normalized_profile() -> None:
    """Профиль с суммой, отличной от единицы, должен отклоняться."""

    profile = ThetaProfile("scn", "prt", 0.5, 0.3, 0.3, 3)
    with pytest.raises(ValueError, match="равна единице"):
        validate_theta_profiles((profile,))


def test_validate_rejects_wrong_selected_k() -> None:
    """Число факторов, отличное от трёх, должно отклоняться."""

    profile = ThetaProfile("scn", "prt", 0.4, 0.3, 0.3, 4)
    with pytest.raises(ValueError, match="selected_k = 3"):
        validate_theta_profiles((profile,))


def test_validate_rejects_duplicate_scenario_id() -> None:
    """Повтор идентификатора сценария должен отклоняться."""

    profiles = (
        ThetaProfile("scn", "prt_1", 0.6, 0.2, 0.2, 3),
        ThetaProfile("scn", "prt_2", 0.2, 0.6, 0.2, 3),
    )
    with pytest.raises(ValueError, match="сценариев должны быть уникальными"):
        validate_theta_profiles(profiles)


def test_dominant_topic_uses_maximum_component() -> None:
    """Доминирующая тема должна соответствовать максимальной компоненте."""

    profile = ThetaProfile("scn", "prt", 0.1, 0.2, 0.7, 3)
    assert profile.dominant_topic == 2


def test_calculate_summaries_counts_dominant_topics(tmp_path: Path) -> None:
    """Сводка должна правильно подсчитать доминирующие факторы."""

    summaries = calculate_topic_summaries(
        load_theta_profiles(_write_reference_input(tmp_path))
    )
    assert [item.dominant_count for item in summaries] == [2, 2, 2]
    assert [item.dominant_share for item in summaries] == pytest.approx(
        [1 / 3, 1 / 3, 1 / 3]
    )


def test_generate_creates_expected_paths(tmp_path: Path) -> None:
    """Генератор должен создать PNG и SVG с установленным именем."""

    _write_reference_input(tmp_path)
    result = generate(project_root=tmp_path, dpi=300)
    assert result.png_path.name == f"{FILE_STEM}.png"
    assert result.svg_path.name == f"{FILE_STEM}.svg"
    assert result.png_path.is_file()
    assert result.svg_path.is_file()


def test_png_and_svg_are_dissertation_ready(tmp_path: Path) -> None:
    """PNG должен быть крупным, а SVG — содержать редактируемые подписи."""

    _write_reference_input(tmp_path)
    result = generate(project_root=tmp_path, dpi=300)
    width, height = _read_png_dimensions(result.png_path)
    svg = result.svg_path.read_text(encoding="utf-8")
    assert width >= 4300
    assert height >= 2300
    for text in (
        "Распределение априорных латентных профилей",
        "Распределение компонент θ₀–θ₂",
        "Доминирующий фактор",
        "Процедурная",
        "трудоёмкость",
        "Операционный",
        "риск",
        "Благоприятные",
        "условия",
        "argmax(θ₀, θ₁, θ₂)",
    ):
        assert text in svg
    assert "<text" in svg
    assert "<path" in svg


def test_cli_main_generates_files(tmp_path: Path) -> None:
    """CLI должен завершаться успешно и создавать оба формата."""

    _write_reference_input(tmp_path)
    from manual_coding_sim.dissertation_figures.figure_4_6_theta_profile_distribution import (
        main,
    )

    exit_code = main(["--project-root", str(tmp_path), "--dpi", "300"])
    assert exit_code == 0
    output_dir = tmp_path / "reports" / "chapter4" / "figures"
    assert (output_dir / f"{FILE_STEM}.png").is_file()
    assert (output_dir / f"{FILE_STEM}.svg").is_file()
