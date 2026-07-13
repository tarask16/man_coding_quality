"""Тесты генератора рисунка 2.2."""

from __future__ import annotations

import struct
from pathlib import Path

import pytest

from manual_coding_sim.dissertation_figures.figure_2_2_encoding_decoding_process import (
    CONTROL_LOOPS,
    ERROR_CHANNELS,
    EXPECTED_CONTROL_SYMBOLS,
    EXPECTED_NODE_SYMBOLS,
    FILE_STEM,
    OUTPUT_DIR,
    PROCESS_NODES,
    generate,
    main,
    validate_process_model,
)


def _read_png_dimensions(path: Path) -> tuple[int, int]:
    """Прочитать размеры PNG из заголовка без внешних библиотек."""

    data = path.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    return struct.unpack(">II", data[16:24])


def test_process_has_required_node_sequence() -> None:
    """Основной процесс должен иметь последовательность M, E_h, C, D_h, M′."""

    validate_process_model()

    assert tuple(node.symbol for node in PROCESS_NODES) == EXPECTED_NODE_SYMBOLS
    assert len(PROCESS_NODES) == 5


def test_error_channels_cover_encoding_recording_and_decoding() -> None:
    """Схема должна охватывать основные места возникновения ошибок."""

    titles = {channel.title for channel in ERROR_CHANNELS}
    targets = {channel.target_symbol for channel in ERROR_CHANNELS}

    assert "Ошибки восприятия" in titles
    assert "Ошибки применения правила" in titles
    assert "Ошибки фиксации результата" in titles
    assert "Ошибки декодирования" in titles
    assert {"E_h", "C", "D_h"}.issubset(targets)


def test_control_loops_have_detection_check_and_correction() -> None:
    """Контуры K_e и K_d должны включать три контрольных действия."""

    assert tuple(loop.symbol for loop in CONTROL_LOOPS) == EXPECTED_CONTROL_SYMBOLS
    for loop in CONTROL_LOOPS:
        assert tuple(loop.actions) == ("обнаружение", "проверка", "исправление")


def test_validation_rejects_broken_process(monkeypatch: pytest.MonkeyPatch) -> None:
    """Проверка должна отклонять процесс без промежуточного представления C."""

    import manual_coding_sim.dissertation_figures.figure_2_2_encoding_decoding_process as module

    monkeypatch.setattr(
        module,
        "PROCESS_NODES",
        tuple(node for node in module.PROCESS_NODES if node.symbol != "C"),
    )

    with pytest.raises(ValueError, match="M → E_h → C → D_h → M′"):
        module.validate_process_model()


def test_generate_creates_png_and_svg(tmp_path: Path) -> None:
    """Генератор должен создать оба обязательных формата рисунка."""

    result = generate(project_root=tmp_path, dpi=300)

    expected_dir = tmp_path / OUTPUT_DIR
    assert result.png_path == expected_dir / f"{FILE_STEM}.png"
    assert result.svg_path == expected_dir / f"{FILE_STEM}.svg"
    assert result.png_path.is_file()
    assert result.svg_path.is_file()
    assert result.png_path.stat().st_size > 130_000
    assert result.svg_path.stat().st_size > 20_000


def test_png_has_dissertation_ready_resolution(tmp_path: Path) -> None:
    """PNG должен иметь разрешение, достаточное для печатного документа."""

    result = generate(project_root=tmp_path, dpi=300)
    width, height = _read_png_dimensions(result.png_path)

    assert width >= 4000
    assert height >= 2100


def test_svg_contains_process_errors_controls_and_delays(tmp_path: Path) -> None:
    """SVG должен сохранять процессные узлы и поясняющие элементы."""

    result = generate(project_root=tmp_path, dpi=300)
    svg = result.svg_path.read_text(encoding="utf-8")

    for text in (
        "Исходное",
        "сообщение",
        "Ручное",
        "кодирование",
        "Кодированное",
        "представление",
        "декодирование",
        "Восстановленное",
        "Ошибки восприятия",
        "Ошибки применения правила",
        "Ошибки фиксации результата",
        "Ошибки декодирования",
        "Контроль кодирования",
        "Контроль декодирования",
        "Δt_e",
        "Δt_c",
        "Δt_d",
        "d(M, M′)",
    ):
        assert text in svg
    assert "<text" in svg
    assert "<path" in svg or "<rect" in svg


def test_cli_reports_both_paths(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """CLI должен завершаться успешно и выводить пути PNG и SVG."""

    exit_code = main(["--project-root", str(tmp_path), "--dpi", "300"])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "Рисунок 2.2 успешно сформирован" in output
    assert ".png" in output
    assert ".svg" in output
