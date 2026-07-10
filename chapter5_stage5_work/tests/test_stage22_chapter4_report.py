"""Тесты итогового отчета главы 4."""

import json
from pathlib import Path

import pytest

from manual_coding_sim.lda.chapter4_report import (
    Chapter4LdaReportBuilder,
    Chapter4LdaReportConfig,
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    """Записать JSON-артефакт для теста отчета."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, ensure_ascii=False)


def _touch(path: Path) -> None:
    """Создать минимальный текстовый артефакт."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("test\n", encoding="utf-8")


def _prepare_report_artifacts(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Создать полный набор искусственных артефактов главы 4."""

    data_dir = tmp_path / "data" / "processed" / "lda"
    models_dir = tmp_path / "models" / "lda"
    reports_dir = tmp_path / "reports" / "chapter4"
    for path in [
        data_dir / "token_map.json",
        data_dir / "dictionary.json",
        data_dir / "corpus_prior.csv",
        data_dir / "corpus_metadata.json",
        data_dir / "leakage_report.json",
        models_dir / "lda_prior.joblib",
        reports_dir / "theta_prior.csv",
        reports_dir / "topic_word.csv",
        reports_dir / "topic_metrics.csv",
        reports_dir / "k_selection_report.csv",
        reports_dir / "k_selection_report.md",
        reports_dir / "topic_stability_report.csv",
        reports_dir / "topic_stability_report.md",
        reports_dir / "topic_interpretation.csv",
        reports_dir / "topic_interpretation.md",
    ]:
        _touch(path)

    for path in [
        data_dir / "corpus_diag.csv",
        data_dir / "dictionary_diag.json",
        data_dir / "token_map_diag.json",
        data_dir / "corpus_metadata_diag.json",
        data_dir / "corpus_full.csv",
        data_dir / "dictionary_full.json",
        data_dir / "token_map_full.json",
        data_dir / "corpus_metadata_full.json",
        models_dir / "lda_diag.joblib",
        models_dir / "lda_full.joblib",
        reports_dir / "theta_diag.csv",
        reports_dir / "theta_full.csv",
        reports_dir / "topic_word_diag.csv",
        reports_dir / "topic_word_full.csv",
    ]:
        _touch(path)

    _write_json(
        reports_dir / "lda_prior_metadata.json",
        {"model_name": "LDA_prior", "n_components": 2, "random_state": 42},
    )
    _write_json(
        reports_dir / "topic_metrics.json",
        {
            "perplexity": 10.0,
            "mean_coherence": -0.2,
            "topic_diversity": 0.8,
            "n_components": 2,
            "top_n": 3,
        },
    )
    _write_json(
        reports_dir / "k_selection_report.json",
        {
            "recommended_k": 2,
            "selection_method": "weighted_min_max_score",
            "candidates": [{"k": 2}, {"k": 3}],
        },
    )
    _write_json(
        reports_dir / "topic_stability_report.json",
        {
            "mean_stability": 0.9,
            "min_stability": 0.8,
            "random_states": [11, 42],
            "analysis_method": "topic_word_cosine_similarity",
        },
    )
    _write_json(
        reports_dir / "topic_interpretation.json",
        {
            "topic_count": 2,
            "top_n": 3,
            "topics": [
                {"suggested_factor_name": "Фактор 1"},
                {"suggested_factor_name": "Фактор 2"},
            ],
        },
    )
    _write_json(
        reports_dir / "lda_diagnostic_metadata.json",
        {
            "diag": {"diagnostic_only": True},
            "full": {"diagnostic_only": True},
        },
    )
    return data_dir, models_dir, reports_dir


def _read_json(path: Path) -> dict[str, object]:
    """Прочитать JSON-отчет."""

    with path.open("r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def test_report_builder_creates_json_and_markdown(tmp_path: Path) -> None:
    """Построитель должен создавать итоговые JSON- и Markdown-отчеты."""

    data_dir, models_dir, reports_dir = _prepare_report_artifacts(tmp_path)

    result = Chapter4LdaReportBuilder().build_from_artifacts(
        data_dir=data_dir,
        models_dir=models_dir,
        reports_dir=reports_dir,
        selected_k=2,
    )

    assert result.report_json_path.exists()
    assert result.report_md_path.exists()
    payload = _read_json(result.report_json_path)
    assert payload["status"] == "completed"
    assert payload["selected_k"] == 2
    assert payload["methodological_constraints"]["lda_prior_uses_only_prior_features"] is True
    assert "lda_diagnostic_models" in payload["completed_steps"]
    assert "Фактор 1" in result.report_md_path.read_text(encoding="utf-8")


def test_report_builder_requires_mandatory_artifacts(tmp_path: Path) -> None:
    """Отчет не должен создаваться при отсутствии обязательного артефакта."""

    data_dir, models_dir, reports_dir = _prepare_report_artifacts(tmp_path)
    (reports_dir / "topic_metrics.json").unlink()

    with pytest.raises(FileNotFoundError, match="topic_metrics_json"):
        Chapter4LdaReportBuilder().build_from_artifacts(
            data_dir=data_dir,
            models_dir=models_dir,
            reports_dir=reports_dir,
            selected_k=2,
        )

    assert not (reports_dir / "chapter4_lda_report.json").exists()


def test_report_builder_can_skip_diagnostic_artifacts(tmp_path: Path) -> None:
    """Диагностические артефакты можно сделать необязательными."""

    data_dir, models_dir, reports_dir = _prepare_report_artifacts(tmp_path)
    for path in [
        reports_dir / "lda_diagnostic_metadata.json",
        reports_dir / "theta_diag.csv",
        reports_dir / "theta_full.csv",
    ]:
        path.unlink()

    result = Chapter4LdaReportBuilder(
        Chapter4LdaReportConfig(require_diagnostic_artifacts=False)
    ).build_from_artifacts(
        data_dir=data_dir,
        models_dir=models_dir,
        reports_dir=reports_dir,
        selected_k=2,
    )

    payload = _read_json(result.report_json_path)
    assert result.diagnostic_artifacts_included is False
    assert "lda_diagnostic_models" not in payload["completed_steps"]


def test_report_builder_respects_overwrite_false(tmp_path: Path) -> None:
    """Построитель должен соблюдать запрет перезаписи итогового отчета."""

    data_dir, models_dir, reports_dir = _prepare_report_artifacts(tmp_path)
    builder = Chapter4LdaReportBuilder(Chapter4LdaReportConfig(overwrite=False))
    builder.build_from_artifacts(
        data_dir=data_dir,
        models_dir=models_dir,
        reports_dir=reports_dir,
        selected_k=2,
    )

    with pytest.raises(FileExistsError):
        builder.build_from_artifacts(
            data_dir=data_dir,
            models_dir=models_dir,
            reports_dir=reports_dir,
            selected_k=2,
        )
