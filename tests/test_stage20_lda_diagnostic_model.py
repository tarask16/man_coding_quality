"""Тесты диагностических моделей LDA_diag и LDA_full."""

import csv
import json
from pathlib import Path

import joblib
import pytest

from manual_coding_sim.lda.config import LdaTokenizationConfig
from manual_coding_sim.lda.lda_diagnostic_model import (
    LdaDiagnosticModel,
    LdaDiagnosticModelConfig,
)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Записать тестовый CSV-файл."""

    with path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    """Прочитать CSV-файл как список словарей."""

    with path.open("r", encoding="utf-8", newline="") as file_obj:
        return list(csv.DictReader(file_obj))


def _read_json(path: Path) -> dict[str, object]:
    """Прочитать JSON-файл."""

    with path.open("r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def _prior_rows() -> list[dict[str, object]]:
    """Вернуть тестовые априорные признаки."""

    return [
        {
            "run_id": "r001",
            "protocol_id": "p001",
            "scenario_id": "s001",
            "has_control": 1,
            "message_length": 10,
            "procedure_type": "simple",
            "operator_skill": "high",
        },
        {
            "run_id": "r002",
            "protocol_id": "p002",
            "scenario_id": "s002",
            "has_control": 1,
            "message_length": 12,
            "procedure_type": "simple",
            "operator_skill": "high",
        },
        {
            "run_id": "r003",
            "protocol_id": "p003",
            "scenario_id": "s003",
            "has_control": 0,
            "message_length": 40,
            "procedure_type": "complex",
            "operator_skill": "low",
        },
        {
            "run_id": "r004",
            "protocol_id": "p004",
            "scenario_id": "s004",
            "has_control": 0,
            "message_length": 45,
            "procedure_type": "complex",
            "operator_skill": "low",
        },
        {
            "run_id": "r005",
            "protocol_id": "p005",
            "scenario_id": "s005",
            "has_control": 1,
            "message_length": 14,
            "procedure_type": "simple",
            "operator_skill": "medium",
        },
        {
            "run_id": "r006",
            "protocol_id": "p006",
            "scenario_id": "s006",
            "has_control": 0,
            "message_length": 50,
            "procedure_type": "complex",
            "operator_skill": "medium",
        },
    ]


def _diagnostic_rows() -> list[dict[str, object]]:
    """Вернуть тестовые диагностические признаки."""

    return [
        {
            "run_id": "r001",
            "protocol_id": "p001",
            "scenario_id": "s001",
            "control_complexity": "low",
            "operator_stress": 0.1,
        },
        {
            "run_id": "r002",
            "protocol_id": "p002",
            "scenario_id": "s002",
            "control_complexity": "low",
            "operator_stress": 0.2,
        },
        {
            "run_id": "r003",
            "protocol_id": "p003",
            "scenario_id": "s003",
            "control_complexity": "high",
            "operator_stress": 0.8,
        },
        {
            "run_id": "r004",
            "protocol_id": "p004",
            "scenario_id": "s004",
            "control_complexity": "high",
            "operator_stress": 0.9,
        },
        {
            "run_id": "r005",
            "protocol_id": "p005",
            "scenario_id": "s005",
            "control_complexity": "mid",
            "operator_stress": 0.3,
        },
        {
            "run_id": "r006",
            "protocol_id": "p006",
            "scenario_id": "s006",
            "control_complexity": "mid",
            "operator_stress": 0.7,
        },
    ]


def _fact_rows() -> list[dict[str, object]]:
    """Вернуть тестовые фактические признаки для LDA_full."""

    return [
        {
            "run_id": "r001",
            "protocol_id": "p001",
            "scenario_id": "s001",
            "errors_total": 0,
            "time_overrun": 0,
        },
        {
            "run_id": "r002",
            "protocol_id": "p002",
            "scenario_id": "s002",
            "errors_total": 1,
            "time_overrun": 0,
        },
        {
            "run_id": "r003",
            "protocol_id": "p003",
            "scenario_id": "s003",
            "errors_total": 5,
            "time_overrun": 1,
        },
        {
            "run_id": "r004",
            "protocol_id": "p004",
            "scenario_id": "s004",
            "errors_total": 6,
            "time_overrun": 1,
        },
        {
            "run_id": "r005",
            "protocol_id": "p005",
            "scenario_id": "s005",
            "errors_total": 2,
            "time_overrun": 0,
        },
        {
            "run_id": "r006",
            "protocol_id": "p006",
            "scenario_id": "s006",
            "errors_total": 4,
            "time_overrun": 1,
        },
    ]


def _prepare_input_files(
    tmp_path: Path,
) -> tuple[Path, Path, Path]:
    """Сохранить входные CSV-файлы для диагностических тестов."""

    prior_path = tmp_path / "prior_features.csv"
    diagnostic_path = tmp_path / "diagnostic_features.csv"
    fact_path = tmp_path / "fact_features.csv"
    _write_csv(prior_path, _prior_rows())
    _write_csv(diagnostic_path, _diagnostic_rows())
    _write_csv(fact_path, _fact_rows())
    return prior_path, diagnostic_path, fact_path


def _diagnostic_model(kind: str, overwrite: bool = True) -> LdaDiagnosticModel:
    """Создать диагностическую модель с быстрыми параметрами обучения."""

    return LdaDiagnosticModel(
        LdaDiagnosticModelConfig(
            diagnostic_kind=kind,
            n_components=2,
            tokenization=LdaTokenizationConfig(df_min=1, df_max_ratio=1.0),
            max_iter=5,
            random_state=42,
            overwrite=overwrite,
        )
    )


def test_lda_diag_creates_separate_artifacts(tmp_path: Path) -> None:
    """LDA_diag должна создавать отдельные диагностические артефакты."""

    prior_path, diagnostic_path, _ = _prepare_input_files(tmp_path)
    result = _diagnostic_model("diag").fit_from_csv(
        prior_features_path=prior_path,
        extension_features_path=diagnostic_path,
        data_dir=tmp_path / "data" / "processed" / "lda",
        models_dir=tmp_path / "models" / "lda",
        reports_dir=tmp_path / "reports" / "chapter4",
    )

    assert result.model_name == "LDA_diag"
    assert result.model_path.name == "lda_diag.joblib"
    assert result.theta_path.name == "theta_diag.csv"
    assert result.topic_word_path.name == "topic_word_diag.csv"
    assert result.corpus_path.name == "corpus_diag.csv"
    assert result.dictionary_path.name == "dictionary_diag.json"
    assert result.token_map_path.name == "token_map_diag.json"
    assert result.corpus_metadata_path.name == "corpus_metadata_diag.json"
    assert result.document_count == 6
    assert result.token_count > 0
    assert result.allowed_for_apriori_forecast is False
    assert joblib.load(result.model_path).n_components == 2


def test_lda_full_is_marked_as_diagnostic_only(tmp_path: Path) -> None:
    """LDA_full должна явно помечаться как недопустимая для прогноза."""

    prior_path, _, fact_path = _prepare_input_files(tmp_path)
    result = _diagnostic_model("full").fit_from_csv(
        prior_features_path=prior_path,
        extension_features_path=fact_path,
        data_dir=tmp_path / "data" / "processed" / "lda",
        models_dir=tmp_path / "models" / "lda",
        reports_dir=tmp_path / "reports" / "chapter4",
    )

    assert result.model_name == "LDA_full"
    assert result.model_path.name == "lda_full.joblib"
    assert result.theta_path.name == "theta_full.csv"
    assert result.topic_word_path.name == "topic_word_full.csv"
    metadata = _read_json(result.metadata_path)
    assert metadata["diagnostic_only"] is True
    assert metadata["allowed_for_apriori_forecast"] is False
    assert metadata["models"]["full"]["diagnostic_only"] is True
    assert metadata["models"]["full"]["allowed_for_apriori_forecast"] is False


def test_diagnostic_models_do_not_write_prior_artifacts(tmp_path: Path) -> None:
    """Диагностические модели не должны создавать файлы LDA_prior."""

    prior_path, diagnostic_path, fact_path = _prepare_input_files(tmp_path)
    data_dir = tmp_path / "data" / "processed" / "lda"
    models_dir = tmp_path / "models" / "lda"
    reports_dir = tmp_path / "reports" / "chapter4"

    _diagnostic_model("diag").fit_from_csv(
        prior_features_path=prior_path,
        extension_features_path=diagnostic_path,
        data_dir=data_dir,
        models_dir=models_dir,
        reports_dir=reports_dir,
    )
    _diagnostic_model("full").fit_from_csv(
        prior_features_path=prior_path,
        extension_features_path=fact_path,
        data_dir=data_dir,
        models_dir=models_dir,
        reports_dir=reports_dir,
    )

    assert not (models_dir / "lda_prior.joblib").exists()
    assert not (reports_dir / "theta_prior.csv").exists()
    assert not (reports_dir / "topic_word.csv").exists()
    assert (reports_dir / "theta_diag.csv").exists()
    assert (reports_dir / "theta_full.csv").exists()


def test_diagnostic_theta_rows_are_normalized(tmp_path: Path) -> None:
    """Строки theta_diag должны быть нормированы и помечены как diagnostic_only."""

    prior_path, diagnostic_path, _ = _prepare_input_files(tmp_path)
    result = _diagnostic_model("diag").fit_from_csv(
        prior_features_path=prior_path,
        extension_features_path=diagnostic_path,
        data_dir=tmp_path / "data" / "processed" / "lda",
        models_dir=tmp_path / "models" / "lda",
        reports_dir=tmp_path / "reports" / "chapter4",
    )

    rows = _read_csv(result.theta_path)
    theta_columns = [column for column in rows[0] if column.startswith("theta_")]
    assert theta_columns == ["theta_0", "theta_1"]
    for row in rows:
        theta_sum = sum(float(row[column]) for column in theta_columns)
        assert theta_sum == pytest.approx(1.0, abs=1e-9)
        assert row["model_name"] == "LDA_diag"
        assert row["diagnostic_only"] == "true"
        assert row["allowed_for_apriori_forecast"] == "false"


def test_diagnostic_token_map_contains_prefixed_extension_features(tmp_path: Path) -> None:
    """Карта токенизации должна разделять prior и diagnostic/full признаки."""

    prior_path, diagnostic_path, _ = _prepare_input_files(tmp_path)
    result = _diagnostic_model("diag").fit_from_csv(
        prior_features_path=prior_path,
        extension_features_path=diagnostic_path,
        data_dir=tmp_path / "data" / "processed" / "lda",
        models_dir=tmp_path / "models" / "lda",
        reports_dir=tmp_path / "reports" / "chapter4",
    )

    token_map = _read_json(result.token_map_path)
    feature_columns = token_map["feature_columns"]
    assert "has_control" in feature_columns
    assert "diag_control_complexity" in feature_columns
    assert "diag_operator_stress" in feature_columns
    assert token_map["diagnostic_kind"] == "diag"
    assert token_map["model_name"] == "LDA_diag"


def test_diagnostic_metadata_aggregates_diag_and_full(tmp_path: Path) -> None:
    """Единый metadata-файл должен содержать сведения о diag и full."""

    prior_path, diagnostic_path, fact_path = _prepare_input_files(tmp_path)
    data_dir = tmp_path / "data" / "processed" / "lda"
    models_dir = tmp_path / "models" / "lda"
    reports_dir = tmp_path / "reports" / "chapter4"

    _diagnostic_model("diag").fit_from_csv(
        prior_features_path=prior_path,
        extension_features_path=diagnostic_path,
        data_dir=data_dir,
        models_dir=models_dir,
        reports_dir=reports_dir,
    )
    result = _diagnostic_model("full").fit_from_csv(
        prior_features_path=prior_path,
        extension_features_path=fact_path,
        data_dir=data_dir,
        models_dir=models_dir,
        reports_dir=reports_dir,
    )

    metadata = _read_json(result.metadata_path)
    assert set(metadata["models"]) == {"diag", "full"}
    assert metadata["models"]["diag"]["model_name"] == "LDA_diag"
    assert metadata["models"]["full"]["model_name"] == "LDA_full"


def test_diagnostic_model_rejects_forbidden_prior_columns(tmp_path: Path) -> None:
    """Фактические признаки не должны попадать в prior-часть диагностик."""

    prior_rows = _prior_rows()
    prior_rows[0]["q_acc"] = 0.9
    prior_path = tmp_path / "prior_features.csv"
    diagnostic_path = tmp_path / "diagnostic_features.csv"
    _write_csv(prior_path, prior_rows)
    _write_csv(diagnostic_path, _diagnostic_rows())

    with pytest.raises(Exception):
        _diagnostic_model("diag").fit_from_csv(
            prior_features_path=prior_path,
            extension_features_path=diagnostic_path,
            data_dir=tmp_path / "data" / "processed" / "lda",
            models_dir=tmp_path / "models" / "lda",
            reports_dir=tmp_path / "reports" / "chapter4",
        )


def test_diagnostic_model_rejects_identifier_mismatch(tmp_path: Path) -> None:
    """Модель должна обнаруживать несовпадение protocol_id между входами."""

    prior_path, diagnostic_path, _ = _prepare_input_files(tmp_path)
    diagnostic_rows = _diagnostic_rows()
    diagnostic_rows[0]["protocol_id"] = "other_protocol"
    _write_csv(diagnostic_path, diagnostic_rows)

    with pytest.raises(ValueError, match="Несовпадение идентификатора"):
        _diagnostic_model("diag").fit_from_csv(
            prior_features_path=prior_path,
            extension_features_path=diagnostic_path,
            data_dir=tmp_path / "data" / "processed" / "lda",
            models_dir=tmp_path / "models" / "lda",
            reports_dir=tmp_path / "reports" / "chapter4",
        )


def test_diagnostic_config_rejects_invalid_kind() -> None:
    """Конфигурация должна запрещать неизвестные виды диагностики."""

    with pytest.raises(ValueError, match="diagnostic_kind"):
        LdaDiagnosticModelConfig(diagnostic_kind="prior", n_components=2).validate()


def test_diagnostic_model_respects_overwrite_flag(tmp_path: Path) -> None:
    """При overwrite=False диагностические отчеты нельзя перезаписать."""

    prior_path, diagnostic_path, _ = _prepare_input_files(tmp_path)
    kwargs = {
        "prior_features_path": prior_path,
        "extension_features_path": diagnostic_path,
        "data_dir": tmp_path / "data" / "processed" / "lda",
        "models_dir": tmp_path / "models" / "lda",
        "reports_dir": tmp_path / "reports" / "chapter4",
    }
    _diagnostic_model("diag").fit_from_csv(**kwargs)

    with pytest.raises(FileExistsError):
        _diagnostic_model("diag", overwrite=False).fit_from_csv(**kwargs)
