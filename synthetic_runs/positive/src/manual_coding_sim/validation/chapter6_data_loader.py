"""Загрузка и структурная проверка входных артефактов главы 6.

Модуль реализует этап 2 программного контура главы 6. Он проверяет
прогнозные артефакты главы 5, латентный профиль главы 4 и фактические
данные главы 3. Формулы, веса и значения прогноза главы 5 не изменяются.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from manual_coding_sim.validation.chapter6_config import Chapter6ValidationConfig


PREDICTED_CRITERIA: tuple[str, ...] = (
    "q_acc_pred",
    "q_time_pred",
    "q_effort_pred",
    "q_res_pred",
    "q_rep_pred",
    "q_fit_pred",
)
FACTUAL_CRITERIA: tuple[str, ...] = (
    "q_acc",
    "q_time",
    "q_effort",
    "q_res",
    "q_rep",
    "q_fit",
)
THETA_COLUMNS: tuple[str, ...] = ("theta_0", "theta_1", "theta_2")
Q_PRED_TOLERANCE: float = 1e-10
UNIT_INTERVAL_TOLERANCE: float = 1e-12


class Chapter6DataLoadError(ValueError):
    """Ошибка загрузки или проверки входных артефактов главы 6."""


@dataclass(frozen=True)
class CsvArtifactContract:
    """Контракт обязательного CSV-артефакта."""

    name: str
    input_names: tuple[str, ...]
    required_columns: tuple[str, ...]
    finite_columns: tuple[str, ...] = ()
    unit_interval_columns: tuple[str, ...] = ()
    allow_protocol_id_reconstruction: bool = False
    require_normalized_columns: bool = False


@dataclass(frozen=True)
class ArtifactValidationResult:
    """Результат проверки одного CSV-артефакта."""

    name: str
    path: str
    row_count: int
    column_count: int
    reconstructed_key_columns: tuple[str, ...]
    unique_keys: bool
    finite_values: bool
    unit_interval_values: bool
    key_set_aligned: bool

    def to_dict(self) -> dict[str, Any]:
        """Преобразовать результат в JSON-совместимый словарь."""

        return {
            "name": self.name,
            "path": self.path,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "reconstructed_key_columns": list(self.reconstructed_key_columns),
            "unique_keys": self.unique_keys,
            "finite_values": self.finite_values,
            "unit_interval_values": self.unit_interval_values,
            "key_set_aligned": self.key_set_aligned,
        }


@dataclass(frozen=True)
class Chapter6InputValidationReport:
    """Машинно-читаемый отчет проверки входов главы 6."""

    stage: int
    report_type: str
    passed: bool
    expected_row_count: int
    join_keys: tuple[str, ...]
    checked_csv_count: int
    checked_json_count: int
    artifact_checks: Mapping[str, ArtifactValidationResult]
    q_pred_consistency: Mapping[str, Any]
    chapter5_prediction_report: Mapping[str, Any]
    chapter5_acceptance: Mapping[str, Any]
    method_note: str

    def to_dict(self) -> dict[str, Any]:
        """Преобразовать отчет в JSON-совместимый словарь."""

        return {
            "stage": self.stage,
            "report_type": self.report_type,
            "passed": self.passed,
            "expected_row_count": self.expected_row_count,
            "join_keys": list(self.join_keys),
            "checked_csv_count": self.checked_csv_count,
            "checked_json_count": self.checked_json_count,
            "artifact_checks": {
                name: result.to_dict()
                for name, result in self.artifact_checks.items()
            },
            "q_pred_consistency": dict(self.q_pred_consistency),
            "chapter5_prediction_report": dict(self.chapter5_prediction_report),
            "chapter5_acceptance": dict(self.chapter5_acceptance),
            "method_note": self.method_note,
        }


@dataclass(frozen=True)
class Chapter6LoadedInputs:
    """Проверенные входные таблицы и отчеты главы 6."""

    q_pred: pd.DataFrame
    q_pred_components: pd.DataFrame
    prediction_uncertainty: pd.DataFrame
    normalized_prior_features: pd.DataFrame
    latent_quality_component: pd.DataFrame
    theta_prior: pd.DataFrame
    quality_targets: pd.DataFrame
    fact_features: pd.DataFrame
    chapter5_prediction_report: Mapping[str, Any]
    chapter5_acceptance_report: Mapping[str, Any]
    validation_report: Chapter6InputValidationReport
    report_json_path: Path | None = None
    report_markdown_path: Path | None = None


class Chapter6DataLoader:
    """Загрузчик и валидатор входных артефактов этапа 2."""

    def __init__(
        self,
        config: Chapter6ValidationConfig,
        project_root: str | Path = ".",
    ) -> None:
        """Сохранить конфигурацию и корень проекта."""

        self.config = config
        self.project_root = Path(project_root)
        self.config.validate()

    def load(self) -> Chapter6LoadedInputs:
        """Загрузить все входы и выполнить структурную проверку."""

        contracts = self._build_csv_contracts()
        q_pred, q_pred_result = self._load_csv(
            contracts[0],
            reference_keys=None,
            protocol_lookup=None,
        )
        reference_keys = self._key_set(q_pred)
        protocol_lookup = self._build_protocol_lookup(q_pred)

        loaded_tables: dict[str, pd.DataFrame] = {"q_pred": q_pred}
        artifact_checks: dict[str, ArtifactValidationResult] = {
            "q_pred": q_pred_result
        }
        for contract in contracts[1:]:
            table, result = self._load_csv(
                contract,
                reference_keys=reference_keys,
                protocol_lookup=protocol_lookup,
            )
            loaded_tables[contract.name] = table
            artifact_checks[contract.name] = result

        prediction_report = self._load_json(
            ("chapter5_prediction_report_path", "chapter5_prediction_report"),
            "итоговый отчет главы 5",
        )
        acceptance_report = self._load_json(
            ("chapter5_acceptance_report_path", "chapter5_acceptance_report"),
            "отчет приемки главы 5",
        )

        prediction_summary = self._validate_prediction_report(prediction_report)
        acceptance_summary = self._validate_acceptance_report(acceptance_report)
        q_pred_consistency = self._validate_q_pred_consistency(
            loaded_tables["q_pred"],
            loaded_tables["q_pred_components"],
            loaded_tables["prediction_uncertainty"],
        )

        report = Chapter6InputValidationReport(
            stage=2,
            report_type="chapter6_input_validation_report",
            passed=True,
            expected_row_count=self._expected_row_count(),
            join_keys=self._join_keys(),
            checked_csv_count=len(contracts),
            checked_json_count=2,
            artifact_checks=artifact_checks,
            q_pred_consistency=q_pred_consistency,
            chapter5_prediction_report=prediction_summary,
            chapter5_acceptance=acceptance_summary,
            method_note=(
                "Фактические показатели загружены только для внешней проверки. "
                "Формулы, веса, нормировки и значения Q_pred главы 5 не изменялись."
            ),
        )

        return Chapter6LoadedInputs(
            q_pred=loaded_tables["q_pred"],
            q_pred_components=loaded_tables["q_pred_components"],
            prediction_uncertainty=loaded_tables["prediction_uncertainty"],
            normalized_prior_features=loaded_tables["normalized_prior_features"],
            latent_quality_component=loaded_tables["latent_quality_component"],
            theta_prior=loaded_tables["theta_prior"],
            quality_targets=loaded_tables["quality_targets"],
            fact_features=loaded_tables["fact_features"],
            chapter5_prediction_report=prediction_report,
            chapter5_acceptance_report=acceptance_report,
            validation_report=report,
        )

    def load_and_save_report(self) -> Chapter6LoadedInputs:
        """Проверить входы, сохранить отчеты и вернуть загруженные данные."""

        loaded = self.load()
        json_path, markdown_path = self.save_validation_report(
            loaded.validation_report
        )
        return Chapter6LoadedInputs(
            q_pred=loaded.q_pred,
            q_pred_components=loaded.q_pred_components,
            prediction_uncertainty=loaded.prediction_uncertainty,
            normalized_prior_features=loaded.normalized_prior_features,
            latent_quality_component=loaded.latent_quality_component,
            theta_prior=loaded.theta_prior,
            quality_targets=loaded.quality_targets,
            fact_features=loaded.fact_features,
            chapter5_prediction_report=loaded.chapter5_prediction_report,
            chapter5_acceptance_report=loaded.chapter5_acceptance_report,
            validation_report=loaded.validation_report,
            report_json_path=json_path,
            report_markdown_path=markdown_path,
        )

    def save_validation_report(
        self,
        report: Chapter6InputValidationReport,
    ) -> tuple[Path, Path]:
        """Сохранить JSON- и Markdown-отчеты этапа 2."""

        json_path = self._resolve_output_path(
            ("input_validation_report_json_path", "input_validation_report_json"),
            Path("reports/chapter6/chapter6_input_validation_report.json"),
        )
        markdown_path = self._resolve_output_path(
            ("input_validation_report_md_path", "input_validation_report_md"),
            Path("reports/chapter6/chapter6_input_validation_report.md"),
        )
        json_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(
            json.dumps(report.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        markdown_path.write_text(
            self.render_markdown(report),
            encoding="utf-8",
        )
        return json_path, markdown_path

    @staticmethod
    def render_markdown(report: Chapter6InputValidationReport) -> str:
        """Сформировать человекочитаемый отчет этапа 2."""

        lines = [
            "# Проверка входных артефактов главы 6",
            "",
            f"- Этап: `{report.stage}`",
            f"- Результат: `{'пройден' if report.passed else 'не пройден'}`",
            f"- Ожидаемое число сценариев: `{report.expected_row_count}`",
            f"- Ключи: `{', '.join(report.join_keys)}`",
            f"- Проверено CSV: `{report.checked_csv_count}`",
            f"- Проверено JSON: `{report.checked_json_count}`",
            "",
            "## CSV-артефакты",
            "",
            "| Артефакт | Строк | Колонок | Восстановленные ключи | Ключи согласованы |",
            "|---|---:|---:|---|---|",
        ]
        for item in report.artifact_checks.values():
            restored = ", ".join(item.reconstructed_key_columns) or "—"
            lines.append(
                f"| `{item.name}` | {item.row_count} | {item.column_count} | "
                f"{restored} | {'да' if item.key_set_aligned else 'нет'} |"
            )
        lines.extend(
            [
                "",
                "## Согласованность прогноза",
                "",
                "- Максимальное расхождение Q_pred: "
                f"`{report.q_pred_consistency['max_abs_difference']:.12g}`",
                "- Приемка главы 5: "
                f"`{report.chapter5_acceptance['accepted']}`",
                "",
                "## Методическое ограничение",
                "",
                report.method_note,
                "",
            ]
        )
        return "\n".join(lines)

    def _build_csv_contracts(self) -> tuple[CsvArtifactContract, ...]:
        """Сформировать фиксированный набор CSV-контрактов этапа 2."""

        keys = self._join_keys()
        return (
            CsvArtifactContract(
                name="q_pred",
                input_names=("q_pred_path", "q_pred"),
                required_columns=keys + PREDICTED_CRITERIA + ("q_pred",),
                finite_columns=PREDICTED_CRITERIA + ("q_pred",),
                unit_interval_columns=PREDICTED_CRITERIA + ("q_pred",),
            ),
            CsvArtifactContract(
                name="q_pred_components",
                input_names=("q_pred_components_path", "q_pred_components"),
                required_columns=keys + PREDICTED_CRITERIA + ("q_latent",),
                finite_columns=PREDICTED_CRITERIA + ("q_latent",),
                unit_interval_columns=PREDICTED_CRITERIA + ("q_latent",),
            ),
            CsvArtifactContract(
                name="prediction_uncertainty",
                input_names=(
                    "prediction_uncertainty_path",
                    "prediction_uncertainty",
                ),
                required_columns=keys
                + ("q_pred", "uncertainty_score", "q_pred_lower", "q_pred_upper")
                + THETA_COLUMNS,
                finite_columns=(
                    "q_pred",
                    "uncertainty_score",
                    "q_pred_lower",
                    "q_pred_upper",
                )
                + THETA_COLUMNS,
                unit_interval_columns=(
                    "q_pred",
                    "uncertainty_score",
                    "q_pred_lower",
                    "q_pred_upper",
                )
                + THETA_COLUMNS,
            ),
            CsvArtifactContract(
                name="normalized_prior_features",
                input_names=(
                    "normalized_prior_features_path",
                    "normalized_prior_features",
                ),
                required_columns=keys,
                require_normalized_columns=True,
            ),
            CsvArtifactContract(
                name="latent_quality_component",
                input_names=(
                    "latent_quality_component_path",
                    "latent_quality_component",
                ),
                required_columns=keys + THETA_COLUMNS + ("q_latent",),
                finite_columns=THETA_COLUMNS + ("q_latent",),
                unit_interval_columns=THETA_COLUMNS + ("q_latent",),
            ),
            CsvArtifactContract(
                name="theta_prior",
                input_names=("theta_prior_path", "theta_prior"),
                required_columns=keys + THETA_COLUMNS,
                finite_columns=THETA_COLUMNS,
                unit_interval_columns=THETA_COLUMNS,
            ),
            CsvArtifactContract(
                name="quality_targets",
                input_names=("quality_targets_path", "quality_targets"),
                required_columns=keys + FACTUAL_CRITERIA + ("integral_quality",),
                finite_columns=FACTUAL_CRITERIA + ("integral_quality",),
                unit_interval_columns=FACTUAL_CRITERIA + ("integral_quality",),
                allow_protocol_id_reconstruction=True,
            ),
            CsvArtifactContract(
                name="fact_features",
                input_names=("fact_features_path", "fact_features"),
                required_columns=keys,
                allow_protocol_id_reconstruction=True,
            ),
        )

    def _load_csv(
        self,
        contract: CsvArtifactContract,
        reference_keys: set[tuple[Any, ...]] | None,
        protocol_lookup: Mapping[Any, Any] | None,
    ) -> tuple[pd.DataFrame, ArtifactValidationResult]:
        """Загрузить и проверить один CSV-артефакт."""

        path = self._resolve_input_path(contract.input_names)
        if not path.exists():
            raise FileNotFoundError(f"Обязательный входной файл не найден: {path}")
        try:
            table = pd.read_csv(path)
        except Exception as error:
            raise Chapter6DataLoadError(
                f"Не удалось прочитать CSV-файл {path}: {error}"
            ) from error

        reconstructed: list[str] = []
        if (
            "protocol_id" in self._join_keys()
            and "protocol_id" not in table.columns
            and contract.allow_protocol_id_reconstruction
        ):
            table = self._reconstruct_protocol_id(table, path, protocol_lookup)
            reconstructed.append("protocol_id")

        self._require_columns(table, contract.required_columns, path)
        self._validate_row_count(table, path)
        self._validate_keys(table, path)
        self._validate_finite_values(table, contract.finite_columns, path)
        self._validate_unit_interval(table, contract.unit_interval_columns, path)

        if contract.require_normalized_columns:
            normalized_columns = [
                column for column in table.columns if column.endswith("_norm")
            ]
            if not normalized_columns:
                raise Chapter6DataLoadError(
                    "В normalized_prior_features.csv отсутствуют колонки *_norm. "
                    f"Файл: {path}"
                )
            self._validate_finite_values(table, normalized_columns, path)
            self._validate_unit_interval(table, normalized_columns, path)

        key_set_aligned = True
        if reference_keys is not None:
            current_keys = self._key_set(table)
            key_set_aligned = current_keys == reference_keys
            if not key_set_aligned:
                missing = len(reference_keys - current_keys)
                extra = len(current_keys - reference_keys)
                raise Chapter6DataLoadError(
                    "Множество ключей входного артефакта не совпадает с "
                    f"q_pred.csv: отсутствует {missing}, лишних {extra}. Файл: {path}"
                )

        result = ArtifactValidationResult(
            name=contract.name,
            path=str(path),
            row_count=len(table),
            column_count=len(table.columns),
            reconstructed_key_columns=tuple(reconstructed),
            unique_keys=True,
            finite_values=True,
            unit_interval_values=True,
            key_set_aligned=key_set_aligned,
        )
        return table, result

    def _load_json(
        self,
        input_names: Sequence[str],
        description: str,
    ) -> Mapping[str, Any]:
        """Загрузить обязательный JSON-отчет."""

        path = self._resolve_input_path(input_names)
        if not path.exists():
            raise FileNotFoundError(f"Не найден {description}: {path}")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise Chapter6DataLoadError(
                f"Не удалось прочитать {description} {path}: {error}"
            ) from error
        if not isinstance(payload, Mapping):
            raise Chapter6DataLoadError(
                f"Корень JSON-файла должен быть объектом. Файл: {path}"
            )
        return payload

    def _validate_prediction_report(
        self,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Проверить итоговый отчет главы 5."""

        expected = self._expected_row_count()
        row_count = payload.get("row_count")
        if row_count != expected:
            raise Chapter6DataLoadError(
                "Итоговый отчет главы 5 содержит неверное число строк: "
                f"ожидалось {expected}, получено {row_count}."
            )
        method_safety = payload.get("method_safety")
        if not isinstance(method_safety, Mapping):
            raise Chapter6DataLoadError(
                "В итоговом отчете главы 5 отсутствует раздел method_safety."
            )
        self._require_true(method_safety, "apriori_only", "априорный режим")
        self._require_true(
            method_safety,
            "leakage_check_passed",
            "контроль методической утечки",
        )
        forbidden_count = int(method_safety.get("forbidden_column_count", -1))
        if forbidden_count != 0:
            raise Chapter6DataLoadError(
                "Итоговый отчет главы 5 содержит запрещенные признаки."
            )
        return {
            "stage": payload.get("stage"),
            "row_count": row_count,
            "apriori_only": True,
            "leakage_check_passed": True,
            "forbidden_column_count": forbidden_count,
        }

    def _validate_acceptance_report(
        self,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Проверить отчет финальной приемки главы 5."""

        if payload.get("accepted") is not True:
            raise Chapter6DataLoadError(
                "Финальная приемка главы 5 не подтверждена: accepted != true."
            )
        checks = payload.get("checks")
        if not isinstance(checks, Mapping) or not checks:
            raise Chapter6DataLoadError(
                "В отчете приемки главы 5 отсутствует непустой раздел checks."
            )
        failed = [name for name, value in checks.items() if value is not True]
        if failed:
            raise Chapter6DataLoadError(
                "В приемке главы 5 есть отрицательные проверки: "
                + ", ".join(failed)
            )
        expected = self._expected_row_count()
        row_counts = payload.get("row_counts", {})
        if isinstance(row_counts, Mapping):
            invalid = {
                name: value for name, value in row_counts.items() if value != expected
            }
            if invalid:
                raise Chapter6DataLoadError(
                    "В отчете приемки главы 5 обнаружены несогласованные "
                    f"размеры таблиц: {invalid}."
                )
        method_safety = payload.get("method_safety")
        if not isinstance(method_safety, Mapping):
            raise Chapter6DataLoadError(
                "В отчете приемки главы 5 отсутствует раздел method_safety."
            )
        self._require_true(method_safety, "apriori_only", "априорный режим")
        self._require_true(
            method_safety,
            "leakage_check_passed",
            "контроль методической утечки",
        )
        self._require_true(
            method_safety,
            "full_pipeline_completed",
            "полный программный контур главы 5",
        )
        if int(method_safety.get("forbidden_column_count", -1)) != 0:
            raise Chapter6DataLoadError(
                "Отчет приемки главы 5 фиксирует запрещенные признаки."
            )
        return {
            "stage": payload.get("stage"),
            "accepted": True,
            "check_count": len(checks),
            "all_checks_passed": True,
            "apriori_only": True,
            "leakage_check_passed": True,
            "full_pipeline_completed": True,
        }

    def _validate_q_pred_consistency(
        self,
        q_pred: pd.DataFrame,
        q_pred_components: pd.DataFrame,
        prediction_uncertainty: pd.DataFrame,
    ) -> dict[str, Any]:
        """Проверить совпадение прогнозов между артефактами главы 5."""

        keys = list(self._join_keys())
        merged = q_pred[keys + ["q_pred"]].merge(
            prediction_uncertainty[keys + ["q_pred"]].rename(
                columns={"q_pred": "q_pred_uncertainty"}
            ),
            on=keys,
            how="inner",
            validate=self._merge_validation(),
        )
        differences = (merged["q_pred"] - merged["q_pred_uncertainty"]).abs()
        max_difference = float(differences.max()) if not differences.empty else 0.0
        if max_difference > Q_PRED_TOLERANCE:
            raise Chapter6DataLoadError(
                "Q_pred не совпадает между q_pred.csv и "
                "prediction_uncertainty.csv: максимальное расхождение "
                f"{max_difference:.12g}."
            )

        compared = q_pred[keys + list(PREDICTED_CRITERIA)].merge(
            q_pred_components[keys + list(PREDICTED_CRITERIA)],
            on=keys,
            how="inner",
            suffixes=("_q_pred", "_components"),
            validate=self._merge_validation(),
        )
        component_differences: dict[str, float] = {}
        for column in PREDICTED_CRITERIA:
            difference = (
                compared[f"{column}_q_pred"]
                - compared[f"{column}_components"]
            ).abs()
            maximum = float(difference.max()) if not difference.empty else 0.0
            component_differences[column] = maximum
            if maximum > Q_PRED_TOLERANCE:
                raise Chapter6DataLoadError(
                    "Частный прогноз не совпадает между q_pred.csv и "
                    f"q_pred_components.csv: {column}."
                )
        return {
            "passed": True,
            "tolerance": Q_PRED_TOLERANCE,
            "max_abs_difference": max_difference,
            "partial_criteria_max_abs_difference": component_differences,
        }

    def _resolve_input_path(self, names: Sequence[str]) -> Path:
        """Разрешить путь входного артефакта относительно корня проекта."""

        configured = self._first_attribute(self.config.inputs, names)
        path = Path(configured)
        return path if path.is_absolute() else self.project_root / path

    def _resolve_output_path(
        self,
        names: Sequence[str],
        fallback: Path,
    ) -> Path:
        """Разрешить выходной путь без зависимости от имени каталога этапа 1."""

        value = self._first_attribute(self.config.outputs, names, required=False)
        path = Path(value) if value is not None else fallback
        return path if path.is_absolute() else self.project_root / path

    @staticmethod
    def _first_attribute(
        obj: object,
        names: Sequence[str],
        *,
        required: bool = True,
    ) -> Any:
        """Вернуть первое существующее именованное свойство объекта."""

        for name in names:
            if hasattr(obj, name):
                return getattr(obj, name)
        if required:
            raise Chapter6DataLoadError(
                "В конфигурации отсутствует обязательный параметр: "
                + " / ".join(names)
            )
        return None

    def _expected_row_count(self) -> int:
        """Вернуть ожидаемое число строк из конфигурации этапа 1."""

        return int(self.config.merge.expected_row_count)

    def _join_keys(self) -> tuple[str, ...]:
        """Вернуть составной ключ проверки."""

        keys = getattr(self.config.merge, "key_columns", None)
        if keys is None:
            keys = getattr(self.config.merge, "keys", None)
        if keys is None:
            raise Chapter6DataLoadError(
                "В конфигурации отсутствуют ключи объединения."
            )
        return tuple(keys)

    def _merge_validation(self) -> str:
        """Вернуть режим проверки объединения."""

        value = getattr(self.config.merge, "validation", None)
        if value is None:
            value = getattr(self.config.merge, "validate_mode", None)
        if value is None:
            raise Chapter6DataLoadError(
                "В конфигурации отсутствует режим проверки объединения."
            )
        return str(value)

    def _validate_row_count(self, table: pd.DataFrame, path: Path) -> None:
        """Проверить ожидаемое число сценариев."""

        expected = self._expected_row_count()
        if len(table) != expected:
            raise Chapter6DataLoadError(
                "Неверное число строк во входном артефакте: "
                f"ожидалось {expected}, получено {len(table)}. Файл: {path}"
            )

    def _validate_keys(self, table: pd.DataFrame, path: Path) -> None:
        """Проверить заполненность и уникальность составного ключа."""

        keys = list(self._join_keys())
        if table[keys].isna().any().any():
            raise Chapter6DataLoadError(
                f"В ключевых колонках обнаружены пропуски. Файл: {path}"
            )
        duplicates = table.duplicated(keys, keep=False)
        if duplicates.any():
            raise Chapter6DataLoadError(
                "Составной ключ scenario_id, protocol_id не уникален. "
                f"Число дублированных строк: {int(duplicates.sum())}. Файл: {path}"
            )

    @staticmethod
    def _validate_finite_values(
        table: pd.DataFrame,
        columns: Sequence[str],
        path: Path,
    ) -> None:
        """Проверить конечность обязательных числовых колонок."""

        if not columns:
            return
        numeric = table[list(columns)].apply(pd.to_numeric, errors="coerce")
        if numeric.isna().any().any():
            raise Chapter6DataLoadError(
                "В обязательных числовых колонках обнаружены NaN или "
                f"нечисловые значения. Файл: {path}"
            )
        if not np.isfinite(numeric.to_numpy(dtype=float)).all():
            raise Chapter6DataLoadError(
                "В обязательных числовых колонках обнаружены inf или -inf. "
                f"Файл: {path}"
            )

    @staticmethod
    def _validate_unit_interval(
        table: pd.DataFrame,
        columns: Sequence[str],
        path: Path,
    ) -> None:
        """Проверить принадлежность указанных колонок диапазону [0; 1]."""

        for column in columns:
            values = pd.to_numeric(table[column], errors="coerce")
            if values.isna().any():
                raise Chapter6DataLoadError(
                    f"Колонка {column} содержит нечисловые или пустые "
                    f"значения. Файл: {path}"
                )
            if (
                (values < -UNIT_INTERVAL_TOLERANCE).any()
                or (values > 1.0 + UNIT_INTERVAL_TOLERANCE).any()
            ):
                raise Chapter6DataLoadError(
                    f"Значения колонки {column} выходят за диапазон [0; 1]. "
                    f"Файл: {path}"
                )

    def _reconstruct_protocol_id(
        self,
        table: pd.DataFrame,
        path: Path,
        protocol_lookup: Mapping[Any, Any] | None,
    ) -> pd.DataFrame:
        """Восстановить protocol_id по однозначному соответствию scenario_id."""

        if "scenario_id" not in table.columns:
            raise Chapter6DataLoadError(
                "Невозможно восстановить protocol_id: отсутствует scenario_id. "
                f"Файл: {path}"
            )
        if protocol_lookup is None:
            raise Chapter6DataLoadError(
                "Невозможно восстановить protocol_id без проверенного "
                f"q_pred.csv. Файл: {path}"
            )
        restored = table.copy()
        restored["protocol_id"] = restored["scenario_id"].map(protocol_lookup)
        if restored["protocol_id"].isna().any():
            missing = sorted(
                restored.loc[
                    restored["protocol_id"].isna(), "scenario_id"
                ].astype(str).unique()
            )
            hint = ""
            if any(value.startswith("A_TEST") for value in missing):
                hint = (
                    " Обнаружен контрольный сценарий A_TEST*. Вероятно, "
                    "регрессионный тест главы 3 перезаписал data/processed. "
                    "Восстановите вычислительные артефакты из Git или "
                    "повторно выполните конвейеры глав 3–5."
                )
            raise Chapter6DataLoadError(
                "Для части scenario_id невозможно восстановить protocol_id: "
                + ", ".join(missing[:10])
                + ("..." if len(missing) > 10 else "")
                + hint
            )
        return restored

    @staticmethod
    def _build_protocol_lookup(q_pred: pd.DataFrame) -> dict[Any, Any]:
        """Построить однозначное соответствие scenario_id → protocol_id."""

        pairs = q_pred[["scenario_id", "protocol_id"]].drop_duplicates()
        counts = pairs.groupby("scenario_id", dropna=False)["protocol_id"].nunique()
        if (counts != 1).any():
            raise Chapter6DataLoadError(
                "В q_pred.csv обнаружено неоднозначное соответствие "
                "scenario_id → protocol_id."
            )
        return dict(zip(pairs["scenario_id"], pairs["protocol_id"], strict=True))

    def _key_set(self, table: pd.DataFrame) -> set[tuple[Any, ...]]:
        """Вернуть множество составных ключей таблицы."""

        return set(
            table[list(self._join_keys())].itertuples(index=False, name=None)
        )

    @staticmethod
    def _require_columns(
        table: pd.DataFrame,
        required_columns: Sequence[str],
        path: Path,
    ) -> None:
        """Проверить наличие обязательных колонок."""

        missing = [column for column in required_columns if column not in table]
        if missing:
            raise Chapter6DataLoadError(
                "Во входном артефакте отсутствуют обязательные колонки: "
                + ", ".join(missing)
                + f". Файл: {path}"
            )

    @staticmethod
    def _require_true(
        mapping: Mapping[str, Any],
        key: str,
        description: str,
    ) -> None:
        """Потребовать истинное логическое значение в JSON-отчете."""

        if mapping.get(key) is not True:
            raise Chapter6DataLoadError(
                f"В отчете главы 5 не подтверждено условие: {description}."
            )


__all__ = [
    "ArtifactValidationResult",
    "Chapter6DataLoadError",
    "Chapter6DataLoader",
    "Chapter6InputValidationReport",
    "Chapter6LoadedInputs",
    "FACTUAL_CRITERIA",
    "PREDICTED_CRITERIA",
    "THETA_COLUMNS",
]
