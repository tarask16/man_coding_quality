"""Итоговый отчет программного блока главы 4.

Модуль собирает уже созданные артефакты LDA-анализа в единый JSON- и
Markdown-отчет. Он не обучает модели и не пересчитывает метрики. Его задача —
проверить полноту результатов главы 4 и зафиксировать трассируемую сводку для
текста диссертации.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping, Sequence


@dataclass(frozen=True)
class Chapter4LdaReportConfig:
    """Параметры формирования итогового отчета главы 4."""

    encoding: str = "utf-8"
    overwrite: bool = True
    require_diagnostic_artifacts: bool = True


@dataclass(frozen=True)
class Chapter4LdaReportResult:
    """Результат формирования итогового отчета главы 4."""

    report_json_path: Path
    report_md_path: Path
    selected_k: int
    completed_steps: tuple[str, ...]
    required_artifact_count: int
    diagnostic_artifacts_included: bool


class Chapter4LdaReportBuilder:
    """Формирует итоговый отчет по артефактам главы 4."""

    def __init__(self, config: Chapter4LdaReportConfig | None = None) -> None:
        """Создать построитель итогового отчета."""

        self.config = config or Chapter4LdaReportConfig()

    def build_from_artifacts(
        self,
        data_dir: str | Path,
        models_dir: str | Path,
        reports_dir: str | Path,
        selected_k: int,
    ) -> Chapter4LdaReportResult:
        """Проверить артефакты главы 4 и сохранить итоговый отчет."""

        if selected_k < 2:
            msg = "selected_k должен быть не меньше 2."
            raise ValueError(msg)

        data_path = Path(data_dir)
        models_path = Path(models_dir)
        reports_path = Path(reports_dir)
        reports_path.mkdir(parents=True, exist_ok=True)

        report_json_path = reports_path / "chapter4_lda_report.json"
        report_md_path = reports_path / "chapter4_lda_report.md"
        self._ensure_can_write([report_json_path, report_md_path])

        required_artifacts = self._required_artifacts(
            data_dir=data_path,
            models_dir=models_path,
            reports_dir=reports_path,
        )
        optional_artifacts = self._diagnostic_artifacts(
            data_dir=data_path,
            models_dir=models_path,
            reports_dir=reports_path,
        )
        self._check_required_artifacts(required_artifacts)
        diagnostic_included = self._check_diagnostic_artifacts(optional_artifacts)

        payload = self._build_payload(
            selected_k=selected_k,
            required_artifacts=required_artifacts,
            diagnostic_artifacts=optional_artifacts if diagnostic_included else {},
            reports_dir=reports_path,
        )
        self._write_json(report_json_path, payload)
        self._write_markdown(report_md_path, payload)

        return Chapter4LdaReportResult(
            report_json_path=report_json_path,
            report_md_path=report_md_path,
            selected_k=selected_k,
            completed_steps=tuple(payload["completed_steps"]),
            required_artifact_count=len(required_artifacts),
            diagnostic_artifacts_included=diagnostic_included,
        )

    def _required_artifacts(
        self,
        data_dir: Path,
        models_dir: Path,
        reports_dir: Path,
    ) -> dict[str, Path]:
        """Вернуть перечень обязательных артефактов главы 4."""

        return {
            "token_map": data_dir / "token_map.json",
            "dictionary": data_dir / "dictionary.json",
            "corpus_prior": data_dir / "corpus_prior.csv",
            "corpus_metadata": data_dir / "corpus_metadata.json",
            "leakage_report": data_dir / "leakage_report.json",
            "lda_prior_model": models_dir / "lda_prior.joblib",
            "theta_prior": reports_dir / "theta_prior.csv",
            "topic_word": reports_dir / "topic_word.csv",
            "lda_prior_metadata": reports_dir / "lda_prior_metadata.json",
            "topic_metrics_json": reports_dir / "topic_metrics.json",
            "topic_metrics_csv": reports_dir / "topic_metrics.csv",
            "k_selection_json": reports_dir / "k_selection_report.json",
            "k_selection_csv": reports_dir / "k_selection_report.csv",
            "k_selection_md": reports_dir / "k_selection_report.md",
            "topic_stability_json": reports_dir / "topic_stability_report.json",
            "topic_stability_csv": reports_dir / "topic_stability_report.csv",
            "topic_stability_md": reports_dir / "topic_stability_report.md",
            "topic_interpretation_json": reports_dir / "topic_interpretation.json",
            "topic_interpretation_csv": reports_dir / "topic_interpretation.csv",
            "topic_interpretation_md": reports_dir / "topic_interpretation.md",
        }

    def _diagnostic_artifacts(
        self,
        data_dir: Path,
        models_dir: Path,
        reports_dir: Path,
    ) -> dict[str, Path]:
        """Вернуть перечень диагностических артефактов главы 4."""

        return {
            "corpus_diag": data_dir / "corpus_diag.csv",
            "dictionary_diag": data_dir / "dictionary_diag.json",
            "token_map_diag": data_dir / "token_map_diag.json",
            "corpus_metadata_diag": data_dir / "corpus_metadata_diag.json",
            "corpus_full": data_dir / "corpus_full.csv",
            "dictionary_full": data_dir / "dictionary_full.json",
            "token_map_full": data_dir / "token_map_full.json",
            "corpus_metadata_full": data_dir / "corpus_metadata_full.json",
            "lda_diag_model": models_dir / "lda_diag.joblib",
            "lda_full_model": models_dir / "lda_full.joblib",
            "theta_diag": reports_dir / "theta_diag.csv",
            "theta_full": reports_dir / "theta_full.csv",
            "topic_word_diag": reports_dir / "topic_word_diag.csv",
            "topic_word_full": reports_dir / "topic_word_full.csv",
            "lda_diagnostic_metadata": reports_dir / "lda_diagnostic_metadata.json",
        }

    def _check_required_artifacts(self, artifacts: Mapping[str, Path]) -> None:
        """Убедиться, что все обязательные артефакты существуют."""

        missing = [name for name, path in artifacts.items() if not path.exists()]
        if missing:
            msg = "Не найдены обязательные артефакты главы 4: "
            msg += ", ".join(sorted(missing))
            raise FileNotFoundError(msg)

    def _check_diagnostic_artifacts(self, artifacts: Mapping[str, Path]) -> bool:
        """Проверить наличие диагностических артефактов."""

        missing = [name for name, path in artifacts.items() if not path.exists()]
        if not missing:
            return True
        if self.config.require_diagnostic_artifacts:
            msg = "Не найдены диагностические артефакты главы 4: "
            msg += ", ".join(sorted(missing))
            raise FileNotFoundError(msg)
        return False

    def _build_payload(
        self,
        selected_k: int,
        required_artifacts: Mapping[str, Path],
        diagnostic_artifacts: Mapping[str, Path],
        reports_dir: Path,
    ) -> dict[str, object]:
        """Сформировать JSON-структуру итогового отчета."""

        prior_metadata = self._read_json(reports_dir / "lda_prior_metadata.json")
        topic_metrics = self._read_json(reports_dir / "topic_metrics.json")
        k_selection = self._read_json(reports_dir / "k_selection_report.json")
        topic_stability = self._read_json(reports_dir / "topic_stability_report.json")
        topic_interpretation = self._read_json(reports_dir / "topic_interpretation.json")
        diagnostic_metadata = None
        diagnostic_metadata_path = reports_dir / "lda_diagnostic_metadata.json"
        if diagnostic_metadata_path.exists():
            diagnostic_metadata = self._read_json(diagnostic_metadata_path)

        completed_steps = [
            "corpus_prior",
            "k_selection",
            "lda_prior",
            "topic_metrics",
            "topic_stability",
            "topic_interpretation",
        ]
        if diagnostic_artifacts:
            completed_steps.append("lda_diagnostic_models")

        return {
            "report_name": "chapter4_lda_report",
            "status": "completed",
            "selected_k": selected_k,
            "completed_steps": completed_steps,
            "methodological_constraints": {
                "lda_prior_uses_only_prior_features": True,
                "fact_features_forbidden_in_lda_prior": True,
                "quality_targets_forbidden_in_lda_prior": True,
                "diagnostic_models_allowed_for_apriori_forecast": False,
            },
            "prior_model": prior_metadata,
            "topic_metrics": self._compact_metrics(topic_metrics),
            "k_selection": self._compact_k_selection(k_selection),
            "topic_stability": self._compact_stability(topic_stability),
            "topic_interpretation": self._compact_interpretation(topic_interpretation),
            "diagnostic_models": diagnostic_metadata,
            "artifacts": self._paths_to_strings(required_artifacts),
            "diagnostic_artifacts": self._paths_to_strings(diagnostic_artifacts),
        }

    def _compact_metrics(self, payload: Mapping[str, object]) -> dict[str, object]:
        """Сформировать компактную сводку метрик LDA."""

        return {
            "perplexity": payload.get("perplexity"),
            "mean_coherence": payload.get("mean_coherence"),
            "topic_diversity": payload.get("topic_diversity"),
            "n_components": payload.get("n_components"),
            "top_n": payload.get("top_n"),
        }

    def _compact_k_selection(self, payload: Mapping[str, object]) -> dict[str, object]:
        """Сформировать компактную сводку выбора ``K``."""

        return {
            "recommended_k": payload.get("recommended_k"),
            "selection_method": payload.get("selection_method"),
            "candidate_count": len(payload.get("candidates", [])),
        }

    def _compact_stability(self, payload: Mapping[str, object]) -> dict[str, object]:
        """Сформировать компактную сводку устойчивости тем."""

        return {
            "mean_stability": payload.get("mean_stability"),
            "min_stability": payload.get("min_stability"),
            "random_states": payload.get("random_states"),
            "analysis_method": payload.get("analysis_method"),
        }

    def _compact_interpretation(self, payload: Mapping[str, object]) -> dict[str, object]:
        """Сформировать компактную сводку интерпретации тем."""

        topics = payload.get("topics", [])
        factor_names = []
        if isinstance(topics, list):
            factor_names = [
                str(topic.get("suggested_factor_name", ""))
                for topic in topics
                if isinstance(topic, dict)
            ]
        return {
            "topic_count": payload.get("topic_count"),
            "top_n": payload.get("top_n"),
            "factor_names": factor_names,
        }

    def _paths_to_strings(self, artifacts: Mapping[str, Path]) -> dict[str, str]:
        """Преобразовать пути артефактов в строки."""

        return {name: str(path) for name, path in sorted(artifacts.items())}

    def _read_json(self, path: Path) -> dict[str, object]:
        """Прочитать JSON-файл отчета."""

        if not path.exists():
            msg = f"JSON-артефакт не найден: {path}"
            raise FileNotFoundError(msg)
        with path.open("r", encoding=self.config.encoding) as file_obj:
            return json.load(file_obj)

    def _write_json(self, path: Path, payload: Mapping[str, object]) -> None:
        """Сохранить итоговый JSON-отчет."""

        with path.open("w", encoding=self.config.encoding) as file_obj:
            json.dump(payload, file_obj, ensure_ascii=False, indent=2, sort_keys=True)
            file_obj.write("\n")

    def _write_markdown(self, path: Path, payload: Mapping[str, object]) -> None:
        """Сохранить итоговый Markdown-отчет."""

        metrics = payload["topic_metrics"]
        stability = payload["topic_stability"]
        interpretation = payload["topic_interpretation"]
        factor_names = interpretation.get("factor_names", [])
        lines = [
            "# Отчет программного блока главы 4",
            "",
            f"Статус: {payload['status']}.",
            f"Выбранное число латентных факторов K: {payload['selected_k']}.",
            "",
            "## Выполненные этапы",
            "",
        ]
        for step in payload["completed_steps"]:
            lines.append(f"- {step}")
        lines.extend(
            [
                "",
                "## Метрики основной модели LDA_prior",
                "",
                f"- Perplexity: {metrics.get('perplexity')}",
                f"- Mean coherence: {metrics.get('mean_coherence')}",
                f"- Topic diversity: {metrics.get('topic_diversity')}",
                "",
                "## Устойчивость тем",
                "",
                f"- Mean stability: {stability.get('mean_stability')}",
                f"- Min stability: {stability.get('min_stability')}",
                "",
                "## Интерпретированные латентные факторы",
                "",
            ]
        )
        if factor_names:
            for name in factor_names:
                lines.append(f"- {name}")
        else:
            lines.append("- Нет интерпретированных факторов.")
        lines.extend(
            [
                "",
                "## Методические ограничения",
                "",
                "- LDA_prior обучается только по априорным признакам.",
                "- Фактические признаки и целевые показатели качества запрещены для LDA_prior.",
                "- LDA_diag и LDA_full имеют только диагностический статус.",
                "",
            ]
        )
        with path.open("w", encoding=self.config.encoding) as file_obj:
            file_obj.write("\n".join(lines))

    def _ensure_can_write(self, paths: Sequence[Path]) -> None:
        """Проверить возможность записи итоговых отчетов."""

        if self.config.overwrite:
            return
        existing_paths = [path for path in paths if path.exists()]
        if existing_paths:
            msg = "Запрещена перезапись существующих отчетов главы 4: "
            msg += ", ".join(str(path) for path in existing_paths)
            raise FileExistsError(msg)
