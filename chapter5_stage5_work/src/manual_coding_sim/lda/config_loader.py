"""Загрузка YAML-конфигурации для программного блока главы 4.

Модуль преобразует человекочитаемый YAML-файл в ``Chapter4RunnerConfig``.
Загрузчик не запускает LDA-модель; его задача — проверить структуру
конфигурации, привести типы и применить явные CLI-переопределения.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from manual_coding_sim.lda.chapter4_runner import Chapter4RunnerConfig
from manual_coding_sim.lda.config import (
    Chapter4LdaConfig,
    LdaInputPaths,
    LdaModelConfig,
    LdaOutputPaths,
    LdaTokenizationConfig,
)
from manual_coding_sim.lda.paths import resolve_project_path


@dataclass(frozen=True)
class Chapter4ConfigOverrides:
    """Явные переопределения параметров конфигурации.

    Такие значения обычно приходят из CLI и должны иметь приоритет над YAML,
    потому что пользователь задает их непосредственно при запуске.
    """

    overwrite: bool | None = None
    skip_diagnostic: bool = False
    selected_k: int | None = None


@dataclass(frozen=True)
class Chapter4ConfigLoadResult:
    """Результат загрузки YAML-конфигурации главы 4."""

    config_path: Path
    runner_config: Chapter4RunnerConfig


class Chapter4ConfigLoader:
    """Загружает YAML-конфигурацию единого запуска главы 4."""

    def load(
        self,
        config_path: str | Path,
        overrides: Chapter4ConfigOverrides | None = None,
    ) -> Chapter4ConfigLoadResult:
        """Прочитать YAML-файл и вернуть готовую конфигурацию runner-а."""

        path = Path(config_path)
        if not path.exists():
            msg = f"Файл конфигурации не найден: {path}"
            raise FileNotFoundError(msg)

        with path.open("r", encoding="utf-8") as file_obj:
            payload = yaml.safe_load(file_obj) or {}
        if not isinstance(payload, dict):
            msg = "Корневой элемент YAML-конфига должен быть словарем."
            raise ValueError(msg)

        runner_config = self._build_runner_config(payload, overrides or Chapter4ConfigOverrides())
        return Chapter4ConfigLoadResult(config_path=path, runner_config=runner_config)

    def validate_required_inputs(
        self,
        runner_config: Chapter4RunnerConfig,
        project_root: str | Path = ".",
    ) -> None:
        """Проверить наличие входных CSV-файлов, нужных для выбранного запуска."""

        root = Path(project_root)
        inputs = runner_config.chapter4.inputs
        required_paths = [inputs.prior_features_path]
        if runner_config.chapter4.build_lda_diag:
            required_paths.append(inputs.diagnostic_features_path)
        if runner_config.chapter4.build_lda_full:
            required_paths.append(inputs.fact_features_path)

        missing_paths = [
            str(resolve_project_path(root, path))
            for path in required_paths
            if not resolve_project_path(root, path).exists()
        ]
        if missing_paths:
            msg = "Не найдены обязательные входные файлы главы 4: " + ", ".join(
                missing_paths
            )
            raise FileNotFoundError(msg)

    def _build_runner_config(
        self,
        payload: dict[str, Any],
        overrides: Chapter4ConfigOverrides,
    ) -> Chapter4RunnerConfig:
        """Собрать ``Chapter4RunnerConfig`` из словаря YAML."""

        input_section = _section(payload, "input", "inputs")
        output_section = _section(payload, "output", "outputs")
        tokenization_section = _section(payload, "tokenization")
        lda_section = _section(payload, "lda", "model")
        diagnostics_section = _section(payload, "diagnostics")
        safety_section = _section(payload, "safety")
        runner_section = _section(payload, "runner")

        model_config = LdaModelConfig(
            k_values=_tuple_of_ints(lda_section.get("k_values"), (3, 4, 5, 6, 7, 8)),
            selected_k=_optional_int(lda_section.get("selected_k")),
            doc_topic_prior=_optional_float(lda_section.get("doc_topic_prior")),
            topic_word_prior=_optional_float(lda_section.get("topic_word_prior")),
            learning_method=str(lda_section.get("learning_method", "batch")),
            max_iter=_positive_int(lda_section.get("max_iter", 100), "max_iter"),
            random_seeds=_tuple_of_ints(
                lda_section.get("random_seeds"),
                (11, 42, 77, 101),
            ),
        )
        if overrides.selected_k is not None:
            model_config = LdaModelConfig(
                k_values=model_config.k_values,
                selected_k=overrides.selected_k,
                doc_topic_prior=model_config.doc_topic_prior,
                topic_word_prior=model_config.topic_word_prior,
                learning_method=model_config.learning_method,
                max_iter=model_config.max_iter,
                random_seeds=model_config.random_seeds,
            )

        build_lda_diag = _bool(diagnostics_section.get("build_lda_diag", True))
        build_lda_full = _bool(diagnostics_section.get("build_lda_full", True))
        if overrides.skip_diagnostic:
            build_lda_diag = False
            build_lda_full = False

        overwrite_value = _bool(runner_section.get("overwrite", True))
        if overrides.overwrite is not None:
            overwrite_value = overrides.overwrite

        chapter4_config = Chapter4LdaConfig(
            inputs=LdaInputPaths(
                protocols_path=_path(input_section.get("protocols"), "data/processed/protocols.csv"),
                prior_features_path=_path(
                    input_section.get("prior_features"),
                    "data/processed/prior_features.csv",
                ),
                diagnostic_features_path=_path(
                    input_section.get("diagnostic_features"),
                    "data/processed/diagnostic_features.csv",
                ),
                fact_features_path=_path(
                    input_section.get("fact_features"),
                    "data/processed/fact_features.csv",
                ),
                quality_targets_path=_path(
                    input_section.get("quality_targets"),
                    "data/processed/quality_targets.csv",
                ),
            ),
            outputs=LdaOutputPaths(
                data_dir=_path(output_section.get("data_dir"), "data/processed/lda"),
                models_dir=_path(output_section.get("models_dir"), "models/lda"),
                reports_dir=_path(output_section.get("reports_dir"), "reports/chapter4"),
            ),
            tokenization=LdaTokenizationConfig(
                df_min=_positive_int(tokenization_section.get("df_min", 2), "df_min"),
                df_max_ratio=_float(
                    tokenization_section.get("df_max_ratio", 0.95),
                    "df_max_ratio",
                ),
                numeric_strategy=str(
                    tokenization_section.get("numeric_strategy", "quantile")
                ),
                numeric_bins=_positive_int(
                    tokenization_section.get("numeric_bins", 3),
                    "numeric_bins",
                ),
            ),
            model=model_config,
            build_lda_diag=build_lda_diag,
            build_lda_full=build_lda_full,
            forbid_fact_features_in_prior=_bool(
                safety_section.get("forbid_fact_features_in_prior", True)
            ),
            forbid_quality_targets_in_prior=_bool(
                safety_section.get("forbid_quality_targets_in_prior", True)
            ),
            require_corpus_hash=_bool(safety_section.get("require_corpus_hash", True)),
        )
        return Chapter4RunnerConfig(
            chapter4=chapter4_config,
            top_n=_positive_int(runner_section.get("top_n", 10), "top_n"),
            overwrite=overwrite_value,
        )


def load_chapter4_runner_config(
    config_path: str | Path,
    overrides: Chapter4ConfigOverrides | None = None,
) -> Chapter4RunnerConfig:
    """Загрузить YAML-конфиг и вернуть только конфигурацию runner-а."""

    return Chapter4ConfigLoader().load(config_path, overrides).runner_config


def _section(payload: dict[str, Any], *names: str) -> dict[str, Any]:
    """Вернуть словарную секцию YAML по одному из допустимых имен."""

    for name in names:
        value = payload.get(name)
        if value is None:
            continue
        if not isinstance(value, dict):
            msg = f"Секция '{name}' должна быть словарем."
            raise ValueError(msg)
        return value
    return {}


def _path(value: Any, default: str) -> Path:
    """Преобразовать значение YAML в путь."""

    if value is None:
        return Path(default)
    return Path(str(value))


def _bool(value: Any) -> bool:
    """Преобразовать YAML-значение в логический флаг."""

    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "1", "on"}:
            return True
        if normalized in {"false", "no", "0", "off"}:
            return False
    return bool(value)


def _positive_int(value: Any, name: str) -> int:
    """Преобразовать значение в положительное целое число."""

    result = int(value)
    if result < 1:
        msg = f"{name} должен быть положительным целым числом."
        raise ValueError(msg)
    return result


def _optional_int(value: Any) -> int | None:
    """Преобразовать необязательное значение в целое число."""

    if value is None:
        return None
    return int(value)


def _float(value: Any, name: str) -> float:
    """Преобразовать значение в число с плавающей точкой."""

    try:
        return float(value)
    except TypeError as exc:
        msg = f"{name} должен быть числом."
        raise ValueError(msg) from exc


def _optional_float(value: Any) -> float | None:
    """Преобразовать необязательное значение в число с плавающей точкой."""

    if value is None:
        return None
    return float(value)


def _tuple_of_ints(value: Any, default: tuple[int, ...]) -> tuple[int, ...]:
    """Преобразовать YAML-список в кортеж целых чисел."""

    if value is None:
        return default
    if isinstance(value, int):
        return (value,)
    if not isinstance(value, list | tuple):
        msg = "Ожидался список целых чисел."
        raise ValueError(msg)
    return tuple(int(item) for item in value)
