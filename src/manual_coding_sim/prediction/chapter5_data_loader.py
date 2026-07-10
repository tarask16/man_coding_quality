"""Загрузка и объединение входных данных главы 5.

Модуль читает априорные признаки, латентный профиль ``theta_prior`` и
интерпретацию тем LDA_prior. Основная задача этапа 3 — получить единый
табличный вход для последующих расчетов главы 5 и остановить выполнение при
методически некорректной структуре данных.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from manual_coding_sim.prediction.chapter5_config import (
    THETA_COLUMNS,
    Chapter5InputPaths,
)
from manual_coding_sim.prediction.chapter5_leakage_guard import Chapter5LeakageGuard
from manual_coding_sim.prediction.paths import resolve_project_path


@dataclass(frozen=True)
class Chapter5InputContract:
    """Описание обязательных входных файлов главы 5."""

    prior_features_path: Path
    theta_prior_path: Path
    topic_interpretation_path: Path


@dataclass(frozen=True)
class Chapter5InputValidationReport:
    """Краткий отчет о проверке входных данных главы 5."""

    scenario_count: int
    theta_row_count: int
    prior_feature_row_count: int
    topic_count: int
    merge_key_columns: tuple[str, ...]
    prior_has_protocol_id: bool
    selected_k: int


@dataclass(frozen=True)
class Chapter5LoadedInputs:
    """Загруженные и объединенные входные данные главы 5."""

    prior_features: pd.DataFrame
    theta_prior: pd.DataFrame
    topic_interpretation: dict[str, Any]
    merged_data: pd.DataFrame
    validation_report: Chapter5InputValidationReport


class Chapter5DataLoadError(ValueError):
    """Ошибка загрузки или структурной проверки входных данных главы 5."""


class Chapter5DataLoader:
    """Загружает входные артефакты для последующего расчета ``Q_pred``."""

    def __init__(
        self,
        paths: Chapter5InputPaths | None = None,
        project_root: str | Path = ".",
        expected_topic_count: int = 3,
        theta_sum_tolerance: float = 1e-4,
    ) -> None:
        """Создать загрузчик входов главы 5.

        ``project_root`` позволяет одинаково запускать загрузчик из тестов, CLI и
        пользовательского рабочего каталога. Относительные пути из конфигурации
        разрешаются относительно этого корня проекта.
        """

        self.paths = paths or Chapter5InputPaths()
        self.project_root = Path(project_root)
        self.expected_topic_count = expected_topic_count
        self.theta_sum_tolerance = theta_sum_tolerance
        self.leakage_guard = Chapter5LeakageGuard()

    def describe_expected_inputs(self) -> Chapter5InputContract:
        """Вернуть описание обязательных входных файлов без чтения данных."""

        return Chapter5InputContract(
            prior_features_path=self.paths.prior_features_path,
            theta_prior_path=self.paths.theta_prior_path,
            topic_interpretation_path=self.paths.topic_interpretation_path,
        )

    def load(self) -> Chapter5LoadedInputs:
        """Загрузить, проверить и объединить входные данные главы 5."""

        prior_features = self.load_prior_features()
        theta_prior = self.load_theta_prior()
        topic_interpretation = self.load_topic_interpretation()
        merged_data, merge_keys = self.merge_inputs(prior_features, theta_prior)
        report = Chapter5InputValidationReport(
            scenario_count=int(merged_data["scenario_id"].nunique()),
            theta_row_count=int(theta_prior.shape[0]),
            prior_feature_row_count=int(prior_features.shape[0]),
            topic_count=int(topic_interpretation["topic_count"]),
            merge_key_columns=merge_keys,
            prior_has_protocol_id="protocol_id" in prior_features.columns,
            selected_k=int(theta_prior["selected_k"].iloc[0]),
        )
        return Chapter5LoadedInputs(
            prior_features=prior_features,
            theta_prior=theta_prior,
            topic_interpretation=topic_interpretation,
            merged_data=merged_data,
            validation_report=report,
        )

    def load_prior_features(self) -> pd.DataFrame:
        """Прочитать ``prior_features.csv`` и проверить априорную структуру."""

        path = self._resolve(self.paths.prior_features_path)
        prior_features = self._read_csv(path, "априорных признаков")
        self._require_columns(prior_features, ("scenario_id",), path)
        self.leakage_guard.require_safe_dataframe(
            prior_features,
            source_name=str(path),
        )
        self._require_prior_columns(prior_features, path)
        self._require_no_duplicate_keys(prior_features, self._prior_key_columns(prior_features), path)
        return prior_features

    def load_theta_prior(self) -> pd.DataFrame:
        """Прочитать ``theta_prior.csv`` и проверить латентный профиль."""

        path = self._resolve(self.paths.theta_prior_path)
        theta_prior = self._read_csv(path, "латентного профиля")
        required_columns = ("scenario_id", "protocol_id", "selected_k", *THETA_COLUMNS)
        self._require_columns(theta_prior, required_columns, path)
        self._require_no_duplicate_keys(theta_prior, ("scenario_id", "protocol_id"), path)
        self._require_selected_k(theta_prior, path)
        self._require_theta_simplex(theta_prior, path)
        return theta_prior

    def load_topic_interpretation(self) -> dict[str, Any]:
        """Прочитать ``topic_interpretation.json`` и проверить пригодность LDA_prior."""

        path = self._resolve(self.paths.topic_interpretation_path)
        if not path.exists():
            msg = f"Файл интерпретации тем главы 4 не найден: {path}"
            raise FileNotFoundError(msg)
        with path.open("r", encoding="utf-8") as file_obj:
            payload = json.load(file_obj)
        if not isinstance(payload, dict):
            msg = f"Файл интерпретации тем должен содержать JSON-объект: {path}"
            raise Chapter5DataLoadError(msg)
        self._validate_topic_interpretation(payload, path)
        return payload

    def merge_inputs(
        self,
        prior_features: pd.DataFrame,
        theta_prior: pd.DataFrame,
    ) -> tuple[pd.DataFrame, tuple[str, ...]]:
        """Объединить ``prior_features`` и ``theta_prior`` по идентификаторам сценария."""

        merge_keys = self._merge_key_columns(prior_features)
        merged_data = theta_prior.merge(
            prior_features,
            how="inner",
            on=list(merge_keys),
            suffixes=("", "_prior"),
            validate="one_to_one",
        )
        if merged_data.shape[0] != theta_prior.shape[0]:
            self._raise_merge_mismatch(prior_features, theta_prior, merged_data, merge_keys)
        return merged_data, merge_keys

    def _resolve(self, path: Path) -> Path:
        """Разрешить путь относительно корня проекта."""

        return resolve_project_path(self.project_root, path)

    @staticmethod
    def _read_csv(path: Path, description: str) -> pd.DataFrame:
        """Прочитать CSV-файл с русскоязычной диагностикой."""

        if not path.exists():
            msg = f"Файл {description} главы 5 не найден: {path}"
            raise FileNotFoundError(msg)
        try:
            return pd.read_csv(path)
        except Exception as error:  # noqa: BLE001 - нужен понятный внешний текст ошибки.
            msg = f"Не удалось прочитать файл {description}: {path}"
            raise Chapter5DataLoadError(msg) from error

    @staticmethod
    def _require_columns(df: pd.DataFrame, required_columns: tuple[str, ...], path: Path) -> None:
        """Проверить наличие обязательных колонок."""

        missing_columns = [column for column in required_columns if column not in df.columns]
        if missing_columns:
            joined = ", ".join(missing_columns)
            msg = f"Во входном файле отсутствуют обязательные колонки: {joined}. Файл: {path}"
            raise Chapter5DataLoadError(msg)

    @staticmethod
    def _require_prior_columns(df: pd.DataFrame, path: Path) -> None:
        """Проверить, что таблица содержит хотя бы один априорный признак."""

        prior_columns = [column for column in df.columns if column.startswith("prior_")]
        if not prior_columns:
            msg = f"Файл априорных признаков не содержит колонок prior_*: {path}"
            raise Chapter5DataLoadError(msg)

    @staticmethod
    def _prior_key_columns(df: pd.DataFrame) -> tuple[str, ...]:
        """Вернуть ключи уникальности таблицы априорных признаков."""

        if "protocol_id" in df.columns:
            return ("scenario_id", "protocol_id")
        return ("scenario_id",)

    def _merge_key_columns(self, prior_features: pd.DataFrame) -> tuple[str, ...]:
        """Выбрать ключи объединения с учетом структуры ``prior_features``."""

        if "protocol_id" in prior_features.columns:
            return ("scenario_id", "protocol_id")
        return ("scenario_id",)

    @staticmethod
    def _require_no_duplicate_keys(df: pd.DataFrame, key_columns: tuple[str, ...], path: Path) -> None:
        """Запретить дубли идентификаторов, нарушающие однозначное объединение."""

        duplicate_mask = df.duplicated(list(key_columns), keep=False)
        if duplicate_mask.any():
            duplicate_values = df.loc[duplicate_mask, list(key_columns)].head(5).to_dict("records")
            msg = (
                "Во входном файле обнаружены дубли идентификаторов, "
                f"объединение главы 5 неоднозначно. Ключи: {key_columns}. "
                f"Примеры: {duplicate_values}. Файл: {path}"
            )
            raise Chapter5DataLoadError(msg)

    def _require_selected_k(self, theta_prior: pd.DataFrame, path: Path) -> None:
        """Проверить, что ``selected_k`` согласован с конфигурацией главы 5."""

        selected_values = sorted(theta_prior["selected_k"].dropna().unique().tolist())
        if selected_values != [self.expected_topic_count]:
            msg = (
                "Файл theta_prior.csv содержит несогласованное число факторов: "
                f"ожидалось {self.expected_topic_count}, найдено {selected_values}. Файл: {path}"
            )
            raise Chapter5DataLoadError(msg)

    def _require_theta_simplex(self, theta_prior: pd.DataFrame, path: Path) -> None:
        """Проверить неотрицательность и сумму компонент ``theta``."""

        theta_values = theta_prior[list(THETA_COLUMNS)]
        if theta_values.isna().any().any():
            msg = f"В theta_prior.csv обнаружены пропуски в компонентах theta. Файл: {path}"
            raise Chapter5DataLoadError(msg)
        if (theta_values < 0).any().any():
            msg = f"Компоненты theta должны быть неотрицательными. Файл: {path}"
            raise Chapter5DataLoadError(msg)
        theta_sums = theta_values.sum(axis=1)
        bad_rows = theta_sums.sub(1.0).abs() > self.theta_sum_tolerance
        if bad_rows.any():
            examples = theta_sums[bad_rows].head(5).round(6).tolist()
            msg = (
                "Сумма компонентов theta должна быть близка к 1. "
                f"Примеры некорректных сумм: {examples}. Файл: {path}"
            )
            raise Chapter5DataLoadError(msg)

    def _validate_topic_interpretation(self, payload: dict[str, Any], path: Path) -> None:
        """Проверить JSON-интерпретацию факторов главы 4."""

        if payload.get("model_name") != "LDA_prior":
            msg = f"Для главы 5 допустима только интерпретация модели LDA_prior. Файл: {path}"
            raise Chapter5DataLoadError(msg)
        if payload.get("allowed_for_apriori_forecast") is not True:
            msg = f"Интерпретация тем не разрешена для априорного прогноза. Файл: {path}"
            raise Chapter5DataLoadError(msg)
        topic_count = int(payload.get("topic_count", -1))
        if topic_count != self.expected_topic_count:
            msg = (
                "Количество тем в topic_interpretation.json не совпадает с конфигурацией: "
                f"ожидалось {self.expected_topic_count}, найдено {topic_count}. Файл: {path}"
            )
            raise Chapter5DataLoadError(msg)
        topics = payload.get("topics")
        if not isinstance(topics, list) or len(topics) != self.expected_topic_count:
            msg = f"Список topics должен содержать {self.expected_topic_count} элементов. Файл: {path}"
            raise Chapter5DataLoadError(msg)
        topic_ids = sorted(int(topic.get("topic_id", -1)) for topic in topics)
        if topic_ids != list(range(self.expected_topic_count)):
            msg = f"Идентификаторы тем должны идти от 0 до {self.expected_topic_count - 1}. Файл: {path}"
            raise Chapter5DataLoadError(msg)

    @staticmethod
    def _raise_merge_mismatch(
        prior_features: pd.DataFrame,
        theta_prior: pd.DataFrame,
        merged_data: pd.DataFrame,
        merge_keys: tuple[str, ...],
    ) -> None:
        """Сформировать подробную ошибку несовпадения ключей входных таблиц."""

        key_list = list(merge_keys)
        prior_keys = prior_features[key_list].drop_duplicates()
        theta_keys = theta_prior[key_list].drop_duplicates()
        missing_in_prior = theta_keys.merge(prior_keys, how="left", on=key_list, indicator=True)
        missing_in_prior = missing_in_prior[missing_in_prior["_merge"] == "left_only"]
        missing_in_theta = prior_keys.merge(theta_keys, how="left", on=key_list, indicator=True)
        missing_in_theta = missing_in_theta[missing_in_theta["_merge"] == "left_only"]
        msg = (
            "Не удалось однозначно объединить theta_prior.csv и prior_features.csv. "
            f"Ключи объединения: {merge_keys}. "
            f"Строк theta_prior: {theta_prior.shape[0]}, строк prior_features: {prior_features.shape[0]}, "
            f"строк после объединения: {merged_data.shape[0]}. "
            f"Примеры ключей, отсутствующих в prior_features: "
            f"{missing_in_prior[key_list].head(5).to_dict('records')}. "
            f"Примеры ключей, отсутствующих в theta_prior: "
            f"{missing_in_theta[key_list].head(5).to_dict('records')}."
        )
        raise Chapter5DataLoadError(msg)
