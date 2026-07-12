"""Генерация синтетических контрольных датасетов для проверки гипотезы.

Положительный контроль содержит заранее заданный независимый вклад
латентного профиля в фактическое качество. Нулевой контроль исключает
этот вклад. Оба набора предназначены только для проверки методики и
программного контура, а не для доказательства практической эффективности.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

SEED = 20260712
ROW_COUNT = 600
CLASS_COUNT = 200
TOPIC_QUALITY = np.asarray([0.20, 0.55, 0.88], dtype=float)
QUALITY_NAMES = ("acc", "time", "effort", "res", "rep", "fit")
QUALITY_OFFSETS = np.asarray([0.025, -0.020, 0.010, -0.030, 0.020, -0.005])
OBSERVED_WEIGHTS = np.asarray([0.55, 0.55, 0.55, 0.55, 0.55, 0.55])
LATENT_WEIGHTS = 1.0 - OBSERVED_WEIGHTS

PRIOR_NORM_COLUMNS = (
    "prior_alpha_ratio_norm",
    "prior_attention_deficit_norm",
    "prior_condition_mean_adjusted_attention_norm",
    "prior_condition_noise_adjustment_norm",
    "prior_condition_noise_level_norm",
    "prior_condition_time_pressure_norm",
    "prior_condition_total_adjusted_time_norm",
    "prior_control_expected_detection_rate_norm",
    "prior_control_intensity_norm",
    "prior_control_marker_ratio_norm",
    "prior_digit_ratio_norm",
    "prior_expected_error_probability_norm",
    "prior_manual_operation_count_norm",
    "prior_mean_complexity_norm",
    "prior_mean_message_criticality_norm",
    "prior_memory_load_index_norm",
    "prior_message_entropy_estimate_norm",
    "prior_message_length_norm",
    "prior_operator_attention_norm",
    "prior_operator_error_susceptibility_norm",
    "prior_operator_fatigue_norm",
    "prior_operator_skill_norm",
    "prior_operator_total_estimated_time_norm",
    "prior_procedure_branch_count_norm",
    "prior_procedure_steps_norm",
    "prior_repetition_expected_count_norm",
    "prior_symbol_group_count_norm",
    "prior_time_pressure_index_norm",
    "prior_total_nominal_time_norm",
    "prior_verification_required_norm",
)

HEADER = [
    "scenario_id", "protocol_id", "run_id", "alternative_id", "q_pred", "q_fact",
    "integral_quality", "q_pred_class", "q_fact_class", "q_acc_pred", "q_time_pred",
    "q_effort_pred", "q_res_pred", "q_rep_pred", "q_fit_pred", "q_acc", "q_time",
    "q_effort", "q_res", "q_rep", "q_fit", "q_latent", "theta_0", "theta_1",
    "theta_2", "theta_dominant_topic", "theta_dominant_value", "theta_entropy",
    "lda_instability", "input_missing_share", "uncertainty_score", "interval_radius",
    "q_pred_lower", "q_pred_upper",
    "q_acc_feature_component", "q_acc_latent_component", "q_acc_observed_weight",
    "q_acc_latent_weight", "q_time_feature_component", "q_time_latent_component",
    "q_time_observed_weight", "q_time_latent_weight", "q_effort_feature_component",
    "q_effort_latent_component", "q_effort_observed_weight", "q_effort_latent_weight",
    "q_res_feature_component", "q_res_latent_component", "q_res_observed_weight",
    "q_res_latent_weight", "q_rep_feature_component", "q_rep_latent_component",
    "q_rep_observed_weight", "q_rep_latent_weight", "q_fit_feature_component",
    "q_fit_latent_component", "q_fit_observed_weight", "q_fit_latent_weight",
    *PRIOR_NORM_COLUMNS,
    "document_index", "selected_k", "random_state", "latent_direction_score",
    "fact_duration_sec", "fact_error_count", "fact_recheck_count", "fact_reject_count",
    "fact_success",
]


def _clip(value: float) -> float:
    """Ограничить значение единичным интервалом."""

    return float(np.clip(value, 0.0, 1.0))


def _quality_class(value: float) -> str:
    """Преобразовать качество в класс по порогам главы 6."""

    if value < 0.45:
        return "low"
    if value < 0.70:
        return "medium"
    return "high"


def _theta_from_target(target: float, rng: np.random.Generator) -> np.ndarray:
    """Построить латентный профиль с заданным средним качеством темы."""

    target = float(np.clip(target, TOPIC_QUALITY[0], TOPIC_QUALITY[-1]))
    if target <= TOPIC_QUALITY[1]:
        right = (target - TOPIC_QUALITY[0]) / (TOPIC_QUALITY[1] - TOPIC_QUALITY[0])
        theta = np.asarray([1.0 - right, right, 0.0], dtype=float)
    else:
        right = (target - TOPIC_QUALITY[1]) / (TOPIC_QUALITY[2] - TOPIC_QUALITY[1])
        theta = np.asarray([0.0, 1.0 - right, right], dtype=float)

    # Небольшая примесь всех тем предотвращает искусственные нулевые вероятности.
    background = rng.dirichlet(np.asarray([1.5, 1.5, 1.5], dtype=float))
    theta = 0.94 * theta + 0.06 * background
    theta = theta / theta.sum()
    return theta


def _entropy(theta: np.ndarray) -> float:
    """Рассчитать нормированную энтропию латентного профиля."""

    safe = np.clip(theta, 1e-15, 1.0)
    return float(-(safe * np.log(safe)).sum() / math.log(len(theta)))


def _generate_prior_features(
    prior_core: float,
    rng: np.random.Generator,
) -> dict[str, float]:
    """Сформировать правдоподобный набор нормированных априорных признаков."""

    positive = {
        "prior_alpha_ratio_norm",
        "prior_condition_mean_adjusted_attention_norm",
        "prior_control_expected_detection_rate_norm",
        "prior_control_intensity_norm",
        "prior_control_marker_ratio_norm",
        "prior_operator_attention_norm",
        "prior_operator_skill_norm",
        "prior_verification_required_norm",
    }
    negative = {
        "prior_attention_deficit_norm",
        "prior_condition_noise_adjustment_norm",
        "prior_condition_noise_level_norm",
        "prior_condition_time_pressure_norm",
        "prior_condition_total_adjusted_time_norm",
        "prior_digit_ratio_norm",
        "prior_expected_error_probability_norm",
        "prior_manual_operation_count_norm",
        "prior_mean_complexity_norm",
        "prior_mean_message_criticality_norm",
        "prior_memory_load_index_norm",
        "prior_message_entropy_estimate_norm",
        "prior_message_length_norm",
        "prior_operator_error_susceptibility_norm",
        "prior_operator_fatigue_norm",
        "prior_operator_total_estimated_time_norm",
        "prior_procedure_branch_count_norm",
        "prior_procedure_steps_norm",
        "prior_repetition_expected_count_norm",
        "prior_symbol_group_count_norm",
        "prior_time_pressure_index_norm",
        "prior_total_nominal_time_norm",
    }

    result: dict[str, float] = {}
    for column in PRIOR_NORM_COLUMNS:
        if column in positive:
            center = prior_core
        elif column in negative:
            center = 1.0 - prior_core
        else:
            center = 0.5
        result[column] = _clip(center + rng.normal(0.0, 0.075))

    result["prior_verification_required_norm"] = _clip(
        0.70 * result["prior_control_intensity_norm"]
        + 0.30 * result["prior_mean_message_criticality_norm"]
    )
    result["prior_repetition_expected_count_norm"] = _clip(
        0.55 * result["prior_expected_error_probability_norm"]
        + 0.45 * result["prior_operator_fatigue_norm"]
    )
    return result


def _generate_rows(control_type: str) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Сформировать строки положительного или нулевого контроля."""

    rng = np.random.default_rng(SEED + (0 if control_type == "positive" else 1))
    targets = np.concatenate(
        [
            rng.uniform(0.20, 0.41, CLASS_COUNT),
            rng.uniform(0.49, 0.66, CLASS_COUNT),
            rng.uniform(0.75, 0.91, CLASS_COUNT),
        ]
    )
    rng.shuffle(targets)

    rows: list[dict[str, Any]] = []
    splits: list[dict[str, str]] = []
    split_labels = np.asarray(["development"] * 360 + ["calibration"] * 120 + ["test"] * 120)
    rng.shuffle(split_labels)

    for index, target in enumerate(targets):
        delta = float(np.clip(rng.normal(0.0, 0.17), -0.28, 0.28))
        latent_target = _clip(target - delta)
        theta = _theta_from_target(latent_target, rng)
        latent_core = float(theta @ TOPIC_QUALITY)
        prior_core = _clip(2.0 * target - latent_core)

        feature_components: list[float] = []
        latent_components: list[float] = []
        predicted_criteria: list[float] = []
        factual_criteria: list[float] = []

        common_feature_noise = rng.normal(0.0, 0.035)
        common_latent_noise = rng.normal(0.0, 0.030)
        common_outcome_noise = rng.normal(0.0, 0.012)

        for criterion_index, offset in enumerate(QUALITY_OFFSETS):
            feature = _clip(
                prior_core
                + offset
                + common_feature_noise
                + rng.normal(0.0, 0.018)
            )
            latent = _clip(
                latent_core
                - 0.35 * offset
                + common_latent_noise
                + rng.normal(0.0, 0.015)
            )
            predicted = _clip(
                OBSERVED_WEIGHTS[criterion_index] * feature
                + LATENT_WEIGHTS[criterion_index] * latent
            )

            if control_type == "positive":
                factual_center = (
                    OBSERVED_WEIGHTS[criterion_index] * (prior_core + offset)
                    + LATENT_WEIGHTS[criterion_index]
                    * (latent_core - 0.35 * offset)
                )
            else:
                # В нулевом контроле фактическое качество определяется только
                # априорным компонентом; латентный профиль не дает добавочной пользы.
                factual_center = prior_core + offset

            factual = _clip(
                factual_center
                + common_outcome_noise
                + rng.normal(0.0, 0.010)
            )

            feature_components.append(feature)
            latent_components.append(latent)
            predicted_criteria.append(predicted)
            factual_criteria.append(factual)

        q_pred = float(np.mean(predicted_criteria))
        q_fact = float(np.mean(factual_criteria))
        q_latent = float(np.mean(latent_components))
        absolute_error = abs(q_pred - q_fact)

        # Оценка неопределенности строится до использования фактического результата.
        signal_disagreement = abs(float(np.mean(feature_components)) - q_latent)
        uncertainty_score = _clip(
            0.70 * signal_disagreement
            + 0.20 * _entropy(theta)
            + 0.10 * rng.uniform(0.0, 0.25)
        )
        interval_radius = _clip(0.025 + 0.60 * uncertainty_score)
        q_pred_lower = max(0.0, q_pred - interval_radius)
        q_pred_upper = min(1.0, q_pred + interval_radius)

        prior_features = _generate_prior_features(prior_core, rng)
        dominant_index = int(np.argmax(theta))

        row: dict[str, Any] = {
            "scenario_id": f"syn_{control_type[:3]}_{index:04d}",
            "protocol_id": f"prt_syn_{control_type[:3]}_{index:04d}",
            "run_id": f"synthetic_{control_type}_control_v1",
            "alternative_id": f"alt_syn_{index:04d}",
            "q_pred": q_pred,
            "q_fact": q_fact,
            "integral_quality": q_fact,
            "q_pred_class": _quality_class(q_pred),
            "q_fact_class": _quality_class(q_fact),
            "q_latent": q_latent,
            "theta_0": float(theta[0]),
            "theta_1": float(theta[1]),
            "theta_2": float(theta[2]),
            "theta_dominant_topic": f"theta_{dominant_index}",
            "theta_dominant_value": float(theta[dominant_index]),
            "theta_entropy": _entropy(theta),
            "lda_instability": _clip(0.015 + rng.uniform(0.0, 0.055)),
            "input_missing_share": 0.0,
            "uncertainty_score": uncertainty_score,
            "interval_radius": interval_radius,
            "q_pred_lower": q_pred_lower,
            "q_pred_upper": q_pred_upper,
            "document_index": index,
            "selected_k": 3,
            "random_state": SEED,
            "latent_direction_score": q_latent,
            "fact_duration_sec": float(25.0 + 210.0 * (1.0 - q_fact) + rng.normal(0.0, 8.0)),
            "fact_error_count": int(max(0, round(12.0 * (1.0 - q_fact) + rng.normal(0.0, 1.0)))),
            "fact_recheck_count": int(max(0, round(5.0 * (1.0 - q_fact) + rng.normal(0.0, 0.7)))),
            "fact_reject_count": int(max(0, round(3.0 * (1.0 - q_fact) + rng.normal(0.0, 0.5)))),
            "fact_success": int(q_fact >= 0.55),
        }

        for name, predicted, factual, feature, latent, observed_weight, latent_weight in zip(
            QUALITY_NAMES,
            predicted_criteria,
            factual_criteria,
            feature_components,
            latent_components,
            OBSERVED_WEIGHTS,
            LATENT_WEIGHTS,
            strict=True,
        ):
            row[f"q_{name}_pred"] = predicted
            row[f"q_{name}"] = factual
            row[f"q_{name}_feature_component"] = feature
            row[f"q_{name}_latent_component"] = latent
            row[f"q_{name}_observed_weight"] = float(observed_weight)
            row[f"q_{name}_latent_weight"] = float(latent_weight)

        row.update(prior_features)
        rows.append(row)
        splits.append(
            {
                "scenario_id": row["scenario_id"],
                "protocol_id": row["protocol_id"],
                "split": str(split_labels[index]),
                "synthetic_control": control_type,
                "generation_seed": str(SEED + (0 if control_type == "positive" else 1)),
                "latent_contribution_in_ground_truth": "true" if control_type == "positive" else "false",
            }
        )

    return rows, splits


def _write_csv(path: Path, rows: list[dict[str, Any]], header: list[str]) -> None:
    """Сохранить строки в CSV с фиксированным порядком колонок."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=header, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Сформировать положительный и нулевой синтетические контроли."""

    root = Path(__file__).resolve().parent.parent
    metadata: dict[str, Any] = {
        "dataset_type": "synthetic_methodological_controls",
        "not_empirical_evidence": True,
        "row_count_per_control": ROW_COUNT,
        "generation_seed": SEED,
        "quality_thresholds": {"low_max": 0.45, "high_min": 0.70},
        "positive_control": {
            "latent_contribution_in_ground_truth": True,
            "purpose": "Проверка способности метода обнаруживать известный вклад LDA.",
        },
        "null_control": {
            "latent_contribution_in_ground_truth": False,
            "purpose": "Проверка отсутствия ложного подтверждения при нулевом эффекте.",
        },
    }

    for control_type in ("positive", "null"):
        rows, splits = _generate_rows(control_type)
        control_dir = root / "data" / f"{control_type}_control"
        _write_csv(control_dir / "validation_dataset.csv", rows, HEADER)
        split_header = [
            "scenario_id",
            "protocol_id",
            "split",
            "synthetic_control",
            "generation_seed",
            "latent_contribution_in_ground_truth",
        ]
        _write_csv(control_dir / "split_assignments.csv", splits, split_header)

        split_by_scenario = {item["scenario_id"]: item["split"] for item in splits}
        for split_name in ("development", "calibration", "test"):
            split_rows = [
                row
                for row in rows
                if split_by_scenario[str(row["scenario_id"])] == split_name
            ]
            _write_csv(
                control_dir / f"validation_dataset_{split_name}.csv",
                split_rows,
                HEADER,
            )

    (root / "metadata").mkdir(parents=True, exist_ok=True)
    (root / "metadata" / "generation_parameters.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
