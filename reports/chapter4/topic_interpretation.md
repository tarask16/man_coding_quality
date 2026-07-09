# Интерпретация латентных факторов качества

Модель: `LDA_prior`.
Число интерпретируемых тем: 3.
Число top-токенов на тему: 10.

## Тема 0

**Предварительное название:** Латентный фактор качества 0: prior_control_marker_ratio, prior_digit_ratio, prior_mean_message_criticality

**Исходные признаки:** prior_control_marker_ratio, prior_digit_ratio, prior_mean_message_criticality, prior_service_ratio, prior_condition_mean_adjusted_attention, prior_operator_final_fatigue, prior_operator_mean_attention, prior_operator_total_effort, prior_reference_ratio, prior_symbol_ratio

**Top-токены:**

- `prior_control_marker_ratio__level_low`
- `prior_digit_ratio__level_high`
- `prior_mean_message_criticality__level_high`
- `prior_service_ratio__level_low`
- `prior_condition_mean_adjusted_attention__level_low`
- `prior_operator_final_fatigue__level_high`
- `prior_operator_mean_attention__level_low`
- `prior_operator_total_effort__level_high`
- `prior_reference_ratio__level_high`
- `prior_symbol_ratio__level_low`

**Комментарий:** Тема 0 интерпретируется как фактор, связанный с признаками: prior_control_marker_ratio, prior_digit_ratio, prior_mean_message_criticality, prior_service_ratio, prior_condition_mean_adjusted_attention. Доминирующий токен: prior_control_marker_ratio__level_low. Наиболее характерные состояния: априорный признак prior_control_marker_ratio: уровень low; априорный признак prior_digit_ratio: уровень high; априорный признак prior_mean_message_criticality: уровень high.

## Тема 1

**Предварительное название:** Латентный фактор качества 1: prior_condition_total_adjusted_time, prior_operator_total_estimated_time, prior_total_nominal_time

**Исходные признаки:** prior_condition_total_adjusted_time, prior_operator_total_estimated_time, prior_total_nominal_time, prior_mean_complexity, prior_message_length, prior_step_count, prior_condition_mean_adjusted_attention, prior_operator_final_fatigue, prior_operator_mean_attention, prior_operator_total_effort

**Top-токены:**

- `prior_condition_total_adjusted_time__level_high`
- `prior_operator_total_estimated_time__level_high`
- `prior_total_nominal_time__level_high`
- `prior_mean_complexity__level_mid`
- `prior_message_length__level_mid`
- `prior_step_count__level_mid`
- `prior_condition_mean_adjusted_attention__level_low`
- `prior_operator_final_fatigue__level_high`
- `prior_operator_mean_attention__level_low`
- `prior_operator_total_effort__level_high`

**Комментарий:** Тема 1 интерпретируется как фактор, связанный с признаками: prior_condition_total_adjusted_time, prior_operator_total_estimated_time, prior_total_nominal_time, prior_mean_complexity, prior_message_length. Доминирующий токен: prior_condition_total_adjusted_time__level_high. Наиболее характерные состояния: априорный признак prior_condition_total_adjusted_time: уровень high; априорный признак prior_operator_total_estimated_time: уровень high; априорный признак prior_total_nominal_time: уровень high.

## Тема 2

**Предварительное название:** Латентный фактор качества 2: prior_condition_mean_adjusted_attention, prior_condition_total_adjusted_time, prior_mean_complexity

**Исходные признаки:** prior_condition_mean_adjusted_attention, prior_condition_total_adjusted_time, prior_mean_complexity, prior_message_length, prior_operator_final_fatigue, prior_operator_mean_attention, prior_operator_total_effort, prior_operator_total_estimated_time, prior_step_count, prior_total_nominal_time

**Top-токены:**

- `prior_condition_mean_adjusted_attention__level_high`
- `prior_condition_total_adjusted_time__level_low`
- `prior_mean_complexity__level_low`
- `prior_message_length__level_low`
- `prior_operator_final_fatigue__level_low`
- `prior_operator_mean_attention__level_high`
- `prior_operator_total_effort__level_low`
- `prior_operator_total_estimated_time__level_low`
- `prior_step_count__level_low`
- `prior_total_nominal_time__level_low`

**Комментарий:** Тема 2 интерпретируется как фактор, связанный с признаками: prior_condition_mean_adjusted_attention, prior_condition_total_adjusted_time, prior_mean_complexity, prior_message_length, prior_operator_final_fatigue. Доминирующий токен: prior_condition_mean_adjusted_attention__level_high. Наиболее характерные состояния: априорный признак prior_condition_mean_adjusted_attention: уровень high; априорный признак prior_condition_total_adjusted_time: уровень low; априорный признак prior_mean_complexity: уровень low.
