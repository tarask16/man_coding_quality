# Интерпретация латентных факторов качества

Модель: `LDA_prior`.
Число интерпретируемых тем: 3.
Число top-токенов на тему: 10.

## Тема 0

**Предварительное название:** Латентный фактор качества 0: prior_total_nominal_time, prior_memory_load_index, prior_manual_operation_count

**Исходные признаки:** prior_total_nominal_time, prior_memory_load_index, prior_manual_operation_count, prior_operator_total_estimated_time, prior_condition_total_adjusted_time, prior_message_length, prior_symbol_group_count, prior_operator_attention, prior_procedure_branch_count, prior_operator_error_susceptibility

**Top-токены:**

- `prior_total_nominal_time__level_high`
- `prior_memory_load_index__level_high`
- `prior_manual_operation_count__level_high`
- `prior_operator_total_estimated_time__level_high`
- `prior_condition_total_adjusted_time__level_high`
- `prior_message_length__level_high`
- `prior_symbol_group_count__level_high`
- `prior_operator_attention__level_mid`
- `prior_procedure_branch_count__level_high`
- `prior_operator_error_susceptibility__level_mid`

**Комментарий:** Тема 0 интерпретируется как фактор, связанный с признаками: prior_total_nominal_time, prior_memory_load_index, prior_manual_operation_count, prior_operator_total_estimated_time, prior_condition_total_adjusted_time. Доминирующий токен: prior_total_nominal_time__level_high. Наиболее характерные состояния: априорный признак prior_total_nominal_time: уровень high; априорный признак prior_memory_load_index: уровень high; априорный признак prior_manual_operation_count: уровень high.

## Тема 1

**Предварительное название:** Латентный фактор качества 1: prior_condition_mean_adjusted_attention, prior_attention_deficit, prior_control_intensity

**Исходные признаки:** prior_condition_mean_adjusted_attention, prior_attention_deficit, prior_control_intensity, prior_operator_error_susceptibility, prior_operator_attention, prior_expected_error_probability, prior_condition_time_pressure, prior_time_pressure_index, prior_verification_required, prior_operator_skill

**Top-токены:**

- `prior_condition_mean_adjusted_attention__level_low`
- `prior_attention_deficit__level_high`
- `prior_control_intensity__level_low`
- `prior_operator_error_susceptibility__level_high`
- `prior_operator_attention__level_low`
- `prior_expected_error_probability__level_high`
- `prior_condition_time_pressure__level_mid`
- `prior_time_pressure_index__level_mid`
- `prior_verification_required__absent`
- `prior_operator_skill__level_low`

**Комментарий:** Тема 1 интерпретируется как фактор, связанный с признаками: prior_condition_mean_adjusted_attention, prior_attention_deficit, prior_control_intensity, prior_operator_error_susceptibility, prior_operator_attention. Доминирующий токен: prior_condition_mean_adjusted_attention__level_low. Наиболее характерные состояния: априорный признак prior_condition_mean_adjusted_attention: уровень low; априорный признак prior_attention_deficit: уровень high; априорный признак prior_control_intensity: уровень low.

## Тема 2

**Предварительное название:** Латентный фактор качества 2: prior_condition_time_pressure, prior_time_pressure_index, prior_repetition_expected_count

**Исходные признаки:** prior_condition_time_pressure, prior_time_pressure_index, prior_repetition_expected_count, prior_operator_error_susceptibility, prior_expected_error_probability, prior_operator_fatigue, prior_procedure_branch_count, prior_verification_required, prior_condition_noise_adjustment, prior_condition_noise_level

**Top-токены:**

- `prior_condition_time_pressure__level_low`
- `prior_time_pressure_index__level_low`
- `prior_repetition_expected_count__level_low`
- `prior_operator_error_susceptibility__level_low`
- `prior_expected_error_probability__level_low`
- `prior_operator_fatigue__level_low`
- `prior_procedure_branch_count__level_low`
- `prior_verification_required__absent`
- `prior_condition_noise_adjustment__level_low`
- `prior_condition_noise_level__level_low`

**Комментарий:** Тема 2 интерпретируется как фактор, связанный с признаками: prior_condition_time_pressure, prior_time_pressure_index, prior_repetition_expected_count, prior_operator_error_susceptibility, prior_expected_error_probability. Доминирующий токен: prior_condition_time_pressure__level_low. Наиболее характерные состояния: априорный признак prior_condition_time_pressure: уровень low; априорный признак prior_time_pressure_index: уровень low; априорный признак prior_repetition_expected_count: уровень low.
