# Финальная приемка главы 5

## 1. Статус приемки

- Статус: **пройдена**.
- Тип отчета: `chapter5_final_acceptance_report`.
- Этап: `12`.

## 2. Проверки

| Проверка | Статус |
|---|---:|
| artifact_existence | `True` |
| prior_features_row_count | `True` |
| prior_features_has_protocol_id | `True` |
| prior_features_no_scenario_duplicates | `True` |
| theta_prior_row_count | `True` |
| theta_key_alignment | `True` |
| all_output_row_counts | `True` |
| q_pred_range | `True` |
| partial_criteria_range | `True` |
| uncertainty_range | `True` |
| interval_radius_range | `True` |
| interval_order | `True` |
| no_missing_core_values | `True` |
| leakage_check_passed | `True` |
| no_forbidden_columns | `True` |
| full_pipeline_completed | `True` |
| final_report_stage | `True` |
| final_report_apriori_only | `True` |

## 3. Число строк

| Артефакт | Строк |
|---|---:|
| prior_features | `150` |
| theta_prior | `150` |
| normalized_prior_features | `150` |
| latent_quality_component | `150` |
| q_pred_components | `150` |
| q_pred | `150` |
| prediction_uncertainty | `150` |
| final_report | `150` |

## 4. Диапазоны качества

| Показатель | min | max |
|---|---:|---:|
| q_pred | `0.103445` | `0.889302` |
| uncertainty_score | `0.099417` | `0.538694` |
| interval_radius | `0.014913` | `0.080804` |

## 5. Методическая безопасность

- Априорный режим: `True`.
- Проверка утечки: `True`.
- Число запрещенных колонок: `0`.
- Полный контур выполнен: `True`.

## 6. Заключение

- Все артефакты главы 5 согласованы и готовы к фиксации.
- Расчетный контур использует только априорные признаки и LDA_prior.
- Финальные pytest-проверки и хеширование выполняются внешней процедурой этапа 12.
