# Отчет программного блока главы 4

Статус: completed.
Выбранное число латентных факторов K: 3.

## Выполненные этапы

- corpus_prior
- k_selection
- lda_prior
- topic_metrics
- topic_stability
- topic_interpretation
- lda_diagnostic_models

## Метрики основной модели LDA_prior

- Perplexity: 94.81725487366135
- Mean coherence: -0.5299670207227024
- Topic diversity: 0.9666666666666667

## Устойчивость тем

- Mean stability: 0.8488497859956521
- Min stability: 0.8084181184315237

## Интерпретированные латентные факторы

- Латентный фактор качества 0: prior_total_nominal_time, prior_memory_load_index, prior_manual_operation_count
- Латентный фактор качества 1: prior_condition_mean_adjusted_attention, prior_attention_deficit, prior_control_intensity
- Латентный фактор качества 2: prior_condition_time_pressure, prior_time_pressure_index, prior_repetition_expected_count

## Методические ограничения

- LDA_prior обучается только по априорным признакам.
- Фактические признаки и целевые показатели качества запрещены для LDA_prior.
- LDA_diag и LDA_full имеют только диагностический статус.
