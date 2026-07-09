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

- Perplexity: 58.435311779755935
- Mean coherence: 0.23426872912916163
- Topic diversity: 0.8666666666666667

## Устойчивость тем

- Mean stability: 0.8928944612554363
- Min stability: 0.8473245771174586

## Интерпретированные латентные факторы

- Латентный фактор качества 0: prior_control_marker_ratio, prior_digit_ratio, prior_mean_message_criticality
- Латентный фактор качества 1: prior_condition_total_adjusted_time, prior_operator_total_estimated_time, prior_total_nominal_time
- Латентный фактор качества 2: prior_condition_mean_adjusted_attention, prior_condition_total_adjusted_time, prior_mean_complexity

## Методические ограничения

- LDA_prior обучается только по априорным признакам.
- Фактические признаки и целевые показатели качества запрещены для LDA_prior.
- LDA_diag и LDA_full имеют только диагностический статус.
