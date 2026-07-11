# Этап 9. Расчет неопределенности прогноза Q_pred

## Назначение этапа

Этап 9 добавляет интервальную априорную оценку интегрального прогнозного показателя качества `Q_pred`.
Расчет не использует фактические признаки, целевые значения качества и результаты последующей проверки.

## Реализованные артефакты

- `src/manual_coding_sim/prediction/prediction_uncertainty.py`
- `reports/chapter5/prediction_uncertainty.csv`
- `reports/chapter5/prediction_uncertainty_report.json`

## Формула расчета

Нормированная энтропия латентного профиля:

```text
H_theta = -sum(theta_k * ln(theta_k)) / ln(K)
```

Итоговая неопределенность:

```text
U = w_entropy * H_theta + w_stability * (1 - mean_stability) + w_input * input_missing_share
```

Интервал прогноза:

```text
interval_radius = delta * U
q_pred_lower = max(0, q_pred - interval_radius)
q_pred_upper = min(1, q_pred + interval_radius)
```

## Контрольный запуск

```text
Строк интервального прогноза: 150
Минимальное uncertainty_score: 0.099417
Максимальное uncertainty_score: 0.538694
Минимальный радиус интервала: 0.014913
Максимальный радиус интервала: 0.080804
```

## Проверка тестами

```text
tests/prediction: 61 passed
full pytest: 309 passed
```

## Методические ограничения

- `fact_features.csv` не используется.
- `quality_targets.csv` не используется.
- Интервальная оценка является априорной и опирается только на `theta_prior`, `Q_pred`, параметры устойчивости LDA и техническое качество нормированных входов.
