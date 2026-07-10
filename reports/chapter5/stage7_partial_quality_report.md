# Отчет этапа 7. Расчет частных прогнозных критериев

## Статус

Этап 7 реализован, но закрывается только после внешнего подтверждения проверок пользователем.

## Реализованные изменения

Добавлен расчет частных прогнозных критериев качества:

- `q_acc_pred`;
- `q_time_pred`;
- `q_effort_pred`;
- `q_res_pred`;
- `q_rep_pred`;
- `q_fit_pred`.

Расчет выполняется в модуле:

```text
src/manual_coding_sim/prediction/partial_quality_predictor.py
```

## Расчетная схема

Для каждого частного критерия используется формула:

```text
q_j_pred = lambda_j * B_j(X_prior) + (1 - lambda_j) * L_j(theta_prior)
```

где:

- `B_j(X_prior)` — взвешенная сумма нормированных априорных признаков;
- `L_j(theta_prior)` — латентная составляющая на основе `q_latent`;
- `lambda_j` — вес наблюдаемой априорной части из `configs/chapter5_feature_weights.yaml`.

## Выходные артефакты

Сформированы:

```text
reports/chapter5/q_pred_components.csv
reports/chapter5/q_pred_components_report.json
```

## CLI

Добавлен флаг:

```text
--calculate-partial-criteria
```

Пример запуска:

```powershell
python -m manual_coding_sim.prediction.chapter5_runner `
  --project-root . `
  --config configs\chapter5.yaml `
  --validate-inputs `
  --normalize-inputs `
  --calculate-latent-component `
  --calculate-partial-criteria
```

## Проверка, выполненная при подготовке этапа

```text
tests/prediction: 49 passed
full pytest: 297 passed
```

## Ключевые характеристики результата

```text
q_pred_components.csv: 150 строк, 35 колонок
частных прогнозных критериев: 6
все q_*_pred находятся в диапазоне [0, 1]
пропусков в q_*_pred нет
```

## Методическое ограничение

Интегральная оценка `Q_pred` на этом этапе не рассчитывается. Она должна быть реализована на этапе 8 на основе частных прогнозных критериев и весов `chapter5_quality_weights.yaml`.
