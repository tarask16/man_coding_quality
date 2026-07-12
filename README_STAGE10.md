# Этап 10. Bootstrap-анализ статистической устойчивости

## Назначение

Этап оценивает статистическую устойчивость метрик модели главы 5 и различий
между этой моделью и baseline-моделями этапа 9.

Используется парный cluster-bootstrap по сценариям:

- единица выборки: `scenario_id`;
- число повторов: `1000`;
- уровень доверия: `0.95`;
- генератор случайных чисел: `random_seed = 42`;
- в каждом повторе выбирается 150 сценариев с возвращением;
- для всех моделей применяется одна и та же повторная выборка.

## Методические ограничения

Bootstrap выполняется по уже зафиксированным прогнозам этапа 9:

```text
mean_baseline
prior_only_baseline
theta_only_baseline
chapter5_model
```

Внутри bootstrap-повторов:

- модели не переобучаются;
- out-of-fold прогнозы mean baseline не пересчитываются;
- `q_pred` главы 5 не изменяется;
- пороги `low`, `medium`, `high` не изменяются;
- фактическое качество используется только для внешней проверки.

## Метрики

Для каждой модели рассчитываются percentile-доверительные интервалы:

```text
MAE
RMSE
Spearman
Kendall tau-b
Accuracy
Macro F1
```

Для каждого baseline дополнительно рассчитывается парная разность:

```text
metric_chapter5_model - metric_baseline
```

Интерпретация направления:

- для MAE и RMSE отрицательная разность означает преимущество модели главы 5;
- для Spearman, Kendall, Accuracy и Macro F1 положительная разность означает
  преимущество модели главы 5;
- различие считается статистически устойчивым, если доверительный интервал
  разности не включает ноль.

## Создаваемые и изменяемые файлы

```text
src/manual_coding_sim/validation/bootstrap_analysis.py
src/manual_coding_sim/validation/chapter6_runner.py
tests/validation/test_bootstrap_analysis.py
README_STAGE10.md
STAGE10_MANIFEST.json
```

## Выходные артефакты

```text
reports/chapter6/bootstrap_confidence_intervals.csv
reports/chapter6/bootstrap_model_differences.csv
reports/chapter6/bootstrap_report.json
reports/chapter6/bootstrap_report.md
```

## Команда запуска

```powershell
python -m manual_coding_sim.validation.chapter6_runner `
  --project-root . `
  --config configs\chapter6.yaml `
  --bootstrap-analysis
```

## Контрольные результаты модели главы 5

| Метрика | Значение | 95% CI lower | 95% CI upper |
|---|---:|---:|---:|
| MAE | 0.1592441669 | 0.1405341874 | 0.1779741437 |
| RMSE | 0.1944027110 | 0.1767722536 | 0.2106599040 |
| Spearman | 0.8817689675 | 0.8273055976 | 0.9136315536 |
| Kendall tau-b | 0.6993288591 | 0.6409357447 | 0.7475942570 |
| Accuracy | 0.4466666667 | 0.3666666667 | 0.5266666667 |
| Macro F1 | 0.4225151692 | 0.3667620360 | 0.4772037969 |

## Статистически устойчивые различия

### Сравнение с mean baseline

- mean baseline устойчиво лучше по MAE, RMSE и Accuracy;
- модель главы 5 устойчиво лучше по Spearman, Kendall и Macro F1.

### Сравнение с prior-only baseline

- prior-only baseline устойчиво лучше по MAE и RMSE;
- различия по Spearman, Kendall, Accuracy и Macro F1 неустойчивы на уровне 95%.

### Сравнение с theta-only baseline

- модель главы 5 устойчиво лучше по MAE, RMSE, Accuracy и Macro F1;
- различия по Spearman и Kendall неустойчивы на уровне 95%.

Сводка 18 парных сравнений:

```text
устойчивых преимуществ модели главы 5: 7
устойчивых преимуществ baseline:       5
различий без устойчивого вывода:       6
```

## Тесты этапа

```powershell
pytest tests\validation\test_bootstrap_analysis.py -q
pytest tests\validation -q
pytest -q
```

После полной регрессии необходимо повторно выполнить CLI этапа 10. Для
дополнительной проверки зависимости этапов 9 и 10 рекомендуется совместный
запуск:

```powershell
python -m manual_coding_sim.validation.chapter6_runner `
  --project-root . `
  --config configs\chapter6.yaml `
  --compare-baselines `
  --bootstrap-analysis
```

## Контрольная проверка

```text
тесты этапа 10:         15 passed
все tests/validation:   121 passed
полная регрессия:       438 passed
CLI на 150 сценариях:   успешно
совместный этап 9 + 10: успешно
```

## Критерий закрытия

Этап считается закрытым после успешного выполнения CLI, 15 тестов этапа,
всех тестов `tests/validation`, полной регрессии и повторного совместного
запуска этапов 9 и 10. Переход к этапу 11 выполняется только после отдельного
подтверждения.
