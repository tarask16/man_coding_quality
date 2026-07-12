# Этап 9. Сравнение с базовыми моделями без LDA

## Назначение

Этап сравнивает неизменный интегральный прогноз главы 5 с тремя базовыми
априорными схемами на одном и том же проверочном корпусе из 150 сценариев.

Сравниваются:

1. `mean_baseline` — среднее фактическое качество обучающих fold;
2. `prior_only_baseline` — только априорные feature-компоненты без LDA;
3. `theta_only_baseline` — только латентная компонента `q_latent`;
4. `chapter5_model` — исходный `q_pred` главы 5 без изменений.

## Методическая защита от утечки

Mean baseline формируется по детерминированной 5-fold out-of-fold схеме:

- сценарии перемешиваются с `random_seed = 42`;
- каждый fold содержит 30 проверочных сценариев;
- среднее рассчитывается по оставшимся 120 обучающим сценариям;
- фактические цели текущего fold не участвуют в его прогнозе.

`prior_only_baseline` рассчитывается без `q_fact`:

```text
sum(w_j * q_j_feature_component)
```

Используются фиксированные веса интегрального качества из
`configs/chapter5_quality_weights.yaml`.

`theta_only_baseline` использует неизменный `q_latent`, а `chapter5_model` —
неизменный `q_pred`. Перекалибровка, подбор порогов и изменение модели главы 5
на этапе 9 не выполняются.

## Создаваемые и изменяемые файлы

```text
src/manual_coding_sim/validation/baseline_models.py
src/manual_coding_sim/validation/chapter6_runner.py
tests/validation/test_baseline_models.py
README_STAGE9.md
STAGE9_MANIFEST.json
```

## Выходные артефакты

```text
reports/chapter6/baseline_predictions.csv
reports/chapter6/baseline_comparison.csv
reports/chapter6/baseline_comparison_report.json
reports/chapter6/baseline_comparison_report.md
```

## Рассчитываемые метрики

Для каждой модели рассчитываются:

- MAE;
- RMSE;
- Bias;
- Median Absolute Error;
- Max Absolute Error;
- Pearson;
- Spearman;
- Kendall tau-b;
- R²;
- Accuracy;
- Balanced Accuracy;
- Macro F1;
- Weighted F1;
- распределение прогнозных классов.

## Команда запуска

```powershell
python -m manual_coding_sim.validation.chapter6_runner `
  --project-root . `
  --config configs\chapter6.yaml `
  --compare-baselines
```

## Тесты этапа

```powershell
pytest tests\validation\test_baseline_models.py -q
pytest tests\validation -q
pytest -q
```

После полной регрессии необходимо повторить CLI этапа 9.

## Контрольный результат на расширенном корпусе

| Модель | MAE | RMSE | Bias | Spearman | Kendall | Accuracy | Balanced Accuracy | Macro F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Mean baseline | 0.0972268573 | 0.1129029089 | 0.0000000000 | -0.0717444272 | -0.0512528592 | 0.6666666667 | 0.3333333333 | 0.2666666667 |
| Prior-only baseline | 0.1118221247 | 0.1321130939 | -0.1077100877 | 0.8590515134 | 0.6665771812 | 0.5133333333 | 0.6456462585 | 0.3931547301 |
| Theta-only baseline | 0.3041245325 | 0.3605441916 | -0.2287207972 | 0.8666891862 | 0.6774944072 | 0.3400000000 | 0.6248979592 | 0.3430737206 |
| Chapter 5 model | 0.1592441669 | 0.1944027110 | -0.1493986447 | 0.8817689675 | 0.6993288591 | 0.4466666667 | 0.6678231293 | 0.4225151692 |

## Предварительная интерпретация

Mean baseline показывает минимальные MAE и RMSE, поскольку фактические
значения сосредоточены преимущественно в среднем классе. При этом он практически
не сохраняет ранжирование сценариев и прогнозирует только класс `medium`.
Поэтому его высокая Accuracy обусловлена дисбалансом классов и не означает
полноценного превосходства.

Prior-only baseline лучше полной модели по абсолютной ошибке, но уступает ей по
Spearman, Kendall, Balanced Accuracy и Macro F1. Это означает, что добавление
латентной компоненты в модели главы 5 улучшает ранжирование и классовое
разделение, но одновременно усиливает систематическое занижение абсолютной
шкалы.

Theta-only baseline сохраняет высокий порядок сценариев, но имеет наибольшую
ошибку и сильное отрицательное смещение.

Полная модель главы 5 показывает лучшие значения Spearman, Kendall, Balanced
Accuracy и Macro F1, но не лучшие MAE и RMSE. Статистическую значимость этих
различий необходимо проверить bootstrap-анализом на этапе 10.

## Критерий закрытия

Этап считается закрытым после успешного выполнения CLI, 15 тестов этапа,
всех тестов `tests/validation`, полной регрессии и повторного запуска CLI.
Переход к этапу 10 выполняется только после отдельного подтверждения.
