# Этап 23 — рисунок 5.3

## Назначение

Этап формирует рисунок **«Распределения частных прогнозных критериев»** по артефакту главы 5:

```text
reports/chapter5/q_pred_components.csv
```

На общей шкале `[0; 1]` сопоставляются:

- `q_acc_pred` — точность восстановления;
- `q_time_pred` — временная эффективность;
- `q_effort_pred` — трудоёмкость выполнения;
- `q_res_pred` — результативность контроля;
- `q_rep_pred` — повторяемость результата;
- `q_fit_pred` — соответствие условиям.

Для каждого критерия совмещены violin-график, boxplot, медиана и среднее. Правая панель содержит сводные значения и веса признаковой и латентной компонент.

## Методическая интерпретация

Частные прогнозы рассчитываются как

```text
q_j,pred = α_j · B_j(X_prior,norm) + (1 − α_j) · q_latent.
```

Показанные величины являются априорными оценками, а не фактическими `q_j,fact`. Поскольку одна и та же латентная компонента входит во все шесть критериев, распределения критериев не рассматриваются как независимые.

## Фактические характеристики расширенного корпуса

| Критерий | Среднее | Медиана | Минимум | Максимум | α / (1−α) |
|---|---:|---:|---:|---:|---:|
| `q_acc_pred` | 0,4677 | 0,4678 | 0,0039 | 0,9291 | 0,65 / 0,35 |
| `q_time_pred` | 0,5529 | 0,5844 | 0,0631 | 0,9847 | 0,70 / 0,30 |
| `q_effort_pred` | 0,5072 | 0,5231 | 0,0348 | 0,9219 | 0,70 / 0,30 |
| `q_res_pred` | 0,4523 | 0,4269 | 0,0684 | 0,9607 | 0,60 / 0,40 |
| `q_rep_pred` | 0,5283 | 0,5192 | 0,0993 | 0,9912 | 0,60 / 0,40 |
| `q_fit_pred` | 0,4847 | 0,4329 | 0,0823 | 0,9109 | 0,65 / 0,35 |

## Создаваемые файлы

```text
src/manual_coding_sim/dissertation_figures/figure_5_3_partial_prediction_components.py
tests/dissertation_figures/test_figure_5_3_partial_prediction_components.py
reports/chapter5/figures/figure_5_3_partial_prediction_components.png
reports/chapter5/figures/figure_5_3_partial_prediction_components.svg
docs/dissertation_figures_roadmap.md
README_FIGURE_STAGE_23.md
```

## Генерация

```powershell
python -m manual_coding_sim.dissertation_figures.figure_5_3_partial_prediction_components `
  --project-root . `
  --dpi 300
```

## Локальная проверка

```powershell
python -m pytest `
  tests\dissertation_figures\test_figure_5_3_partial_prediction_components.py `
  -q
```

Подготовлено 10 тестов. В рабочем окружении этапа они не запускались.
