# Этап 26. Рисунок 5.6

## Назначение

Сформировать рисунок «Вклад частных критериев и латентной компоненты в `Q_pred`» по фактическим артефактам главы 5:

- `reports/chapter5/q_pred_components.csv`;
- `reports/chapter5/q_pred.csv`.

Рисунок использует усредненную аддитивную декомпозицию по 150 сценариям. Для каждого частного критерия его вклад в интегральный индекс разделяется на:

1. часть, формируемую нормированными априорными признаками;
2. часть, формируемую общей латентной компонентой `q_latent`.

## Формулы

Для критерия `j`:

```text
q_j,pred = alpha_j * B_j(X_prior,norm) + (1 - alpha_j) * q_latent
```

Интегральный прогноз:

```text
Q_pred = sum_j w_j * q_j,pred
```

Разложение критерия на вклад признаков и латентный вклад:

```text
C_j,feature = w_j * alpha_j * B_j
C_j,latent  = w_j * (1 - alpha_j) * q_latent
C_j,total   = C_j,feature + C_j,latent
```

## Содержание рисунка

Рисунок содержит три панели:

- средние вклады шести критериев с разделением на признаковую и латентную части;
- состав среднего `Q_pred` по критериям;
- суммарное разделение среднего `Q_pred` по источникам.

Для расширенного корпуса получено:

```text
N = 150
mean(Q_pred) = 0.4988435686
feature_total = 0.3520110730
latent_total = 0.1468324957
feature_share = 70.5654 %
latent_share = 29.4346 %
```

Средние полные вклады критериев:

```text
q_acc    = 0.077950
q_time   = 0.092142
q_effort = 0.084537
q_res    = 0.075388
q_rep    = 0.088046
q_fit    = 0.080781
```

## Методическое ограничение

Полученные доли являются аддитивными частями расчетного индекса. Они не должны интерпретироваться как причинные эффекты признаков или факторов и не подтверждают внешнюю точность модели.

## Созданные файлы

```text
src/manual_coding_sim/dissertation_figures/figure_5_6_q_pred_contributions.py
tests/dissertation_figures/test_figure_5_6_q_pred_contributions.py
reports/chapter5/figures/figure_5_6_q_pred_contributions.png
reports/chapter5/figures/figure_5_6_q_pred_contributions.svg
docs/dissertation_figures_roadmap.md
README_FIGURE_STAGE_26.md
```

## Локальная генерация

```powershell
python -m manual_coding_sim.dissertation_figures.figure_5_6_q_pred_contributions `
  --project-root . `
  --dpi 300
```

## Локальные тесты

```powershell
python -m pytest `
  tests\dissertation_figures\test_figure_5_6_q_pred_contributions.py `
  -q
```

В рабочем окружении подготовки этапа тесты не запускались. Тестовый модуль содержит 10 проверок структуры данных, математической согласованности, SVG/PNG и CLI.
