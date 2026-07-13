# Этап 22 — рисунок 5.2

## Назначение

Этап формирует рисунок «Распределение латентной компоненты качества» по априорным латентным профилям `theta_prior.csv`.

Расчёт выполняется по формуле:

\[
q_{\mathrm{latent}}=\frac{-\theta_0-\theta_1+\theta_2+1}{2}.
\]

Поскольку для каждой строки выполняется

\[
\theta_0+\theta_1+\theta_2=1,
\]

в принятой конфигурации направлений `d = (-1, -1, +1)` латентная компонента точно совпадает с `theta_2`:

\[
q_{\mathrm{latent}}=\theta_2.
\]

Это равенство обусловлено выбранной формулой и нормировкой профиля, а не внешней калибровкой качества.

## Входные данные

По умолчанию используется:

```text
reports/chapter4/theta_prior.csv
```

Обязательные колонки:

```text
scenario_id
protocol_id
theta_0
theta_1
theta_2
selected_k
```

## Фактические характеристики расширенного корпуса

```text
N = 150
minimum = 0.010720
Q1 = 0.012092
median = 0.400247
mean = 0.419521
Q3 = 0.747305
maximum = 0.978029
standard deviation = 0.371588
```

Средние направленные члены:

```text
- theta_0 = -0.265381
- theta_1 = -0.315098
+ theta_2 = +0.419521
```

## Выходные файлы

```text
reports/chapter5/figures/figure_5_2_latent_quality_component.png
reports/chapter5/figures/figure_5_2_latent_quality_component.svg
```

PNG формируется с разрешением 300 dpi. SVG сохраняет редактируемые текстовые элементы.

## Генерация

```powershell
python -m manual_coding_sim.dissertation_figures.figure_5_2_latent_quality_component `
  --project-root . `
  --dpi 300
```

Для явно заданного входного файла:

```powershell
python -m manual_coding_sim.dissertation_figures.figure_5_2_latent_quality_component `
  --project-root . `
  --input .\reports\chapter4\theta_prior.csv `
  --dpi 300
```

## Локальные тесты

```powershell
python -m pytest `
  tests\dissertation_figures\test_figure_5_2_latent_quality_component.py `
  -q
```

Подготовлено 10 тестов. В рабочем окружении подготовки этапа они не запускались согласно принятому правилу проекта.

## Критерии визуальной проверки

- отображены гистограмма и горизонтальный boxplot;
- отмечены минимум, медиана, среднее и максимум;
- присутствуют квартильные значения;
- отдельно показаны направления `theta_0`, `theta_1`, `theta_2`;
- явно указано тождество `q_latent = theta_2`;
- отсутствуют обрезка и наложение подписей;
- PNG и SVG содержательно совпадают.
