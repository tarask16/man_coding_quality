# Этап 11. Рисунок 3.3 — распределения априорных признаков

## Назначение

Этап формирует шестипанельный рисунок с распределениями ключевых априорных признаков расширенного вычислительного корпуса:

- `prior_mean_complexity`;
- `prior_mean_message_criticality`;
- `prior_operator_total_estimated_time`;
- `prior_condition_time_pressure`;
- `prior_operator_attention`;
- `prior_expected_error_probability`.

Для дискретных признаков строятся столбчатые распределения, для непрерывных — гистограммы с отметкой медианы. На каждой панели выводятся минимальное, максимальное и среднее значения, а также стандартное отклонение.

## Входные данные

По умолчанию:

```text
data/processed/prior_features.csv
```

Путь можно переопределить аргументом `--input`.

## Выходные артефакты

```text
reports/chapter3/figures/figure_3_3_prior_feature_distributions.png
reports/chapter3/figures/figure_3_3_prior_feature_distributions.svg
```

PNG формируется с разрешением 300 dpi. SVG сохраняет текст редактируемым.

## Команда генерации

```powershell
python -m manual_coding_sim.dissertation_figures.figure_3_3_prior_feature_distributions `
  --project-root . `
  --dpi 300
```

## Тесты этапа

```powershell
python -m pytest `
  tests\dissertation_figures\test_figure_3_3_prior_feature_distributions.py `
  -q
```

## Критерии завершения

- CSV успешно загружен и содержит шесть обязательных колонок;
- значения числовые, конечные и находятся в допустимых диапазонах;
- сформированы PNG и SVG;
- подписи шести панелей читаемы;
- статистические блоки и легенда не перекрывают данные;
- тесты этапа и полная регрессия прошли успешно;
- пользователь подтвердил визуальную корректность рисунка.
