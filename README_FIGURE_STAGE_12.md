# Этап 12. Рисунок 3.4 — распределения фактических показателей качества

## Назначение

Этап формирует единый горизонтальный violin- и boxplot-график для шести частных и одного интегрального показателя фактического качества:

- `q_acc`;
- `q_time`;
- `q_effort`;
- `q_res`;
- `q_rep`;
- `q_fit`;
- `integral_quality`.

Все показатели отображаются в общей шкале `[0; 1]`. Ширина violin-графика характеризует локальную плотность, белый boxplot — межквартильный диапазон, красная линия — медиану, ромб — среднее значение.

## Входные данные

По умолчанию:

```text
data/processed/quality_targets.csv
```

Путь можно переопределить аргументом `--input`.

## Выходные артефакты

```text
reports/chapter3/figures/figure_3_4_quality_target_distributions.png
reports/chapter3/figures/figure_3_4_quality_target_distributions.svg
```

PNG формируется с разрешением 300 dpi. SVG сохраняет текст редактируемым.

## Команда генерации

```powershell
python -m manual_coding_sim.dissertation_figures.figure_3_4_quality_target_distributions `
  --project-root . `
  --dpi 300
```

## Тесты этапа

```powershell
python -m pytest `
  tests\dissertation_figures\test_figure_3_4_quality_target_distributions.py `
  -q
```

## Критерии завершения

- CSV содержит семь обязательных показателей;
- значения числовые, конечные и находятся в диапазоне `[0; 1]`;
- сформированы PNG и SVG;
- все распределения отображаются в единой шкале;
- подписи и средние значения читаемы;
- тесты этапа и полная регрессия прошли успешно;
- пользователь подтвердил визуальную корректность рисунка.
