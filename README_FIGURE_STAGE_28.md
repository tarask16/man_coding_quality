# Этап 28 — рисунок 6.2

## Назначение

Этап формирует рисунок «Остатки интегрального прогноза относительно фактического качества» по данным:

- `reports/chapter5/q_pred.csv`;
- `data/processed/quality_targets.csv`.

Остаток определяется как:

\[
e_i = Q_{pred,i} - Q_{fact,i}.
\]

Отрицательный остаток означает занижение фактического качества априорным индексом, положительный — завышение.

## Состав рисунка

- диаграмма рассеяния остатков относительно `Q_fact`;
- нулевая линия `e = 0`;
- средний остаток `Bias`;
- гауссово-сглаженная тенденция с шириной `h = 0.12`;
- отметки минимального и максимального остатка;
- сводные карточки центра, диапазона, направлений ошибок и связи остатка с `Q_fact`.

## Фактические показатели корпуса

- число сценариев: 150;
- `Bias = -0.149399`;
- медиана остатка: `-0.145240`;
- стандартное отклонение: `0.124805`;
- минимум: `-0.376663`, сценарий `scn_0027`;
- максимум: `+0.096407`, сценарий `scn_0066`;
- `Q_pred < Q_fact`: 132 сценария;
- `Q_pred > Q_fact`: 18 сценариев;
- `Pearson(Q_fact, e) = +0.628566`.

## Методическое ограничение

Сглаженная кривая используется для диагностики структуры систематического смещения. Она не является абсолютной калибровкой модели и не интерпретируется как причинная зависимость.

## Созданные файлы

```text
src/manual_coding_sim/dissertation_figures/
    figure_6_2_residuals_vs_q_fact.py

tests/dissertation_figures/
    test_figure_6_2_residuals_vs_q_fact.py

reports/chapter6/figures/
    residuals_vs_q_fact.png
    residuals_vs_q_fact.svg

docs/
    dissertation_figures_roadmap.md
```

## Генерация

```powershell
python -m manual_coding_sim.dissertation_figures.figure_6_2_residuals_vs_q_fact `
  --project-root . `
  --dpi 300
```

Явное указание входных файлов и ширины сглаживания:

```powershell
python -m manual_coding_sim.dissertation_figures.figure_6_2_residuals_vs_q_fact `
  --project-root . `
  --q-pred .\reports\chapter5\q_pred.csv `
  --q-fact .\data\processed\quality_targets.csv `
  --bandwidth 0.12 `
  --dpi 300
```

## Локальная проверка

```powershell
python -m pytest `
  tests\dissertation_figures\test_figure_6_2_residuals_vs_q_fact.py `
  -q
```

Ожидается 10 успешно пройденных тестов. Тесты на этапе подготовки не запускались в соответствии с правилами проекта.
