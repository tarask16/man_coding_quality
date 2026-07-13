# Этап 14 — рисунок 3.6

## Назначение

Этап формирует блок-схему контура воспроизводимости вычислительного эксперимента:

`конфигурация и seed → управляемый запуск → CSV/JSON-артефакты → контрольные суммы → тесты → отчёт воспроизводимости`.

Дополнительно показан независимый повторный запуск с теми же входными параметрами и сравнением повторных артефактов по SHA-256.

## Создаваемые файлы

- `src/manual_coding_sim/dissertation_figures/figure_3_6_reproducibility_contour.py`;
- `tests/dissertation_figures/test_figure_3_6_reproducibility_contour.py`;
- `reports/chapter3/figures/figure_3_6_reproducibility_contour.png`;
- `reports/chapter3/figures/figure_3_6_reproducibility_contour.svg`.

## Генерация

```powershell
$env:PYTHONPATH = (Resolve-Path .\src).Path
python -m manual_coding_sim.dissertation_figures.figure_3_6_reproducibility_contour `
  --project-root . `
  --dpi 300
```

## Тесты этапа

```powershell
python -m pytest `
  tests\dissertation_figures\test_figure_3_6_reproducibility_contour.py `
  -q
```

## Методическое ограничение

Совпадение конфигурации, схемы, контрольных сумм и результатов тестов подтверждает техническую воспроизводимость вычислительного контура. Это не является самостоятельным доказательством внешней валидности модели и содержательной достоверности научных выводов.
