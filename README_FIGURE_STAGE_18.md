# Этап 18. Рисунок 4.4 — устойчивость латентных факторов

## Назначение

Генератор формирует составной рисунок устойчивости `LDA_prior` к начальному состоянию модели по отчёту `reports/chapter4/topic_stability_report.csv`.

На рисунке представлены:

- тепловая карта попарного среднего сходства запусков с `random_state = 11, 42, 77, 101`;
- минимальное сходство внутри каждой пары;
- интервалы `min_similarity → mean_similarity` для трёх латентных факторов;
- чувствительные пары seed с `min_similarity < 0,70`;
- сводные показатели устойчивости.

## Созданные файлы

```text
src/manual_coding_sim/dissertation_figures/figure_4_4_topic_stability.py
tests/dissertation_figures/test_figure_4_4_topic_stability.py
reports/chapter4/figures/figure_4_4_topic_stability.png
reports/chapter4/figures/figure_4_4_topic_stability.svg
docs/dissertation_figures_roadmap.md
README_FIGURE_STAGE_18.md
```

## Входной файл

```text
reports/chapter4/topic_stability_report.csv
```

Ожидаемые типы строк: `summary`, `topic`, `pairwise_run`.

## Генерация

```powershell
python -m manual_coding_sim.dissertation_figures.figure_4_4_topic_stability `
  --project-root . `
  --dpi 300
```

При необходимости путь к отчёту задаётся явно:

```powershell
python -m manual_coding_sim.dissertation_figures.figure_4_4_topic_stability `
  --project-root . `
  --input .\reports\chapter4\topic_stability_report.csv `
  --dpi 300
```

## Локальные тесты

```powershell
python -m pytest `
  tests\dissertation_figures\test_figure_4_4_topic_stability.py `
  -q
```

Подготовлено 10 тестов. В рабочем окружении этапа тесты не запускались.

## Визуальная проверка

Необходимо подтвердить:

- симметричность тепловой карты;
- единичную диагональ;
- красное выделение чувствительных пар;
- отсутствие наложений подписей тем на шкалу цвета;
- читаемость интервалов по темам;
- совпадение содержания PNG и SVG;
- редактируемость текста в SVG.
