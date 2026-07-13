# Этап 21. Рисунок 5.1 — конвейер построения априорной интегральной оценки

## Назначение

Этап формирует структурную схему полного вычислительного контура главы 5:

`X_prior + theta_prior → LeakageGuard → нормировка → q_latent → частные критерии → Q_pred → uncertainty → финальная приемка`.

Рисунок фиксирует методически существенные ограничения:

- фактические признаки и целевые показатели качества не используются при расчете `Q_pred`;
- `fact_features.csv`, `quality_targets.csv`, `theta_diag.csv` и `theta_full.csv` показаны как запрещенные входы;
- интервальная оценка трактуется как диагностическая неопределенность, а не как доказанная абсолютная калибровка;
- финальная приемка включает 18 проверок и согласование 150 строк во всех основных артефактах.

## Созданные и измененные файлы

```text
src/manual_coding_sim/dissertation_figures/figure_5_1_prediction_pipeline.py
tests/dissertation_figures/test_figure_5_1_prediction_pipeline.py
reports/chapter5/figures/figure_5_1_prediction_pipeline.png
reports/chapter5/figures/figure_5_1_prediction_pipeline.svg
docs/dissertation_figures_roadmap.md
README_FIGURE_STAGE_21.md
```

## Генерация

```powershell
python -m manual_coding_sim.dissertation_figures.figure_5_1_prediction_pipeline `
  --project-root . `
  --dpi 300
```

## Локальные тесты

Тесты подготовлены, но в рабочем окружении этапа не запускались.

```powershell
python -m pytest `
  tests\dissertation_figures\test_figure_5_1_prediction_pipeline.py `
  -q
```

Ожидается 10 успешно пройденных тестов.

## Критерии визуальной проверки

- присутствуют восемь пронумерованных этапов;
- верхний и нижний ряды соединены в единый последовательный путь;
- запрещенные входы отделены от расчетного контура;
- подписи, формулы и имена артефактов не обрезаны;
- PNG и SVG содержательно совпадают;
- текст SVG остается редактируемым.
