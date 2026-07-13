# Этап 1. Рисунок 1.1 — факторы качества сценария применения

## Назначение

Этап реализует отдельный генератор структурной схемы, показывающей связь пяти компонентов сценария
`A = {S, O, U, G, K}` с шестью частными показателями и интегральным качеством сценария.

## Создаваемые файлы

```text
src/manual_coding_sim/dissertation_figures/__init__.py
src/manual_coding_sim/dissertation_figures/common.py
src/manual_coding_sim/dissertation_figures/figure_1_1_quality_factors.py
tests/dissertation_figures/test_figure_1_1_quality_factors.py
docs/dissertation_figures_roadmap.md
README_FIGURE_STAGE_01.md
```

## Формируемые артефакты

```text
reports/dissertation_figures/chapter1/figure_1_1_quality_factors.png
reports/dissertation_figures/chapter1/figure_1_1_quality_factors.svg
```

PNG формируется с разрешением 300 dpi. SVG сохраняет текстовые элементы и пригоден для векторного редактирования.

## Команда запуска

```powershell
python -m manual_coding_sim.dissertation_figures.figure_1_1_quality_factors `
  --project-root . `
  --dpi 300
```

## Критерии завершения

1. CLI завершается с кодом 0.
2. Созданы PNG и SVG с одинаковым именем основы.
3. PNG имеет ширину не менее 3000 пикселей и высоту не менее 1600 пикселей.
4. SVG содержит текст и векторные элементы.
5. Тесты этапа и полная регрессия проходят успешно.
6. Пользователь визуально подтверждает корректность рисунка.
