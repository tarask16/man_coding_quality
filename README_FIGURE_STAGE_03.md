# Этап 3. Рисунок 1.3 — сравнение подходов к оценке качества

## Назначение

Сформировать качественную матрицу сопоставления пяти подходов к априорной оценке качества:

- экспертного;
- балльного;
- многокритериального;
- имитационного;
- латентно-вероятностного.

Сопоставление выполняется по шести методическим критериям. Рисунок прямо маркируется как аналитическое качественное сравнение, а не как результат экспериментального измерения точности методов.

## Создаваемые артефакты

```text
reports/dissertation_figures/chapter1/figure_1_3_methods_comparison.png
reports/dissertation_figures/chapter1/figure_1_3_methods_comparison.svg
```

## Применение архива

```powershell
Set-Location D:\Projects\man_coding_quality
Expand-Archive `
  -Path "$HOME\Downloads\dissertation_figures_stage03_figure_1_3.zip" `
  -DestinationPath . `
  -Force
.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = (Resolve-Path .\src).Path
```

## Генерация артефактов

```powershell
python -m manual_coding_sim.dissertation_figures.figure_1_3_methods_comparison `
  --project-root . `
  --dpi 300
```

## Проверка артефактов

```powershell
Test-Path .\reports\dissertation_figures\chapter1\figure_1_3_methods_comparison.png
Test-Path .\reports\dissertation_figures\chapter1\figure_1_3_methods_comparison.svg
Start-Process .\reports\dissertation_figures\chapter1\figure_1_3_methods_comparison.png
Start-Process .\reports\dissertation_figures\chapter1\figure_1_3_methods_comparison.svg
```

## Тесты этапа

```powershell
python -m pytest `
  tests\dissertation_figures\test_figure_1_3_methods_comparison.py `
  -q
```

## Совокупная проверка рисунков

```powershell
python -m pytest tests\dissertation_figures -q
```

## Полная регрессия

```powershell
python -m pytest -q
```

## Критерий завершения

Этап завершается после генерации PNG и SVG, прохождения тестов этапа и полной регрессии, визуальной проверки обоих форматов и явного подтверждения пользователя.
