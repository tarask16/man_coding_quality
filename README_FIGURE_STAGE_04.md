# Этап 4. Рисунок 1.4 — общая логика разработанного метода

## Назначение

Сформировать блок-схему последовательности научно-методического аппарата:

1. формальная модель сценария;
2. компьютерное моделирование процесса;
3. формирование априорных признаков `X_prior`;
4. построение латентного профиля `LDA_prior`;
5. расчет сравнительного индекса `Q_pred`;
6. внешняя проверка по `Q_fact`.

На рисунке отдельно показано, что `quality_targets.csv` и `fact_features.csv` поступают только во внешний проверочный контур и не используются при построении `Q_pred`.

## Создаваемые артефакты

```text
reports/dissertation_figures/chapter1/figure_1_4_method_logic.png
reports/dissertation_figures/chapter1/figure_1_4_method_logic.svg
```

## Применение архива

```powershell
Set-Location D:\Projects\man_coding_quality
Expand-Archive `
  -Path "$HOME\Downloads\dissertation_figures_stage04_figure_1_4.zip" `
  -DestinationPath . `
  -Force
.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = (Resolve-Path .\src).Path
```

## Генерация артефактов

```powershell
python -m manual_coding_sim.dissertation_figures.figure_1_4_method_logic `
  --project-root . `
  --dpi 300
```

## Проверка артефактов

```powershell
Test-Path .\reports\dissertation_figures\chapter1\figure_1_4_method_logic.png
Test-Path .\reports\dissertation_figures\chapter1\figure_1_4_method_logic.svg
Start-Process .\reports\dissertation_figures\chapter1\figure_1_4_method_logic.png
Start-Process .\reports\dissertation_figures\chapter1\figure_1_4_method_logic.svg
```

## Тесты этапа

```powershell
python -m pytest `
  tests\dissertation_figures\test_figure_1_4_method_logic.py `
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
