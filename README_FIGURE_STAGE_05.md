# Этап 5. Рисунок 2.1 — структура сценария применения

## Назначение

Сформировать структурную схему сценария

```text
A_i = {S_i, O_i, U_i, G_i, K_i}
```

с отображением:

- ручного средства кодирования `S_i`;
- оператора `O_i`;
- условий применения `U_i`;
- класса сообщений `G_i`;
- контрольных процедур `K_i`;
- ключевых параметров каждого компонента;
- подмножеств признаков `X_S`, `X_O`, `X_U`, `X_G`, `X_K`;
- их объединения в априорное описание `X_prior(A_i)`.

Рисунок отдельно фиксирует, что фактические ошибки, фактическое время и результаты выполнения относятся к `X_fact` и `Y_fact` и не входят в априорное описание.

## Создаваемые артефакты

```text
reports/dissertation_figures/chapter2/figure_2_1_scenario_structure.png
reports/dissertation_figures/chapter2/figure_2_1_scenario_structure.svg
```

## Применение архива

```powershell
Set-Location D:\Projects\man_coding_quality
Expand-Archive `
  -Path "$HOME\Downloads\dissertation_figures_stage05_figure_2_1.zip" `
  -DestinationPath . `
  -Force
.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = (Resolve-Path .\src).Path
```

## Генерация артефактов

```powershell
python -m manual_coding_sim.dissertation_figures.figure_2_1_scenario_structure `
  --project-root . `
  --dpi 300
```

## Проверка артефактов

```powershell
Test-Path .\reports\dissertation_figures\chapter2\figure_2_1_scenario_structure.png
Test-Path .\reports\dissertation_figures\chapter2\figure_2_1_scenario_structure.svg
Start-Process .\reports\dissertation_figures\chapter2\figure_2_1_scenario_structure.png
Start-Process .\reports\dissertation_figures\chapter2\figure_2_1_scenario_structure.svg
```

## Тесты этапа

```powershell
python -m pytest `
  tests\dissertation_figures\test_figure_2_1_scenario_structure.py `
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
