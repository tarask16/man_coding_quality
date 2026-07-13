# Этап 7. Рисунок 2.3 — классификация ошибок ручного кодирования

## Назначение

Сформировать иерархическую схему шести основных типов ошибок:

```text
замена;
пропуск;
вставка;
перестановка;
ошибочный выбор правила;
ошибка контроля.
```

Схема дополнительно должна показывать группы факторов-модификаторов:

- свойства процедуры и сообщения;
- характеристики оператора;
- условия применения;
- организацию контроля.

## Создаваемые артефакты

```text
reports/dissertation_figures/chapter2/figure_2_3_error_taxonomy.png
reports/dissertation_figures/chapter2/figure_2_3_error_taxonomy.svg
```

## Применение архива

```powershell
Set-Location D:\Projects\man_coding_quality
Expand-Archive `
  -Path "$HOME\Downloads\dissertation_figures_stage07_figure_2_3.zip" `
  -DestinationPath . `
  -Force
.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = (Resolve-Path .\src).Path
```

## Генерация артефактов

```powershell
python -m manual_coding_sim.dissertation_figures.figure_2_3_error_taxonomy `
  --project-root . `
  --dpi 300
```

## Проверка артефактов

```powershell
Test-Path .\reports\dissertation_figures\chapter2\figure_2_3_error_taxonomy.png
Test-Path .\reports\dissertation_figures\chapter2\figure_2_3_error_taxonomy.svg
Start-Process .\reports\dissertation_figures\chapter2\figure_2_3_error_taxonomy.png
Start-Process .\reports\dissertation_figures\chapter2\figure_2_3_error_taxonomy.svg
```

## Тесты этапа

```powershell
python -m pytest `
  tests\dissertation_figures\test_figure_2_3_error_taxonomy.py `
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
