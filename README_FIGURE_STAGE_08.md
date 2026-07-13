# Этап 8. Рисунок 2.4 — компромисс контрольных процедур

## Назначение

Этап формирует причинно-функциональную схему влияния контрольных процедур на частные критерии и интегральное качество ручного кодирования.

Рисунок показывает три разнонаправленные ветви:

1. повышение вероятности обнаружения и исправления ошибок снижает остаточную ошибку и повышает `q_acc`, `q_res`;
2. дополнительные контрольные действия увеличивают время и снижают `q_time`;
3. повторные операции и проверки увеличивают трудоёмкость и снижают `q_effort`.

Итоговый показатель представлен как агрегирование частных критериев с весами, определяемыми сценарием применения.

## Созданные и изменённые файлы

```text
src/manual_coding_sim/dissertation_figures/figure_2_4_control_tradeoff.py
tests/dissertation_figures/test_figure_2_4_control_tradeoff.py
docs/dissertation_figures_roadmap.md
README_FIGURE_STAGE_08.md
reports/dissertation_figures/chapter2/figure_2_4_control_tradeoff.png
reports/dissertation_figures/chapter2/figure_2_4_control_tradeoff.svg
```

## Выходные форматы

```text
reports/dissertation_figures/chapter2/figure_2_4_control_tradeoff.png
reports/dissertation_figures/chapter2/figure_2_4_control_tradeoff.svg
```

PNG формируется с разрешением 300 dpi. SVG сохраняет текст и геометрические элементы редактируемыми.

## Команды Windows PowerShell

### 1. Перейти в корень проекта

```powershell
Set-Location D:\Projects\man_coding_quality
```

### 2. Распаковать архив этапа

```powershell
Expand-Archive `
  -Path "$HOME\Downloads\dissertation_figures_stage08_figure_2_4.zip" `
  -DestinationPath . `
  -Force
```

### 3. Активировать виртуальное окружение

```powershell
.\.venv\Scripts\Activate.ps1
```

### 4. Установить `PYTHONPATH`

```powershell
$env:PYTHONPATH = (Resolve-Path .\src).Path
```

### 5. Сформировать рисунок

```powershell
python -m manual_coding_sim.dissertation_figures.figure_2_4_control_tradeoff `
  --project-root . `
  --dpi 300
```

### 6. Запустить тесты этапа

```powershell
python -m pytest `
  tests\dissertation_figures\test_figure_2_4_control_tradeoff.py `
  -q
```

Ожидаемый результат:

```text
9 passed
```

### 7. Проверить все реализованные рисунки

```powershell
python -m pytest tests\dissertation_figures -q
```

Ожидаемый результат после этапа 8:

```text
57 passed
```

### 8. Выполнить полную регрессию проекта

```powershell
python -m pytest -q
```

## Критерии визуальной проверки

- три причинные ветви различимы и не перекрываются;
- положительный эффект контроля отделён от временной и трудоёмкостной стоимости;
- формулы остаточной ошибки, времени контроля и интегрального показателя читаемы;
- стрелки направлены от контрольных процедур к частным критериям и далее к `Q(A)`;
- отсутствуют обрезка, выход текста за границы и наложение подписей;
- SVG открывается как редактируемая векторная схема.
