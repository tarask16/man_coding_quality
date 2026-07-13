# Этап 10. Рисунок 3.2 — архитектура программного пакета

## Назначение

Этап формирует компонентную диаграмму программного пакета компьютерного моделирования процессов ручного кодирования.

Рисунок показывает:

1. управляющий компонент `runner`, загружающий конфигурацию и фиксирующий `random_seed`;
2. предметные модули `message`, `procedure`, `operator`, `condition`, `error` и `control`;
3. симулятор `protocol`, формирующий протокол применения `P_i`;
4. компоненты `features` и `quality`, раздельно формирующие признаки и фактические показатели;
5. компонент `dataset`, собирающий проверяемые CSV- и JSON-артефакты;
6. разделение выходов `X_prior`, `X_fact` и `Y_fact`.

## Созданные и изменённые файлы

```text
src/manual_coding_sim/dissertation_figures/figure_3_2_package_architecture.py
tests/dissertation_figures/test_figure_3_2_package_architecture.py
docs/dissertation_figures_roadmap.md
README_FIGURE_STAGE_10.md
reports/chapter3/figures/figure_3_2_package_architecture.png
reports/chapter3/figures/figure_3_2_package_architecture.svg
```

## Выходные форматы

```text
reports/chapter3/figures/figure_3_2_package_architecture.png
reports/chapter3/figures/figure_3_2_package_architecture.svg
```

PNG формируется с разрешением 300 dpi. SVG сохраняет текст, блоки и связи редактируемыми.

## Команды Windows PowerShell

### 1. Перейти в корень проекта

```powershell
Set-Location D:\Projects\man_coding_quality
```

### 2. Распаковать архив этапа

```powershell
Expand-Archive `
  -Path "$HOME\Downloads\dissertation_figures_stage10_figure_3_2.zip" `
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
python -m manual_coding_sim.dissertation_figures.figure_3_2_package_architecture `
  --project-root . `
  --dpi 300
```

### 6. Запустить тесты этапа

```powershell
python -m pytest `
  tests\dissertation_figures\test_figure_3_2_package_architecture.py `
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

Ожидаемый результат после этапа 10:

```text
75 passed
```

### 8. Выполнить полную регрессию проекта

```powershell
python -m pytest -q
```

## Критерии визуальной проверки

- все 11 компонентов присутствуют и читаются;
- `runner` находится в слое оркестрации и управляет шестью предметными модулями;
- предметные модули передают данные в `protocol`;
- `protocol` раздельно связан с `features` и `quality`;
- `dataset` получает `X_prior`, `X_fact`, `Y_fact` и `Q_fact`;
- имена выходных файлов читаемы;
- отсутствуют пересечения стрелок с текстом, обрезка и наложения;
- SVG открывается как редактируемая векторная схема.
