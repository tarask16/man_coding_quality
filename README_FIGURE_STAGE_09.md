# Этап 9. Рисунок 2.5 — разделение данных и токенизация

## Назначение

Этап формирует составную схему разделения информационных множеств и преобразования априорных признаков в документ корпуса `LDA_prior`.

Рисунок показывает:

1. `X_prior` — признаки, известные до выполнения процедуры и разрешённые для априорного контура;
2. `X_fact` — признаки, возникающие во время или после выполнения процедуры;
3. `Y_fact` — фактические частные и интегральные показатели качества;
4. последовательность `X_prior → дискретизация → токенизация → d_i`;
5. перечёркнутые связи от `X_fact` и `Y_fact` к априорному документу;
6. работу `LeakageGuard` и формирование корпуса `D_prior`.

## Созданные и изменённые файлы

```text
src/manual_coding_sim/dissertation_figures/figure_2_5_information_sets_and_tokenization.py
tests/dissertation_figures/test_figure_2_5_information_sets_and_tokenization.py
docs/dissertation_figures_roadmap.md
README_FIGURE_STAGE_09.md
reports/dissertation_figures/chapter2/figure_2_5_information_sets_and_tokenization.png
reports/dissertation_figures/chapter2/figure_2_5_information_sets_and_tokenization.svg
```

## Выходные форматы

```text
reports/dissertation_figures/chapter2/figure_2_5_information_sets_and_tokenization.png
reports/dissertation_figures/chapter2/figure_2_5_information_sets_and_tokenization.svg
```

PNG формируется с разрешением 300 dpi. SVG сохраняет текст, стрелки, блоки и перечёркнутые запрещённые связи редактируемыми.

## Команды Windows PowerShell

### 1. Перейти в корень проекта

```powershell
Set-Location D:\Projects\man_coding_quality
```

### 2. Распаковать архив этапа

```powershell
Expand-Archive `
  -Path "$HOME\Downloads\dissertation_figures_stage09_figure_2_5.zip" `
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
python -m manual_coding_sim.dissertation_figures.figure_2_5_information_sets_and_tokenization `
  --project-root . `
  --dpi 300
```

### 6. Запустить тесты этапа

```powershell
python -m pytest `
  tests\dissertation_figures\test_figure_2_5_information_sets_and_tokenization.py `
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

Ожидаемый результат после этапа 9:

```text
66 passed
```

### 8. Выполнить полную регрессию проекта

```powershell
python -m pytest -q
```

## Критерии визуальной проверки

- три информационных множества визуально разделены;
- только `X_prior` имеет разрешённую связь с дискретизацией;
- связи от `X_fact` и `Y_fact` перечёркнуты и подписаны как запрещённые;
- этапы дискретизации, токенизации и формирования `d_i` читаются слева направо;
- блок `LeakageGuard` связан с априорным конвейером;
- формула корпуса `D_prior` и идентификаторы сценария читаемы;
- отсутствуют обрезка, наложения и выход текста за границы блоков;
- SVG открывается как редактируемая векторная схема.
