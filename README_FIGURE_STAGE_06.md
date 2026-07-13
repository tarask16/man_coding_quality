# Этап 6. Рисунок 2.2 — процесс ручного кодирования и декодирования

## Назначение

Сформировать процессную схему

```text
M → E_h → C → D_h → M′
```

с отображением:

- исходного сообщения `M`;
- ручного кодирования `E_h`;
- кодированного представления `C`;
- ручного декодирования `D_h`;
- восстановленного сообщения `M′`;
- каналов ошибок восприятия, выбора правила, записи и декодирования;
- задержек `Δt_e`, `Δt_c`, `Δt_d`;
- контуров контроля `K_e` и `K_d`;
- обнаружения, проверки и исправления ошибок;
- формирования фактического результата через `d(M, M′)`, число ошибок и общее время.

## Создаваемые артефакты

```text
reports/dissertation_figures/chapter2/figure_2_2_encoding_decoding_process.png
reports/dissertation_figures/chapter2/figure_2_2_encoding_decoding_process.svg
```

## Применение архива

```powershell
Set-Location D:\Projects\man_coding_quality
Expand-Archive `
  -Path "$HOME\Downloads\dissertation_figures_stage06_figure_2_2.zip" `
  -DestinationPath . `
  -Force
.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = (Resolve-Path .\src).Path
```

## Генерация артефактов

```powershell
python -m manual_coding_sim.dissertation_figures.figure_2_2_encoding_decoding_process `
  --project-root . `
  --dpi 300
```

## Проверка артефактов

```powershell
Test-Path .\reports\dissertation_figures\chapter2\figure_2_2_encoding_decoding_process.png
Test-Path .\reports\dissertation_figures\chapter2\figure_2_2_encoding_decoding_process.svg
Start-Process .\reports\dissertation_figures\chapter2\figure_2_2_encoding_decoding_process.png
Start-Process .\reports\dissertation_figures\chapter2\figure_2_2_encoding_decoding_process.svg
```

## Тесты этапа

```powershell
python -m pytest `
  tests\dissertation_figures\test_figure_2_2_encoding_decoding_process.py `
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
