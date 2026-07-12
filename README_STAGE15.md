# Этап 15. Перенос результатов в текст диссертации

## Назначение

Этап автоматически переносит принятые результаты программного контура главы 6 в DOCX-документ диссертации. Источником числовых значений служат только артефакты этапов 2–14. Синтетический holdout не подключается и не используется как основное доказательство.

## Создаваемые и изменяемые файлы

- `src/manual_coding_sim/validation/chapter6_dissertation_updater.py` — обновление таблиц, интерпретаций, выводов и рисунков;
- `src/manual_coding_sim/validation/chapter6_runner.py` — CLI-флаг `--update-dissertation`;
- `tests/validation/test_chapter6_dissertation_updater.py` — 22 теста этапа;
- `requirements-stage15.txt` — зависимость `python-docx`.

## Входной документ

По умолчанию используется файл:

`6. Глава 6. Экспериментальная проверка достоверности априорной оценки качества.docx`

При обновлении на месте исходная версия копируется в:

`reports/chapter6/dissertation/chapter6_before_stage15.docx`

## Формируемые артефакты

- обновленный DOCX главы 6;
- `reports/chapter6/chapter6_results_for_dissertation.md`;
- `reports/chapter6/chapter6_dissertation_update_report.json`;
- `reports/chapter6/chapter6_dissertation_update_report.md`.

## Автоматически заполняемые элементы

- таблица интегральных метрик;
- таблица шести частных критериев;
- матрица ошибок и классификационные метрики;
- показатели интервального прогноза;
- сравнение четырех моделей;
- bootstrap-доверительные интервалы;
- четыре ключевые парные bootstrap-разности;
- топ-10 ошибок;
- итоговые интерпретации разделов 6.4–6.10;
- девять выводов раздела 6.12;
- восемь рисунков этапа 12.

## Методические ограничения

Обновление разрешается только при `accepted = true` в акте этапа 14 и при 150 сценариях. Исходные CSV/JSON и рисунки хэшируются до и после операции. Прогноз главы 5, веса, пороги и формулы не изменяются. Статус гипотезы копируется из отчета этапа 13 и не задается вручную.

## CLI

```powershell
python -m manual_coding_sim.validation.chapter6_runner `
  --project-root . `
  --config configs\chapter6.yaml `
  --update-dissertation
```

Для записи в отдельный файл:

```powershell
python -m manual_coding_sim.validation.chapter6_runner `
  --project-root . `
  --config configs\chapter6.yaml `
  --update-dissertation `
  --chapter6-output-document reports\chapter6\dissertation\chapter6_updated.docx
```

## Критерии закрытия

1. DOCX успешно открыт библиотекой `python-docx` после сохранения.
2. Заполнено девять расчетных таблиц.
3. Вставлено восемь рисунков.
4. Маркеры `[рассчитать]`, `[n]` и `[автозаполнение]` отсутствуют.
5. Созданы Markdown- и JSON-артефакты этапа 15.
6. Исходные расчетные артефакты не изменены.
7. Пройдено 22 теста этапа и полная регрессия.
8. Итоговый DOCX визуально проверен в Word или LibreOffice.
