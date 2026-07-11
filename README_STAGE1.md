# Глава 6 — этап 1. Конфигурационный каркас

## Назначение этапа

Этап 1 создает самостоятельный пакет `manual_coding_sim.validation`, который
фиксирует пути к артефактам глав 3–5 и методические параметры внешней проверки
априорного прогноза. Фактические таблицы на этом этапе не загружаются, метрики
не рассчитываются, программный контур главы 5 не изменяется.

## Созданные файлы

```text
src/manual_coding_sim/validation/__init__.py
src/manual_coding_sim/validation/paths.py
src/manual_coding_sim/validation/chapter6_config.py
src/manual_coding_sim/validation/chapter6_runner.py
configs/chapter6.yaml
tests/validation/test_stage1_chapter6.py
README_STAGE1.md
```

Обновлен журнал реализации:

```text
reports/chapter6.md
```

## Зафиксированные параметры

- ключи объединения: `scenario_id`, `protocol_id`;
- режим объединения: `one_to_one`;
- ожидаемое число сценариев: `150`;
- пороги классов: `0.45` и `0.70`;
- bootstrap-повторы: `1000`;
- доверительный уровень: `0.95`;
- `random_seed`: `42`;
- единица bootstrap-выборки: `scenario_id`.

## Проверка конфигурации

```bash
python -m manual_coding_sim.validation.chapter6_runner \
  --project-root . \
  --config configs/chapter6.yaml \
  --show-config
```

Для вывода абсолютных путей:

```bash
python -m manual_coding_sim.validation.chapter6_runner \
  --project-root . \
  --config configs/chapter6.yaml \
  --show-resolved-paths
```

## Тестирование

Тесты этапа:

```bash
python -m pytest tests/validation/test_stage1_chapter6.py -q
```

Полная регрессионная проверка проекта:

```bash
python -m pytest -q
```

Фактические результаты:

- тесты этапа: **8 passed**;
- полная регрессионная проверка: **325 passed**;
- время полной проверки в рабочем окружении: **11,12 с**.
