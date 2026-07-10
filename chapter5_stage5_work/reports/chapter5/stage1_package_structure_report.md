# Этап 1. Каркас программного модуля главы 5

## Статус

Этап 1 закрыт.

Цель этапа: создать импортируемый каркас подпакета `manual_coding_sim.prediction` для последующей реализации априорной интегральной оценки качества `Q_pred(A)`.

На этом этапе расчет `Q_pred`, нормировка признаков, частные критерии и неопределенность намеренно не реализовывались. Эти задачи относятся к следующим этапам.

## Созданные файлы

### Исходный код

- `src/manual_coding_sim/prediction/__init__.py`
- `src/manual_coding_sim/prediction/paths.py`
- `src/manual_coding_sim/prediction/chapter5_config.py`
- `src/manual_coding_sim/prediction/chapter5_data_loader.py`
- `src/manual_coding_sim/prediction/chapter5_leakage_guard.py`
- `src/manual_coding_sim/prediction/prior_feature_normalizer.py`
- `src/manual_coding_sim/prediction/latent_quality_component.py`
- `src/manual_coding_sim/prediction/partial_quality_predictor.py`
- `src/manual_coding_sim/prediction/integral_quality_predictor.py`
- `src/manual_coding_sim/prediction/prediction_uncertainty.py`
- `src/manual_coding_sim/prediction/chapter5_report_builder.py`
- `src/manual_coding_sim/prediction/chapter5_runner.py`

### Тесты

- `tests/prediction/test_chapter5_config.py`
- `tests/prediction/test_chapter5_data_loader.py`
- `tests/prediction/test_chapter5_leakage_guard.py`
- `tests/prediction/test_chapter5_runner.py`

## Реализованный минимум

1. Создан подпакет `manual_coding_sim.prediction`.
2. Добавлены пути по умолчанию для входов и выходов главы 5.
3. Добавлена базовая конфигурация `Chapter5PredictionConfig`.
4. Добавлен каркас загрузчика `Chapter5DataLoader`.
5. Добавлена первичная проверка запрещенных колонок в `Chapter5LeakageGuard`.
6. Добавлены заглушки расчетных компонентов с русскоязычными сообщениями `NotImplementedError`.
7. Добавлен CLI-каркас `chapter5_runner.py` с русскоязычным консольным выводом.
8. Добавлены базовые тесты импортируемости, конфигурации, защиты от утечки и runner-а.

## Контроль русскоязычности

В новых Python-файлах русскоязычными сделаны:

- docstring-и;
- комментарии по смыслу модулей;
- сообщения исключений;
- консольный вывод CLI-каркаса.

Технические имена модулей, классов, методов, CLI-аргументов и файлов сохранены на английском языке.

## Результаты тестирования

Проверка тестов этапа 1:

```text
pytest tests/prediction -q
10 passed in 0.20s
```

Полная регрессионная проверка проекта:

```text
pytest -q
258 passed in 3.67s
```

## Критерии закрытия

| Критерий | Статус |
|---|---:|
| Каркас подпакета создан | выполнено |
| Импорты не падают | выполнено |
| Базовая конфигурация валидируется | выполнено |
| CLI-каркас запускается | выполнено |
| Консольный вывод на русском языке | выполнено |
| `pytest tests/prediction` проходит | выполнено |
| Полный `pytest` проходит | выполнено |

## Следующий этап

Следующий этап: **Этап 2. Конфигурации главы 5**.

Переход к этапу 2 не выполнялся.
