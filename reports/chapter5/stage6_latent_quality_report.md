# Отчет этапа 6. Расчет латентной компоненты качества

## Статус

Этап 6 реализован. Закрытие этапа требует выполнения контрольных проверок в рабочей среде пользователя.

## Что реализовано

- Добавлен расчет латентной компоненты качества `q_latent` по профилю `theta_prior`.
- Добавлен CLI-флаг `--calculate-latent-component`.
- Добавлены выходные артефакты:
  - `reports/chapter5/latent_quality_component.csv`;
  - `reports/chapter5/latent_quality_component_report.json`.
- Добавлены тесты расчета латентной компоненты.
- Добавлена тестовая фикстура восстановления входных артефактов главы 5 после тестов, чтобы ранние тесты главы 3 не перезаписывали `data/processed/prior_features.csv`.

## Формула этапа

При направлениях факторов `theta_0=-1`, `theta_1=-1`, `theta_2=+1` расчет выполняется как:

```text
latent_direction_score = -theta_0 - theta_1 + theta_2
q_latent = clip((latent_direction_score + 1) / 2, 0, 1)
```

При нормированном профиле `theta_0 + theta_1 + theta_2 = 1` эта форма эквивалентна положительному вкладу фактора благоприятных условий `theta_2`, но в коде сохранена обобщенная форма через направления факторов.

## Полученные показатели

| Показатель | Значение |
|---|---:|
| Строк в `latent_quality_component.csv` | 150 |
| Колонок | 10 |
| Минимальное `q_latent` | 0.010720 |
| Среднее `q_latent` | 0.419521 |
| Максимальное `q_latent` | 0.978029 |
| Доминирует `theta_0` | 34 |
| Доминирует `theta_1` | 46 |
| Доминирует `theta_2` | 70 |

## Проверки, выполненные в контрольной среде

```text
python -m pytest tests/prediction -q
43 passed in 0.44s

python -m pytest -q
291 passed in 5.83s
```

## Измененные файлы этапа 6

- `src/manual_coding_sim/prediction/latent_quality_component.py`
- `src/manual_coding_sim/prediction/chapter5_runner.py`
- `src/manual_coding_sim/prediction/chapter5_config.py`
- `src/manual_coding_sim/prediction/paths.py`
- `src/manual_coding_sim/prediction/__init__.py`
- `configs/chapter5.yaml`
- `tests/conftest.py`
- `tests/prediction/test_latent_quality_component.py`
- `tests/prediction/test_chapter5_runner.py`
- `tests/prediction/test_chapter5_config.py`
- `reports/chapter5/latent_quality_component.csv`
- `reports/chapter5/latent_quality_component_report.json`
