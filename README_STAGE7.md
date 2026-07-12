# Этап 7. Проверка классификации уровней качества

## Назначение

Этап выполняет внешнюю классификационную проверку неизмененного прогноза
главы 5. Значения `q_pred` и `q_fact` переводятся в классы по порогам,
зафиксированным до анализа фактических данных:

- `low`: `Q < 0.45`;
- `medium`: `0.45 <= Q < 0.70`;
- `high`: `Q >= 0.70`.

Пороговые значения, веса и формулы главы 5 на этапе 7 не изменяются.

## Создаваемые и изменяемые файлы

```text
src/manual_coding_sim/validation/classification_validator.py
src/manual_coding_sim/validation/chapter6_runner.py
tests/validation/test_classification_validator.py
README_STAGE7.md
STAGE7_MANIFEST.json
```

## Выходные артефакты

```text
reports/chapter6/classification_predictions.csv
reports/chapter6/confusion_matrix.csv
reports/chapter6/classification_report.json
reports/chapter6/classification_report.md
```

## Рассчитываемые показатели

- Accuracy;
- Balanced Accuracy;
- Macro F1;
- Weighted F1;
- Precision, Recall и F1 для `low`, `medium`, `high`;
- матрица ошибок;
- число критических ошибок `low -> high` и `high -> low`;
- распределения прогнозных и фактических классов.

## Команда запуска

```powershell
python -m manual_coding_sim.validation.chapter6_runner `
  --project-root . `
  --config configs\chapter6.yaml `
  --validate-classification
```

## Тесты этапа

```powershell
pytest tests\validation\test_classification_validator.py -q
pytest tests\validation -q
pytest -q
```

## Контрольный результат на расширенном корпусе

```text
Сценариев:                 150
Accuracy:                  0.4466666667
Balanced Accuracy:         0.6678231293
Macro F1:                  0.4225151692
Weighted F1:               0.5552571280
Критических low -> high:   0
Критических high -> low:   0
```

Матрица ошибок:

| Факт / прогноз | low | medium | high |
|---|---:|---:|---:|
| low | 1 | 0 | 0 |
| medium | 66 | 33 | 1 |
| high | 0 | 16 | 33 |

## Критерий закрытия

Этап считается закрытым после успешного выполнения CLI, тестов этапа,
всех тестов `tests/validation`, полной регрессии и повторного запуска CLI.
Переход к этапу 8 выполняется только после отдельного подтверждения.
