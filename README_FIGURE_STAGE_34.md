# Этап 34 — рисунок 6.8

## Назначение

Этап формирует boxplot абсолютной ошибки интегрального прогноза по доминирующему латентному фактору:

```text
reports/chapter6/figures/error_by_dominant_topic
```

Доминирующая тема определяется по априорному профилю:

\[
\operatorname{dominant\_topic}_i
=
\arg\max_k \theta_{ik}.
\]

Абсолютная ошибка рассчитывается после внешнего сопоставления:

\[
|e_i|=|Q_{\text{pred},i}-Q_{\text{fact},i}|.
\]

## Входные данные

```text
reports/chapter5/q_pred.csv
data/processed/quality_targets.csv
reports/chapter4/theta_prior.csv
```

Таблицы объединяются в режиме `one_to_one` по ключам:

```text
scenario_id
protocol_id
```

Компоненты `theta_0`, `theta_1`, `theta_2` должны находиться в диапазоне `[0; 1]` и суммироваться в единицу.

## Результаты расширенного корпуса

| Доминирующий фактор | n | Mean \(|e|\) | Median \(|e|\) | IQR | Max \(|e|\) |
|---|---:|---:|---:|---:|---:|
| Тема 0 — процедурная трудоёмкость | 34 | 0,250701 | 0,270347 | [0,212932; 0,301112] | 0,374730 |
| Тема 1 — операционный риск | 46 | 0,233306 | 0,250976 | [0,159721; 0,286518] | 0,376663 |
| Тема 2 — благоприятные условия | 70 | 0,066153 | 0,046820 | [0,023597; 0,094442] | 0,220385 |

Общая MAE корпуса:

```text
0,159244
```

Наблюдаемая средняя абсолютная ошибка минимальна в группе темы 2 и максимальна в группе темы 0.

Интерпретация различий только ассоциативная. Доминирующий фактор не рассматривается как причина ошибки, а групповой результат не подтверждает переносимость или абсолютную калибровку модели.

## Созданные файлы

```text
src/manual_coding_sim/dissertation_figures/figure_6_8_error_by_dominant_topic.py
tests/dissertation_figures/test_figure_6_8_error_by_dominant_topic.py
reports/chapter6/figures/error_by_dominant_topic.png
reports/chapter6/figures/error_by_dominant_topic.svg
docs/dissertation_figures_roadmap.md
README_FIGURE_STAGE_34.md
```

## Генерация

```powershell
python -m manual_coding_sim.dissertation_figures.figure_6_8_error_by_dominant_topic `
  --project-root . `
  --dpi 300
```

Явное указание входных файлов:

```powershell
python -m manual_coding_sim.dissertation_figures.figure_6_8_error_by_dominant_topic `
  --project-root . `
  --q-pred .\reports\chapter5\q_pred.csv `
  --q-fact .\data\processed\quality_targets.csv `
  --theta .\reports\chapter4\theta_prior.csv `
  --dpi 300
```

## Локальные тесты

```powershell
python -m pytest `
  tests\dissertation_figures\test_figure_6_8_error_by_dominant_topic.py `
  -q
```

Ожидается:

```text
10 passed
```

Полная локальная проверка рисунков:

```powershell
python -m pytest tests\dissertation_figures -q
```

При сохранении ранее подтверждённых результатов ожидается:

```text
301 passed
```

## Выполненная проверка в рабочем окружении

Тесты не запускались. Проверены:

- успешное формирование PNG и SVG;
- размер PNG `5067 x 2414 px` при 300 dpi;
- наличие редактируемых подписей в SVG;
- наличие трёх обязательных групп и их численностей;
- корректность boxplot, точек, средних и медиан;
- отображение общей MAE;
- наличие методического ограничения об ассоциативной интерпретации;
- отсутствие критических наложений, обрезки и выхода текста за панели.
