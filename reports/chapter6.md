# План доработки программного обеспечения для завершения главы 6

## 1. Назначение

Цель доработки — реализовать автономный программный контур главы 6, предназначенный для экспериментальной проверки достоверности априорной оценки качества ручных средств кодирования информации.

Контур главы 6 должен:

- использовать результаты главы 5 как зафиксированный априорный прогноз;
- использовать фактические показатели только для внешней проверки;
- не изменять веса, нормировки, направления латентных факторов и формулу `Q_pred`;
- рассчитывать интегральные и частные метрики качества прогноза;
- проверять классификацию уровней качества;
- оценивать интервальный прогноз;
- сравнивать модель главы 5 с базовыми моделями без LDA;
- выполнять bootstrap-анализ;
- выявлять сценарии с наибольшими ошибками;
- формировать CSV-, JSON-, Markdown-отчеты и графики;
- проходить финальную автоматизированную приемку.

---

## 2. Исходное состояние

### 2.1. Зафиксированный результат главы 5

Программный контур главы 5 завершен и принят.

Подтверждено:

- сценариев: 150;
- латентных факторов: 3;
- методическая утечка отсутствует;
- `Q_pred` рассчитан;
- шесть частных прогнозных критериев рассчитаны;
- интервалы прогноза сформированы;
- выполнено 18 приемочных проверок;
- финальная приемка пройдена.

### 2.2. Входные артефакты

Прогнозные результаты:

```text
reports/chapter5/q_pred.csv
reports/chapter5/q_pred_components.csv
reports/chapter5/prediction_uncertainty.csv
reports/chapter5/chapter5_prediction_report.json
reports/chapter5/chapter5_acceptance_report.json
reports/chapter5/normalized_prior_features.csv
reports/chapter5/latent_quality_component.csv

Латентный профиль:

reports/chapter4/theta_prior.csv

Фактические данные:

data/processed/quality_targets.csv
data/processed/fact_features.csv

Ключи объединения:

scenario_id
protocol_id
3. Общие правила разработки
Каждый этап выполняется отдельно.
Следующий этап начинается только после прохождения тестов текущего этапа.
Для каждого этапа формируется ZIP-архив со всеми создаваемыми и изменяемыми файлами.
В ZIP-архиве сохраняется структура каталогов проекта.
Все комментарии и docstring-и в Python-файлах оформляются на русском языке.
Идентификаторы, имена файлов, классов, функций и CLI-аргументов могут оставаться на английском языке.
Программный контур главы 5 считается зафиксированным.
Фактические показатели запрещено использовать для изменения модели главы 5.
Все случайные процедуры должны использовать фиксированный random_seed.
Все расчетные результаты сохраняются в машинно-читаемом виде.
Полный контур главы 6 должен запускаться одной CLI-командой.
4. Целевая архитектура
src/manual_coding_sim/validation/
    __init__.py
    paths.py
    chapter6_config.py
    chapter6_data_loader.py
    validation_dataset_builder.py
    integral_quality_validator.py
    partial_criteria_validator.py
    classification_validator.py
    interval_validator.py
    baseline_models.py
    bootstrap_analyzer.py
    prediction_error_analyzer.py
    chapter6_figure_builder.py
    chapter6_report_builder.py
    chapter6_acceptance.py
    chapter6_pipeline.py
    chapter6_runner.py

configs/
    chapter6.yaml

tests/validation/
    test_stage1_chapter6.py
    test_chapter6_data_loader.py
    test_validation_dataset_builder.py
    test_integral_quality_validator.py
    test_partial_criteria_validator.py
    test_classification_validator.py
    test_interval_validator.py
    test_baseline_models.py
    test_bootstrap_analyzer.py
    test_prediction_error_analyzer.py
    test_chapter6_figure_builder.py
    test_chapter6_report_builder.py
    test_chapter6_acceptance.py
    test_chapter6_runner.py

reports/chapter6/
5. Этапы реализации
Этап 0. Аудит исходных данных
Цель

Подтвердить согласованность входных данных расширенного корпуса.

Работы
проверить наличие обязательных файлов;
проверить число строк;
проверить обязательные колонки;
проверить уникальность ключей;
проверить согласованность идентификаторов;
повторно выполнить полный контур главы 5;
зафиксировать хэши файлов.
Критерий завершения

Все основные таблицы содержат 150 сценариев, а приемка главы 5 имеет статус accepted = true.

Статус

Завершен.

Этап 1. Конфигурационный каркас
Файлы
src/manual_coding_sim/validation/__init__.py
src/manual_coding_sim/validation/paths.py
src/manual_coding_sim/validation/chapter6_config.py
src/manual_coding_sim/validation/chapter6_runner.py
configs/chapter6.yaml
tests/validation/test_stage1_chapter6.py
README_STAGE1.md
Функции
хранение путей;
загрузка YAML;
проверка порогов классов;
проверка ключей объединения;
проверка bootstrap-параметров;
CLI для отображения конфигурации.
Архив
chapter6_stage1_validation_config.zip

Статус

Завершен 11.07.2026.

Фактически реализовано:

- создан пакет `manual_coding_sim.validation`;
- зафиксированы входные и выходные пути программного контура;
- добавлена загрузка единой конфигурации `configs/chapter6.yaml`;
- реализована проверка ключей `scenario_id`, `protocol_id` и режима `one_to_one`;
- реализована проверка порогов классов `0.45` и `0.70`;
- реализована проверка параметров bootstrap: 1000 повторов, доверительный уровень 0.95, `random_seed = 42`;
- добавлен CLI для отображения проверенной конфигурации и разрешенных путей;
- тесты этапа: 8 пройдено;
- полная регрессионная проверка: 325 тестов пройдено.

Этап 2. Загрузка входных артефактов
Файлы
src/manual_coding_sim/validation/chapter6_data_loader.py
src/manual_coding_sim/validation/chapter6_config.py
src/manual_coding_sim/validation/chapter6_runner.py
tests/validation/test_chapter6_data_loader.py
Проверки
наличие файлов;
корректность CSV и JSON;
обязательные колонки;
число строк;
уникальность ключей;
отсутствие NaN, inf, -inf;
диапазон показателей [0; 1];
согласованность Q_pred между файлами;
успешная приемка главы 5.
Выходы
reports/chapter6/chapter6_input_validation_report.json
reports/chapter6/chapter6_input_validation_report.md
Архив
chapter6_stage2_data_loader.zip
Этап 3. Формирование проверочного датасета
Файлы
src/manual_coding_sim/validation/validation_dataset_builder.py
src/manual_coding_sim/validation/chapter6_runner.py
tests/validation/test_validation_dataset_builder.py
Операции
объединение по (scenario_id, protocol_id);
режим one_to_one;
контроль числа строк после каждого merge;
добавление прогнозных и фактических критериев;
добавление латентного профиля;
добавление фактических диагностических признаков;
добавление прогнозных интервалов;
расчет прогнозного и фактического классов.
Выход
reports/chapter6/validation_dataset.csv
Архив
chapter6_stage3_validation_dataset.zip
Этап 4. Проверка фактического интегрального качества
Файлы
src/manual_coding_sim/validation/integral_quality_validator.py
src/manual_coding_sim/validation/chapter6_runner.py
tests/validation/test_integral_quality_validator.py
Операции
использовать integral_quality как Q_fact;
независимо пересчитать интегральное качество из шести частных критериев;
применить веса главы 5;
вычислить максимальное и среднее расхождение;
проверить числовой допуск.
Выходы
reports/chapter6/integral_quality_consistency.csv
reports/chapter6/integral_quality_consistency_report.json
reports/chapter6/integral_quality_consistency_report.md
Архив
chapter6_stage4_integral_quality.zip
Этап 5. Метрики интегрального прогноза
Метрики
MAE;
RMSE;
Bias;
Median Absolute Error;
Max Absolute Error;
Pearson;
Spearman;
Kendall;
R².
Поля по сценариям
prediction_error
absolute_error
squared_error
Выходы
reports/chapter6/validation_metrics.json
reports/chapter6/validation_metrics.md
reports/chapter6/integral_prediction_errors.csv
Архив
chapter6_stage5_integral_metrics.zip
Этап 6. Проверка частных критериев
Сопоставления
q_acc_pred    -> q_acc
q_time_pred   -> q_time
q_effort_pred -> q_effort
q_res_pred    -> q_res
q_rep_pred    -> q_rep
q_fit_pred    -> q_fit
Метрики
MAE;
RMSE;
Bias;
Pearson;
Spearman;
Kendall;
R²;
прогнозное среднее;
фактическое среднее.
Выходы
reports/chapter6/partial_criteria_validation.csv
reports/chapter6/partial_criteria_validation_report.json
reports/chapter6/partial_criteria_validation_report.md
Архив
chapter6_stage6_partial_criteria.zip
Этап 7. Проверка классификации
Пороги
low:    Q < 0.45
medium: 0.45 <= Q < 0.70
high:   Q >= 0.70
Метрики
Accuracy;
Balanced Accuracy;
Macro F1;
Weighted F1;
Precision по классам;
Recall по классам;
F1 по классам;
confusion matrix.
Критические ошибки
low -> high
high -> low
Выходы
reports/chapter6/classification_report.json
reports/chapter6/classification_report.md
reports/chapter6/confusion_matrix.csv
reports/chapter6/classification_predictions.csv
Архив
chapter6_stage7_classification.zip
Этап 8. Проверка интервального прогноза
Условие
q_pred_lower <= Q_fact <= q_pred_upper
Метрики
coverage_rate;
mean_interval_width;
median_interval_width;
miss_lower_count;
miss_upper_count;
среднее расстояние до интервала.
Дополнительные срезы
по фактическому классу;
по прогнозному классу;
по доминирующему фактору;
по квантилям uncertainty_score.
Выходы
reports/chapter6/interval_coverage_details.csv
reports/chapter6/interval_coverage_report.json
reports/chapter6/interval_coverage_report.md
Архив
chapter6_stage8_interval_validation.zip
Этап 9. Базовые модели
Модели
Mean baseline.
Prior-only baseline.
Theta-only baseline.
Полная модель главы 5.
Методические требования
Mean baseline строится out-of-fold;
тестовые цели не участвуют в расчете обучающего среднего;
prior-only использует q_*_feature_component;
theta-only использует q_latent;
полная модель использует неизменный Q_pred.
Выходы
reports/chapter6/baseline_predictions.csv
reports/chapter6/baseline_comparison.csv
reports/chapter6/baseline_comparison_report.json
reports/chapter6/baseline_comparison_report.md
Архив
chapter6_stage9_baselines.zip
Этап 10. Bootstrap-анализ
Параметры
resamples: 1000
confidence_level: 0.95
random_seed: 42
sampling_unit: scenario_id

Для финального запуска допускается увеличение до 5000 повторов.

Метрики
MAE;
RMSE;
Bias;
Spearman;
Kendall;
Accuracy;
Macro F1;
coverage rate.
Парные разности
delta_mae
delta_rmse
delta_spearman
delta_accuracy
delta_macro_f1
Выходы
reports/chapter6/bootstrap_confidence_intervals.csv
reports/chapter6/bootstrap_model_differences.csv
reports/chapter6/bootstrap_report.json
reports/chapter6/bootstrap_report.md
Архив
chapter6_stage10_bootstrap.zip
Этап 11. Анализ ошибок
Операции
сформировать top-10 по absolute_error;
проверить связь ошибки с uncertainty_score;
проверить связь ошибки с доминирующей темой;
оценить ошибки по классам;
оценить ошибки по условиям;
оценить связь с фактическими диагностическими признаками.
Выходы
reports/chapter6/top_prediction_errors.csv
reports/chapter6/error_group_analysis.csv
reports/chapter6/prediction_error_analysis.json
reports/chapter6/prediction_error_analysis.md
Архив
chapter6_stage11_error_analysis.zip
Этап 12. Графические материалы
Графики
reports/chapter6/figures/q_pred_vs_q_fact.png
reports/chapter6/figures/residuals_vs_q_fact.png
reports/chapter6/figures/absolute_error_distribution.png
reports/chapter6/figures/confusion_matrix.png
reports/chapter6/figures/baseline_comparison.png
reports/chapter6/figures/prediction_intervals.png
reports/chapter6/figures/error_by_dominant_topic.png
reports/chapter6/figures/partial_criteria_comparison.png
Требования
русскоязычные подписи;
разрешение для вставки в диссертацию;
воспроизводимое построение;
отсутствие ручной подмены данных.
Архив
chapter6_stage12_figures.zip
Этап 13. Итоговый отчет
Файлы
src/manual_coding_sim/validation/chapter6_report_builder.py
src/manual_coding_sim/validation/chapter6_pipeline.py
src/manual_coding_sim/validation/chapter6_runner.py
tests/validation/test_chapter6_report_builder.py
Выходы
reports/chapter6/chapter6_validation_report.json
reports/chapter6/chapter6_validation_report.md
reports/chapter6/chapter6_pipeline_run_report.json
Статусы гипотезы
hypothesis_supported
hypothesis_partially_supported
hypothesis_not_supported
Архив
chapter6_stage13_report_builder.zip
Этап 14. Финальная приемка
Обязательные проверки
Входные файлы найдены.
Все таблицы содержат 150 сценариев.
Merge выполнен без потери строк.
Ключи уникальны.
Показатели находятся в [0; 1].
integral_quality согласован с частными критериями.
Рассчитаны интегральные метрики.
Проверены шесть частных критериев.
Сформирована confusion matrix.
Рассчитаны классификационные метрики.
Проверены интервалы.
Построены baseline-модели.
Mean baseline не содержит утечки.
Выполнен bootstrap.
Рассчитаны интервалы парных разностей.
Сформирован top-10 ошибок.
Построены графики.
Создан JSON-отчет.
Создан Markdown-отчет.
Артефакты главы 5 не изменены.
Параметры главы 5 не подгонялись.
Полный CLI-контур завершен.
Выходы
reports/chapter6/chapter6_acceptance_report.json
reports/chapter6/chapter6_acceptance_report.md
Ожидаемый статус
{
  "stage": 14,
  "report_type": "chapter6_final_acceptance_report",
  "accepted": true,
  "full_pipeline_completed": true,
  "prediction_model_frozen": true,
  "target_leakage_detected": false
}
Архив
chapter6_stage14_final_acceptance.zip
Этап 15. Обновление текста диссертации
Работы
заполнить таблицы главы 6;
перенести значения метрик;
вставить confusion matrix;
вставить результаты baseline;
вставить bootstrap-интервалы;
добавить top-10 ошибок;
вставить графики;
уточнить ограничения;
сформулировать итоговые выводы;
обновить DOCX главы 6.
Выходы
6. Глава 6. Экспериментальная проверка достоверности априорной оценки качества.docx
reports/chapter6/chapter6_results_for_dissertation.md
Архив
chapter6_stage15_dissertation_update.zip
6. Итоговая CLI-команда
python -m manual_coding_sim.validation.chapter6_runner `
  --project-root . `
  --config configs/chapter6.yaml `
  --run-full-pipeline `
  --build-figures `
  --build-report `
  --run-acceptance
7. Итоговый набор артефактов
reports/chapter6/
    validation_dataset.csv
    validation_metrics.json
    validation_metrics.md
    integral_prediction_errors.csv
    partial_criteria_validation.csv
    classification_report.json
    classification_report.md
    confusion_matrix.csv
    interval_coverage_details.csv
    interval_coverage_report.json
    baseline_predictions.csv
    baseline_comparison.csv
    bootstrap_confidence_intervals.csv
    bootstrap_model_differences.csv
    top_prediction_errors.csv
    error_group_analysis.csv
    chapter6_validation_report.json
    chapter6_validation_report.md
    chapter6_pipeline_run_report.json
    chapter6_acceptance_report.json
    chapter6_acceptance_report.md
    figures/
8. Критерий завершения

Программный блок главы 6 считается завершенным, если:

проверены все 150 сценариев;
рассчитаны интегральные и частные метрики;
выполнена классификационная проверка;
проверены интервалы;
построены baseline-модели;
выполнен bootstrap;
проведен анализ ошибок;
сформированы графики;
сформированы итоговые отчеты;
приемка имеет статус accepted = true;
глава 5 не изменена по фактическим результатам;
результаты перенесены в окончательную редакцию главы 6.

---

## 9. Журнал реализации главы 6

### 9.1. Этап 1 — конфигурационный каркас

Программная реализация экспериментальной проверки начата с выделения
самостоятельного пакета `manual_coding_sim.validation`. В конфигурации
зафиксированы все входные артефакты глав 3–5, планируемые выходные артефакты
главы 6, ключи объединения `(scenario_id, protocol_id)`, режим проверки
`one_to_one`, ожидаемое число сценариев 150, пороги классов качества 0,45 и
0,70, а также воспроизводимые параметры bootstrap-анализа.

Конфигурационный слой не загружает фактические показатели и не изменяет
прогноз главы 5. Тем самым на уровне архитектуры сохранено методическое
разделение между построением априорной оценки и ее внешней проверкой.
Конфигурация проверяется до выполнения последующих вычислительных этапов;
ошибка порогов, ключей объединения или bootstrap-параметров блокирует запуск.

Этап подтвержден восемью специализированными тестами. Полная регрессионная
проверка программного комплекса завершена успешно: 325 тестов пройдено.
Следующим разрешенным этапом является этап 2 — загрузка входных артефактов.
