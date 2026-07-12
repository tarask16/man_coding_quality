# Bootstrap-анализ статистической устойчивости

- Этап: **10**
- Статус: **пройден**
- Сценариев: **150**
- Bootstrap-повторов: **1000**
- Уровень доверия: **95.00%**
- Единица выборки: **scenario_id**
- Random seed: **42**

## Доверительные интервалы модели главы 5

| Метрика | Значение | Bootstrap mean | CI lower | CI upper |
|---|---:|---:|---:|---:|
| mae | 0.1592441669 | 0.1591211204 | 0.1405341874 | 0.1779741437 |
| rmse | 0.1944027110 | 0.1940961131 | 0.1767722536 | 0.2106599040 |
| spearman | 0.8817689675 | 0.8785732159 | 0.8273055976 | 0.9136315536 |
| kendall | 0.6993288591 | 0.6999236547 | 0.6409357447 | 0.7475942570 |
| accuracy | 0.4466666667 | 0.4472866667 | 0.3666666667 | 0.5266666667 |
| macro_f1 | 0.4225151692 | 0.4213444855 | 0.3667620360 | 0.4772037969 |

## Парные разности модели главы 5 и baseline

Разность определяется как `metric_chapter5_model - metric_baseline`.

| Baseline | Метрика | Δ | CI lower | CI upper | Вывод |
|---|---|---:|---:|---:|---|
| mean_baseline | mae | 0.0620173095 | 0.0399775161 | 0.0831623328 | baseline_favored |
| mean_baseline | rmse | 0.0814998022 | 0.0620006635 | 0.1010660990 | baseline_favored |
| mean_baseline | spearman | 0.9535133948 | 0.7945381938 | 1.1085640595 | chapter5_model_favored |
| mean_baseline | kendall | 0.7505817183 | 0.6318637319 | 0.8731483828 | chapter5_model_favored |
| mean_baseline | accuracy | -0.2200000000 | -0.3466666667 | -0.0933333333 | baseline_favored |
| mean_baseline | macro_f1 | 0.1558485026 | 0.1001392798 | 0.2115135690 | chapter5_model_favored |
| prior_only_baseline | mae | 0.0474220422 | 0.0332866472 | 0.0611626456 | baseline_favored |
| prior_only_baseline | rmse | 0.0622896172 | 0.0496971742 | 0.0734129167 | baseline_favored |
| prior_only_baseline | spearman | 0.0227174541 | -0.0022452452 | 0.0506598875 | no_stable_difference |
| prior_only_baseline | kendall | 0.0327516779 | -0.0003827908 | 0.0680903849 | no_stable_difference |
| prior_only_baseline | accuracy | -0.0666666667 | -0.1468333333 | 0.0133333333 | no_stable_difference |
| prior_only_baseline | macro_f1 | 0.0293604392 | -0.0283288822 | 0.0915292404 | no_stable_difference |
| theta_only_baseline | mae | -0.1448803656 | -0.1623000456 | -0.1272445902 | chapter5_model_favored |
| theta_only_baseline | rmse | -0.1661414806 | -0.1828177797 | -0.1492711534 | chapter5_model_favored |
| theta_only_baseline | spearman | 0.0150797813 | -0.0071465075 | 0.0360610910 | no_stable_difference |
| theta_only_baseline | kendall | 0.0218344519 | -0.0079383673 | 0.0509785922 | no_stable_difference |
| theta_only_baseline | accuracy | 0.1066666667 | 0.0466666667 | 0.1666666667 | chapter5_model_favored |
| theta_only_baseline | macro_f1 | 0.0794414487 | 0.0331987674 | 0.1261619028 | chapter5_model_favored |

## Сводка статистически устойчивых различий

- Преимуществ модели главы 5: **7**
- Преимуществ baseline: **5**
- Неустойчивых различий: **6**

## Методическое ограничение

Использован парный percentile-bootstrap по сценариям. Внутри повторных выборок применялись уже зафиксированные out-of-fold прогнозы этапа 9. Модели не переобучались, прогноз главы 5 не изменялся.
