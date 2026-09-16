# 结果文件与生成入口核对（2026-09-15）

范围：递归检查 `1.Source Data/` 与 `2.Source Code/`，包括隐藏的模型
checkpoint 和目录内历史结果。源数据、配置输入、手工流程图不要求生成脚本。
仅把代码中实际执行的 `to_csv`、`savefig`、模型保存或对应写出函数算作生成入口；
文件名出现在读取语句或文档中不算。Notebook 的 cell 编号均从 **0** 开始，
行号是该 cell 的 source 行号，不是执行次数。

逐文件明细见 [result_script_inventory.csv](result_script_inventory.csv)。
这是静态来源核对，未重新训练或运行全部 notebook，因此“找到生成入口”不等于
验证了现有文件可逐字节再生。后续归档只移动文件，未改动结果内容或模型代码。

## 核对结论

作者复核确认这些历史文件已经过时。按“当前脚本显式读取则保留，否则归档”
处理后，**13 个文件已归档，2 个仍被使用的旧 R² 表保留原位**。
此前 12 个未找到 writer 的旧派生文件均已归档，不再列为活动结果的脚本缺口。
两个根目录 SHAP 图按作者确认登记为 notebook 导出，不计入缺失。

归档目录为
[`0.Archived/2026-09-15_obsolete_historical_results/`](0.Archived/2026-09-15_obsolete_historical_results/README.md)，
含逐文件 SHA-256、原始/当前路径及读取依赖核对证据。活动 Python、notebook 代码
单元、复现入口和测试的静态检索中未找到这 13 个文件的输入依赖。
文档提及、同名输出语句或通用 Excel 读取能力不算显式调用。

| 类别 | 文件数 | 判定 |
| --- | ---: | --- |
| 当前代码具有明确写出入口的结果 | 155 | 已定位 writer |
| 作者确认来源的根目录 SHAP 导出图 | 2 | 已登记 notebook 出口 |
| 同名 writer 存在，但输出目录不同的 R² 数据表 | 3 | 2 个活动依赖保留，旧 CV 表已归档 |
| 历史 missingness 结果 | 3 | Git 中有历史 writer；两张图的精确版本仍未确认 |
| 未找到生成入口的历史派生文件 | 12 | 作者确认过时，全部已归档 |
| 隐藏 checkpoint | 168 | 均找到动态写出入口，单列以免混入展示结果 |
| 源数据或参考输入 | 4 | 豁免 |
| 超参数配置输入 | 4 | 豁免；不代表已核验调参过程 |
| 流程图及可编辑图源 | 6 | 按作者要求豁免 |
| 空 notebook 占位文件 | 1 | 不计入结果 |

审计原始快照共登记 358 个文件；其中非 checkpoint 的计算结果为 175 个。
13 个移出活动目录后，原审计范围内仍有 345 个文件（含此前已归档的 3 个
missingness 文件）。CSV 保留原始 `artifact`，以 `current_path` 和 `disposition`
记录实际去向；旧的来源状态是历史证据，不表示该文件仍是活动结果。
其他 `.py`、正常 notebook、说明文档、环境锁文件及 `__pycache__` 不属于结果分母。

## 已归档的 12 个历史派生文件

以下为原始路径，均相对于 `1.Source Data/`；现已移至上述归档目录下的同名
相对路径。这些旧文件不再要求补找生成脚本。表中保留最初审计发现供追溯。

| 文件 | 内容及缺口 |
| --- | --- |
| `All_prediction_cleaned.csv` | 清理后的预测表；未找到清理和导出入口 |
| `All_prediction_updated.xlsx` | 派生预测工作簿；未找到导出入口 |
| `error_analysis_forecasting_011525.csv` | 30 行误差分组统计；未找到统计与导出入口 |
| `metric_frame.csv` | 6 行阈值 precision/recall；未找到计算与导出入口 |
| `forecasting_df_with_folds.csv` | 已分配 CV fold 的派生表；未找到该文件的分折导出入口 |
| `forecasting_df_with_folds_5.csv` | 同上，五折版本 |
| `nowcasting_df_with_folds.csv` | 已分配 CV fold 的派生表；未找到该文件的分折导出入口 |
| `nowcasting_df_with_folds_5.csv` | 同上，五折版本 |
| `old_graph/2022_actual_alert_top20_percent.png` | 历史统计地图；未找到单图 writer |
| `old_graph/2022_actual_alert_top30_percent.png` | 同上 |
| `old_graph/2022_forecast_alert_top20_percent.png` | 同上 |
| `old_graph/2022_forecast_alert_top30_percent.png` | 同上 |

四份 fold 表属于派生的分折文件，单独列出其分折来源缺口；这不要求为原始分析
输入补写数据准备脚本。现有地图 panel 生成器输出组合图，不能据此声称找到了
上述四张独立旧地图的精确出口。此次按作者确认归档，未猜测方法或补造脚本。

## SHAP 图的 notebook 出口

按作者确认，两张位于 `2.Source Code/` 根目录的 SHAP 图来自 notebook 保存。
当前保存语句均以 `2.Source Code/` 为工作目录，输出到 `produced_graph/`。

| 文件名 | Notebook（位于 `2.Source Code/`） | 出口：零基 cell / source 行 |
| --- | --- | --- |
| `shap_values_forecasting_phase3_phase4_colored.jpg` | `Figure2_Feature_Importance_Forecasting.ipynb` | cell 5 / 92：`plt.savefig('produced_graph/shap_values_forecasting_phase3_phase4_colored.jpg', ...)` |
| `shap_values_grouped_horizontal_bars.jpg` | `Figure2_Feature_Importance_Forecasting.ipynb` | cell 6 / 37：`plt.savefig('produced_graph/shap_values_grouped_horizontal_bars.jpg', ...)` |
| `shap_values_nowcasting_phase3_phase4_L1.jpg` | `Figure2_Nowcasting_two_layer_feature_importance.ipynb` | cell 4 / 86：`plt.savefig(...)` |
| `shap_values_nowcasting_bar_phase3_phase4_L1.jpg` | `Figure2_Nowcasting_two_layer_feature_importance.ipynb` | cell 4 / 251–252：`plt.savefig(...)` |
| `shap_values_nowcasting_phase3_phase4_colored.jpg` | `Figure2_Nowcasting_two_layer_feature_importance.ipynb` | cell 6 / 90：`plt.savefig(...)` |
| `shap_values_grouped_horizontal_bars_nowcasting.jpg` | `Figure2_Nowcasting_two_layer_feature_importance.ipynb` | cell 7 / 37：`plt.savefig(...)` |

前两张各有根目录和 `produced_graph/` 两个版本，字节不同。Notebook 来源已按
作者说明确认；未将这两个位置的文件描述为同一份精确导出，也未覆盖任一版本。

## 有 writer、但需要保留的路径和版本说明

- `1.Source Data/r2_frame_forecasting.csv`：对应
  `Figure2_Feature_Importance_Forecasting.ipynb` cell 3 / 5。
- `1.Source Data/r2_frame_nowcasting.csv`：对应
  `Figure2_Nowcasting_two_layer_feature_importance.ipynb` cell 3 / 4。
- `1.Source Data/r2_frame_cv.csv`：对应
  `Table1_Contemporaneous_main.ipynb` cell 2 / 6。

三处当前 `to_csv` 都写到 `produced_graph/`，不会直接覆盖 `1.Source Data/`
中的旧文件。Forecasting 和 Nowcasting 两份旧表仍被
`Figure1_multiple_figures.ipynb`（零基 cell 1，第 15–16 行）、
`generate_phase_cumulative_scatter_comparison.py`（43–44、1347 行）以及
`run_replication.py`（34–35 行）显式使用，因此保留原位。
`r2_frame_cv.csv` 没有当前读取入口，已一并归档。
没有验证旧文件与当前代码的数值一致性。特别是 `r2_frame_cv.csv`
只有 1,115 行；notebook 在每折重置预测容器，末尾导出最后一折，不能用它替代
5,575 行完整 CV 预测，也不能替代刚恢复的 random-area-CV 结果。

`produced_graph/missingness_sensitivity/archive/2026-09-03_pre_54row_rerun/`
中的三个旧文件对应历史版本 `1fa8c42` 的
`generate_missingness_sensitivity.py`：`write_outputs` 在 1210–1216 行写 CSV、
1219–1224 行写 PDF、1227–1232 行写 PNG。旧 CSV 与该提交产物逐字节一致；
旧 PDF/PNG 与该提交中的图不同，精确图形版本未确认。当前工作树代码只写修订后的
54 行 removal CSV 和单独的 9 行 indicator CSV，不再保存旧图。

## 运行依赖边界

已检查的活动结果族未发现因此次归档而缺少直接上游文件，但这不是全流程执行通过。
以下默认输入在当前机器存在，却位于 replication package 之外：

- `generate_2022_alert_map_panel.py` 使用上级 codespace 的
  `0.Archived/New_analysis_dataset_for_vis/world_analysis.shp`。
- `generate_country_phase3plus_population.py` 使用上级 codespace 的
  `0.Archived/new_merge_0107.csv`。
- `generate_persistence_baseline_comparison.py` 使用上级 codespace 的
  `0.Archived/new_merge_0108_with_country_code.csv`。

这些结果有生成脚本，但单独拷贝 replication package 未必具备全部运行输入。
另有 `1.Source Data/all_predictions_severity.ipynb` 为零字节占位文件，没有可执行代码；
它不是结果，也不能作为任何结果的生成入口。
