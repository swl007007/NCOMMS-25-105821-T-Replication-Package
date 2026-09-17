# Artifact 生成环境索引（2026-09-17）

## 结论与证据口径

本说明聚焦当前需要复现的结果及本次指定重跑的普通 LOCO。保留两组版本组合：
A（Python 3.11.3 / XGBoost 2.0.3，当前结果）和 C（Python 3.12.3 / XGBoost 3.2.0，仅历史 LOCO）。

本索引于 2026-09-16 调查后整理，并补入同日两套 LOCO 的 A 环境完整重跑。
2026-09-17 按作者决定将两套 A 环境 LOCO 结果及 checkpoint 启用至 produced_graph 根目录；旧 strict 结果及旧 checkpoint 归档至 0.Archived/2026-09-17_pre_windows203_loco/，旧 ordinary 结果保留在原归档。source audit 的版本字段是运行记录；
代码中的 EXPECTED_ENVIRONMENT、冻结 manifest 和 lock 文件是复现契约；
运行记录与复现契约分别标注。
LOCO 重跑前后已核对受保护结果、输入和生成器哈希；没有将其扩展为全包逐产物复现验证。

## 两组版本组合

| 组件         | A：正式冻结主结果及多数扩展分析                                       | C：原 LOCO 结果                                                  |
| ------------ | --------------------------------------------------------------------- | ---------------------------------------------------------------- |
| Python       | 3.11.3                                                                | 3.12.3                                                           |
| NumPy        | 1.26.4                                                                | 2.4.4                                                            |
| pandas       | 2.2.3                                                                 | 3.0.3                                                            |
| scikit-learn | 1.5.2                                                                 | 1.8.0                                                            |
| XGBoost      | 2.0.3                                                                 | 3.2.0                                                            |
| SciPy        | 1.17.1，部分 audit / 冻结契约记录                                     | 本次所查 audit 未记录                                            |
| statsmodels  | 0.14.6，simple baseline audit / 扩展契约记录                          | 未记录                                                           |
| matplotlib   | 3.10.1，部分 audit / 图像记录                                         | strict LOCO audit 为 3.10.9                                      |
| 平台         | 冻结契约和 rescue 等 audit 明确 Windows；部分 A 类 audit 没有平台字段 | strict temporal LOCO 明确 WSL2 Linux；旧普通 LOCO audit 未记平台 |

A/C 是本文的版本组合简称，不是原 audit 的 environment_id。
某族归入 A，不表示它自己的 audit 记录了 A 表中所有组件，也不表示所有任务都使用 XGBoost。
线程数、random_state 和评估协议仍以各生成器/audit 为准；不能仅凭包版本合并结果。
例如 frozen main 使用 default threads / no explicit seed，而 direct rescue 有 seed0 / njobs1 契约。

## 活动结果族与证据

下表路径除另注明外均相对 `2.Source Code/produced_graph/`。
“运行记录”指 source audit 直接记录的版本；不代表已完整重跑通过。

| 结果族                                                                                         | 环境归属 / 证据等级          | 证据及限制                                                                                                                                                                                                     |
| ---------------------------------------------------------------------------------------------- | ---------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `main_result_figure1_v1.json/.csv`                                                           | A，冻结契约                  | JSON 的 environment 和 generator_environment_extensions；生成代码`main_result_figure1_v1.py:41–64,252–261` 序列化常量。主结果来自指定 notebook stored outputs，不是该 JSON 本身重新采集了历史 kernel 环境  |
| `leave_area_out_20pct_random_cv_*`                                                           | A，运行记录                  | `leave_area_out_20pct_random_cv_source_audit.csv` 的 python/numpy/pandas/sklearn/xgboost_version；未记录平台和 SciPy。已恢复的完整六文件族，Nowcasting 行被选作 contemporaneous random-area CV，不能列为废弃 |
| `leave_area_out_20pct_fivefold_*`                                                            | A，运行记录                  | `leave_area_out_20pct_fivefold_source_audit.csv` 同上；这是 strict-temporal new-area fivefold，区别于普通 random-area CV                                                                                     |
| `all_prediction_contemporaneous_random_cv_*`                                                 | A，部分版本运行记录          | 同名 source audit 记录 Python/NumPy/pandas/XGBoost 和 Windows 平台，未记录 sklearn；这是旧 row-level CV，与所选 random-area CV 不同                                                                            |
| `spatial_feature_comparison_*`、`spatial_feature_interpolation_*` 及对应比较图             | A，运行记录                  | `spatial_feature_comparison_source_audit.csv`；Python/NumPy/pandas/sklearn/XGBoost/matplotlib，四条件 × 三模型，并列出产物路径/哈希                                                                         |
| `multinomial_baseline_*`                                                                     | A 的数据/绘图组件，运行记录  | `multinomial_baseline_source_audit.csv`；Python/NumPy/pandas/sklearn/matplotlib；未记录 XGBoost，不应推断用了它                                                                                              |
| `persistence_baseline_*`                                                                     | A 的数据/绘图组件，运行记录  | `persistence_baseline_source_audit.csv`；Python/NumPy/pandas/sklearn/matplotlib；未记录 XGBoost                                                                                                              |
| `simple_baseline_comparison_*`、`phase3plus_precision_recall_simple_baseline_comparison.*` | A 的统计/绘图组件，运行记录  | `simple_baseline_comparison_source_audit.csv`；Python/NumPy/pandas/SciPy/sklearn/statsmodels/matplotlib，亦引用已保存的 persistence/multinomial/spatial 结果；并非所有行都由同一进程重新拟合                 |
| `phase_cumulative_*` 散点图族                                                                | A，运行记录 + 上游记录       | `phase_cumulative_scatter_source_audit.csv` 的 F/N 行记录 Python/NumPy/pandas/sklearn/XGBoost，并有 formal_environment_passed；Contemporaneous 行是复制的上游 audit，不是绘图进程重新采集的环境              |
| `strict_temporal_loco_*`                                                                     | A，运行记录         | 当前 source audit 记录 A 核心组合及 Windows；旧 C 结果已归档，比较证据见下节                                                                              |
| `phase4_rescue_classifier/*`                                                                 | A，运行记录                  | `phase4_rescue_source_audit.csv`；Windows、Python/NumPy/pandas/SciPy/sklearn/XGBoost/matplotlib、XGBoost DLL hash，run_status=complete。配置中的硬编码版本不是独立现场记录                                   |
| `direct_phase3_vs_phase45_rescue/*`                                                          | A，运行记录                  | 同名 source audit；同上，reference_environment_id 为 windows_py3113_xgb203_direct_seed0_njobs1                                                                                                                 |
| `direct_phase3_vs_phase45_rescue_contemporaneous/*`                                          | A，运行记录                  | 同名 source audit；同上；正式入口检查 environment                                                                                                                                                              |
| `conflict_perturbation_10pct/*`                                                              | A，经校验契约                | `audit.json`：environment 复制 EXPECTED_ENVIRONMENT，入口先 validate_environment；platform 是现场 Windows 值。status=exploratory_completed，不是正式主结果                                                   |
| `selected_figure1_repeated_area_refit_metrics.csv`                                           | A refit 契约；CSV 为混合来源 | 两个 refit 行 metric_source 嵌入冻结 environment_id；当前生成器 validate_environment 强制 A。CSV 也含 baseline 和 frozen reference 行，没有逐组件运行版本字段                                                  |

普通 LOCO 活动族 `leave_one_country_out_*` 及对应 precision-recall 图同属 A，运行版本见其 source audit。

## 历史归档与 checkpoint

以下 A8 指 `0.Archived/2026-09-08_obsolete_presentation/2.Source Code/produced_graph/`。

| 历史族                                             | 记录环境 / 边界                                                                                                 |
| -------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| A8 的 leave_one_country_out                        | 原结果 C 核心版本；平台未记录。按作者要求在 A 下完整重跑，新结果已启用，比较证据及历史归档保留            |
| A8 的 leave_area_out_20pct_random_cv 备份          | A；与已恢复的活动 source audit 一致，活动族保留                                                                 |
| .leave_one_country_out_checkpoints（29 manifests） | 记录输入/代码 hash、seed、protocol，无软件版本字段                                                              |
| .strict_temporal_loco_checkpoints（27 manifests）  | 另记录 cutoff、n_jobs、严格训练边界等，仍无软件版本字段；不可自动用最终 source audit 给每份 checkpoint 认证环境 |

## LOCO 在 A 环境下的重跑验收（2026-09-16）

保持 seed=0、每个 estimator n_jobs=1、workers=4；全量国家，使用全新 checkpoint。
验收采用未舍入绝对差值 ≤0.001；指标显示三位小数。双方缺失视为同为未定义，
单方缺失视为失败；没有筛选或缩减评估样本。主表与详细表分别判定。

| 协议            | 主表（8 单元格） | 详细表                 | 最大主表绝对差 |
| --------------- | ---------------- | ---------------------- | -------------: |
| strict_temporal | FAIL：8/8 超界   | FAIL：1027/5168 不通过 |       0.014842 |
| ordinary        | FAIL：8/8 超界   | FAIL：156/232 不通过   |       0.036668 |

strict temporal 每模型 1,170 行、646 地区、27 国；普通 LOCO 每模型 5,575 行、1,198 地区、29 国。
完整预测键和真值保持一致，新主表已从保存预测独立复算。原结果及 10% holdout 哈希不变。

结果、运行版本、命令、日志和可重跑比较脚本见
[LOCO 重跑比较](<2.Source%20Code/produced_graph/loco_windows203_rerun_20260916/README.md>)。
全精度差值见同目录 metric_comparison.csv，统计见 comparison_summary.json。

限制：当前共用 LOCO 脚本与原审计记录的 hash 不同，未找到原脚本快照。
因此这是当前代码在 A 环境下对旧结果的再现检查，不能把差异纯归因于环境。
两套重跑结果现为活动 LOCO 结果；普通 LOCO 旧原件保持归档。普通 LOCO 的 Nowcasting 行对应 contemporaneous results（using Nowcasting dataset）。
活动 checkpoint 为本次 A 重跑副本；上表所述旧 checkpoint 已移至 2026-09-17_pre_windows203_loco。
迁移原路径、活动副本与 SHA-256 见该归档的 promotion_manifest.json；source audit 保留原始 rerun 路径。

## 当前可用解释器与使用建议

2026-09-16 同会话实际核验：

- Windows A 核心包可正常导入，解释器为
  `C:\Users\swl00\AppData\Local\NCOMMSFigure1\venvs\xgb203-generators\Scripts\python.exe`。
  Python 3.11.3、NumPy 1.26.4、pandas 2.2.3、SciPy 1.17.1、
  sklearn 1.5.2、XGBoost 2.0.3、statsmodels 0.14.6 已核验。
- WSL `/home/swl007007/.venvs/ipcch-geo/bin/python` 的已安装
  NumPy/pandas/sklearn/XGBoost 版本与 C 核心组合一致；不是 A 的替代品。
  artifact audit 通常没有 executable 字段，因此版本匹配不证明历史进程一定来自这个路径。
- 仓库目前无 .venv；裸 python3 / pytest 指向 /usr/bin，系统 Python 缺
  xgboost、statsmodels、shap，且 NumPy/pandas 等也不同于 A。
- requirements.txt 是通用兼容范围，不是全部 artifacts 的统一锁文件。
  `2.Source Code/main_result_figure1_v1_environment_lock.txt` 只针对 A 的冻结/扩展环境，
  不包含全部 notebook/地理/SHAP 工作流依赖，不覆盖 C。

复现时先按 artifact 的 source audit / 冻结契约选择环境，再核对种子、线程及评估口径。
手动导出图表不再要求追补历史环境；当前两套 LOCO 均选用 A 环境重跑结果。
