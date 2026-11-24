# PlotLab 绘图工作台

**一句话**：把数据拖进 DeepSeek Harness，说一句「按分组画箱线图比较 score 并标显著性」，自动剖析+清洗+出图+统计；字体、字号、颜色在页面里实时调；最后带走 300 dpi 图片、可复现的 Python/R 代码和一份报告。

## 界面

- **左栏**：数据集卡（行列/类型/来源/列清单）→ 列角色映射（X/Y/分组/颜色/大小/时间/事件）→ **AI 需求**输入框 → 数据质量分（0–100）与逐项问题、一键清洗、推荐图表。
- **中栏**：22 种图表画廊（比较 / 关系 / 分布 / 统计）→ 图表参数 → **样式面板**（字体、标题/轴标题/刻度/图例字号、配色板、逐组自定义颜色、主题、画布尺寸、DPI、网格、边框、刻度方向、标签旋转、对数轴、坐标范围、图例位置、显著性标注样式与多重校正、误差线、点大小/线宽/柱宽）。
- **右栏**：实时预览（300 dpi，改动即重绘）+ 统计面板 + Python/R 代码 + 历史记录 + 确定性自检。

把数据文件直接拖到页面任意位置即可导入；也支持对话里让 Agent 用 `plotlab` 工具做同样的事（页面按钮和 Agent 工具走同一套实现）。

## 四条核心能力

1. **拖数据 + 一句话需求 → 自动分析清洗**
   上传 CSV/TSV/TXT/XLSX/JSON/GZ/ZIP（自动识别编码/分隔符/sheet，自动判断长表 tidy 还是 基因×样本矩阵，自动识别列类型与分组列），页面「AI 需求」解析图表/分组列/统计方法/配色/字号/画布/清洗等关键词，自动执行清洗（空列、重复行、数字文本、缺失填补、异常值缩尾、log/标准化……每步都有中文理由），再选图、出图、做统计。需求会写进工作台状态，Agent 在对话里能直接看到并接手。
2. **页面调字体/字号/颜色**
   样式面板的所有改动（字体、四类字号、配色板、逐组颜色、透明度、主题、画布、图例、显著性标注…）实时重渲染预览图；一张图一个 `style` 字典，可随导出代码一并带走。
3. **辅助分析 + 统计分析**
   剖析给出质量分与检查项（缺失、重复、常量列、ID 列、数字文本、异常值、分组均衡、样本量、共线、负值、规模）；统计含描述统计、组间比较（自动 Welch t / Mann-Whitney / ANOVA / Kruskal + 事后两两比较 + 效应量 + 多重校正）、前提检查（Shapiro、Levene）、相关（Pearson/Spearman/Kendall）、线性/逻辑回归、卡方/Fisher、正态性、ROC/AUC（bootstrap CI + Youden 阈值）、生存（KM + log-rank + Cox HR）、PCA、多重检验校正。
4. **输出含关键步骤代码和图片**
   每张图都生成「读取数据 → 重放清洗步骤 → 作图 → 统计检验」的完整 Python 脚本（另有 ggplot2 片段）；导出支持 PNG（默认 300 dpi，按需 PDF/JPG）、`.py`/`.R`、`report.md`、英文 `methods.md`、`protocol.json`（溯源：软件版本、参数、清洗记录）、清洗后 CSV、统计表 CSV、以及一键 `bundle.zip`。

## 22 种图表

| 组 | 图表 |
|---|---|
| 比较 | 柱状图（均值/中位数/求和 ± SD/SE/CI95，可叠散点）、柱散点图、箱线图、小提琴图、雨云图、点图（颜色+大小编码）、堆叠柱、饼/环图 |
| 关系 | 散点图（回归线 + CI + 相关系数）、气泡图、折线图（误差带）、面积图、热图（z-score/聚类/数值标注）、相关矩阵（星号显著性）、散点矩阵 |
| 分布 | 直方图、密度图、QQ 图 |
| 统计 | ROC 曲线、KM 生存曲线、森林图、PCA 散点（置信椭圆/方差解释） |

## Agent 工具

工具名 `plotlab`，典型流程：

```text
plotlab action=open path=/path/to/data.csv      # 或 action=demo name=clinical
plotlab action=profile                          # 质量分 + 问题 + 推荐
plotlab action=auto need="按 group 分组画箱线图比较 score 并标显著性，配色用色盲友好"
plotlab action=stat stat=compare                # 统计分析
plotlab action=export what=bundle               # 图 + 代码 + 报告 + 数据 zip
```

其它 action：`inspect / preview / clean / reset_data / roles / charts / plot / code / records / audit / suggest / need / samples / formats / reset / engine_restart`。返回的 `figure.file` 是 PNG 绝对路径，可直接用 `read_image` 查看。

## 安装

见 [INSTALL.md](INSTALL.md)。简版：

```bash
# 在 DeepSeek Harness 里
plugin_manager install_bundle target=/absolute/path/to/plugins/dsh-plotlab
```

插件依赖本机 Python 3 及 numpy / pandas / scipy / scikit-learn / matplotlib / seaborn / statsmodels / lifelines / openpyxl（脚本会自动探测解释器并给出可读错误）。

## 已知限制

- 图片默认 PNG；PDF/JPG 是位图转换，SVG 需要矢量重绘，暂不提供。
- 生存/ROC/回归等统计需要对应的列（时间+事件、0/1 标签、数值列），不满足时会明确报错或给出警告而不是编造结果。
- 图内文字默认英文（跨机器字体兼容）；需要中文可在样式里选 `Arial Unicode MS`/`苹方` 等中文字体，本机已安装。
- 单图渲染目标是 1.5 秒内（demo 规模）；超大矩阵会自动取方差 top 特征并在提示里说明。

## 目录

```
index.js             Host 入口：/plotlab 路由 + plotlab 工具 + prompt section
lib/impl.mjs         HTTP / Agent 工具共用的操作实现
lib/engine.mjs       Python worker 管理（探测/排队/超时/重启）
lib/pl_engine.py     worker 主循环（JSON-lines）
lib/pl_registry.py   命令分发：数据集/图表/统计/需求解析
lib/pl_common.py     结果信封、风格系统、调色板、出图
lib/pl_prep.py       角色推断与 tidy/wide 数据装配
lib/pl_state.py      数据集容器与内置示例
lib/pl_clean.py      剖析 + 清洗
lib/pl_codesnippet.py 关键步骤代码生成（22 图族的 Python/ggplot2 片段兜底）
lib/pl_io.py         全格式导入
lib/pl_charts.py     22 种图表 + 代码片段
lib/pl_stats.py      统计分析
lib/pl_codegen.py    完整可复现脚本
lib/pl_export.py     图片/代码/报告/Methods/protocol/bundle
client.js            浏览器工作台 UI
```

## License

MIT
