# dsh-plotlab 内部契约（PlotLab 绘图工作台）

> 本文件是唯一事实来源。实现与契约冲突时以契约为准；确需改协议先改本文件。
> 参考实现（只读，不要改）：`../dsh-geobulk/`（Host 路由 + Python worker + agent 工具 + Client 整页三合一）。

## 0. 定位

在 DeepSeek Harness Web GUI 里提供一个「拖数据进来 → 说需求 → 自动剖析清洗 → 出图 → 页面调字体/字号/颜色 → 统计 → 带走图和关键代码」的绘图工作台，
网页形态参考 Hiplot / 仙桃（左侧图表画廊 + 中间参数表单 + 右侧实时预览）。

四条硬需求对应实现：

1. **拖数据 + 提需求自动分析清洗**：页面拖拽上传 CSV/TSV/XLSX/JSON → 后端 `profile` 剖析 → 页面「AI 需求」框输入一句话 → `auto` 命令解析需求（图表/分组/统计/风格关键词）→ 自动清洗 + 选图 + 出图；需求同时写入数据集，Agent 在对话里可直接接着做（prompt section 会带上）。
2. **页面调字体/字号/颜色**：`style` 全量入参化，页面任意样式改动 → `plot` 重渲染（PNG 300dpi），毫秒级刷新预览。
3. **辅助分析 + 统计分析**：`profile`（质量分 0-100 + 逐项问题 + 建议）、`clean`（缺失/异常/类型/去重）、`stat`（描述、组间比较、相关、回归、卡方、正态性、ROC、生存、PCA，含效应量与多重校正）。
4. **输出含关键步骤代码和图片**：每张图都返回 `code.python`（pandas+matplotlib/seaborn 可直接跑）与 `code.r`（ggplot2），`export` 可导出 PNG / 代码 / 清洗后数据 / 统计表 / report.md / methods.md / protocol.json / bundle.zip。

## 1. 目录与文件归属

```
plugins/dsh-plotlab/
  DESIGN.md                       # Lead  本契约
  package.json / cordis.patch.yml / icon.svg / locale/{zh,en}.json   # Lead
  index.js                        # Lead  Host 入口：/plotlab 前缀路由 + plotlab 工具 + prompt section
  lib/impl.mjs                    # Lead  唯一操作实现（HTTP + Agent 工具共用）
  lib/engine.mjs                  # Lead  Python worker 管理（探测/排队/超时/重启）
  lib/pl_engine.py                # Lead  worker 主循环 + 命令分发（协议 §7）
  lib/pl_registry.py              # Lead  调度：数据集状态、图表/统计注册、结果整形
  lib/pl_common.py                # Lead  EngineError / table / figure / result / 风格 / 调色板 / save_fig
  lib/pl_prep.py                  # Lead  角色解析 + 数据装配（tidy/wide）
  lib/pl_state.py                 # Lead  Dataset 容器、demo 数据、历史记录
  lib/pl_clean.py                 # Lead  profile 剖析 + clean 清洗
  lib/pl_io.py                    # io   全格式导入（csv/tsv/txt/xlsx/json/zip/gz/url）
  lib/pl_charts.py                # chart 22 个图表渲染 + 每图 python/R 代码片段
  lib/pl_stats.py                 # stat  统计分析（含显著性标注辅助）
  lib/pl_codegen.py               # Lead  整篇可复现脚本拼装（清洗步骤 + 图 + 统计）
  lib/pl_export.py                # Lead  export 产物（数据/表/报告/methods/protocol/bundle）
  client.js                       # Lead  浏览器 UI（Hiplot 式工作台）
  test/                           # Lead + 各模块自测
  examples/                       # demo 数据与示例图
```

模块之间**只允许** `import pl_common`、`import pl_prep`（图表/统计模块额外只允许 `pl_common`、`pl_prep`）；
**禁止** import `pl_registry` / `pl_engine` / `pl_state`（registry 反向 import 它们，会循环）。
`pl_charts.py` / `pl_stats.py` / `pl_io.py` 顶层 import 只允许：标准库 + numpy/pandas/scipy/sklearn/matplotlib/seaborn/lifelines/statsmodels/openpyxl + pl_common + pl_prep。

## 2. 数据模型（worker 内存）

```python
dataset = {
  "id": "ds1", "name": "expr.csv", "source": "/abs/path.csv",
  "kind": "tidy" | "matrix",         # tidy=一行一个观测；matrix=一行一个特征（基因/代谢物），其余列是样本
  "df": DataFrame,                   # 当前（清洗后）数据
  "raw_df": DataFrame,               # 导入后未清洗数据（类型已规整）
  "columns": [ {"key","dtype","n_unique","missing","missing_pct","examples",...} ],  # dtype ∈ numeric/category/datetime/text
  "roles": {"x":None,"y":None,"group":None,"color":None,"size":None,"facet":None,"time":None,"event":None},
  "group_col": None|str, "groups": [..]|None,
  "cleaning": [ {"op","params","rows_before","rows_after","notes"} ],
  "history": [ {"id","chart","params","style","roles","roles_label","figure_id","at"} ],
  "need": "" | "用户最新一句话需求",
  "notes": [], "warnings": [],
  "created_at": iso, "steps": [..]
}
```

- `df` 的行/列定义随 `kind` 变；所有图与统计都通过 `pl_prep.build_data(dataset, chart_id, params, roles)` 拿标准装配结果（§3），**不要直接猜列**。
- matrix 模式：`df.iloc[:,0]` 是特征 id 列，其余列（可多选）是样本；样本名 = 列名。可挂样本表（列：sample, group, ...）由 `pl_io` 合并到 `sample_info`。
- 时间/生存列在 tidy 模式按列选取；matrix 模式不做生存。

## 3. 角色与数据装配（pl_prep）

```python
ROLE_KEYS = ("x","y","group","color","size","facet","time","event","id")

def infer_roles(dataset, chart_id=None) -> dict      # 图表相关的默认角色
def recommend_roles(dataset, chart_id) -> dict       # UI/auto 用：最合理的默认映射
def build_data(dataset, chart_id, params, roles) -> dict
```

`build_data` 返回（图表/统计统一消费这个 dict）：

```python
{
  "kind": "tidy"|"matrix",
  "tidy": DataFrame | None,   # 标准列：__x,__y,__group,__color,__size,__facet,__time,__event,__id
                              # 只含 chart 声明需要的列；需要列缺失→bad_request；行内有缺失→丢弃并记 n_dropped
  "labels": {"x":"原始列名","y":..,"group":..,"color":..,"size":..},
  "levels": {"group":[...], "color":[...], "facet":[...]},   # 保序（类别按首次出现；数值分组按数值）
  "n_dropped": int,
  "wide": {                    # 仅 matrix 数据集
     "matrix": np.ndarray (n_samples, n_features), "features": [...], "samples": [...],
     "sample_groups": np.ndarray|None, "group_name": str|None,
  } | None,
  "numeric_cols": [...], "categorical_cols": [...],
  "dataset": {"id","name","kind","n_rows","n_cols"},
  "params": {...}, "roles": {...},
  "notes": [...],
}
```

角色语义：
- `x`：分类/数值横轴；`y`：数值纵轴；`group`：分组（颜色/分面/统计分组）；`color`：第二分组或连续着色；`size`：气泡大小（数值）；`facet`：分面；`time`：时间轴；`event`：生存事件 0/1。
- 图表元数据声明 `roles: {"x":"required"|"optional"|"none", ...}`，`needs` 声明 `tidy` 或 `wide`。
- matrix 图表（heatmap/corr/pca/volcano?）只用 `wide`；其余用 `tidy`。

## 4. 风格 schema（全图表共用）

```python
STYLE_DEFAULT = {
  "font": "auto",            # auto|Arial|Helvetica|Times New Roman|PingFang SC|Songti SC|Hiragino Sans GB|Arial Unicode MS|Georgia|DejaVu Sans
  "title_size": 14, "label_size": 12, "tick_size": 10, "legend_size": 10, "text_size": 10,
  "title_bold": True,
  "fig_width": 6.2, "fig_height": 4.4, "dpi": 300,
  "theme": "classic",        # classic|minimal|grid|void|nature|lancet|jama
  "palette": "okabe_ito",    # okabe_ito|npg|lancet|jama|set1|set2|tableau|viridis|blue_red|grey|pastel|custom
  "colors": {},              # {分组名: "#RRGGBB"} 覆盖调色板
  "alpha": 0.85, "line_width": 1.4, "point_size": 28, "bar_width": 0.72, "edge_width": 0.8,
  "grid": "none",            # none|x|y|both
  "frame": "left_bottom",    # left_bottom|none|full  （轴线保留哪几条）
  "tick_dir": "out",
  "legend_pos": "right",     # right|top|bottom|none
  "legend_title": "",
  "title_text": "", "x_label": "", "y_label": "",   # 空=自动
  "x_rotate": 0, "y_rotate": 0,
  "x_log": False, "y_log": False,
  "x_lim": [None, None], "y_lim": [None, None],
  "error": "se",             # sd|se|ci95|none（bar/line）
  "jitter": True, "jitter_size": 2.2, "jitter_alpha": 0.55,
  "stat_annot": True,        # 图上有组间比较时是否标显著性
  "stat_method": "auto",     # auto|welch|student|wilcox|anova|kruskal
  "p_adjust": "bh",          # none|bh|bonferroni
  "annot_style": "stars",    # stars|p|both
  "show_n": False,           # 刻度/图例带 n=
  "legend_inside": False,
}
def resolve_style(style) -> dict      # 类型/范围校验 + 补默认，非法→bad_request
def style_rc(style) -> dict           # rcParams（供 plt.rc_context）
def font_family(style) -> str
def palette_colors(style, levels) -> dict[level, hex]
```

render 期间统一用 `with pl_common.style_context(style):` 包住建图。

## 5. 图表目录（pl_charts）

每个图表：

```python
{"id":"box","title":"箱线图","group":"比较","desc":"...",
 "needs":"tidy","roles":{"x":"required","y":"required","group":"optional",...},  # 未写=不支持
 "params":[{"key","label","type","default",...}],   # type: number|select|boolean|text
 "stat":True,                                        # 是否自动带组间显著性
 "defaults":{"x":...}}                               # 由 pl_prep.recommend_roles 处理
```

`pl_charts.py` 导出：

```python
CHARTS = [...]                                  # 上面结构，顺序即 UI 顺序
def render(chart_id, data, params, style, outdir) -> dict
def python_code(chart_id, data, params, style) -> str     # 只画图那一段（含数据准备），可独立运行
def r_code(chart_id, data, params, style) -> str          # ggplot2 片段
```

`render` 返回：

```python
{
  "figures": [ {"id","title","file","width","height","bytes"} ],   # pl_common.save_fig 产物
  "summary": {"bullets": ["一句话结论", ...], "key_numbers": {"n":..., ...}},
  "tables": [pl_common.table(...), ...],
  "warnings": [...],
  "stats": {...} | None,       # 供图注/报告复用（如 {"method":"Welch t","p":0.001,"effect":"d=0.8"}）
  "annotations": {...},
}
```

图表清单（id / 中文名 / 组 / needs / 关键角色）：

比较类
1. `bar` 柱状图 tidy x,y,group（聚合 mean/median/sum + 误差线 + 可选散点）
2. `bar_dot` 柱散点图 tidy x,y,group
3. `box` 箱线图 tidy x,y,group（+jitter）
4. `violin` 小提琴图 tidy x,y,group（+箱）
5. `raincloud` 雨云图 tidy x,y,group（半小提琴+箱+散点）
6. `dotplot` 点图 tidy x,y,color,size（分类矩阵点图）
7. `stack` 堆叠柱 tidy x,y,group（比例/绝对值）
8. `pie` 饼图/环图 tidy x,y

关系类
9. `scatter` 散点图 tidy x,y,color,size（回归线+CI+相关注释）
10. `bubble` 气泡图 tidy x,y,size,color
11. `line` 折线图 tidy x,y,group（误差带/点）
12. `area` 面积图 tidy x,y,group
13. `heatmap` 热图 matrix（zscore/聚类/数值标注）
14. `corr` 相关矩阵 matrix 或 tidy 多数值列（Pearson/Spearman + 星号）
15. `pairplot` 散点矩阵 tidy（数值列 ≤6，按 group 着色）

分布类
16. `hist` 直方图 tidy x,color（分组叠加/密度）
17. `density` 密度图 tidy x,color
18. `qq` QQ 图 tidy x（正态性）

统计类
19. `roc` ROC 曲线 tidy y(分数),group(0/1 标签) 或 params 指定
20. `km` 生存曲线 tidy time,event,group
21. `forest` 森林图 tidy（或由 stat 结果传入；HR/OR/β + CI）
22. `pca` PCA 散点 matrix（按分组着色 + 置信椭圆 + 碎石图副图）
23. `volcano` 火山图 tidy（需 logfc/p 列或由两组宽表算；明确 prerequisites，缺列→明确报错）

默认参数（图内实现细节由图表模块定，但必须支持 style 的所有键）。

## 6. 统计目录（pl_stats）

```python
STATS_CATALOG = [ {"id","title","group","desc","params":[...],"roles":{...}} ]
def run(stat_id, dataset, data, params, style) -> {"title","method","summary":{"bullets":[],"key_numbers":{}},
                                                   "tables":[table...], "figures":[figure...]|[],
                                                   "report":"markdown 片段", "annotation":{...}|None,
                                                   "warnings":[]}
```

必做 id：
- `describe` 描述统计（n/mean/sd/median/IQR/min/max/missing；按分组）
- `compare` 组间比较（自动选检验：2 组 Welch t / Mann-Whitney；>2 组 ANOVA / Kruskal；paired 可选配对；事后两两比较 + 多重校正；效应量 Cohen's d / eta² / rank-biserial；返回 annotation 供图上标星）
- `assumption` 前提检查（Shapiro-Wilk 正态性、Levene 方差齐性、样本量、异常值比例、建议用参数/非参数）
- `corr` 相关性（Pearson/Spearman/Kendall 矩阵 + p 值 + 显著性星号 + 强相关对 top 表）
- `regression` 线性回归 OLS（y ~ x 多列；系数/SE/t/p/CI/R²/adj R²/F；可选标准化系数；二分类 y 自动 logistic → OR）
- `chi2` 列联表卡方/Fisher + Cramér's V + 期望频数警告
- `normality` 单变量正态性（Shapiro/Wilks、偏度峰度、QQ 图）
- `roc` ROC + AUC + 95%CI（bootstrap 或 DeLong 近似）+ 最佳阈值（Youden）+ 曲线图
- `survival` KM + log-rank + Cox HR（单变量/多变量）+ 生存曲线图
- `pca` PCA（解释方差、载荷、得分图）
- `adjust` 多重检验校正（BH/Bonferroni/Holm）

数值约定：p 值保留有效位数（`p<1e-4` 用科学计数），表格列 `{"key","label","type"}` + `rows` 为 dict 列表；所有输出必须 `jsonable`。

## 7. worker 协议（pl_engine.py）

启动：`python3 -u lib/pl_engine.py --workdir <abs_dir>`，`cwd=lib/`；stdin/stdout JSON-lines。
成功 `{"id","ok":true,"result":{}}`；失败 `{"id","ok":false,"error":{"code","message","trace"?}}`；进度 `{"id","event":"progress","message","pct"}`。
stdout 只允许协议行；日志走 stderr。`code` ∈ `bad_request|not_found|prerequisite|unsupported|ambiguous|internal|timeout|no_python|engine_exited|busy|disposed`。

命令表（args 都是 dict；省略 dataset=活动数据集）：

| cmd | args | 返回 |
|---|---|---|
| `ping` | {} | `{python,platform,workdir,packages:{numpy,pandas,scipy,sklearn,matplotlib,seaborn,statsmodels,lifelines,openpyxl},fonts:[...],charts:[...],stats:true,capabilities:{charts,stats,formats}}` |
| `open` | {path?,url?,name?,data_format?,kind?,delimiter?,sheet?,header?,index_col?,sample_sheet?,feature_col?,transpose?} | `{dataset,preview}` |
| `inspect` | 同上探测子集 | `{format,n_rows,n_cols,kind,columns:[...],candidates:[],notes:[]}` |
| `demo` | {name?} | `{dataset,preview}` |
| `list` / `select` / `close` / `reset` | {id?} | `{datasets,active}` |
| `preview` | {dataset?,limit?} | `{dataset,columns,n_rows,n_cols,rows:[...],roles,group_col,groups,cleaning,need,history}` |
| `profile` | {dataset?} | `{score,level,checks:[{id,level,title,detail,fix?}],columns,summary,numeric_stats,group_summary,corr,recommend:{charts:[],stats:[]}}` |
| `clean` | {dataset?,plan:[{op,params}]?,auto?:true} | `{cleaning,report:{applied:[],skipped:[],warnings:[]},preview}` |
| `roles` | {dataset?,roles} | `{roles,recommend}` |
| `charts` | {dataset?} | `{charts:[...],styles:{...元数据},themes,palettes,fonts}` |
| `plot` | {dataset?,chart,params,style,roles,code?:true} | `{figure,chart,params,style,summary,tables,warnings,code:{python,r}}` |
| `stat` | {dataset?,stat,params,roles} | §6 run 返回 |
| `auto` | {dataset?,need} | `{plan:{cleaning,chart,params,style,roles,stats},figure,...,need}` |
| `code` | {dataset?,what:"plot"|"session",chart?,params?,style?,roles?} | `{python,r,filename}` |
| `export` | {dataset?,what,png_path?,out_path?} | `{file,dir,files,bytes}` |
| `records` | {dataset?} | `{history}` |
| `audit` / `suggest` | {dataset?} | 同 geobulk 语义 |

## 8. HTTP / Agent 工具（impl.mjs）

前缀 `/plotlab`，只接受回环请求：
`GET /health|/state|/charts|/stats|/figure?id=|/activity`，`POST /dataset/open|/dataset/upload|/dataset/demo|/dataset/select|/dataset/close|/preview|/profile|/clean|/roles|/plot|/stat|/auto|/code|/export|/need|/reset|/engine/restart`。

Agent 工具名 `plotlab`，actions 与上表命令同名（外加 `charts/stats/export/records/audit/suggest`），参数直接映射；`plot` 返回 `figure.file` 绝对路径，Agent 可直接 `read_image`。
prompt section：数据集打开时报告 name/shape/kind/已做清洗/最新 need/建议下一步。

## 9. 代码生成与导出（pl_codegen / pl_export）

`python_code(chart...)` 返回片段，`pl_codegen.build_script(dataset, chart, params, style, roles, stats)` 输出完整可跑脚本：

```python
# ==== 1. 读取数据 ====
df = pd.read_csv(r"...", sep=",", encoding="utf-8")
# ==== 2. 数据清洗（与工作台一致）====
# [clean] drop_empty ...
df = df.dropna(...)
# ==== 3. 统计 ====
...
# ==== 4. 作图 ====
...
plt.savefig("plotlab_figure.png", dpi=300, bbox_inches="tight")
```

export what ∈ `figure|code|report|methods|protocol|cleaned|stats|bundle`；`bundle`=zip（图+代码+报告+数据+统计表）。
图默认 PNG 300dpi；用户明确要 PDF/SVG 时才给（可加 `format`）。

## 10. 自测（提交前必须跑）

```bash
cd plugins/dsh-plotlab/lib && python3 -m py_compile *.py
cd .. && python3 test/py_plotlab_test.py      # 覆盖：导入(2类)→profile→clean→全部图表各渲一次→样式改动→统计→代码→导出
```
测试离线、确定、5 分钟内跑完、打印 `PASS` 且 exit 0。图内文字默认英文（服务端字体虽支持中文，但跨机器会方块；用户显式选中文/中文需求时用中文）。

## 11. 环境事实

- python3 = /Library/Frameworks/Python.framework/Versions/3.14/bin/python3；已装 numpy/pandas/scipy/sklearn/matplotlib/seaborn/statsmodels/lifelines/openpyxl。
- matplotlib 必须 Agg；`~/.matplotlib` 不可写 → 设置 `MPLBACKEND=Agg`、`MPLCONFIGDIR=<可写临时目录>`。
- 字体：Arial / Helvetica / Times New Roman / Georgia / DejaVu Sans / PingFang SC / Songti SC / Hiragino Sans GB / Arial Unicode MS 可用（本机）。
- 不联网（除用户显式 URL 导入）；不安装新依赖。
