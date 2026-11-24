# 安装 PlotLab 绘图工作台

## 1. 前置条件

- DeepSeek Harness（桌面版或 Web）。
- 本机 Python 3（3.9+；开发机为 3.14），并有：
  `numpy pandas scipy scikit-learn matplotlib seaborn statsmodels lifelines openpyxl`
  检查：
  ```bash
  python3 -c "import numpy,pandas,scipy,sklearn,matplotlib,seaborn,statsmodels,lifelines,openpyxl;print('ok')"
  ```
  缺包时插件不会崩，页面会显示可读的引擎错误（`action=status` / 页面顶栏「引擎异常」可看原因）。

## 2. 安装插件

在 DeepSeek Harness 里（Agent 或插件管理）：

```
plugin_manager install_bundle target=/absolute/path/to/plugins/dsh-plotlab
plugin_manager list_bundles
```

或者手动把插件目录链接进 profile：

```bash
# 以 desktop profile 为例
ln -s "/absolute/path/to/plugins/dsh-plotlab" ~/.dsh/profiles/desktop/node_modules/dsh-plotlab
```

并在 `~/.dsh/profiles/desktop/package.json` 的 `dependencies` / `dsh.profile.bundles` 里加上 `dsh-plotlab`，
然后重启 DeepSeek Harness。

安装后：左侧栏出现「绘图」入口，主区域可切换到 PlotLab 整页；对话里出现 `plotlab` 工具。

## 3. 验证

```bash
# 页面路由（把 57105 换成你实际 GUI 端口）
curl http://127.0.0.1:57105/plotlab/health
curl http://127.0.0.1:57105/plotlab/state
```

`health.ready=true` 且 `engine.python` 有路径即正常。页面里点「示例」→「临床队列示例」，应出现预览图。
也可以用 Agent：`plotlab action=demo name=clinical` → `plotlab action=auto need="按分组画箱线图比较 score"`。

## 4. 配置（可选）

插件 config 支持：

| 键 | 说明 |
|---|---|
| `python` | 指定 Python 解释器绝对路径（默认自动探测：PATH → macOS 框架版 → Homebrew） |
| `workdir` | 工作目录（默认 `~/.dsh/plotlab`；存图片/上传/导出） |
| `timeoutMs` | 单次引擎请求默认超时（毫秒） |

可在插件配置里写，也可设环境变量 `DSH_PLOTLAB_PYTHON`。

## 5. 卸载

`plugin_manager remove_bundle target=dsh-plotlab`，或删除 profile 里的链接与配置项后重启。

## 6. 常见问题

- **引擎异常：找不到 Python/依赖** → 在插件 config 里显式指定 `python`，或用 `DSH_PLOTLAB_PYTHON` 环境变量。
- **页面没有出现 PlotLab** → 确认 bundle 已启用（`plugin_manager list_bundles`），宿主需要重启以加载新插件 JS。
- **中文标题变方块** → 样式面板里把字体改为 `Arial Unicode MS` / `PingFang SC` / `Songti SC`（本机需已安装）。
- **图很慢** → 通常是矩阵太大；热图/PCA 会取方差 top 特征，可先在参数里减少特征或先过滤数据。
