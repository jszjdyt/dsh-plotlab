/**
 * dsh-plotlab — 工作台统计口径契约（口径卡的数据源，UI / Agent / 导出共用）
 *
 * 与 bioflow 的 run 级契约、dsh-geobulk 的数据集口径同一套哲学：
 *   只搬运引擎真实回传的状态字段，取不到就写 null 并进 caveats——
 *   不补默认值、不猜方法名、不把缺失说成 0。
 *
 * 绘图插件没有「样本分组」语义，口径要回答的是**图本身的可信度**：
 *   数据集现状（行列/列类型/缺失/清洗前后行数/source）
 *   + 最近一次 figure run（chart / 角色映射 / 检验方法 / 多重校正 / 样式覆盖）
 *   + 最近一次统计与最近一次导出的绑定状态（figure_run_id 是否存在、和当前图是否同一张）。
 *
 * 注意：本文件由 impl.mjs 用带 rev 的 URL 动态 import（同 impl 的 HMR 策略），
 * 因此这里不要 import 其它 lib 模块，保持零依赖、可在 node 单测里直接 import。
 */

/** 插件版本（= package.json version；测试会比对，防止漂移）。 */
export const PLUGIN_VERSION = '0.1.0'

/** 契约 schema 标识（与 dsh-geobulk 同一代口径契约）。 */
export const CONTRACT_SCHEMA = 'dsh-workbench.stats_contract/1'

// ---------------------------------------------------------------------------
// 基础归一化：null/'' 一律 null，不编默认值
// ---------------------------------------------------------------------------

function text(value) {
  if (value === null || value === undefined) return ''
  return String(value).replace(/\s+/g, ' ').trim()
}

function nullableText(value, max) {
  const out = text(value)
  if (!out) return null
  if (Number.isFinite(Number(max)) && Number(max) > 0 && out.length > Number(max)) {
    return out.slice(0, Number(max) - 1) + '…'
  }
  return out
}

function num(value) {
  if (value === null || value === undefined || value === '') return null
  const number = Number(value)
  return Number.isFinite(number) ? number : null
}

function boolOrNull(value) {
  if (value === true) return true
  if (value === false) return false
  return null
}

function objectOrNull(value) {
  return value && typeof value === 'object' && !Array.isArray(value) ? value : null
}

function stringList(value) {
  if (!Array.isArray(value)) return []
  const out = []
  for (const item of value) {
    const entry = text(item)
    if (entry) out.push(entry)
  }
  return out
}

/** 分组名→数量：只保留有限数值，取不到的计 null（reference 的 n_per_group 同口径）。 */
function countMap(value) {
  const raw = objectOrNull(value)
  if (!raw) return {}
  const out = {}
  for (const key of Object.keys(raw)) {
    const name = text(key)
    if (!name) continue
    const count = num(raw[key])
    out[name] = count === null ? null : count
  }
  return out
}

const ROLE_KEYS = ['x', 'y', 'group', 'color', 'size', 'facet', 'time', 'event']

function roleMap(value) {
  const raw = objectOrNull(value) || {}
  const out = {}
  for (const key of ROLE_KEYS) out[key] = nullableText(raw[key])
  return out
}

// ---------------------------------------------------------------------------
// 口径契约
// ---------------------------------------------------------------------------

/**
 * 数据集 summary（pl_registry.dataset_summary 的返回）→ 统计口径契约。
 *
 * @param {object} dataset 引擎 status/records 回传的 dataset summary
 * @param {{plugin?: string, version?: string}} [extra] 插件标识（默认 dsh-plotlab）
 * @returns {object} 契约对象（可 JSON 序列化；所有取不到的字段为 null）
 */
export function buildWorkbenchContract(dataset, extra = {}) {
  const d = dataset && typeof dataset === 'object' ? dataset : {}
  const plugin = text(extra.plugin) || 'dsh-plotlab'
  const version = text(extra.version) || PLUGIN_VERSION

  const columnKindsRaw = objectOrNull(d.column_kinds) || {}
  const columnsTotal = num(d.columns_count) !== null ? num(d.columns_count) : num(d.n_cols)
  const columnKinds = {
    numeric: num(columnKindsRaw.numeric),
    categorical: num(columnKindsRaw.categorical),
    time: num(columnKindsRaw.time),
    // event 不是 dtype：只有显式角色映射时才存在，取不到写 null（不是 0）
    event: num(columnKindsRaw.event),
    other: num(columnKindsRaw.other),
    total: columnsTotal,
  }

  const missingRaw = objectOrNull(d.missing) || {}
  const missing = {
    cells: num(missingRaw.cells),
    missing: num(missingRaw.missing),
    columns_with_missing: num(missingRaw.columns_with_missing),
    rate: num(missingRaw.rate),
  }

  const cleanRaw = objectOrNull(d.clean_summary) || {}
  const cleaning = {
    steps: num(cleanRaw.steps),
    ops: stringList(cleanRaw.ops),
    rows_import: num(cleanRaw.rows_import),
    rows_current: num(cleanRaw.rows_current),
    rows_delta: num(cleanRaw.rows_delta),
    columns_import: num(cleanRaw.columns_import),
    columns_current: num(cleanRaw.columns_current),
    logged_rows_delta: num(cleanRaw.logged_rows_delta),
    changed_rows: boolOrNull(cleanRaw.changed_rows),
    trace_complete: boolOrNull(cleanRaw.trace_complete),
  }

  const roles = objectOrNull(d.roles) || {}
  const groups = {
    key: text(d.group_col) || text(roles.group) || null,
    levels: stringList(d.groups),
    counts: countMap(d.group_counts),
  }

  const datasetDigest = nullableText(d.data_digest)
  const datasetView = {
    id: text(d.id) || null,
    name: text(d.name) || null,
    kind: text(d.kind) || null,
    source: text(d.source) || null,
    n_rows: num(d.n_rows),
    n_cols: num(d.n_cols),
    columns: columnKinds,
    declared: {
      time_column: nullableText(d.time_col),
      event_column: nullableText(d.event_col),
    },
    missing,
    cleaning,
    groups,
    data_digest: datasetDigest,
    created_at: nullableText(d.created_at),
  }

  // ---- 最近一次 figure run -------------------------------------------------
  const figureRaw = objectOrNull(d.last_figure)
  let figure = null
  if (figureRaw) {
    const inferenceRaw = objectOrNull(figureRaw.inference)
    let inference = null
    if (inferenceRaw) {
      const method = nullableText(inferenceRaw.method, 60)
      const pAdjust = nullableText(inferenceRaw.p_adjust)
      const nContrasts = num(inferenceRaw.n_contrasts)
      if (method || pAdjust || nContrasts !== null) {
        inference = {
          method,
          p_adjust: pAdjust,
          // 图上的显著性标注用固定星号阈值，引擎不记录可调 α → null，不补 0.05
          alpha: num(inferenceRaw.alpha),
          factor: nullableText(inferenceRaw.factor),
          kind: nullableText(inferenceRaw.kind),
          n_contrasts: nContrasts,
          n_per_group: countMap(inferenceRaw.n_per_group),
        }
      }
    }
    const figureDigest = nullableText(figureRaw.data_digest)
    const styleOverrides = stringList(figureRaw.style_overrides)
    figure = {
      figure_run_id: nullableText(figureRaw.figure_run_id),
      chart: nullableText(figureRaw.chart),
      chart_title: nullableText(figureRaw.chart_title),
      at: nullableText(figureRaw.at),
      data_digest: figureDigest,
      data_matches_current: figureDigest && datasetDigest ? figureDigest === datasetDigest : null,
      roles: roleMap(figureRaw.roles),
      params: objectOrNull(figureRaw.params) || {},
      style_overrides: styleOverrides,
      n_style_overrides: styleOverrides.length,
      inference,
      n_figures: num(figureRaw.n_figures),
      artifact_available: boolOrNull(figureRaw.artifact_available),
    }
  }

  // ---- 最近一次统计 --------------------------------------------------------
  const statRaw = objectOrNull(d.last_stat)
  let stat = null
  if (statRaw) {
    stat = {
      stat: nullableText(statRaw.stat),
      title: nullableText(statRaw.title),
      method: nullableText(statRaw.method, 120),
      p_adjust: nullableText(statRaw.p_adjust),
      alpha: num(statRaw.alpha),
      params: objectOrNull(statRaw.params) || {},
      at: nullableText(statRaw.at),
      ok: boolOrNull(statRaw.ok),
      error: nullableText(statRaw.error, 160),
    }
  }

  // ---- 最近一次导出（figure_run_id 绑定） -----------------------------------
  const exportRaw = objectOrNull(d.last_export)
  let exported = null
  if (exportRaw) {
    const exportedRunId = nullableText(exportRaw.figure_run_id)
    const currentRunId = figure ? figure.figure_run_id : null
    exported = {
      what: nullableText(exportRaw.what),
      file: nullableText(exportRaw.file),
      at: nullableText(exportRaw.at),
      status: nullableText(exportRaw.status),
      verified: boolOrNull(exportRaw.verified),
      chart: nullableText(exportRaw.chart),
      figure_run_id: exportedRunId,
      data_matches_current_figure: exportedRunId && currentRunId ? exportedRunId === currentRunId : null,
      diffs: stringList(exportRaw.diffs),
      notes: stringList(exportRaw.notes),
      n_files: num(exportRaw.n_files),
    }
  }

  // ---- caveats：真实可判定的规则 -------------------------------------------
  const caveats = []
  const add = (code, level, message) => caveats.push({ code, level, text: message })
  const hasDataset = Boolean(datasetView.id || text(d.name)) || datasetView.n_rows !== null

  if (!hasDataset) {
    add('no_dataset', 'warn', '还没有数据集：口径卡没有可评估的对象，先导入数据。')
  } else {
    // 分组与 n：数据集现状
    if (!groups.key && !groups.levels.length) {
      add('no_groups', 'info', '没有分组列：图上只能做描述，任何组间比较都不成立。')
    } else if (Object.keys(groups.counts).length) {
      const entries = Object.keys(groups.counts).map((name) => ({ name, count: groups.counts[name] }))
      const low = entries.filter((item) => item.count !== null && item.count < 2)
      if (low.length) {
        add('group_singleton', 'warn',
          `分组「${low[0].name}」n=${low[0].count}：该组没有重复，组间检验/箱线图的推断不成立。`)
      }
    }

    // 缺失是否处理
    if (missing.missing === null && missing.rate === null) {
      add('missing_unknown', 'info', '缺失率取不到（数据集没有列摘要）：引用前需要人工核对缺失处理。')
    } else if (missing.missing && missing.missing > 0) {
      const rateText = missing.rate === null ? '' : `（${(missing.rate * 100).toFixed(2)}%）`
      const handled = cleaning.ops.some((op) => ['impute', 'dropna', 'drop_empty', 'filter', 'aggregate', 'outlier'].includes(op))
      if (handled) {
        add('missing_after_cleaning', 'warn',
          `清洗后仍有 ${missing.missing} 个缺失单元格${rateText}：作图和统计会按行删除缺失样本，n 会小于原始行数。`)
      } else {
        add('missing_unhandled', 'warn',
          `数据有 ${missing.missing} 个缺失单元格${rateText}，没有找到填补/删除的清洗记录：作图时会静默丢行，n 需要人工核对。`)
      }
    }

    // 清洗行数留痕
    if (cleaning.changed_rows === true) {
      if (!cleaning.steps) {
        add('cleaning_unlogged', 'warn',
          `行数从 ${cleaning.rows_import} 变为 ${cleaning.rows_current}，但没有清洗计划记录：无法解释数据被谁改动。`)
      } else if (cleaning.trace_complete === false) {
        add('cleaning_trace_incomplete', 'warn',
          `清洗记录只解释了 Δ=${cleaning.logged_rows_delta} 行，实际 Δ=${cleaning.rows_delta}：中间可能有未记录的改动。`)
      } else {
        add('cleaning_rows_changed', 'info',
          `清洗改变了行数：${cleaning.rows_import} → ${cleaning.rows_current}（${cleaning.steps} 步已记录）。`)
      }
    } else if (cleaning.steps === 0) {
      add('no_cleaning', 'info', '没有清洗记录：图基于导入原样数据（类型/缺失/重复问题未处理）。')
    }

    // 最近 figure run
    if (!figure) {
      add('no_figure', 'info', '还没有成功的 figure run：图表层面的证据为空，先出图再引用。')
    } else {
      if (!figure.figure_run_id) {
        add('figure_run_id_missing', 'warn',
          '最近一次图没有 figure_run_id（旧记录）：导出/统计无法回查到具体版本。')
      }
      if (figure.artifact_available === false) {
        add('figure_artifact_missing', 'warn',
          '最近 figure run 的图记录已被清理（只剩历史条目）：图与参数不可回查，需重新出图。')
      }
      if (figure.data_matches_current === false) {
        add('figure_stale', 'warn',
          '最近一次图的数据指纹与当前数据不一致（清洗/改数据后没有重画）：先重画再引用。')
      }
      if (!figure.inference) {
        add('figure_no_inference', 'info', '最近一次图没有统计检验层：图上的差异只能当描述，不能当推断结论。')
      } else {
        const perGroup = figure.inference.n_per_group || {}
        const low = Object.keys(perGroup).filter((name) => perGroup[name] !== null && perGroup[name] < 2)
        if (low.length) {
          add('figure_n_below_2', 'warn',
            `图上检验的分组「${low[0]}」n=${perGroup[low[0]]}：该对比没有重复，不能作为推断结论。`)
        }
        if (figure.inference.p_adjust && figure.inference.alpha === null) {
          add('alpha_unknown', 'info',
            `多重校正方法已记录（${figure.inference.p_adjust}），但显著性水平 α 未记录（图内星号阈值为固定 0.001/0.01/0.05）。`)
        }
        if (text(d.kind) === 'matrix') {
          add('matrix_inference', 'info',
            '矩阵数据（特征×样本）的组间检验只有显式声明独立样本（independent_samples="sample"）后才成立：样本内多特征仍属技术重复。')
        }
      }
      if (figure.style_overrides.length) {
        add('style_overridden', 'info',
          `最近一次图覆盖了 ${figure.style_overrides.length} 项样式（${figure.style_overrides.slice(0, 6).join('、')}）：不是默认样式，复现以导出代码为准。`)
      }
    }

    // 最近统计
    if (stat) {
      if (stat.ok === false) {
        add('last_stat_failed', 'warn',
          `最近一次统计「${stat.stat || '?'}」失败（${stat.error || '无错误信息'}）：该统计结果不可引用。`)
      } else if (stat.p_adjust && stat.alpha === null) {
        add('stat_alpha_unknown', 'info',
          `最近统计记录了多重校正（${stat.p_adjust}），但没有记录显著性水平 α。`)
      }
    }

    // 最近导出与 figure_run_id 绑定
    if (!exported) {
      add('no_export', 'info', '还没有导出记录：当前图/统计没有被固定成可交付文件。')
    } else {
      if (!exported.status) {
        add('export_status_missing', 'warn',
          '最近导出没有绑定状态（旧协议）：无法确认导出的图和请求的是同一张。')
      } else if (exported.status === 'rerendered') {
        add('export_rerendered', 'warn',
          '最近导出与请求不一致，已按请求重画：导出的文件与此前预览的图不是同一张。')
      } else if (exported.status === 'legacy') {
        add('export_legacy', 'warn',
          '最近导出是旧记录（无 figure_run_id，按请求 + 数据指纹派生）：回查不到具体 figure run。')
      } else if (exported.status === 'absent') {
        add('export_without_figure', 'info', '最近一次导出不含图（例如只导代码/统计表），没有图形绑定。')
      }
      const wantsFigure = ['figure', 'bundle'].includes(String(exported.what || ''))
      if (wantsFigure && !exported.figure_run_id) {
        add('export_no_figure_run_id', 'warn',
          '最近导出的图没有 figure_run_id：证据链断在导出这一步，无法回查导出的是哪张图。')
      }
      if (exported.data_matches_current_figure === false) {
        add('export_figure_mismatch', 'warn',
          `最近导出绑定 figure_run_id=${exported.figure_run_id || '?'}，当前最近图为 ${figure && figure.figure_run_id ? figure.figure_run_id : '?'}：导出的不是当前预览这张图。`)
      }
      if (exported.diffs.length) {
        add('export_diffs', 'warn',
          `导出绑定发现 ${exported.diffs.length} 项差异：${exported.diffs.slice(0, 2).join('；')}`)
      }
    }
  }

  return {
    schema: CONTRACT_SCHEMA,
    plugin,
    plugin_version: version,
    dataset: datasetView,
    figure,
    stat,
    export: exported,
    caveats,
    ok_to_cite: hasDataset && !caveats.some((item) => item.level === 'warn'),
    generated_at: new Date().toISOString(),
  }
}
