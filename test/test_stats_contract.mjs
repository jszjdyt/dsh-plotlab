/**
 * 工作台统计口径契约测试（node:test）
 *
 * 绘图插件的口径卡回答的是「这张图本身能不能被引用」：
 *   数据集现状（行列/列类型/缺失/清洗前后行数/source）
 *   + 最近 figure run（chart / 角色映射 / 检验方法 / 多重校正 / 样式覆盖）
 *   + 最近统计与最近导出的 figure_run_id 绑定状态。
 *
 * 四条用例钉死：齐全 / 缺陷 / 导出绑定状态 / 空数据集。
 * 规则同 dsh-geobulk：只搬运真实字段，取不到写 null + caveat，不编默认值。
 *
 * 运行：node --test test/test_stats_contract.mjs
 */
import test from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'

import { buildWorkbenchContract, PLUGIN_VERSION, CONTRACT_SCHEMA } from '../lib/stats_contract.mjs'
// 纯函数层自注册到 globalThis.__PL_TEST__（node 环境没有 window）
import '../client.js'

const CLIENT = globalThis.__PL_TEST__ || {}
const packageJson = JSON.parse(readFileSync(new URL('../package.json', import.meta.url), 'utf8'))

function baseFigure(overrides = {}) {
  return {
    figure_run_id: 'fr_abc123',
    chart: 'box',
    chart_title: '箱线图',
    at: '2026-11-27T10:20:00+08:00',
    data_digest: 'digest-current',
    roles: { x: 'group', y: 'score', group: 'group', color: '', size: '', facet: '', time: '', event: '' },
    params: { stat_annot: true },
    style_overrides: ['title_size'],
    inference: {
      method: 'Welch t-test',
      p_adjust: 'bh',
      alpha: null,
      factor: 'group',
      kind: 'group_comparison',
      n_contrasts: 1,
      n_per_group: { Control: 48, Treatment: 48 },
    },
    n_figures: 1,
    artifact_available: true,
    ...overrides,
  }
}

function baseDataset(overrides = {}) {
  return {
    id: 'ds1',
    name: '临床队列',
    kind: 'tidy',
    source: '/tmp/clinical.csv',
    n_rows: 96,
    n_cols: 10,
    columns_count: 10,
    column_kinds: { numeric: 6, categorical: 4, time: 0, event: null, other: 0 },
    missing: { cells: 960, missing: 0, columns_with_missing: 0, rate: 0 },
    clean_summary: {
      steps: 1,
      ops: ['impute'],
      rows_import: 96,
      rows_current: 96,
      rows_delta: 0,
      columns_import: 10,
      columns_current: 10,
      logged_rows_delta: 0,
      changed_rows: false,
      trace_complete: true,
    },
    roles: { x: 'group', y: 'score', group: 'group' },
    group_col: 'group',
    groups: ['Control', 'Treatment'],
    group_counts: { Control: 48, Treatment: 48 },
    time_col: null,
    event_col: null,
    data_digest: 'digest-current',
    created_at: '2026-11-27T10:00:00+08:00',
    history_count: 3,
    last_figure: baseFigure(),
    last_stat: {
      stat: 'compare',
      title: '组间比较',
      method: 'Welch t-test',
      params: { adjust: 'bh' },
      alpha: null,
      p_adjust: 'bh',
      at: '2026-11-27T10:21:00+08:00',
      ok: true,
      error: null,
    },
    last_export: {
      what: 'bundle',
      file: '/tmp/exports/bundle.zip',
      at: '2026-11-27T10:22:00+08:00',
      status: 'reused',
      verified: true,
      chart: 'box',
      figure_run_id: 'fr_abc123',
      data_digest: 'digest-current',
      diffs: [],
      notes: [],
      n_files: 6,
    },
    ...overrides,
  }
}

test('口径 1) 齐全：行列/列类型/缺失/清洗/分组/最近图检验/导出绑定都可回查', () => {
  const contract = buildWorkbenchContract(baseDataset())
  assert.equal(contract.schema, CONTRACT_SCHEMA)
  assert.equal(contract.schema, 'dsh-workbench.stats_contract/1')
  assert.equal(contract.plugin, 'dsh-plotlab')
  assert.equal(contract.plugin_version, PLUGIN_VERSION)
  // 版本必须与 package.json 同步（防止常量漂移）
  assert.equal(PLUGIN_VERSION, packageJson.version)

  // 数据集层面
  assert.equal(contract.dataset.id, 'ds1')
  assert.equal(contract.dataset.source, '/tmp/clinical.csv')
  assert.equal(contract.dataset.n_rows, 96)
  assert.equal(contract.dataset.n_cols, 10)
  assert.deepEqual(contract.dataset.columns, {
    numeric: 6, categorical: 4, time: 0, event: null, other: 0, total: 10,
  })
  assert.equal(contract.dataset.missing.rate, 0)
  assert.equal(contract.dataset.cleaning.steps, 1)
  assert.equal(contract.dataset.cleaning.changed_rows, false)
  assert.deepEqual(contract.dataset.groups.counts, { Control: 48, Treatment: 48 })

  // 图表层面：figure run / 检验方法 / 校正 / 角色映射 / 样式覆盖
  assert.ok(contract.figure)
  assert.equal(contract.figure.figure_run_id, 'fr_abc123')
  assert.equal(contract.figure.chart, 'box')
  assert.equal(contract.figure.data_matches_current, true)
  assert.equal(contract.figure.roles.x, 'group')
  assert.equal(contract.figure.roles.y, 'score')
  assert.deepEqual(contract.figure.style_overrides, ['title_size'])
  assert.equal(contract.figure.inference.method, 'Welch t-test')
  assert.equal(contract.figure.inference.p_adjust, 'bh')
  assert.equal(contract.figure.inference.alpha, null, '图上 α 不可调/未记录 → null，不补 0.05')
  assert.equal(contract.figure.inference.kind, 'group_comparison')
  assert.deepEqual(contract.figure.inference.n_per_group, { Control: 48, Treatment: 48 })

  // 最近统计 + 导出绑定（figure_run_id 与当前图一致）
  assert.equal(contract.stat.method, 'Welch t-test')
  assert.equal(contract.export.status, 'reused')
  assert.equal(contract.export.figure_run_id, 'fr_abc123')
  assert.equal(contract.export.data_matches_current_figure, true)

  // 无 warn；info 只提示「α 未记录 / 样式被覆盖」这类非缺陷项
  assert.deepEqual(contract.caveats.filter((item) => item.level === 'warn'), [])
  assert.equal(contract.ok_to_cite, true)
  assert.doesNotThrow(() => JSON.stringify(contract), '契约必须可 JSON 序列化')

  // client 归一化：契约 → UI 模型（字段名换 camelCase，null 保持 null）
  const ui = CLIENT.normalizeStatsContract(contract)
  assert.ok(ui)
  assert.equal(ui.okToCite, true)
  assert.equal(ui.figure.inference.method, 'Welch t-test')
  assert.equal(ui.figure.styleOverrides[0], 'title_size')
  assert.equal(ui.exportInfo.status, 'reused')
  assert.equal(ui.dataset.columnKinds.numeric, 6)
  assert.equal(ui.figure.inference.alpha, null)
})

test('口径 2) 缺陷：分组 n<2 / 缺失未处理 / 行数无留痕 / 图过期与记录缺失 / 统计失败 → 全部 warn', () => {
  const contract = buildWorkbenchContract(baseDataset({
    group_counts: { Control: 1, Treatment: 3 },
    missing: { cells: 960, missing: 12, columns_with_missing: 3, rate: 0.0125 },
    clean_summary: {
      steps: 0,
      ops: [],
      rows_import: 100,
      rows_current: 96,
      rows_delta: -4,
      columns_import: 10,
      columns_current: 10,
      logged_rows_delta: 0,
      changed_rows: true,
      trace_complete: false,
    },
    last_figure: baseFigure({
      figure_run_id: null,
      data_digest: 'digest-old',
      artifact_available: false,
      style_overrides: [],
      inference: {
        method: 'Welch t-test', p_adjust: 'bh', alpha: null, factor: 'group',
        kind: 'group_comparison', n_contrasts: 1, n_per_group: { Control: 1, Treatment: 3 },
      },
    }),
    last_stat: {
      stat: 'compare', title: '组间比较', method: '', params: {}, alpha: null,
      p_adjust: null, at: '2026-11-27T10:21:00+08:00', ok: false,
      error: '至少 1 组有效样本少于 3 个',
    },
    last_export: null,
  }))
  const codes = contract.caveats.map((item) => item.code)
  assert.ok(codes.includes('group_singleton'), codes.join(','))
  assert.ok(codes.includes('missing_unhandled'), codes.join(','))
  assert.ok(codes.includes('cleaning_unlogged'), codes.join(','))
  assert.ok(codes.includes('figure_run_id_missing'), codes.join(','))
  assert.ok(codes.includes('figure_artifact_missing'), codes.join(','))
  assert.ok(codes.includes('figure_stale'), codes.join(','))
  assert.ok(codes.includes('figure_n_below_2'), codes.join(','))
  assert.ok(codes.includes('last_stat_failed'), codes.join(','))
  assert.ok(codes.includes('no_export') && contract.caveats.find((item) => item.code === 'no_export').level === 'info')
  assert.equal(contract.figure.data_matches_current, false)
  assert.equal(contract.ok_to_cite, false)
})

test('口径 3) 导出绑定状态：重画/不一致/旧记录/无图 分别给出可引用性结论', () => {
  const rerendered = buildWorkbenchContract(baseDataset({
    last_export: {
      what: 'figure',
      status: 'rerendered',
      verified: true,
      chart: 'bar',
      figure_run_id: 'fr_other',
      data_digest: 'digest-current',
      diffs: ['params 快照不一致（图表参数与记录不同）'],
      notes: ['导出请求与最近一次渲染不一致，已按请求重画'],
      n_files: 1,
      at: '2026-11-27T10:22:00+08:00',
    },
  }))
  const rerenderedCodes = rerendered.caveats.map((item) => item.code)
  assert.ok(rerenderedCodes.includes('export_rerendered'), rerenderedCodes.join(','))
  assert.ok(rerenderedCodes.includes('export_figure_mismatch'), rerenderedCodes.join(','))
  assert.ok(rerenderedCodes.includes('export_diffs'), rerenderedCodes.join(','))
  assert.equal(rerendered.export.data_matches_current_figure, false)
  assert.equal(rerendered.ok_to_cite, false)

  const legacy = buildWorkbenchContract(baseDataset({
    last_export: {
      what: 'bundle',
      status: 'legacy',
      figure_run_id: null,
      diffs: [],
      n_files: 4,
      at: '2026-11-27T10:22:00+08:00',
    },
  }))
  const legacyCodes = legacy.caveats.map((item) => item.code)
  assert.ok(legacyCodes.includes('export_legacy'), legacyCodes.join(','))
  assert.ok(legacyCodes.includes('export_no_figure_run_id'), legacyCodes.join(','))

  const absent = buildWorkbenchContract(baseDataset({
    last_export: {
      what: 'code',
      status: 'absent',
      figure_run_id: null,
      diffs: [],
      n_files: 2,
      at: '2026-11-27T10:22:00+08:00',
    },
  }))
  assert.equal(absent.export.status, 'absent')
  assert.equal(absent.export.figure_run_id, null)
  assert.ok(absent.caveats.some((item) => item.code === 'export_without_figure' && item.level === 'info'))
  assert.ok(!absent.caveats.some((item) => item.code === 'export_no_figure_run_id'),
    '只导代码/统计表时不应要求 figure_run_id')

  // normalizeDataset 透传：stats_contract 原样带过，UI 字段规范化
  const ui = CLIENT.normalizeDataset({
    dataset: { id: 'ds1', name: ' 临床队列 ', n_rows: '96', stats_contract: rerendered },
  })
  assert.equal(ui.id, 'ds1')
  assert.equal(ui.name, '临床队列')
  assert.equal(ui.n_rows, 96)
  assert.ok(ui.stats_contract)
  assert.equal(ui.stats_contract.exportInfo.status, 'rerendered')
  assert.equal(ui.stats_contract.caveats.length > 0, true)
})

test('口径 4) 空数据集不抛错：字段为 null/空，且明确 no_dataset', () => {
  const contract = buildWorkbenchContract({})
  assert.equal(contract.schema, 'dsh-workbench.stats_contract/1')
  assert.equal(contract.plugin, 'dsh-plotlab')
  assert.equal(contract.dataset.id, null)
  assert.equal(contract.dataset.n_rows, null)
  assert.equal(contract.dataset.columns.numeric, null)
  assert.equal(contract.dataset.columns.event, null)
  assert.equal(contract.dataset.missing.rate, null)
  assert.equal(contract.dataset.cleaning.steps, null)
  assert.deepEqual(contract.dataset.groups.levels, [])
  assert.equal(contract.figure, null)
  assert.equal(contract.stat, null)
  assert.equal(contract.export, null)
  assert.ok(contract.caveats.some((item) => item.code === 'no_dataset' && item.level === 'warn'))
  assert.equal(contract.ok_to_cite, false)
  assert.equal(CLIENT.normalizeStatsContract(null), null)
  assert.deepEqual(CLIENT.normalizeDataset({}), {
    id: '', name: '', kind: '', source: '', n_rows: null, n_cols: null, group_col: '',
    groups: [], roles: {}, cleaning: [], notes: [], warnings: [],
    need: '', history_count: null, data_digest: '', stats_contract: null,
  })
  assert.equal(CLIENT.normalizeDataset({}).columns, undefined, '没带列摘要时不要编造空数组（会清空页面列清单）')
})
