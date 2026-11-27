/**
 * dsh-plotlab — Host 业务实现（唯一操作实现）
 *
 * 一条操作只写一次：浏览器 UI 走 HTTP 路由，Agent 走 plotlab 工具，
 * 两边入口都调同一批 op* 函数（它们再转成 Python worker 命令）。
 *
 * 本文件按需用带 rev 的 URL 动态 import（见 index.js 的 loadImpl），
 * 因此不要在 lib 模块之间写静态 import；需要 engine.mjs 时用 loadEngineMod()。
 */

import { createReadStream, createWriteStream, existsSync, mkdirSync, statSync } from 'node:fs'
import path from 'node:path'
import os from 'node:os'
import { randomUUID } from 'node:crypto'
import { pipeline } from 'node:stream/promises'

const PREFIX = '/plotlab'
const VERSION = '0.1.0'
const MAX_JSON_BODY = 4 * 1024 * 1024
const MAX_UPLOAD_BYTES = 2 * 1024 * 1024 * 1024
const ACTIVITY_LIMIT = 200
const DEFAULT_TIMEOUT = 90 * 1000
const PLOT_TIMEOUT = 4 * 60 * 1000
const AUTO_TIMEOUT = 6 * 60 * 1000
const EXPORT_TIMEOUT = 3 * 60 * 1000
const STAT_TIMEOUT = 5 * 60 * 1000

export const ACTIONS = [
  'status', 'open', 'demo', 'inspect', 'preview', 'profile', 'clean', 'reset_data', 'roles',
  'charts', 'plot', 'stat', 'auto', 'code', 'export', 'records', 'audit', 'suggest', 'need',
  'samples', 'formats', 'reset', 'engine_restart',
]

export const DATA_FORMATS = ['auto', 'csv', 'tsv', 'txt', 'xlsx', 'json', 'archive', 'gz']
export const EXPORT_KINDS = ['figure', 'code', 'report', 'methods', 'protocol', 'cleaned', 'stats', 'bundle']

const LOOPBACK_HOSTS = new Set(['127.0.0.1', 'localhost', '[::1]', '::1', '0.0.0.0'])

export function hostIsLoopback(req) {
  const raw = String((req && req.headers && req.headers.host) || '')
  if (!raw) return true
  const name = raw.startsWith('[') ? raw.slice(0, raw.indexOf(']') + 1) : raw.split(':')[0]
  return LOOPBACK_HOSTS.has(name.toLowerCase())
}

function sendJson(res, status, value) {
  const body = Buffer.from(JSON.stringify(value), 'utf8')
  res.writeHead(status, {
    'content-type': 'application/json; charset=utf-8',
    'cache-control': 'no-store',
    'content-length': String(body.length),
  })
  res.end(body)
}

function opError(code, message, status = 400, details = null) {
  const body = { ok: false, code, error: message }
  if (details && typeof details === 'object') Object.assign(body, details)
  return { status, body }
}

function toFailure(error) {
  const code = (error && error.code) || 'internal'
  const message = String((error && error.message) || error)
  const details = error && error.details && typeof error.details === 'object' ? error.details : null
  if (code === 'prerequisite' || code === 'bad_request' || code === 'not_found' || code === 'unsupported' || code === 'ambiguous') {
    return opError(code, message, 400, details)
  }
  if (code === 'timeout') return opError(code, message, 504, details)
  if (code === 'no_python' || code === 'engine_exited' || code === 'internal' || code === 'disposed') {
    return opError(code, message, 500, details)
  }
  if (code === 'busy') return opError(code, message, 429, details)
  return opError(code, message, 500, details)
}

function readBody(req, limit) {
  return new Promise((resolve, reject) => {
    const chunks = []
    let size = 0
    req.on('data', (chunk) => {
      size += chunk.length
      if (size > limit) {
        reject(Object.assign(new Error(`请求体超过上限（${Math.round(limit / 1024)} KB）`), { code: 'bad_request' }))
        req.destroy()
        return
      }
      chunks.push(chunk)
    })
    req.on('end', () => resolve(Buffer.concat(chunks)))
    req.on('error', (error) => reject(Object.assign(error, { code: 'bad_request' })))
  })
}

async function readJson(req) {
  const buffer = await readBody(req, MAX_JSON_BODY)
  if (!buffer.length) return {}
  try {
    const value = JSON.parse(buffer.toString('utf8'))
    return value && typeof value === 'object' ? value : {}
  } catch (error) {
    throw Object.assign(new Error(`请求 JSON 解析失败：${error.message}`), { code: 'bad_request' })
  }
}

function sanitizeName(name) {
  const base = String(name || 'upload')
  return path.basename(base).replace(/[^\w.\-\u4e00-\u9fa5]+/g, '_').slice(0, 160) || 'upload'
}

function nowIso() {
  return new Date().toISOString()
}

function defaultWorkdir() {
  return path.join(os.homedir(), '.dsh', 'plotlab')
}

async function loadEngineMod() {
  return import(new URL(`./engine.mjs?rev=${Date.now()}-${Math.random().toString(36).slice(2)}`, import.meta.url).href)
}

// ---------------------------------------------------------------------------
// 引擎
// ---------------------------------------------------------------------------

async function ensureEngine(runtime, { allowStart = true } = {}) {
  if (runtime.engine) {
    if (runtime.engine.running) return runtime.engine
    if (!allowStart) return runtime.engine
  } else {
    const mod = await loadEngineMod()
    const config = runtime.config && typeof runtime.config === 'object' ? runtime.config : {}
    runtime.engine = mod.createEngine({
      python: typeof config.python === 'string' && config.python.trim() ? config.python.trim() : undefined,
      workdir: typeof config.workdir === 'string' && config.workdir.trim() ? config.workdir.trim() : defaultWorkdir(),
      timeout: Number(config.timeoutMs) > 0 ? Number(config.timeoutMs) : DEFAULT_TIMEOUT,
      config,
    })
    if (typeof runtime.ctx?.effect === 'function') {
      const engine = runtime.engine
      runtime.ctx.effect(() => () => { engine.dispose().catch(() => {}) }, 'dsh-plotlab: engine')
    }
  }
  if (!allowStart) return runtime.engine
  const now = Date.now()
  if (runtime.engineFail && now - runtime.engineFail.at < 60_000) throw runtime.engineFail.error
  try {
    await runtime.engine.start()
    runtime.engineFail = null
  } catch (error) {
    runtime.engineFail = { at: now, error }
    throw error
  }
  return runtime.engine
}

export async function engineStatus(runtime) {
  let engine = runtime.engine
  if (!engine) {
    try {
      await ensureEngine(runtime, { allowStart: false })
      engine = runtime.engine
    } catch (error) {
      return {
        ready: false, mode: 'none', error: String(error && error.message ? error.message : error),
        stack: String(error && error.stack ? error.stack : '').slice(0, 3000),
        python: null, packages: {}, fonts: [],
      }
    }
  }
  if (!engine) return { ready: false, mode: 'none', error: 'Python 引擎未初始化', python: null, packages: {}, fonts: [] }
  const status = typeof engine.status === 'function' ? engine.status() : { ready: false }
  if (!status.ready && !status.error && runtime.engineFail) {
    status.error = String(runtime.engineFail.error && runtime.engineFail.error.message
      ? runtime.engineFail.error.message
      : runtime.engineFail.error)
  }
  if (status.ready && !runtime.engineInfo) {
    try {
      runtime.engineInfo = await engine.request('ping', {}, { skipStart: true })
      status.packages = runtime.engineInfo.packages || status.packages
      status.fonts = runtime.engineInfo.fonts || status.fonts
      status.charts = runtime.engineInfo.charts || []
    } catch { /* ping 失败不影响状态 */ }
  }
  if (runtime.engineInfo) {
    status.packages = runtime.engineInfo.packages || status.packages
    status.fonts = runtime.engineInfo.fonts || status.fonts
    status.formats = runtime.engineInfo.formats || status.formats
  }
  return status
}

async function engineRequest(runtime, cmd, args, timeout) {
  const engine = await ensureEngine(runtime)
  return engine.request(cmd, args, timeout ? { timeout } : {})
}

export function logActivity(runtime, entry) {
  if (!runtime.activity) runtime.activity = []
  const item = { at: nowIso(), ...entry }
  runtime.activity.push(item)
  if (runtime.activity.length > ACTIVITY_LIMIT) runtime.activity.splice(0, runtime.activity.length - ACTIVITY_LIMIT)
  return item
}

export function activityOf(runtime, since = 0) {
  const list = (runtime.activity || []).slice(Number(since) || 0)
  return { items: list, total: (runtime.activity || []).length }
}

/** 记住最近一次状态，供 prompt section 使用。 */
function rememberState(runtime, payload) {
  if (!payload || typeof payload !== 'object') return
  const dataset = payload.dataset
  if (dataset && dataset.id) {
    runtime.lastState = {
      id: dataset.id,
      name: dataset.name,
      kind: dataset.kind,
      n_rows: dataset.n_rows,
      n_cols: dataset.n_cols,
      group_col: dataset.group_col,
      cleaning: (dataset.cleaning || []).length,
      need: dataset.need || runtime.lastState?.need || '',
      history: dataset.history_count || 0,
    }
    return
  }
  if (payload.active !== undefined || payload.datasets) {
    const active = (payload.datasets || []).find((item) => item.id === payload.active)
    runtime.lastState = active
      ? {
          id: active.id, name: active.name, kind: active.kind, n_rows: active.n_rows, n_cols: active.n_cols,
          group_col: active.group_col, cleaning: (active.cleaning || []).length,
          need: active.need || '', history: active.history_count || 0,
        }
      : null
  }
}


// ---------------------------------------------------------------------------
// P0 修复：冻结数据版本 + 不可变 run 记录（产物落 runs/<run_id>/figures）
// ---------------------------------------------------------------------------

/** 解析出明确的数据集 id：显式给了就用它；没给就当场把活动数据集冻结成 id。 */
async function resolvePlotDatasetId(runtime, selector) {
  const explicit = selector === undefined || selector === null ? '' : String(selector).trim()
  if (explicit) return explicit
  const state = await engineRequest(runtime, 'state', {}, 15_000)
  const active = state && (state.active || (state.dataset && state.dataset.id))
  if (!active) throw Object.assign(new Error('还没有数据集，请先 open 或 demo'), { code: 'bad_request' })
  return String(active)
}

function plotArtifacts(result) {
  const out = []
  for (const fig of (result && result.figures) || []) {
    if (fig && fig.file) out.push({ kind: 'figure', source: fig.file, name: fig.filename || path.basename(fig.file) })
  }
  if (result && result.figure && result.figure.file) {
    out.push({ kind: 'figure', source: result.figure.file, name: result.figure.filename || path.basename(result.figure.file) })
  }
  return out
}

/** 复用已提交 run 时，把图路径换回不可变副本。 */
function rehydratePlot(result, task) {
  const cached = ((task.run && task.run.artifacts) || []).filter((item) => item.kind === 'figure')
  if (!cached.length) return { ...(result || {}), run_id: task.runId, run_reused: true }
  const next = { ...(result || {}), run_id: task.runId, run_reused: true }
  const rebuilt = cached.map((item, index) => {
    const original = ((result && result.figures) || [])[index] || (result && result.figure) || {}
    return { ...original, file: item.file, filename: item.name, sha256: item.sha256, bytes: item.bytes }
  })
  next.figures = rebuilt
  next.figure = rebuilt[0]
  return next
}

// ---------------------------------------------------------------------------
// 操作实现（HTTP 与 Agent 工具共用）
// ---------------------------------------------------------------------------

export async function opHealth(runtime) {
  const status = await engineStatus(runtime)
  return {
    ok: true,
    version: VERSION,
    engine: status,
    ready: Boolean(status.ready),
    workdir: runtime.engine ? runtime.engine.workdir : null,
    datasets: runtime.lastState ? 1 : 0,
  }
}

export async function opState(runtime) {
  const status = await engineStatus(runtime)
  let state
  try {
    state = await engineRequest(runtime, 'state', {}, 15_000)
  } catch (error) {
    state = { datasets: [], active: null, engine: { python: status.python, mode: 'none' }, error: String(error && error.message ? error.message : error) }
  }
  const payload = { ...state, engine: { ...(state.engine || {}), ...status, ready: Boolean(status.ready) } }
  rememberState(runtime, payload)
  return payload
}

export async function opCharts(runtime, args = {}) {
  const result = await engineRequest(runtime, 'charts', pickDataset(args), 30_000)
  if (result.dataset) rememberState(runtime, result)
  return result
}

export async function opStatsCatalog(runtime) {
  const result = await engineRequest(runtime, 'charts', {}, 30_000)
  return { stats: result.stats || [], styles: result.styles, roles: result.roles }
}

export async function opPreview(runtime, args = {}) {
  const result = await engineRequest(runtime, 'preview', pickDataset(args), 30_000)
  if (result.dataset) rememberState(runtime, result)
  return result
}

export async function opInspect(runtime, args = {}) {
  const payload = {}
  for (const key of ['path', 'url', 'name', 'data_format', 'delimiter', 'encoding', 'sheet', 'header', 'max_rows', 'kind', 'member', 'transpose']) {
    if (args[key] !== undefined && args[key] !== null && args[key] !== '') payload[key] = args[key]
  }
  if (!payload.path && !payload.url) throw Object.assign(new Error('需要 path 或 url'), { code: 'bad_request' })
  return engineRequest(runtime, 'inspect', payload, 60_000)
}

export async function opOpen(runtime, args = {}, actor = 'user') {
  const payload = {}
  for (const key of ['path', 'url', 'name', 'data_format', 'delimiter', 'encoding', 'sheet', 'header',
    'index_col', 'kind', 'feature_col', 'transpose', 'sample_sheet', 'max_rows', 'member']) {
    if (args[key] !== undefined && args[key] !== null && args[key] !== '') payload[key] = args[key]
  }
  if (!payload.path && !payload.url) throw Object.assign(new Error('需要 path 或 url'), { code: 'bad_request' })
  if (payload.path) {
    const text = String(payload.path)
    if (text.startsWith('~/')) payload.path = path.join(os.homedir(), text.slice(2))
  }
  const result = await engineRequest(runtime, 'open', payload, 3 * 60 * 1000)
  rememberState(runtime, result)
  logActivity(runtime, { kind: 'open', actor, dataset: result.dataset && result.dataset.id, message: `导入 ${result.dataset && result.dataset.name}` })
  return result
}

export async function opDemo(runtime, args = {}, actor = 'user') {
  const result = await engineRequest(runtime, 'demo', { name: args.name || args.demo || 'clinical' }, 60_000)
  rememberState(runtime, result)
  logActivity(runtime, { kind: 'demo', actor, dataset: result.dataset && result.dataset.id, message: `载入示例 ${result.dataset && result.dataset.name}` })
  return result
}

export async function opSelect(runtime, args = {}) {
  const result = await engineRequest(runtime, 'select', { dataset: args.dataset }, 30_000)
  rememberState(runtime, result)
  return result
}

export async function opClose(runtime, args = {}) {
  const result = await engineRequest(runtime, 'close', { dataset: args.dataset }, 30_000)
  rememberState(runtime, result)
  return result
}

export async function opProfile(runtime, args = {}) {
  const result = await engineRequest(runtime, 'profile', pickDataset(args), 90_000)
  if (result.dataset) rememberState(runtime, result)
  logActivity(runtime, { kind: 'profile', actor: args.actor || 'user', dataset: result.dataset && result.dataset.id, message: `剖析：质量分 ${result.score}/100` })
  return result
}

export async function opClean(runtime, args = {}) {
  const payload = { ...pickDataset(args), auto: Boolean(args.auto) }
  if (Array.isArray(args.plan)) payload.plan = args.plan
  else if (args.plan === null || args.plan === undefined) payload.auto = true
  const result = await engineRequest(runtime, 'clean', payload, 2 * 60 * 1000)
  if (result.dataset) rememberState(runtime, result)
  logActivity(runtime, { kind: 'clean', actor: args.actor || 'user', dataset: result.dataset && result.dataset.id, message: `清洗 ${(result.applied || []).length} 步` })
  return result
}

export async function opResetData(runtime, args = {}) {
  const result = await engineRequest(runtime, 'reset_data', pickDataset(args), 60_000)
  if (result.dataset) rememberState(runtime, result)
  return result
}

export async function opRoles(runtime, args = {}) {
  const result = await engineRequest(runtime, 'roles', { ...pickDataset(args), roles: args.roles || {} }, 30_000)
  return result
}

export async function opPlot(runtime, args = {}) {
  const payload = { chart: String(args.chart || args.kind || ''), params: args.params || {}, style: args.style || {}, roles: args.roles || undefined }
  if (!payload.chart) throw Object.assign(new Error('需要 chart（图表 id）'), { code: 'bad_request' })
  const engine = await ensureEngine(runtime)
  // P0-2：先冻结数据版本，之后只认这个 id（UI 切活动数据集不会把图画到别的数据上）
  const datasetId = await resolvePlotDatasetId(runtime, args.dataset)
  const task = await engine.withTask({
    op: `plot:${payload.chart}`,
    dataset: datasetId,
    params: { chart: payload.chart, params: payload.params, style: payload.style, roles: payload.roles || null },
    label: payload.chart,
    timeoutMs: PLOT_TIMEOUT,
    reuse: args.reuse !== false,
  }, async (ctx) => {
    const result = await ctx.request('plot', { ...payload, dataset: ctx.frozen.id }, { timeout: PLOT_TIMEOUT })
    return { result, artifacts: plotArtifacts(result) }
  })
  const result = task.reused ? rehydratePlot(task.result, task) : { ...(task.result || {}), run_id: task.runId, run_reused: false }
  rememberState(runtime, result)
  logActivity(runtime, {
    kind: 'plot', actor: args.actor || 'user', dataset: task.frozen ? task.frozen.id : datasetId,
    message: `${result.chart_title} → ${result.figure && result.figure.filename}`,
    figure: result.figure,
  })
  runtime.lastPlot = result
  return result
}

export async function opStat(runtime, args = {}) {
  const stat = String(args.stat || '')
  if (!stat) throw Object.assign(new Error('需要 stat（统计 id）'), { code: 'bad_request' })
  const engine = await ensureEngine(runtime)
  const datasetId = await resolvePlotDatasetId(runtime, args.dataset)
  const task = await engine.withTask({
    op: `stat:${stat}`,
    dataset: datasetId,
    params: { stat, params: args.params || {}, style: args.style || {}, roles: args.roles || null },
    label: stat,
    timeoutMs: STAT_TIMEOUT,
    reuse: args.reuse !== false,
  }, async (ctx) => {
    const result = await ctx.request('stat', { stat, params: args.params || {}, style: args.style || {}, roles: args.roles || undefined, dataset: ctx.frozen.id }, { timeout: STAT_TIMEOUT })
    return { result, artifacts: plotArtifacts(result) }
  })
  const result = task.reused ? rehydratePlot(task.result, task) : { ...(task.result || {}), run_id: task.runId, run_reused: false }
  logActivity(runtime, { kind: 'stat', actor: args.actor || 'user', dataset: task.frozen ? task.frozen.id : datasetId, message: `${result.title || stat}` })
  return result
}

export async function opAuto(runtime, args = {}) {
  const need = String(args.need || '').trim()
  if (!need) throw Object.assign(new Error('需要 need（一句中文需求，例如「按分组画箱线图并标显著性」）'), { code: 'bad_request' })
  const engine = await ensureEngine(runtime)
  const datasetId = await resolvePlotDatasetId(runtime, args.dataset)
  const task = await engine.withTask({
    op: 'auto',
    dataset: datasetId,
    params: { need, params: args.params || {}, style: args.style || {}, roles: args.roles || null },
    label: `auto:${need.slice(0, 40)}`,
    timeoutMs: AUTO_TIMEOUT,
    // auto 可能带自动清洗（会改数据版本），不做幂等复用
    reuse: false,
  }, async (ctx) => {
    const result = await ctx.request('auto', { dataset: ctx.frozen.id, need, params: args.params || {}, style: args.style || {}, roles: args.roles || undefined }, { timeout: AUTO_TIMEOUT })
    return { result, artifacts: plotArtifacts(result) }
  })
  const result = { ...(task.result || {}), run_id: task.runId, run_reused: false }
  rememberState(runtime, result)
  logActivity(runtime, {
    kind: 'auto', actor: args.actor || 'user', dataset: task.frozen ? task.frozen.id : datasetId,
    message: `需求：${need} → ${result.chart_title}`,
    figure: result.figure,
  })
  runtime.lastPlot = result
  return result
}

export async function opCode(runtime, args = {}) {
  const payload = { ...pickDataset(args), what: args.what || 'plot' }
  for (const key of ['chart', 'params', 'style', 'roles']) {
    if (args[key] !== undefined) payload[key] = args[key]
  }
  return engineRequest(runtime, 'code', payload, 90_000)
}

export async function opExport(runtime, args = {}) {
  const engine = await ensureEngine(runtime)
  const datasetId = await resolvePlotDatasetId(runtime, args.dataset)
  const payload = { what: args.what || 'bundle' }
  for (const key of ['chart', 'params', 'style', 'roles', 'out_path', 'format', 'stat', 'stat_params', 'figure_run_id']) {
    if (args[key] !== undefined && args[key] !== null && args[key] !== '') payload[key] = args[key]
  }
  if (args.img_format) payload.format = args.img_format
  const task = await engine.withTask({
    op: `export:${payload.what}`,
    dataset: datasetId,
    params: { what: payload.what, chart: payload.chart || null, format: payload.format || null, out_path: payload.out_path || null },
    label: `export:${payload.what}`,
    timeoutMs: EXPORT_TIMEOUT,
    reuse: false,
  }, async (ctx) => {
    const result = await ctx.request('export', { ...payload, dataset: ctx.frozen.id }, { timeout: EXPORT_TIMEOUT })
    const artifacts = plotArtifacts(result)
    if (result && result.file) artifacts.push({ kind: 'file', source: result.file, name: path.basename(result.file), link: true })
    return { result, artifacts }
  })
  const result = { ...(task.result || {}), run_id: task.runId, run_reused: false }
  logActivity(runtime, { kind: 'export', actor: args.actor || 'user', dataset: task.frozen ? task.frozen.id : datasetId, message: `导出 ${result.what} → ${result.file}` })
  return result
}

export async function opRecords(runtime, args = {}) {
  const payload = pickDataset(args)
  if (args.figure_run_id) payload.figure_run_id = args.figure_run_id
  return engineRequest(runtime, 'records', payload, 30_000)
}

export async function opAudit(runtime, args = {}) {
  return engineRequest(runtime, 'audit', pickDataset(args), 90_000)
}

export async function opSuggest(runtime, args = {}) {
  return engineRequest(runtime, 'suggest', pickDataset(args), 30_000)
}

export async function opNeed(runtime, args = {}) {
  const result = await engineRequest(runtime, 'need', { ...pickDataset(args), need: args.need || '' }, 30_000)
  if (result.dataset) rememberState(runtime, result)
  return result
}

export async function opSamples(runtime, args = {}) {
  const payload = { ...pickDataset(args), op: args.op || 'list' }
  for (const key of ['path', 'column', 'samples', 'name']) {
    if (args[key] !== undefined) payload[key] = args[key]
  }
  const result = await engineRequest(runtime, 'samples', payload, 60_000)
  return result
}

export async function opFormats(runtime) {
  const status = await engineStatus(runtime)
  return { formats: DATA_FORMATS, engine: status, export_kinds: EXPORT_KINDS }
}

export async function opReset(runtime) {
  const result = await engineRequest(runtime, 'reset', {}, 30_000)
  runtime.lastState = null
  runtime.lastPlot = null
  runtime.activity = []
  return result
}

export async function opEngineRestart(runtime) {
  const engine = runtime.engine
  if (engine) {
    try { await engine.dispose() } catch { /* ignore */ }
  }
  runtime.engine = null
  runtime.engineInfo = null
  runtime.engineFail = null
  runtime.lastState = null
  runtime.lastPlot = null
  try {
    const started = await ensureEngine(runtime)  // 真正重启并 ping 握手
    runtime.engineInfo = await started.request('ping', {}, { skipStart: true })
  } catch (error) {
    const status = await engineStatus(runtime)
    return { ok: false, engine: { ...status, error: String(error && error.message ? error.message : error) } }
  }
  runtime.lastState = null
  const status = await engineStatus(runtime)
  return { ok: Boolean(status.ready), engine: status }
}

function pickDataset(args = {}) {
  const payload = {}
  if (args.dataset) payload.dataset = args.dataset
  return payload
}

// ---------------------------------------------------------------------------
// Agent 工具
// ---------------------------------------------------------------------------

async function runToolInner(runtime, args = {}) {
  const action = String(args.action || 'status')
  const actor = 'agent'
  switch (action) {
    case 'status': {
      const state = await opState(runtime)
      const active = (state.datasets || []).find((item) => item.id === state.active)
      const ready = state.engine && state.engine.ready
      const python = state.engine && state.engine.python
      const lines = []
      if (active) {
        lines.push(`PlotLab 工作台就绪：当前数据集 ${active.name}（${active.n_rows} 行 × ${active.n_cols} 列，${active.kind}）` +
          `，已清洗 ${(active.cleaning || []).length} 步，已出图 ${active.history_count || 0} 张` +
          (active.need ? `，最新需求「${active.need}」` : '') + '。')
      } else {
        lines.push(`PlotLab 工作台就绪但没有数据集（Python ${python || '?'}${ready ? '' : '，引擎未就绪'}）。` +
          '可以 action=open 导入本地文件、action=demo 载入示例，或让用户在页面拖拽数据。')
      }
      const suggestions = await opSuggest(runtime, {}).catch(() => ({ suggestions: [] }))
      if ((suggestions.suggestions || []).length) lines.push(`建议下一步：${suggestions.suggestions.map((s) => s.title).join('、')}`)
      return { message: lines.join('\n'), ...state, suggestions: suggestions.suggestions }
    }
    case 'open': {
      const result = await opOpen(runtime, args, actor)
      const dataset = result.dataset
      const profile = await opProfile(runtime, { dataset: dataset.id }).catch(() => null)
      const rec = profile ? (profile.recommend.charts || []).slice(0, 3).map((c) => `${c.title}（${c.id}）`).join('、') : ''
      return {
        message: `已导入「${dataset.name}」：${dataset.n_rows} 行 × ${dataset.n_cols} 列（${dataset.kind}）` +
          `${dataset.group_col ? `，分组列 ${dataset.group_col}` : ''}` +
          `${profile ? `，数据质量 ${profile.score}/100` : ''}。` +
          (rec ? `推荐图表：${rec}。` : '') +
          '下一步：action=auto 传 need 一句话需求（自动清洗+出图+统计），或 action=plot 指定图表。',
        ...compactDatasetResult(result),
        profile: profile ? { score: profile.score, level: profile.level, problems: (profile.checks || []).filter((c) => c.level !== 'ok').map((c) => c.title), recommend: profile.recommend } : null,
      }
    }
    case 'demo': {
      const result = await opDemo(runtime, args, actor)
      return {
        message: `已载入示例「${result.dataset.name}」：${result.dataset.n_rows} 行 × ${result.dataset.n_cols} 列。` +
          '可以用 action=auto need="按分组画箱线图并标显著性" 试试自动分析出图。',
        ...compactDatasetResult(result),
      }
    }
    case 'inspect': {
      const result = await opInspect(runtime, args)
      return {
        message: `格式 ${result.format}：${result.n_rows || '?'} 行 × ${result.n_cols || '?'} 列，判定 ${result.kind || '?'}` +
          `${(result.candidates || []).length ? `，候选 ${(result.candidates || []).join('、')}` : ''}`,
        ...result,
      }
    }
    case 'preview': {
      const result = await opPreview(runtime, args)
      return {
        message: `${result.n_rows} 行 × ${result.n_cols} 列（${result.kind}）` +
          `${result.group_col ? `，分组 ${result.group_col}` : ''}` +
          `；已清洗 ${(result.cleaning || []).length} 步；已出图 ${(result.history || []).length} 张。`,
        dataset: result.dataset, columns: (result.columns || []).slice(0, 40), rows: (result.rows || []).slice(0, 12),
        roles: result.roles, n_rows: result.n_rows, n_cols: result.n_cols, kind: result.kind,
        cleaning: result.cleaning, need: result.need, history: result.history,
      }
    }
    case 'profile': {
      const result = await opProfile(runtime, args)
      return {
        message: `数据质量 ${result.score}/100（${result.level}）：` +
          (result.summary && result.summary.bullets ? result.summary.bullets.join('；') : '') +
          `\n推荐图表：${(result.recommend.charts || []).slice(0, 5).map((c) => c.title + '（' + c.id + '）').join('、')}`,
        score: result.score, level: result.level, checks: result.checks, summary: result.summary,
        recommend: result.recommend, group_col: result.group_col,
      }
    }
    case 'clean': {
      const result = await opClean(runtime, { ...args, actor })
      return {
        message: `清洗完成：应用 ${(result.applied || []).length} 步` +
          (result.applied && result.applied.length ? `（${result.applied.map((a) => a.op).join('→')}）` : '') +
          `，现在 ${result.n_rows} 行 × ${result.n_cols} 列。` +
          ((result.warnings || []).length ? ` 提示：${(result.warnings || []).join('；')}` : ''),
        applied: result.applied,
        skipped: (result.skipped || []).slice(0, 10),
        warnings: result.warnings,
        dataset: result.dataset,
        columns: (result.columns || []).slice(0, 40),
      }
    }
    case 'reset_data': {
      const result = await opResetData(runtime, args)
      return { message: `已恢复到导入时的数据：${result.n_rows} 行 × ${result.n_cols} 列。`, ...result }
    }
    case 'roles': {
      const result = await opRoles(runtime, args)
      return { message: `角色映射已更新：${Object.entries(result.roles || {}).map(([k, v]) => `${k}=${v}`).join('，') || '（空）'}`, ...result }
    }
    case 'charts': {
      const result = await opCharts(runtime, args)
      const groups = {}
      for (const chart of result.charts || []) {
        groups[chart.group] = groups[chart.group] || []
        groups[chart.group].push(chart.id)
      }
      return {
        message: `${(result.charts || []).length} 种图表：` + Object.entries(groups).map(([g, ids]) => `${g}:${ids.join('/')}`).join('；') +
          `\n${(result.stats || []).length} 项统计：${(result.stats || []).map((s) => s.id).join('、')}`,
        charts: (result.charts || []).map((c) => ({ id: c.id, title: c.title, group: c.group, roles: c.roles, params: (c.params || []).map((p) => p.key) })),
        stats: (result.stats || []).map((s) => ({ id: s.id, title: s.title, group: s.group })),
        fonts: result.styles && result.styles.fonts,
        palettes: result.styles && result.styles.palettes,
      }
    }
    case 'plot': {
      const result = await opPlot(runtime, { ...args, actor })
      const code = result.code && result.code.python ? result.code.python : ''
      return {
        message: `已出图：${result.chart_title}（${result.elapsed_ms}ms）\n图片：${result.figure.file}\n` +
          ((result.summary && result.summary.bullets) ? result.summary.bullets.join('；') + '\n' : '') +
          ((result.warnings || []).length ? `提示：${(result.warnings || []).slice(0, 5).join('；')}` : '') +
          (code ? '\n（已生成可复现代码，可用 action=code 取完整脚本或 action=export what=code 导出）' : ''),
        figure: result.figure,
        figures: result.figures,
        figure_run_id: result.figure_run_id,
        data_digest: result.data_digest,
        chart: result.chart,
        chart_title: result.chart_title,
        summary: result.summary,
        warnings: result.warnings,
        roles: result.roles,
        params: result.params,
        style: shrinkStyle(result.style),
        tables: (result.tables || []).slice(0, 4),
        code_info: { python_chars: code.length, r_chars: (result.code && result.code.r || '').length },
        dataset: result.dataset,
      }
    }
    case 'stat': {
      const result = await opStat(runtime, { ...args, actor })
      return {
        message: `${result.title || result.stat}（${result.method || ''}）\n` +
          ((result.summary && result.summary.bullets) || []).join('；') +
          ((result.figures || []).length ? `\n图：${result.figures.map((f) => f.file).join('、')}` : ''),
        title: result.title,
        method: result.method,
        summary: result.summary,
        tables: result.tables,
        figures: result.figures,
        report: result.report,
        warnings: result.warnings,
      }
    }
    case 'auto': {
      const result = await opAuto(runtime, { ...args, actor })
      const plan = result.plan || {}
      const cleaning = result.cleaning && (result.cleaning.applied || []).length
        ? `清洗 ${(result.cleaning.applied || []).map((a) => a.op).join('→')}`
        : '无需额外清洗'
      const stats = (result.stats || []).map((s) => `【${s.title}】${(s.summary && s.summary.bullets || []).slice(0, 2).join('；')}`).join('\n')
      return {
        message: `按需求「${result.need}」执行：\n` +
          `计划：${(plan.steps || []).map((s) => s.label).join(' → ')}\n` +
          `${cleaning}\n图表：${result.chart_title}（${result.chart}）\n图片：${result.figure && result.figure.file}\n` +
          ((result.summary && result.summary.bullets) ? result.summary.bullets.join('；') + '\n' : '') +
          (stats ? stats + '\n' : '') +
          `（风格：${describeStyle(result.style)}；可用 action=export what=bundle 导出图片+代码+报告）`,
        plan: { chart: result.chart, chart_title: result.chart_title, steps: plan.steps, rationale: plan.rationale, roles: result.roles },
        cleaning: result.cleaning ? { applied: (result.cleaning.applied || []).map((a) => a.op), warnings: result.cleaning.warnings, n_rows: result.cleaning.n_rows } : null,
        figure: result.figure,
        figures: result.figures,
        summary: result.summary,
        style: shrinkStyle(result.style),
        stats: (result.stats || []).map((s) => ({ stat: s.stat, title: s.title, method: s.method, summary: s.summary, tables: (s.tables || []).slice(0, 3) })),
        warnings: result.warnings,
        dataset: result.dataset,
      }
    }
    case 'code': {
      const result = await opCode(runtime, args)
      const python = String(result.python || '')
      return {
        message: `已生成 ${result.chart ? result.chart + ' 的' : ''}可复现 Python 代码（${python.split('\n').length} 行）` +
          (result.r ? '，另附 R/ggplot2 片段' : '') + '。\n```python\n' + truncate(python, 6000) + '\n```',
        chart: result.chart,
        python,
        r: result.r,
        filename: result.filename,
      }
    }
    case 'export': {
      const result = await opExport(runtime, { ...args, actor })
      return {
        message: `已导出 ${result.what} → ${result.file}\n文件：${(result.files || []).map((f) => f.label || f.path).join('、')}`,
        what: result.what,
        file: result.file,
        dir: result.dir,
        files: result.files,
        bytes: result.bytes,
        figure_run_id: result.figure_run_id,
        data_digest: result.data_digest,
        figure_binding: result.figure_binding,
      }
    }
    case 'records': {
      const result = await opRecords(runtime, args)
      return {
        message: `历史记录 ${result.count} 条：` + (result.history || []).slice(-5).map((h) => `${h.chart_title}（${h.at}）`).join('；'),
        ...result,
      }
    }
    case 'audit': {
      const result = await opAudit(runtime, args)
      const bad = (result.checks || []).filter((c) => c.level !== 'ok')
      return {
        message: `自检得分 ${result.score}/100（${result.level}）` +
          (bad.length ? `；待改进：${bad.map((c) => c.title).join('、')}` : ''),
        ...result,
      }
    }
    case 'suggest': {
      const result = await opSuggest(runtime, args)
      return { message: (result.suggestions || []).map((s) => `${s.title}：${s.detail}`).join('\n'), ...result }
    }
    case 'need': {
      const result = await opNeed(runtime, { ...args, actor })
      return { message: `已记录需求：「${result.need}」。下一步可以 action=auto need="${result.need}" 直接执行，或在页面点「分析并出图」。`, ...result }
    }
    case 'samples': {
      const result = await opSamples(runtime, args)
      return { message: `样本表已更新（${(result.sample_info || []).length} 个样本）`, ...result }
    }
    case 'formats': {
      const result = await opFormats(runtime)
      return {
        message: `支持导入格式：${(result.formats || []).join('、')}；导出：${(result.export_kinds || []).join('、')}。` +
          `Python ${result.engine && result.engine.version ? result.engine.version : '?'}，可用字体 ${((result.engine && result.engine.fonts) || []).length} 种。`,
        ...result,
      }
    }
    case 'reset': {
      const result = await opReset(runtime)
      return { message: '工作台已清空（数据集/历史/图）。', ...result }
    }
    case 'engine_restart': {
      const result = await opEngineRestart(runtime)
      return { message: result.ok ? 'Python 引擎已重启。' : `引擎重启失败：${result.engine && result.engine.error}`, ...result }
    }
    default:
      throw Object.assign(new Error(`未知 action "${action}"，可用：${ACTIONS.join('/')}`), { code: 'bad_request' })
  }
}

/** 对外入口：先跑业务，再把结果裁剪成 lossless JSON（undefined/NaN 一律转 null）。 */
export async function runTool(runtime, args = {}) {
  return sanitizeValue(await runToolInner(runtime, args), 0)
}

function compactDatasetResult(result) {
  return {
    dataset: result.dataset,
    preview: result.preview ? {
      n_rows: result.preview.n_rows, n_cols: result.preview.n_cols, kind: result.preview.kind,
      roles: result.preview.roles, group_col: result.preview.group_col, groups: result.preview.groups,
      columns: (result.preview.columns || []).slice(0, 40), rows: (result.preview.rows || []).slice(0, 8),
    } : undefined,
    notes: result.notes,
    warnings: result.warnings,
  }
}

function shrinkStyle(style) {
  if (!style) return undefined
  const keys = ['font', 'title_size', 'label_size', 'tick_size', 'legend_size', 'fig_width', 'fig_height',
    'dpi', 'theme', 'palette', 'legend_pos', 'error', 'x_rotate', 'y_log', 'stat_annot', 'p_adjust']
  const out = {}
  for (const key of keys) if (style[key] !== undefined) out[key] = style[key]
  return out
}

function describeStyle(style) {
  if (!style) return ''
  return `${style.font} / ${style.palette} / 标题 ${style.title_size}pt / ${style.fig_width}×${style.fig_height}in @${style.dpi}dpi`
}

/**
 * 工具输出必须是 lossless JSON：递归去掉 undefined、把 NaN/Infinity 转 null、
 * Date 转 ISO 字符串。否则宿主会拒绝整个工具结果（实测 demo 返回 warnings: undefined 即触发）。
 */
function sanitizeValue(value, depth) {
  if ((depth || 0) > 20) return null
  if (value === undefined || value === null) return null
  const type = typeof value
  if (type === 'number') return Number.isFinite(value) ? value : null
  if (type === 'string' || type === 'boolean') return value
  if (type === 'bigint') return String(value)
  if (Array.isArray(value)) return value.map((item) => sanitizeValue(item, (depth || 0) + 1))
  if (value instanceof Date) return value.toISOString()
  if (type === 'object') {
    if (value instanceof Map) {
      const out = {}
      for (const [key, item] of value.entries()) out[String(key)] = sanitizeValue(item, (depth || 0) + 1)
      return out
    }
    const out = {}
    for (const key of Object.keys(value)) {
      const item = value[key]
      if (item === undefined) continue
      out[key] = sanitizeValue(item, (depth || 0) + 1)
    }
    return out
  }
  return String(value)
}

function truncate(text, limit) {
  const value = String(text || '')
  return value.length <= limit ? value : value.slice(0, limit) + `\n... (共 ${value.length} 字符，完整版用 action=code 或 action=export what=code)`
}

function buildToolDefinition(hostRuntime) {
  const def = {
    name: 'plotlab',
    description:
      'PlotLab 绘图工作台（本机 Host 插件，带可视化页面与 Python/matplotlib 计算引擎）。用于：拖入 CSV/TSV/XLSX/JSON 数据后做数据剖析与清洗、按一句话需求自动选图出图、22 种图表（柱状/箱线/小提琴/雨云/散点/气泡/折线/热图/相关/PCA/ROC/生存/森林/直方/密度/QQ 等）、页面/接口级样式调整（字体/字号/颜色/配色/画布/图例/显著性标注）、统计分析（描述/组间比较/相关/回归/卡方/正态性/ROC/生存/PCA）、导出 300dpi PNG + 可复现 Python/R 代码 + report.md/methods.md/protocol.json。' +
      'action: status=工作台状态与建议; open=导入本地文件或 URL（path/url，可带 kind/data_format/sheet/delimiter/transpose/sample_sheet）; demo=载入示例（name=clinical/expression/dose）; inspect=只探测格式; preview=表格预览与列类型; profile=数据质量剖析（0-100 分+逐项问题+推荐图表）; clean=清洗（auto=true 自动，或 plan=[{op,params}]）; reset_data=恢复导入时数据; roles=设置列角色（x/y/group/color/size/facet/time/event）; charts=图表与统计目录; plot=按图表 id 出图（chart + params + style + roles）; stat=统计分析（stat + params）; auto=一句话需求自动清洗+选图+出图+统计（need 必填）; code=生成可复现完整脚本（what=plot/session）; export=导出（what=figure/code/report/methods/protocol/cleaned/stats/bundle）; records=历史记录; audit=确定性自检; suggest=建议下一步; need=记录用户需求; samples=样本表; formats=支持格式; reset=清空; engine_restart=重启引擎。' +
      '用户提到画图/作图/出图/可视化/图表/统计检验/相关性/生存曲线/ROC/字体字号配色调整时，优先用本工具驱动这个工作台（与页面按钮同一套实现），不要自己写脚本重跑；返回的 figure.file 是 PNG 绝对路径，可直接用 read_image 查看。' +
      '典型流程：open 或 demo → profile → auto(need) 或 plot → stat → export what=bundle。',
    parameters: {
      action: {
        type: 'string',
        enum: ACTIONS,
        description: '要执行的操作',
      },
      dataset: { type: 'string', description: '数据集 id（如 ds1）；省略则用当前活动数据集' },
      path: { type: 'string', description: '本地文件/目录绝对路径（action=open/inspect/samples op=join）' },
      url: { type: 'string', description: 'http(s)/ftp 直链（action=open/inspect）' },
      name: { type: 'string', description: '数据集/示例名（demo: clinical/expression/dose）' },
      data_format: { type: 'string', enum: DATA_FORMATS, description: '强制格式（默认 auto）' },
      delimiter: { type: 'string', description: '分隔符（默认自动嗅探）' },
      encoding: { type: 'string', description: '编码（默认自动尝试 utf-8/gbk）' },
      sheet: { type: 'string', description: 'Excel 的 sheet 名' },
      header: { type: 'boolean', description: '首行是否为表头（默认 true）' },
      kind: { type: 'string', enum: ['auto', 'tidy', 'matrix'], description: '数据结构：tidy=一行一个观测；matrix=一行一个特征、其余列为样本' },
      feature_col: { type: 'string', description: 'matrix 数据的特征 id 列（默认首列）' },
      transpose: { type: 'boolean', description: '矩阵转置（样本×特征 转成 特征×样本）' },
      sample_sheet: { type: 'string', description: '样本信息表路径（含 sample + group 列，用于矩阵数据的分组着色）' },
      max_rows: { type: 'number', description: '最多读取行数（默认 20 万）' },
      member: { type: 'string', description: '压缩包内的文件名' },
      need: { type: 'string', description: '一句话自然语言需求（action=auto/need），例如「按分组画箱线图比较 score 并标显著性，配色用色盲友好」' },
      chart: { type: 'string', description: '图表 id（action=plot/code/export；可选值见 action=charts）' },
      params: { type: 'object', description: '图表/统计参数（如 bar: {agg:"mean",error:"se"}；scatter: {add_fit:true,add_corr:true}；compare: {paired:false}）', additionalProperties: true },
      style: {
        type: 'object',
        description: '样式覆盖：font/title_size/label_size/tick_size/legend_size/fig_width/fig_height/dpi/theme/palette/colors{分组名:"#RRGGBB"}/grid/frame/legend_pos/error/x_rotate/x_log/y_log/stat_annot/p_adjust/annot_style 等',
        additionalProperties: true,
      },
      roles: { type: 'object', description: '列角色映射 {x,y,group,color,size,facet,time,event}', additionalProperties: true },
      stat: { type: 'string', description: '统计 id（action=stat/export；可选值见 action=charts 返回的 stats）' },
      stat_params: { type: 'object', description: '统计参数（如 {paired:true}、{method:"spearman"}、{positive:"Yes"}）', additionalProperties: true },
      what: { type: 'string', enum: EXPORT_KINDS, description: 'action=code: plot/session；action=export: 导出类型' },
      img_format: { type: 'string', enum: ['png', 'pdf', 'jpg'], description: '图片格式（默认 png，300dpi）' },
      out_path: { type: 'string', description: '导出目录路径（可选）' },
      figure_run_id: { type: 'string', description: '导出绑定的 figure run id（action=plot/records 返回；按 id 可回查/固定某一次渲染的历史版本）' },
      op: { type: 'string', description: 'action=samples 子操作：list/join/set_group/update' },
      column: { type: 'string', description: 'action=samples op=set_group 的列名' },
      samples: { type: 'array', description: 'action=samples op=update 的样本列表', items: { type: 'object' } },
      plan: { type: 'array', description: 'action=clean 的清洗计划 [{op,params}]（省略则 auto=true 自动计划）', items: { type: 'object' } },
      auto: { type: 'boolean', description: 'action=clean 是否自动生成计划（默认 true）' },
      limit: { type: 'number', description: '预览行数（默认 30）' },
    },
    output: {
      schema: { type: 'object' },
      render: (_args, value) => {
        const text = value && typeof value.message === 'string' ? value.message : JSON.stringify(value)
        return [{ type: 'text', text: String(text) }]
      },
    },
    execute: async (args) => {
      const fresh = await import(new URL(`./impl.mjs?rev=${Date.now()}-${Math.random().toString(36).slice(2)}`, import.meta.url).href)
      const result = await fresh.runTool(hostRuntime, args || {})
      return sanitizeValue(result, 0)
    },
  }
  const reservedKeywords = new Set(['type', 'enum', 'const', 'multipleOf', 'maximum', 'exclusiveMaximum', 'minimum', 'exclusiveMinimum', 'maxLength', 'minLength', 'pattern', 'maxItems', 'minItems', 'uniqueItems', 'maxContains', 'minContains', 'maxProperties', 'minProperties', 'required', 'dependentRequired', 'properties', 'patternProperties', 'additionalProperties', 'propertyNames', 'items', 'prefixItems', 'additionalItems', 'contains', 'title', 'description', 'default', 'deprecated', 'readOnly', 'writeOnly', 'examples', 'format', 'contentEncoding', 'contentMediaType', 'contentSchema', 'if', 'then', 'else', 'allOf', 'anyOf', 'oneOf', 'not'])
  const clash = Object.keys(def.parameters).filter((key) => reservedKeywords.has(key))
  if (clash.length) throw new Error(`[dsh-plotlab] 参数名 ${clash.join('、')} 是 JSON Schema 保留关键字，请改名`)
  return def
}

const TOOL_SCHEMA_VERSION = 1

export function ensureToolRegistered(runtime) {
  if (runtime.toolRegistered && runtime.toolSchemaVersion === TOOL_SCHEMA_VERSION) return runtime.toolRegistered
  try {
    const ctx = runtime.ctx
    const tools = ctx && typeof ctx.get === 'function' ? ctx.get('tools') : ctx && ctx.tools
    if (!tools || typeof tools.register !== 'function') {
      runtime.toolRegistered = false
      return false
    }
    if (runtime.toolRegistered) {
      if (typeof runtime.toolRegistered !== 'function') return runtime.toolRegistered
      try { runtime.toolRegistered() } catch (error) {
        try { console.warn('[dsh-plotlab] old tool dispose failed:', error && error.message ? error.message : error) } catch { /* ignore */ }
      }
      runtime.toolRegistered = false
    }
    const dispose = tools.register(buildToolDefinition(runtime))
    runtime.toolRegistered = typeof dispose === 'function' ? dispose : true
    runtime.toolSchemaVersion = TOOL_SCHEMA_VERSION
    return true
  } catch (error) {
    runtime.toolRegistered = false
    try { console.warn('[dsh-plotlab] agent tool unavailable:', error && error.message ? error.message : error) } catch { /* ignore */ }
    return false
  }
}

// ---------------------------------------------------------------------------
// Prompt section
// ---------------------------------------------------------------------------

export function renderPromptSection(runtime) {
  const state = runtime.lastState
  const lines = ['PlotLab 绘图工作台（Host 插件 dsh-plotlab）已加载。']
  if (state) {
    lines.push(
      `当前数据集：${state.name || state.id}（${state.n_rows} 行 × ${state.n_cols} 列，${state.kind}）` +
      `${state.group_col ? `，分组列 ${state.group_col}` : ''}，已清洗 ${state.cleaning || 0} 步，已出图 ${state.history || 0} 张。`,
    )
    if (state.need) lines.push(`用户最新需求：「${state.need}」—— 优先用 plotlab 工具 action=auto 执行（自动清洗+出图+统计）。`)
  }
  lines.push(
    '用户要画图/作图/出图/改字体字号颜色/做统计检验/导出图片或代码时，优先调用 plotlab 工具操作这个工作台' +
      '（与页面按钮同一套实现），不要另写脚本重画；返回的 figure.file 是 PNG 绝对路径，可直接用 read_image 查看。' +
      '风格要求（字体/字号/颜色/配色/画布）通过 style 参数传入，例如 {"palette":"npg","title_size":16,"colors":{"Control":"#4E79A7"}}。',
  )
  return lines.join('\n')
}

export function ensurePromptSection(runtime) {
  if (runtime.promptRegistered) return runtime.promptRegistered
  try {
    const ctx = runtime.ctx
    const systemPrompt = ctx && typeof ctx.get === 'function' ? ctx.get('systemPrompt') : ctx && ctx.systemPrompt
    if (!systemPrompt || typeof systemPrompt.section !== 'function') {
      runtime.promptRegistered = false
      return false
    }
    const dispose = systemPrompt.section({
      name: 'dsh-plotlab-workbench',
      order: 768,
      text: () => renderPromptSection(runtime),
    })
    runtime.promptRegistered = typeof dispose === 'function' ? dispose : true
    return true
  } catch (error) {
    runtime.promptRegistered = false
    try { console.warn('[dsh-plotlab] prompt section unavailable:', error && error.message ? error.message : error) } catch { /* ignore */ }
    return false
  }
}

// ---------------------------------------------------------------------------
// 上传
// ---------------------------------------------------------------------------

async function handleUpload(runtime, req, res) {
  // 浏览器不允许非 ASCII 请求头，客户端会 encodeURIComponent；这里解码回中文文件名
  let rawName = String(req.headers['x-dsh-filename'] || 'upload.csv')
  try { rawName = decodeURIComponent(rawName) } catch { /* 保留原值 */ }
  const filename = sanitizeName(rawName)
  const engine = await ensureEngine(runtime)
  const dest = path.join(engine.dirs.uploads, `${Date.now().toString(36)}-${randomUUID().slice(0, 8)}-${filename}`)
  let size = 0
  await pipeline(
    req,
    async function* count(source) {
      for await (const chunk of source) {
        size += chunk.length
        if (size > MAX_UPLOAD_BYTES) throw Object.assign(new Error('文件超过 2 GiB 上限'), { code: 'bad_request' })
        yield chunk
      }
    },
    createWriteStream(dest),
  )
  const result = await opOpen(runtime, { path: dest, name: filename }, 'user')
  return { ok: true, ...result, upload: { path: dest, bytes: size } }
}

// ---------------------------------------------------------------------------
// 图片服务
// ---------------------------------------------------------------------------

const CONTENT_TYPES = { '.png': 'image/png', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.pdf': 'application/pdf', '.svg': 'image/svg+xml' }

async function serveFigure(runtime, url, res) {
  const id = url.searchParams.get('id')
  const rawPath = url.searchParams.get('path')
  let file = ''
  if (rawPath) {
    file = path.resolve(String(rawPath))
  } else if (id && runtime.lastPlot) {
    const figures = runtime.lastPlot.figures || (runtime.lastPlot.figure ? [runtime.lastPlot.figure] : [])
    const hit = figures.find((item) => item.id === id)
    if (hit) file = hit.file
  }
  if (!file && id) {
    // P0-3：lastPlot 是内存态；重启后按图 id 去已提交的 runs/<run_id>/figures 里回查
    try {
      const engine = await ensureEngine(runtime, { allowStart: false })
      const artifact = engine && (engine.resolveArtifact(id, 'figure') ||
        engine.resolveArtifact(`${id}.png`, 'figure'))
      const byPrefix = artifact || (() => {
        try {
          const runs = engine.listRuns({ limit: 200 })
          for (const run of runs) {
            const hit = (run.artifacts || []).find((item) => item.kind === 'figure' && item.name.startsWith(`${id}_`))
            if (hit) return hit
          }
        } catch { /* ignore */ }
        return null
      })()
      if (byPrefix && byPrefix.file) file = byPrefix.file
    } catch { /* 引擎未就绪时保持 not_found */ }
  }
  if (!file) throw Object.assign(new Error('图片不存在或已过期，请重新出图'), { code: 'not_found' })
  const roots = []
  try {
    const engine = await ensureEngine(runtime)
    roots.push(path.resolve(engine.workdir))
  } catch { /* 引擎未启动 */ }
  const inside = roots.some((root) => file === root || file.startsWith(root + path.sep))
  if (!inside) throw Object.assign(new Error('拒绝：只能读取工作台工作目录内的图片'), { code: 'forbidden' })
  let stat
  try { stat = statSync(file) } catch {
    throw Object.assign(new Error('图片文件已被清理，请重新生成'), { code: 'not_found' })
  }
  const ext = path.extname(file).toLowerCase()
  res.writeHead(200, {
    'content-type': CONTENT_TYPES[ext] || 'application/octet-stream',
    'cache-control': 'private, max-age=3600',
    'content-length': String(stat.size),
  })
  await pipeline(createReadStream(file), res)
}

// ---------------------------------------------------------------------------
// 路由
// ---------------------------------------------------------------------------

export async function handle(runtime, req, res) {
  ensureToolRegistered(runtime)
  ensurePromptSection(runtime)
  if (!hostIsLoopback(req)) {
    return sendJson(res, 403, { ok: false, code: 'forbidden', error: '拒绝：只接受本机（回环地址）请求' })
  }
  const url = new URL(req.url || '/', 'http://127.0.0.1')
  if (!url.pathname.startsWith(PREFIX)) {
    return sendJson(res, 404, { ok: false, code: 'not_found', error: '未知路由' })
  }
  const route = url.pathname.slice(PREFIX.length) || '/'
  const method = (req.method || 'GET').toUpperCase()
  const key = `${method} ${route}`
  try {
    switch (key) {
      case 'GET /health':
        return sendJson(res, 200, await opHealth(runtime))
      case 'GET /state':
        return sendJson(res, 200, await opState(runtime))
      case 'GET /charts':
        return sendJson(res, 200, await opCharts(runtime, { dataset: url.searchParams.get('dataset') || undefined }))
      case 'GET /stats':
        return sendJson(res, 200, await opStatsCatalog(runtime))
      case 'GET /formats':
        return sendJson(res, 200, await opFormats(runtime))
      case 'GET /activity':
        return sendJson(res, 200, activityOf(runtime, Number(url.searchParams.get('since') || 0)))
      case 'GET /figure':
        return await serveFigure(runtime, url, res)
      case 'POST /dataset/open':
        return sendJson(res, 200, await opOpen(runtime, await readJson(req), 'user'))
      case 'POST /dataset/demo':
        return sendJson(res, 200, await opDemo(runtime, await readJson(req), 'user'))
      case 'POST /dataset/upload':
        return sendJson(res, 200, await handleUpload(runtime, req, res))
      case 'POST /dataset/select':
        return sendJson(res, 200, await opSelect(runtime, await readJson(req)))
      case 'POST /dataset/close':
        return sendJson(res, 200, await opClose(runtime, await readJson(req)))
      case 'POST /inspect':
        return sendJson(res, 200, await opInspect(runtime, await readJson(req)))
      case 'POST /preview':
        return sendJson(res, 200, await opPreview(runtime, await readJson(req)))
      case 'POST /profile':
        return sendJson(res, 200, await opProfile(runtime, await readJson(req)))
      case 'POST /clean':
        return sendJson(res, 200, await opClean(runtime, await readJson(req)))
      case 'POST /reset-data':
        return sendJson(res, 200, await opResetData(runtime, await readJson(req)))
      case 'POST /roles':
        return sendJson(res, 200, await opRoles(runtime, await readJson(req)))
      case 'POST /plot':
        return sendJson(res, 200, await opPlot(runtime, await readJson(req)))
      case 'POST /stat':
        return sendJson(res, 200, await opStat(runtime, await readJson(req)))
      case 'POST /auto':
        return sendJson(res, 200, await opAuto(runtime, await readJson(req)))
      case 'POST /code':
        return sendJson(res, 200, await opCode(runtime, await readJson(req)))
      case 'POST /export':
        return sendJson(res, 200, await opExport(runtime, await readJson(req)))
      case 'POST /records':
        return sendJson(res, 200, await opRecords(runtime, await readJson(req)))
      case 'POST /audit':
        return sendJson(res, 200, await opAudit(runtime, await readJson(req)))
      case 'POST /suggest':
        return sendJson(res, 200, await opSuggest(runtime, await readJson(req)))
      case 'POST /need':
        return sendJson(res, 200, await opNeed(runtime, await readJson(req)))
      case 'POST /samples':
        return sendJson(res, 200, await opSamples(runtime, await readJson(req)))
      case 'POST /reset':
        return sendJson(res, 200, await opReset(runtime))
      case 'POST /engine/restart':
        return sendJson(res, 200, await opEngineRestart(runtime))
      default:
        return sendJson(res, 404, { ok: false, code: 'not_found', error: `未知路由 ${key}` })
    }
  } catch (error) {
    const failure = toFailure(error)
    if (failure && failure.body) return sendJson(res, failure.status, failure.body)
    return sendJson(res, 500, { ok: false, code: 'internal', error: String(error && error.message ? error.message : error) })
  }
}
