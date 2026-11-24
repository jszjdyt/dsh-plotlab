/**
 * dsh-plotlab — Host 半边入口（薄且稳定）
 *
 * 只做四件事：
 *   1. 把 `/plotlab/*` 的 exact 路由挂到 webServer，全部转给 lib/impl.mjs；
 *   2. 注册 `plotlab` agent 工具（和 UI 同一套操作实现）；
 *   3. 注册一段 system prompt section，让 Agent 知道工作台与当前数据集/需求；
 *   4. 插件卸载时关掉 Python worker。
 *
 * 业务逻辑都在 lib/impl.mjs / lib/pl_engine.py 里。为了改完 lib 代码不用重启宿主进程，
 * impl 用带 `?rev=<mtime>` 的 URL 动态 import；同一个文件版本只 import 一次。
 */

import { statSync } from 'node:fs'

const PREFIX = '/plotlab'
const VERSION = '0.1.0'

/** 每次调用都拿一份新的 impl（带 rev 查询参数绕开 ESM 缓存）。 */
const IMPL_URL = new URL('./lib/impl.mjs', import.meta.url)

/** 当前 impl.mjs 的版本标识：文件没改就不换模块实例。 */
function implVersion() {
  try { return String(statSync(IMPL_URL).mtimeMs) } catch { return 'static' }
}

let implCache = { version: '', promise: null }

function loadImpl() {
  const version = implVersion()
  if (implCache.promise && implCache.version === version) return implCache.promise
  const url = new URL(`?rev=${encodeURIComponent(version)}`, IMPL_URL)
  const promise = import(url.href)
  implCache = { version, promise }
  promise.catch(() => {
    if (implCache.promise === promise) implCache = { version: '', promise: null }
  })
  return promise
}

export const name = 'dsh-plotlab'

export function apply(ctx, config) {
  const runtime = {
    version: VERSION,
    config: config && typeof config === 'object' ? config : {},
    ctx,
    invocation: 'apply',
  }

  // ---- HTTP 路由：页面通过同源 fetch 调这里 ----
  ctx.inject(['webServer'], (scoped) => {
    const webServer = scoped.webServer
    const dispose = webServer.register({
      kind: 'prefix',
      path: PREFIX,
      handler: (req, res) => dispatch(runtime, req, res),
    })
    scoped.effect(() => () => {
      try { dispose() } catch { /* ignore */ }
    }, 'dsh-plotlab: routes')
  })

  // ---- agent 工具 ----
  ctx.inject(['tools'], () => {
    loadImpl()
      .then((impl) => impl.ensureToolRegistered(runtime))
      .catch((error) => {
        try { console.warn('[dsh-plotlab] agent tool unavailable:', error && error.message ? error.message : error) } catch { /* ignore */ }
      })
  })

  // ---- system prompt section ----
  ctx.inject(['systemPrompt'], () => {
    loadImpl()
      .then((impl) => impl.ensurePromptSection(runtime))
      .catch((error) => {
        try { console.warn('[dsh-plotlab] prompt section unavailable:', error && error.message ? error.message : error) } catch { /* ignore */ }
      })
  })

  // ---- 卸载时回收 Python worker ----
  ctx.effect(() => () => {
    const engine = runtime.engine
    if (engine && typeof engine.dispose === 'function') {
      try { engine.dispose() } catch { /* ignore */ }
    }
  }, 'dsh-plotlab: engine teardown')
}

async function dispatch(runtime, req, res) {
  try {
    const impl = await loadImpl()
    await impl.handle(runtime, req, res)
  } catch (error) {
    const message = error && error.stack ? String(error.stack) : String(error)
    try {
      if (!res.headersSent) {
        const body = JSON.stringify({ ok: false, code: 'internal', error: message })
        res.writeHead(500, { 'content-type': 'application/json; charset=utf-8', 'cache-control': 'no-store' })
        res.end(body)
      } else {
        res.end()
      }
    } catch { /* ignore */ }
    try { console.error('[dsh-plotlab] dispatch failed:', error) } catch { /* ignore */ }
  }
}
