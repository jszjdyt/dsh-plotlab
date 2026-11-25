/**
 * P0-2 / P0-3 回归测试（PlotLab）：出图前冻结数据版本、切数据集不漂移、
 * 同参数重复出图幂等、不同参数产生新 run_id、图落不可变 run 目录可回查。
 *
 * 真实 lib/engine.mjs（队列 + run store + withTask）+ 真实 lib/impl.mjs，
 * Python 侧换成只讲 JSON-lines 的假 worker。
 *
 * 运行：node test/test_p0_state_plot.mjs
 */

import { mkdtempSync, existsSync, readFileSync, readdirSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import path from 'node:path'
import { createEngine } from '../lib/engine.mjs'
import * as impl from '../lib/impl.mjs'

const PLUGIN = 'plotlab'
const TMP = mkdtempSync(path.join(tmpdir(), `dsh-p0-state-${PLUGIN}-`))
const workerPath = path.join(TMP, 'fake_worker.py')

const FAKE_WORKER = `#!/usr/bin/env python3
import json, os, sys, time

def arg(name, default=None):
    try:
        return sys.argv[sys.argv.index(name) + 1]
    except Exception:
        return default

WORKDIR = arg('--workdir', '.')

def note(cmd, args):
    try:
        with open(os.path.join(WORKDIR, 'calls.jsonl'), 'a', encoding='utf-8') as handle:
            handle.write(json.dumps({'cmd': cmd, 'args': args}, ensure_ascii=False) + '\\n')
    except Exception:
        pass

def reply(msg_id, result=None, error=None):
    payload = {'id': msg_id, 'ok': error is None}
    if error is None:
        payload['result'] = result if result is not None else {}
    else:
        payload['error'] = {'code': 'internal', 'message': error}
    sys.stdout.write(json.dumps(payload) + '\\n')
    sys.stdout.flush()

DATASETS = [
    {'id': 'ds1', 'name': 'A', 'kind': 'tidy', 'n_rows': 20, 'n_cols': 4, 'group_col': 'group', 'cleaning': []},
    {'id': 'ds2', 'name': 'B', 'kind': 'tidy', 'n_rows': 30, 'n_cols': 5, 'group_col': 'group', 'cleaning': []},
]
ACTIVE = 'ds1'

def figure(name, kind):
    target = os.path.join(WORKDIR, 'figures', name)
    os.makedirs(os.path.dirname(target), exist_ok=True)
    with open(target, 'wb') as handle:
        handle.write(b'\\x89PNG\\r\\n\\x1a\\n fake')
    return {'id': 'f_' + kind, 'file': target, 'filename': name, 'width': 100, 'height': 80, 'bytes': 16, 'dpi': 300}

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        msg = json.loads(line)
    except Exception:
        continue
    cmd = msg.get('cmd')
    args = msg.get('args') or {}
    note(cmd, args)
    if cmd == 'ping':
        reply(msg.get('id'), {'mode': 'builtin', 'packages': {}, 'fonts': [], 'platform': 'test'})
    elif cmd in ('state', 'list'):
        reply(msg.get('id'), {'datasets': DATASETS, 'active': ACTIVE, 'engine': {'mode': 'builtin'}})
    elif cmd == 'select':
        ACTIVE = str(args.get('dataset') or ACTIVE)
        reply(msg.get('id'), {'active': ACTIVE})
    elif cmd in ('plot', 'stat', 'auto'):
        kind = str(args.get('chart') or args.get('stat') or cmd)
        name = 'f_%s_%s.png' % (kind, str(int(time.time() * 1000))[-6:])
        payload = figure(name, kind)
        payload['chart'] = kind
        payload['chart_title'] = kind
        payload['dataset'] = {'id': args.get('dataset') or ACTIVE, 'n_rows': 20, 'n_cols': 4}
        snapshot = dict(payload)
        payload['figure'] = dict(snapshot)
        payload['figures'] = [snapshot]
        if cmd == 'auto':
            payload['need'] = args.get('need')
        reply(msg.get('id'), payload)
        # 模拟「第一条命令跑完后用户在 UI 里切了数据集」
        if os.path.exists(os.path.join(WORKDIR, 'switch-dataset')):
            ACTIVE = 'ds2'
    elif cmd == 'export':
        what = str(args.get('what') or 'bundle')
        target = os.path.join(WORKDIR, 'exports', what + '.zip')
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(target, 'w', encoding='utf-8') as handle:
            handle.write('bundle')
        reply(msg.get('id'), {'what': what, 'file': target, 'bytes': 6})
    elif cmd == 'shutdown':
        reply(msg.get('id'), {'bye': True})
        break
    else:
        reply(msg.get('id'), {'echo': cmd})
`

writeFileSync(workerPath, FAKE_WORKER, { mode: 0o755 })

let passed = 0
let failed = 0
const failures = []

function check(name, condition, detail = '') {
  if (condition) {
    passed += 1
    console.log(`  PASS ${name}${detail ? `  [${detail}]` : ''}`)
  } else {
    failed += 1
    failures.push(`${name}${detail ? `：${detail}` : ''}`)
    console.log(`  FAIL ${name}${detail ? `  [${detail}]` : ''}`)
  }
}

function readCalls(workdir) {
  const file = path.join(workdir, 'calls.jsonl')
  if (!existsSync(file)) return []
  return readFileSync(file, 'utf8').split('\n').filter(Boolean).map((line) => {
    try { return JSON.parse(line) } catch { return { cmd: '?', args: {} } }
  })
}

function stagingLeftovers(workdir) {
  try { return readdirSync(path.join(workdir, 'staging')) } catch { return [] }
}

async function main() {
  const workdir = path.join(TMP, 'work')
  const engine = createEngine({
    workdir,
    engineScript: workerPath,
    python: process.env.DSH_TEST_PYTHON || 'python3',
  })
  const runtime = { engine, config: {}, ctx: null, activity: [], jobs: new Map() }
  check('0 引擎与 run store 就绪', (await engine.start()).mode === 'builtin' && Boolean(engine.store))

  // -------------------------------------------------------------------------
  // 1. 出图前冻结数据版本：图命令期间 active 被切成 ds2，仍必须用 ds1
  // -------------------------------------------------------------------------
  writeFileSync(path.join(workdir, 'switch-dataset'), '1')
  const first = await impl.opPlot(runtime, { chart: 'box', params: { metric: 'score' } })
  const frozenByActive = first.run_id && engine.getRun(first.run_id).dataset.id
  const plotCalls = readCalls(workdir).filter((item) => item.cmd === 'plot')
  check('1a 出图命令显式带冻结的 dataset（不读活动数据集）',
    plotCalls.length === 1 && plotCalls[0].args.dataset === 'ds1',
    JSON.stringify(plotCalls.map((item) => item.args.dataset)))
  check('1b 返回 run_id 与图路径', Boolean(first.run_id) && Boolean(first.figure && first.figure.file && frozenByActive === 'ds1'),
    `run=${first.run_id} dataset=${frozenByActive}`)

  // -------------------------------------------------------------------------
  // 2. 图进不可变 run 目录：可回查、带哈希
  // -------------------------------------------------------------------------
  const run = engine.getRun(first.run_id)
  const artifact = run && run.artifacts.find((item) => item.kind === 'figure')
  check('2a 图被提交进 runs/<run_id>/figures', Boolean(artifact) && artifact.file.startsWith(run.dir) &&
    existsSync(artifact.file), artifact ? artifact.file : 'no-artifact')
  check('2b 产物带 sha256', Boolean(artifact) && /^[0-9a-f]{64}$/.test(artifact.sha256 || ''),
    artifact && artifact.sha256 ? artifact.sha256.slice(0, 12) : 'no-hash')
  check('2c 按图 id 能回查已提交产物', (() => {
    const hit = engine.resolveArtifact(artifact.name, 'figure')
    return Boolean(hit) && hit.file === artifact.file
  })(), artifact ? artifact.name : 'n/a')

  // -------------------------------------------------------------------------
  // 3. 同参数重复出图幂等（复用同一 run，不重复渲染）
  // -------------------------------------------------------------------------
  const plotCountBefore = readCalls(workdir).filter((item) => item.cmd === 'plot').length
  const second = await impl.opPlot(runtime, { chart: 'box', params: { metric: 'score' }, dataset: 'ds1' })
  const plotCountAfter = readCalls(workdir).filter((item) => item.cmd === 'plot').length
  check('3a 同参数重复出图复用同一 run_id', second.run_id === first.run_id && second.run_reused === true,
    `first=${first.run_id} second=${second.run_id} reused=${second.run_reused}`)
  check('3b 幂等复用时不再渲染', plotCountAfter === plotCountBefore,
    `before=${plotCountBefore} after=${plotCountAfter}`)
  check('3c 幂等复用的图路径指向不可变副本', Boolean(second.figure && second.figure.file) &&
    existsSync(second.figure.file) && second.figure.file.startsWith(run.dir), second.figure && second.figure.file)

  // -------------------------------------------------------------------------
  // 4. 不同参数 → 新 run_id
  // -------------------------------------------------------------------------
  const third = await impl.opPlot(runtime, { chart: 'box', params: { metric: 'other' }, dataset: 'ds1' })
  check('4a 不同参数产生新 run_id', third.run_id !== first.run_id, `a=${first.run_id} b=${third.run_id}`)

  // -------------------------------------------------------------------------
  // 5. auto / export 同样进 run 记录；失败不留 staging
  // -------------------------------------------------------------------------
  const auto = await impl.opAuto(runtime, { need: '按 group 画箱线图比较 score' })
  check('5a auto 有 run_id 与图产物', Boolean(auto.run_id) && existsSync(auto.figure.file), `run=${auto.run_id}`)
  const exported = await impl.opExport(runtime, { what: 'bundle' })
  const exportRun = engine.getRun(exported.run_id)
  check('5b export 登记产物与哈希', Boolean(exportRun) && exportRun.artifacts.length > 0 &&
    exportRun.artifacts.every((item) => /^[0-9a-f]{64}$/.test(item.sha256 || '')), exported.run_id)
  check('5c staging 无残留目录', stagingLeftovers(workdir).length === 0, JSON.stringify(stagingLeftovers(workdir)))

  // -------------------------------------------------------------------------
  // 6. 非法请求（没有数据集）明确失败，不产生已提交 run
  // -------------------------------------------------------------------------
  const before = engine.listRuns({ limit: 100 }).filter((item) => item.status === 'committed').length
  let threw = false
  try {
    await impl.opPlot(runtime, { chart: 'box', dataset: 'ds-never' })
  } catch (error) {
    threw = Boolean(error && ['not_found', 'bad_request'].includes(error.code))
  }
  const after = engine.listRuns({ limit: 100 }).filter((item) => item.status === 'committed').length
  check('6a 不存在的数据集明确失败', threw === true)
  check('6b 失败请求不新增已提交 run', after === before, `before=${before} after=${after}`)

  await engine.dispose()
  try { rmSync(TMP, { recursive: true, force: true }) } catch { /* ignore */ }

  console.log(`\n== P0-2/P0-3 状态回归（${PLUGIN}）：${passed} 通过 / ${failed} 失败 ==`)
  if (failed) {
    for (const item of failures) console.log(`  ✗ ${item}`)
    process.exit(1)
  }
}

main().catch((error) => {
  console.error('测试本身异常：', error)
  try { rmSync(TMP, { recursive: true, force: true }) } catch { /* ignore */ }
  process.exit(1)
})
