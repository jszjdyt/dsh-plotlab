/**
 * dsh-plotlab — 浏览器端 UI（Client 半边）
 *
 * 形态：window.__ModuleLoader__.load({ id:'dsh-plotlab', factory(require){...} }），
 * 只 require('react')，不碰任何 @deepseek-ai Client 包（宿主版本一变就会炸）。
 * 数据面：全部走同源相对路径 /plotlab/*（见 DESIGN.md §7/§8）。
 *
 * 文件分两层：
 *   1) 纯函数层（到 factory 之前）：默认样式、样式字段表、参数默认值、列/角色匹配、
 *      API 客户端、格式化、图 URL、请求体拼装。不碰 DOM，node 里挂到 globalThis.__PL_TEST__ 供单测。
 *   2) 浏览器层（factory 内）：React 组件 + 样式 + 槽注册。
 *
 * 约定：类名一律 pl- 前缀；颜色只用 --dsw-alias-* 主题变量，数据色用 JS 常量 PALETTE。
 */
;(function () {
  'use strict'

  var NS = 'dsh-plotlab'
  var API = '/plotlab'
  var PLUGIN_VERSION = '0.1.0'

  /** 数据色（不参与主题）：色盲友好 Okabe-Ito。 */
  var PALETTE = ['#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#D55E00', '#CC79A7', '#7F7F7F']

  /** 样式默认值（与后端 pl_common.STYLE_DEFAULT 一致；后端会再校验一遍）。 */
  var DEFAULT_STYLE = {
    font: 'auto',
    title_size: 14,
    label_size: 12,
    tick_size: 10,
    legend_size: 10,
    text_size: 10,
    title_bold: true,
    fig_width: 6.2,
    fig_height: 4.4,
    dpi: 300,
    theme: 'classic',
    palette: 'okabe_ito',
    colors: {},
    alpha: 0.85,
    line_width: 1.4,
    point_size: 28,
    bar_width: 0.72,
    edge_width: 0.8,
    grid: 'none',
    frame: 'left_bottom',
    tick_dir: 'out',
    legend_pos: 'right',
    legend_title: '',
    title_text: '',
    x_label: '',
    y_label: '',
    x_rotate: 0,
    y_rotate: 0,
    x_log: false,
    y_log: false,
    x_lim: [null, null],
    y_lim: [null, null],
    error: 'se',
    jitter: true,
    jitter_size: 2.2,
    jitter_alpha: 0.55,
    stat_annot: true,
    stat_method: 'auto',
    p_adjust: 'bh',
    annot_style: 'stars',
    show_n: false,
    legend_inside: false,
  }

  var STYLE_GROUP_LABELS = {
    font: '字体与字号',
    color: '颜色',
    canvas: '画布与导出',
    axis: '坐标轴',
    legend: '图例与标题',
    stat: '统计与图形元素',
  }

  var THEMES = [
    { id: 'classic', label: '经典（全框）' },
    { id: 'minimal', label: '极简（左/下轴）' },
    { id: 'grid', label: '网格' },
    { id: 'void', label: '无框' },
    { id: 'nature', label: 'Nature' },
    { id: 'lancet', label: 'Lancet' },
    { id: 'jama', label: 'JAMA' },
  ]

  var PALETTES = [
    { id: 'okabe_ito', label: 'Okabe-Ito（色盲友好）' },
    { id: 'npg', label: 'NPG（Nature）' },
    { id: 'lancet', label: 'Lancet' },
    { id: 'jama', label: 'JAMA' },
    { id: 'set1', label: 'Set1' },
    { id: 'set2', label: 'Set2' },
    { id: 'tableau', label: 'Tableau' },
    { id: 'pastel', label: '马卡龙' },
    { id: 'grey', label: '灰度' },
    { id: 'viridis', label: 'Viridis（连续）' },
    { id: 'blue_red', label: '蓝红（连续）' },
    { id: 'sequential', label: 'Blues（连续）' },
  ]

  var FONTS = [
    { id: 'auto', label: '自动（Arial + 中文回退）' },
    { id: 'Arial', label: 'Arial' },
    { id: 'Helvetica', label: 'Helvetica' },
    { id: 'Times New Roman', label: 'Times New Roman（衬线）' },
    { id: 'Georgia', label: 'Georgia（衬线）' },
    { id: 'DejaVu Sans', label: 'DejaVu Sans' },
    { id: 'Arial Unicode MS', label: 'Arial Unicode MS（中文）' },
    { id: 'PingFang SC', label: '苹方（中文）' },
    { id: 'Songti SC', label: '宋体（中文）' },
    { id: 'Hiragino Sans GB', label: '冬青黑体（中文）' },
  ]

  var ROLE_DEFS = [
    { key: 'x', label: 'X 轴 / 分类', hint: '柱状图的组、散点图的横轴', kind: 'any' },
    { key: 'y', label: 'Y 轴 / 数值', hint: '要展示的数值列', kind: 'numeric' },
    { key: 'group', label: '分组', hint: '颜色/比较分组（如 group/condition）', kind: 'any' },
    { key: 'color', label: '颜色', hint: '第二分组或连续着色', kind: 'any' },
    { key: 'size', label: '大小', hint: '气泡大小', kind: 'numeric' },
    { key: 'facet', label: '分面', hint: '按列拆子图', kind: 'any' },
    { key: 'time', label: '时间', hint: '生存时间（月/天）', kind: 'numeric' },
    { key: 'event', label: '事件', hint: '生存事件 0/1', kind: 'any' },
  ]

  /** 样式面板字段表：group 决定分组，type 决定控件。 */
  var STYLE_FIELDS = [
    { key: 'font', group: 'font', label: '字体', type: 'select', options: FONTS },
    { key: 'title_size', group: 'font', label: '标题字号', type: 'number', min: 6, max: 40, step: 1, unit: 'pt' },
    { key: 'label_size', group: 'font', label: '轴标题字号', type: 'number', min: 6, max: 40, step: 1, unit: 'pt' },
    { key: 'tick_size', group: 'font', label: '刻度字号', type: 'number', min: 5, max: 40, step: 1, unit: 'pt' },
    { key: 'legend_size', group: 'font', label: '图例字号', type: 'number', min: 5, max: 40, step: 1, unit: 'pt' },
    { key: 'text_size', group: 'font', label: '注释字号', type: 'number', min: 5, max: 40, step: 1, unit: 'pt' },
    { key: 'title_bold', group: 'font', label: '标题加粗', type: 'boolean' },

    { key: 'palette', group: 'color', label: '配色板', type: 'select', options: PALETTES },
    { key: 'colors', group: 'color', label: '自定义分组颜色', type: 'colors' },
    { key: 'alpha', group: 'color', label: '透明度', type: 'number', min: 0.05, max: 1, step: 0.05 },

    { key: 'theme', group: 'canvas', label: '主题', type: 'select', options: THEMES },
    { key: 'fig_width', group: 'canvas', label: '画布宽', type: 'number', min: 2, max: 20, step: 0.2, unit: 'in' },
    { key: 'fig_height', group: 'canvas', label: '画布高', type: 'number', min: 2, max: 20, step: 0.2, unit: 'in' },
    { key: 'dpi', group: 'canvas', label: '导出分辨率', type: 'select', options: [
      { id: 150, label: '150 dpi（草稿）' }, { id: 300, label: '300 dpi（推荐）' }, { id: 600, label: '600 dpi（印刷）' },
    ] },

    { key: 'grid', group: 'axis', label: '网格线', type: 'select', options: [
      { id: 'none', label: '无' }, { id: 'x', label: '仅 X' }, { id: 'y', label: '仅 Y' }, { id: 'both', label: '全网格' },
    ] },
    { key: 'frame', group: 'axis', label: '边框', type: 'select', options: [
      { id: 'left_bottom', label: '左+下轴' }, { id: 'full', label: '全框' }, { id: 'none', label: '无框' },
    ] },
    { key: 'tick_dir', group: 'axis', label: '刻度方向', type: 'select', options: [
      { id: 'out', label: '朝外' }, { id: 'in', label: '朝内' },
    ] },
    { key: 'x_rotate', group: 'axis', label: 'X 标签旋转', type: 'select', options: [
      { id: 0, label: '0°' }, { id: 30, label: '30°' }, { id: 45, label: '45°' }, { id: 90, label: '90°' },
    ] },
    { key: 'y_rotate', group: 'axis', label: 'Y 标签旋转', type: 'select', options: [
      { id: 0, label: '0°' }, { id: 45, label: '45°' }, { id: 90, label: '90°' },
    ] },
    { key: 'x_log', group: 'axis', label: 'X 轴对数', type: 'boolean' },
    { key: 'y_log', group: 'axis', label: 'Y 轴对数', type: 'boolean' },
    { key: 'x_lim', group: 'axis', label: 'X 轴范围', type: 'range' },
    { key: 'y_lim', group: 'axis', label: 'Y 轴范围', type: 'range' },

    { key: 'legend_pos', group: 'legend', label: '图例位置', type: 'select', options: [
      { id: 'right', label: '右侧' }, { id: 'top', label: '上方' }, { id: 'bottom', label: '下方' }, { id: 'none', label: '不显示' },
    ] },
    { key: 'legend_title', group: 'legend', label: '图例标题', type: 'text', placeholder: '默认用列名' },
    { key: 'title_text', group: 'legend', label: '图标题', type: 'text', placeholder: '默认自动生成' },
    { key: 'x_label', group: 'legend', label: 'X 轴标题', type: 'text', placeholder: '默认用列名' },
    { key: 'y_label', group: 'legend', label: 'Y 轴标题', type: 'text', placeholder: '默认用列名' },
    { key: 'show_n', group: 'legend', label: '刻度显示 n=', type: 'boolean' },

    { key: 'stat_annot', group: 'stat', label: '显著性标注', type: 'boolean' },
    { key: 'stat_method', group: 'stat', label: '检验方法', type: 'select', options: [
      { id: 'auto', label: '自动' }, { id: 'welch', label: 'Welch t' }, { id: 'student', label: 'Student t' },
      { id: 'wilcox', label: 'Wilcoxon 秩和' }, { id: 'anova', label: 'ANOVA' }, { id: 'kruskal', label: 'Kruskal-Wallis' },
    ] },
    { key: 'p_adjust', group: 'stat', label: '多重校正', type: 'select', options: [
      { id: 'bh', label: 'BH (FDR)' }, { id: 'bonferroni', label: 'Bonferroni' }, { id: 'holm', label: 'Holm' }, { id: 'none', label: '不校正' },
    ] },
    { key: 'annot_style', group: 'stat', label: '标注样式', type: 'select', options: [
      { id: 'stars', label: '星号' }, { id: 'p', label: 'p 值' }, { id: 'both', label: '星号+p' },
    ] },
    { key: 'error', group: 'stat', label: '误差线', type: 'select', options: [
      { id: 'se', label: '标准误 SE' }, { id: 'sd', label: '标准差 SD' }, { id: 'ci95', label: '95% CI' }, { id: 'none', label: '不显示' },
    ] },
    { key: 'jitter', group: 'stat', label: '散点抖动', type: 'boolean' },
    { key: 'point_size', group: 'stat', label: '点大小', type: 'number', min: 4, max: 200, step: 2 },
    { key: 'line_width', group: 'stat', label: '线宽', type: 'number', min: 0.2, max: 8, step: 0.1 },
    { key: 'bar_width', group: 'stat', label: '柱宽', type: 'number', min: 0.1, max: 1, step: 0.05 },
  ]

  var STYLE_GROUPS = ['font', 'color', 'canvas', 'axis', 'legend', 'stat']

  // ==========================================================================
  // 纯函数层
  // ==========================================================================

  function isNil(value) {
    return value === null || value === undefined
  }

  function clone(value) {
    try { return JSON.parse(JSON.stringify(value)) } catch (error) { return value }
  }

  function classNames() {
    var out = []
    for (var index = 0; index < arguments.length; index += 1) {
      var item = arguments[index]
      if (!item) continue
      if (typeof item === 'string') out.push(item)
      else if (typeof item === 'object') {
        for (var key in item) if (Object.prototype.hasOwnProperty.call(item, key) && item[key]) out.push(key)
      }
    }
    return out.join(' ')
  }

  function formatNumber(value, digits) {
    if (isNil(value) || value === '') return '—'
    var number = Number(value)
    if (!isFinite(number)) return '—'
    var precision = isNil(digits) ? 3 : digits
    if (number === 0) return '0'
    var abs = Math.abs(number)
    if (abs >= 100000 || abs < 0.001) return number.toExponential(1)
    return String(Number(number.toFixed(precision)))
  }

  function formatP(value) {
    if (isNil(value) || value === '') return '—'
    var number = Number(value)
    if (!isFinite(number)) return '—'
    if (number < 0.0001) return '< 1e-4'
    if (number < 0.001) return number.toExponential(1)
    return String(Number(number.toFixed(4)))
  }

  function formatBytes(bytes) {
    var value = Number(bytes) || 0
    if (value < 1024) return value + ' B'
    if (value < 1024 * 1024) return (value / 1024).toFixed(1) + ' KB'
    return (value / 1024 / 1024).toFixed(1) + ' MB'
  }

  function formatTime(iso) {
    if (!iso) return ''
    try {
      var date = new Date(iso)
      if (isNaN(date.getTime())) return String(iso)
      var pad = function (n) { return n < 10 ? '0' + n : String(n) }
      return date.getFullYear() + '-' + pad(date.getMonth() + 1) + '-' + pad(date.getDate()) + ' ' +
        pad(date.getHours()) + ':' + pad(date.getMinutes())
    } catch (error) { return String(iso) }
  }

  function pStars(p) {
    var value = Number(p)
    if (!isFinite(value)) return ''
    if (value < 0.001) return '***'
    if (value < 0.01) return '**'
    if (value < 0.05) return '*'
    return 'ns'
  }

  function chartGroups(charts) {
    var order = []
    var map = {}
    ;(charts || []).forEach(function (chart) {
      var group = chart.group || '其他'
      if (!map[group]) { map[group] = []; order.push(group) }
      map[group].push(chart)
    })
    return order.map(function (group) { return { group: group, charts: map[group] } })
  }

  function findChart(charts, id) {
    var list = charts || []
    for (var index = 0; index < list.length; index += 1) {
      if (list[index].id === id) return list[index]
    }
    return null
  }

  function findStat(stats, id) {
    var list = stats || []
    for (var index = 0; index < list.length; index += 1) {
      if (list[index].id === id) return list[index]
    }
    return null
  }

  function defaultParams(meta) {
    var out = {}
    var params = (meta && meta.params) || []
    params.forEach(function (param) {
      if (param && param.key) out[param.key] = isNil(param.default) ? null : clone(param.default)
    })
    return out
  }

  function defaultRolesFor(meta) {
    var out = {}
    var roles = (meta && meta.recommended_roles) || {}
    for (var key in roles) {
      if (Object.prototype.hasOwnProperty.call(roles, key) && roles[key]) out[key] = roles[key]
    }
    return out
  }

  function mergeStyle(base, patch) {
    var out = clone(base || DEFAULT_STYLE)
    if (patch && typeof patch === 'object') {
      for (var key in patch) {
        if (!Object.prototype.hasOwnProperty.call(patch, key)) continue
        if (key === 'colors' && patch.colors && typeof patch.colors === 'object') {
          out.colors = Object.assign({}, out.colors || {}, patch.colors)
        } else if (key === 'x_lim' || key === 'y_lim') {
          out[key] = Array.isArray(patch[key]) ? patch[key].slice(0, 2) : [null, null]
        } else if (Object.prototype.hasOwnProperty.call(DEFAULT_STYLE, key)) {
          out[key] = patch[key]
        }
      }
    }
    return out
  }

  function styleFieldsByGroup(group) {
    return STYLE_FIELDS.filter(function (field) { return field.group === group })
  }

  function columnsForRole(columns, roleKey) {
    var def = null
    for (var index = 0; index < ROLE_DEFS.length; index += 1) {
      if (ROLE_DEFS[index].key === roleKey) { def = ROLE_DEFS[index]; break }
    }
    var kind = def ? def.kind : 'any'
    return (columns || []).filter(function (column) {
      if (kind === 'numeric') return column.dtype === 'numeric'
      return column.dtype !== 'text' || (roleKey === 'x' || roleKey === 'group')
    })
  }

  function figureUrl(figure, revision) {
    if (!figure) return ''
    var file = figure.file || ''
    if (!file) return figure.url || ''
    return API + '/figure?path=' + encodeURIComponent(file) + '&v=' + encodeURIComponent(String(revision || figure.id || ''))
  }

  function downloadUrl(figure, revision) {
    return figureUrl(figure, revision)
  }

  function buildPlotRequest(state) {
    var body = { chart: state.chart, params: state.params || {}, style: state.style || {}, roles: state.roles || {} }
    if (state.dataset && state.dataset.id) body.dataset = state.dataset.id
    return body
  }

  function buildExportRequest(state, what) {
    var body = { what: what }
    if (state.dataset && state.dataset.id) body.dataset = state.dataset.id
    if (state.chart) body.chart = state.chart
    if (state.params) body.params = state.params
    if (state.style) body.style = state.style
    if (state.roles) body.roles = state.roles
    if (what === 'stats' && state.stat) { body.stat = state.stat; body.stat_params = state.statParams || {} }
    return body
  }

  function countProblems(profile) {
    if (!profile || !profile.checks) return 0
    return profile.checks.filter(function (check) { return check.level !== 'ok' }).length
  }

  function levelTone(level) {
    if (level === 'ok' || level === 'good') return 'ok'
    if (level === 'warn') return 'warn'
    return 'bad'
  }

  function tableColumns(table) {
    return (table && table.columns) || []
  }

  function formatCell(column, row) {
    var value = row ? row[column.key] : undefined
    if (isNil(value)) return '—'
    // 后端统计表已把数值格式化成显示字符串（如 "12.3" / "< 1e-4"），原样展示
    if (typeof value === 'string') return value
    if (column.type === 'p') return formatP(value)
    if (column.type === 'num') return formatNumber(value)
    return String(value)
  }

  function dedupeFigures(figures) {
    var seen = {}
    var out = []
    ;(figures || []).forEach(function (figure) {
      if (!figure) return
      var key = figure.id || figure.file
      if (!key || seen[key]) return
      seen[key] = true
      out.push(figure)
    })
    return out
  }

  function truncate(text, limit) {
    var value = String(text || '')
    if (value.length <= limit) return value
    return value.slice(0, limit) + '\n...（共 ' + value.length + ' 字符）'
  }

  function figureKey(figure) {
    if (!figure) return ''
    return String(figure.id || '') + ':' + String(figure.file || '')
  }

  /** 简易 API 客户端：注入 fetch 便于单测。 */
  function createApi(fetchImpl, baseUrl) {
    var base = baseUrl || API
    var request = fetchImpl || (typeof fetch !== 'undefined' ? fetch.bind(typeof window !== 'undefined' ? window : globalThis) : null)

    function call(path, options) {
      if (!request) return Promise.reject(new Error('当前环境没有 fetch'))
      var init = Object.assign({ headers: { 'content-type': 'application/json' } }, options || {})
      return request(base + path, init).then(function (response) {
        var contentType = (response.headers && response.headers.get && response.headers.get('content-type')) || ''
        if (contentType.indexOf('application/json') < 0) {
          return response.text().then(function (text) {
            throw Object.assign(new Error(text || ('HTTP ' + response.status)), { status: response.status })
          })
        }
        return response.json().then(function (payload) {
          if (payload && payload.ok === false) {
            throw Object.assign(new Error(payload.error || '请求失败'), { code: payload.code, details: payload })
          }
          return payload
        })
      })
    }

    function get(path, query) {
      var search = ''
      if (query) {
        var parts = []
        for (var key in query) {
          if (!Object.prototype.hasOwnProperty.call(query, key)) continue
          if (isNil(query[key]) || query[key] === '') continue
          parts.push(encodeURIComponent(key) + '=' + encodeURIComponent(query[key]))
        }
        if (parts.length) search = '?' + parts.join('&')
      }
      return call(path + search, { method: 'GET' })
    }

    function post(path, body) {
      return call(path, { method: 'POST', body: JSON.stringify(body || {}) })
    }

    return {
      base: base,
      call: call,
      get: get,
      post: post,
      health: function () { return get('/health') },
      state: function () { return get('/state') },
      charts: function (datasetId) { return get('/charts', { dataset: datasetId }) },
      stats: function () { return get('/stats') },
      formats: function () { return get('/formats') },
      preview: function (datasetId) { return post('/preview', { dataset: datasetId }) },
      profile: function (datasetId) { return post('/profile', { dataset: datasetId }) },
      clean: function (datasetId, plan, auto) { return post('/clean', { dataset: datasetId, plan: plan || undefined, auto: auto !== false }) },
      resetData: function (datasetId) { return post('/reset-data', { dataset: datasetId }) },
      roles: function (datasetId, roles) { return post('/roles', { dataset: datasetId, roles: roles }) },
      plot: function (body) { return post('/plot', body) },
      stat: function (body) { return post('/stat', body) },
      auto: function (datasetId, need) { return post('/auto', { dataset: datasetId, need: need }) },
      need: function (datasetId, need) { return post('/need', { dataset: datasetId, need: need }) },
      code: function (body) { return post('/code', body) },
      exportAll: function (body) { return post('/export', body) },
      records: function (datasetId) { return post('/records', { dataset: datasetId }) },
      audit: function (datasetId) { return post('/audit', { dataset: datasetId }) },
      suggest: function (datasetId) { return post('/suggest', { dataset: datasetId }) },
      demo: function (name) { return post('/dataset/demo', { name: name || 'clinical' }) },
      demoList: function () { return post('/dataset/demo', { name: '__list__' }) },
      openPath: function (payload) { return post('/dataset/open', payload) },
      select: function (datasetId) { return post('/dataset/select', { dataset: datasetId }) },
      close: function (datasetId) { return post('/dataset/close', { dataset: datasetId }) },
      reset: function () { return post('/reset', {}) },
      restart: function () { return post('/engine/restart', {}) },
      upload: function (file, extra) {
        if (!request) return Promise.reject(new Error('当前环境没有 fetch'))
        var headers = { 'x-dsh-filename': encodeURIComponent(file.name || 'upload.csv') }
        if (extra && extra.kind) headers['x-dsh-kind'] = String(extra.kind)
        return request(base + '/dataset/upload', { method: 'POST', headers: headers, body: file }).then(function (response) {
          return response.json().then(function (payload) {
            if (payload && payload.ok === false) throw Object.assign(new Error(payload.error || '上传失败'), { code: payload.code, details: payload })
            return payload
          })
        })
      },
      figureUrl: figureUrl,
    }
  }

  // == 纯函数层结束 ==
  var PL_TEST = {
    NS: NS,
    API: API,
    DEFAULT_STYLE: DEFAULT_STYLE,
    STYLE_FIELDS: STYLE_FIELDS,
    ROLE_DEFS: ROLE_DEFS,
    THEMES: THEMES,
    PALETTES: PALETTES,
    FONTS: FONTS,
    chartGroups: chartGroups,
    findChart: findChart,
    findStat: findStat,
    defaultParams: defaultParams,
    defaultRolesFor: defaultRolesFor,
    mergeStyle: mergeStyle,
    styleFieldsByGroup: styleFieldsByGroup,
    columnsForRole: columnsForRole,
    figureUrl: figureUrl,
    buildPlotRequest: buildPlotRequest,
    buildExportRequest: buildExportRequest,
    countProblems: countProblems,
    levelTone: levelTone,
    formatNumber: formatNumber,
    formatP: formatP,
    pStars: pStars,
    formatBytes: formatBytes,
    formatTime: formatTime,
    tableColumns: tableColumns,
    formatCell: formatCell,
    dedupeFigures: dedupeFigures,
    createApi: createApi,
    clone: clone,
    classNames: classNames,
  }

  // ==========================================================================
  // 浏览器层
  // ==========================================================================

  function factory(require) {
    var React = require('react')
    var useState = React.useState
    var useEffect = React.useEffect
    var useRef = React.useRef
    var useMemo = React.useMemo
    var useCallback = React.useCallback

    function h(type, props) {
      var children = Array.prototype.slice.call(arguments, 2)
      return React.createElement.apply(React, [type, props].concat(children))
    }

    var T = function (key) { return String(key) }

    function warn(message, error) {
      try { console.warn('[dsh-plotlab] ' + message, error || '') } catch (ignored) { /* ignore */ }
    }

    // ----------------------------------------------------------------------
    // 样式表（生命周期跟随组件）
    // ----------------------------------------------------------------------
    var CSS = [
      '.pl-root{position:relative;display:flex;flex-direction:column;height:100%;min-height:72vh;overflow:hidden;isolation:isolate;border-radius:12px;background:var(--dsw-alias-bg-base);color:var(--dsw-alias-label-primary);font-size:13px;line-height:1.5;font-variant-numeric:tabular-nums}',
      '.pl-root *,.pl-root *::before,.pl-root *::after{box-sizing:border-box}',
      '.pl-root button,.pl-root input,.pl-root select,.pl-root textarea{font-family:inherit;font-variant-numeric:tabular-nums}',
      '.pl-root ::selection{background:color-mix(in srgb,var(--dsw-alias-brand-primary) 24%,transparent)}',

      /* 顶栏 */
      '.pl-top{display:flex;align-items:center;gap:10px;flex-wrap:wrap;flex:none;min-height:52px;padding:9px 14px;border-bottom:1px solid var(--dsw-alias-border-l1);background:var(--dsw-alias-bg-layer-1)}',
      '.pl-brand{display:flex;align-items:center;gap:9px;min-width:0}',
      '.pl-mark{display:inline-flex;align-items:center;justify-content:center;flex:none;width:30px;height:30px;border-radius:9px;border:1px solid color-mix(in srgb,var(--dsw-alias-brand-primary) 34%,var(--dsw-alias-border-l1));background:linear-gradient(135deg,color-mix(in srgb,var(--dsw-alias-brand-primary) 18%,var(--dsw-alias-bg-layer-1)),var(--dsw-alias-bg-layer-1));color:var(--dsw-alias-brand-primary)}',
      '.pl-title-block{display:flex;flex-direction:column;gap:0;min-width:0}',
      '.pl-title{font-size:14px;font-weight:650;letter-spacing:.01em;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}',
      '.pl-subtitle{font-size:11px;color:var(--dsw-alias-label-secondary);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}',
      '.pl-badge{display:inline-flex;align-items:center;gap:6px;height:22px;padding:0 9px;border:1px solid var(--dsw-alias-border-l1);border-radius:999px;background:var(--dsw-alias-bg-layer-2);color:var(--dsw-alias-label-secondary);font-size:11px;line-height:1;white-space:nowrap}',
      '.pl-badge b{font-weight:600;color:var(--dsw-alias-label-primary)}',
      '.pl-badge-ok{border-color:color-mix(in srgb,var(--dsw-alias-state-success-primary) 40%,var(--dsw-alias-border-l1));background:color-mix(in srgb,var(--dsw-alias-state-success-primary) 9%,var(--dsw-alias-bg-layer-1));color:var(--dsw-alias-state-success-primary)}',
      '.pl-badge-warn{border-color:color-mix(in srgb,var(--dsw-alias-state-warn-primary) 42%,var(--dsw-alias-border-l1));background:color-mix(in srgb,var(--dsw-alias-state-warn-primary) 9%,var(--dsw-alias-bg-layer-1));color:var(--dsw-alias-state-warn-primary)}',
      '.pl-badge-bad{border-color:color-mix(in srgb,var(--dsw-alias-state-error-primary) 42%,var(--dsw-alias-border-l1));background:color-mix(in srgb,var(--dsw-alias-state-error-primary) 9%,var(--dsw-alias-bg-layer-1));color:var(--dsw-alias-state-error-primary)}',
      '.pl-dot{width:6px;height:6px;border-radius:50%;flex:none;background:currentColor;box-shadow:0 0 0 2px color-mix(in srgb,currentColor 18%,transparent)}',
      '.pl-top-actions{display:flex;align-items:center;gap:6px;flex-wrap:wrap;margin-left:auto}',
      '.pl-sep{width:1px;height:18px;background:var(--dsw-alias-border-l1);flex:none}',

      /* 按钮 / 输入 */
      '.pl-btn{display:inline-flex;align-items:center;justify-content:center;gap:5px;height:28px;padding:0 10px;border:1px solid var(--dsw-alias-border-l1);border-radius:8px;background:var(--dsw-alias-bg-layer-2);color:var(--dsw-alias-label-primary);font:inherit;font-size:12px;line-height:1;cursor:pointer;white-space:nowrap;transition:border-color .14s ease,background .14s ease,opacity .14s ease}',
      '.pl-btn:hover:not(:disabled){border-color:var(--dsw-alias-border-l2);background:var(--dsw-alias-bg-overlay)}',
      '.pl-btn:disabled{opacity:.45;cursor:default}',
      '.pl-btn-primary{background:var(--dsw-alias-brand-primary);border-color:var(--dsw-alias-brand-primary);color:var(--dsw-alias-bg-base);box-shadow:0 1px 2px color-mix(in srgb,var(--dsw-alias-brand-primary) 30%,transparent)}',
      '.pl-btn-primary:hover:not(:disabled){opacity:.9;background:var(--dsw-alias-brand-primary)}',
      '.pl-btn-ghost{background:transparent;border-color:transparent;color:var(--dsw-alias-label-secondary)}',
      '.pl-btn-ghost:hover:not(:disabled){background:var(--dsw-alias-bg-layer-2);border-color:var(--dsw-alias-border-l1);color:var(--dsw-alias-label-primary)}',
      '.pl-btn-sm{height:24px;padding:0 8px;font-size:11.5px;border-radius:6px}',
      '.pl-btn-block{width:100%}',
      '.pl-input{height:28px;padding:0 9px;border:1px solid var(--dsw-alias-border-l1);border-radius:8px;background:var(--dsw-alias-bg-base);color:var(--dsw-alias-label-primary);font:inherit;font-size:12px;outline:none;min-width:0;transition:border-color .14s ease,box-shadow .14s ease}',
      '.pl-input:focus{border-color:color-mix(in srgb,var(--dsw-alias-brand-primary) 48%,var(--dsw-alias-border-l1));box-shadow:0 0 0 2px color-mix(in srgb,var(--dsw-alias-brand-primary) 12%,transparent)}',
      '.pl-input::placeholder{color:var(--dsw-alias-label-tertiary)}',
      '.pl-select{cursor:pointer;padding-right:4px}',
      '.pl-textarea{width:100%;min-height:62px;padding:7px 9px;border:1px solid var(--dsw-alias-border-l1);border-radius:9px;background:var(--dsw-alias-bg-base);color:var(--dsw-alias-label-primary);font:inherit;font-size:12px;line-height:1.5;resize:vertical;outline:none}',
      '.pl-textarea:focus{border-color:color-mix(in srgb,var(--dsw-alias-brand-primary) 48%,var(--dsw-alias-border-l1));box-shadow:0 0 0 2px color-mix(in srgb,var(--dsw-alias-brand-primary) 12%,transparent)}',
      '.pl-check{display:inline-flex;align-items:center;gap:6px;font-size:12px;color:var(--dsw-alias-label-secondary);cursor:pointer;user-select:none}',
      '.pl-check input{accent-color:var(--dsw-alias-brand-primary);width:14px;height:14px;cursor:pointer}',
      '.pl-field{display:flex;align-items:center;justify-content:space-between;gap:8px;min-width:0}',
      '.pl-field-col{flex-direction:column;align-items:stretch;gap:3px}',
      '.pl-field-label{flex:0 0 auto;font-size:11.5px;color:var(--dsw-alias-label-secondary);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}',
      '.pl-field .pl-input,.pl-field .pl-select{flex:1 1 auto;max-width:132px;height:25px;font-size:11.5px}',
      '.pl-field-col .pl-input,.pl-field-col .pl-select{max-width:none;width:100%}',
      '.pl-range{display:flex;align-items:center;gap:4px;flex:1 1 auto;min-width:0}',
      '.pl-range .pl-input{max-width:none;width:100%;height:24px;font-size:11px}',
      '.pl-range span{color:var(--dsw-alias-label-tertiary);font-size:11px}',

      /* 骨架 */
      '.pl-body{display:grid;grid-template-columns:284px minmax(360px,1fr) 452px;flex:1 1 auto;min-height:0;overflow:hidden}',
      '@media (max-width:1280px){.pl-body{grid-template-columns:250px minmax(320px,1fr) 380px}}',
      '@media (max-width:1080px){.pl-body{grid-template-columns:1fr;overflow-y:auto;display:flex;flex-direction:column}.pl-left,.pl-center,.pl-right{width:auto;max-height:none;border-right:none;border-left:none}}',
      '.pl-col{display:flex;flex-direction:column;min-height:0;min-width:0}',
      '.pl-left{border-right:1px solid var(--dsw-alias-border-l1);background:var(--dsw-alias-bg-layer-1);overflow-y:auto}',
      '.pl-center{background:var(--dsw-alias-bg-base);overflow-y:auto}',
      '.pl-right{border-left:1px solid var(--dsw-alias-border-l1);background:var(--dsw-alias-bg-layer-1);overflow-y:auto}',
      '.pl-section{display:flex;flex-direction:column;gap:8px;padding:11px 12px;border-bottom:1px solid var(--dsw-alias-border-l1)}',
      '.pl-section-head{display:flex;align-items:center;justify-content:space-between;gap:8px;min-height:20px}',
      '.pl-section-title{display:flex;align-items:center;gap:6px;font-size:11px;font-weight:650;letter-spacing:.05em;color:var(--dsw-alias-label-secondary);text-transform:uppercase}',
      '.pl-section-title svg{opacity:.75}',
      '.pl-section-note{font-size:10.5px;color:var(--dsw-alias-label-tertiary);white-space:nowrap}',
      '.pl-hint{font-size:11px;color:var(--dsw-alias-label-tertiary);line-height:1.5}',
      '.pl-muted{font-size:11.5px;color:var(--dsw-alias-label-secondary);line-height:1.5}',
      '.pl-divider{height:1px;background:var(--dsw-alias-border-l1);margin:2px 0}',
      '.pl-row{display:flex;align-items:center;gap:6px;flex-wrap:wrap;min-width:0}',

      /* 数据卡 */
      '.pl-datacard{display:flex;flex-direction:column;gap:5px;padding:10px 11px;border:1px solid var(--dsw-alias-border-l1);border-radius:12px;background:linear-gradient(160deg,color-mix(in srgb,var(--dsw-alias-brand-primary) 5%,var(--dsw-alias-bg-base)),var(--dsw-alias-bg-base) 60%)}',
      '.pl-datacard-name{font-size:13px;font-weight:650;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}',
      '.pl-datacard-meta{display:flex;flex-wrap:wrap;gap:3px 8px;font-size:11px;color:var(--dsw-alias-label-tertiary)}',
      '.pl-datacard-meta b{font-weight:600;color:var(--dsw-alias-label-secondary)}',
      '.pl-source{font-size:10.5px;color:var(--dsw-alias-label-tertiary);word-break:break-all;font-family:ui-monospace,SFMono-Regular,Menlo,monospace}',

      /* 列清单 */
      '.pl-cols{display:flex;flex-direction:column;gap:3px;max-height:190px;overflow-y:auto;margin-top:2px}',
      '.pl-colrow{display:flex;align-items:center;gap:7px;padding:3px 6px;border-radius:6px;font-size:11.5px}',
      '.pl-colrow:hover{background:var(--dsw-alias-bg-layer-2)}',
      '.pl-coldot{width:7px;height:7px;border-radius:50%;flex:none}',
      '.pl-coldot-numeric{background:var(--dsw-alias-brand-primary)}',
      '.pl-coldot-category{background:var(--dsw-alias-state-success-primary)}',
      '.pl-coldot-text{background:var(--dsw-alias-label-tertiary)}',
      '.pl-coldot-datetime{background:var(--dsw-alias-state-warn-primary)}',
      '.pl-colname{flex:1 1 auto;min-width:0;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;font-family:ui-monospace,SFMono-Regular,Menlo,monospace}',
      '.pl-colmeta{flex:0 0 auto;font-size:10px;color:var(--dsw-alias-label-tertiary)}',
      '.pl-tag{display:inline-flex;align-items:center;height:17px;padding:0 6px;border-radius:5px;background:var(--dsw-alias-bg-layer-2);border:1px solid var(--dsw-alias-border-l1);color:var(--dsw-alias-label-tertiary);font-size:10px;white-space:nowrap}',
      '.pl-tag-num{color:var(--dsw-alias-brand-primary);border-color:color-mix(in srgb,var(--dsw-alias-brand-primary) 30%,var(--dsw-alias-border-l1))}',
      '.pl-tag-cat{color:var(--dsw-alias-state-success-primary);border-color:color-mix(in srgb,var(--dsw-alias-state-success-primary) 30%,var(--dsw-alias-border-l1))}',

      /* 需求卡 */
      '.pl-need{border:1px solid color-mix(in srgb,var(--dsw-alias-brand-primary) 30%,var(--dsw-alias-border-l1));border-radius:12px;background:color-mix(in srgb,var(--dsw-alias-brand-primary) 4%,var(--dsw-alias-bg-base));padding:10px 11px;display:flex;flex-direction:column;gap:7px}',
      '.pl-plan{display:flex;flex-direction:column;gap:5px;padding:8px 10px;border:1px dashed color-mix(in srgb,var(--dsw-alias-brand-primary) 32%,var(--dsw-alias-border-l1));border-radius:10px;background:var(--dsw-alias-bg-base)}',
      '.pl-plan-steps{display:flex;flex-wrap:wrap;gap:4px}',
      '.pl-step{display:inline-flex;align-items:center;gap:4px;height:21px;padding:0 7px;border-radius:6px;background:var(--dsw-alias-bg-layer-2);border:1px solid var(--dsw-alias-border-l1);font-size:10.5px;color:var(--dsw-alias-label-secondary)}',
      '.pl-step b{color:var(--dsw-alias-brand-primary);font-weight:600}',
      '.pl-rationale{font-size:11px;color:var(--dsw-alias-label-secondary);line-height:1.5;margin:0;padding-left:14px}',
      '.pl-rationale li::marker{color:var(--dsw-alias-brand-primary)}',

      /* 质量卡 */
      '.pl-score{display:flex;align-items:center;gap:10px}',
      '.pl-score-num{font-size:26px;font-weight:300;line-height:1}',
      '.pl-score-num small{font-size:11px;color:var(--dsw-alias-label-tertiary);margin-left:1px}',
      '.pl-score-bar{flex:1 1 auto;height:6px;border-radius:999px;background:var(--dsw-alias-bg-layer-2);overflow:hidden}',
      '.pl-score-bar span{display:block;height:100%;border-radius:999px;background:linear-gradient(90deg,var(--dsw-alias-state-warn-primary),var(--dsw-alias-state-success-primary))}',
      '.pl-checks{display:flex;flex-direction:column;gap:3px;max-height:150px;overflow-y:auto}',
      '.pl-check-row{display:flex;align-items:flex-start;gap:6px;padding:4px 6px;border-radius:6px;font-size:11px;line-height:1.4}',
      '.pl-check-row:hover{background:var(--dsw-alias-bg-layer-2)}',
      '.pl-check-dot{width:6px;height:6px;border-radius:50%;flex:none;margin-top:4px}',
      '.pl-check-ok{background:var(--dsw-alias-state-success-primary)}',
      '.pl-check-warn{background:var(--dsw-alias-state-warn-primary)}',
      '.pl-check-fail{background:var(--dsw-alias-state-error-primary)}',
      '.pl-check-text{flex:1 1 auto;min-width:0;color:var(--dsw-alias-label-secondary)}',
      '.pl-check-text b{display:block;font-weight:600;color:var(--dsw-alias-label-primary)}',
      '.pl-fix{font-size:10.5px;color:var(--dsw-alias-label-tertiary)}',

      /* 图表画廊 */
      '.pl-gallery{display:flex;flex-direction:column;gap:9px}',
      '.pl-gallery-group{display:flex;flex-direction:column;gap:5px}',
      '.pl-gallery-label{font-size:10.5px;color:var(--dsw-alias-label-tertiary);font-weight:600;letter-spacing:.05em}',
      '.pl-gallery-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(84px,1fr));gap:5px}',
      '.pl-chart{display:flex;flex-direction:column;align-items:center;justify-content:center;gap:3px;height:58px;padding:4px;border:1px solid var(--dsw-alias-border-l1);border-radius:9px;background:var(--dsw-alias-bg-layer-1);color:var(--dsw-alias-label-secondary);font:inherit;font-size:11px;cursor:pointer;transition:border-color .14s ease,background .14s ease,color .14s ease}',
      '.pl-chart:hover:not(:disabled){border-color:var(--dsw-alias-border-l2);color:var(--dsw-alias-label-primary);background:var(--dsw-alias-bg-layer-2)}',
      '.pl-chart-on{border-color:color-mix(in srgb,var(--dsw-alias-brand-primary) 52%,var(--dsw-alias-border-l1));background:color-mix(in srgb,var(--dsw-alias-brand-primary) 9%,var(--dsw-alias-bg-layer-1));color:var(--dsw-alias-brand-primary);box-shadow:0 0 0 1px color-mix(in srgb,var(--dsw-alias-brand-primary) 24%,transparent)}',
      '.pl-chart svg{opacity:.85}',
      '.pl-chart-on svg{opacity:1}',

      /* 参数 / 样式 */
      '.pl-card{display:flex;flex-direction:column;gap:8px;padding:11px 12px;border:1px solid var(--dsw-alias-border-l1);border-radius:12px;background:var(--dsw-alias-bg-layer-1)}',
      '.pl-card-head{display:flex;align-items:center;justify-content:space-between;gap:8px}',
      '.pl-card-title{display:flex;align-items:center;gap:6px;font-size:11.5px;font-weight:650;color:var(--dsw-alias-label-secondary)}',
      '.pl-card-body{display:flex;flex-direction:column;gap:8px}',
      '.pl-grid2{display:grid;grid-template-columns:1fr 1fr;gap:6px 14px}',
      '.pl-grid3{display:grid;grid-template-columns:repeat(3,1fr);gap:6px 12px}',
      '@media (max-width:1400px){.pl-grid3{grid-template-columns:1fr 1fr}}',
      '.pl-style-group{display:flex;flex-direction:column;gap:6px}',
      '.pl-style-head{display:flex;align-items:center;justify-content:space-between;gap:6px;padding:5px 2px;border:none;background:transparent;color:var(--dsw-alias-label-secondary);font:inherit;font-size:11px;font-weight:650;letter-spacing:.04em;cursor:pointer}',
      '.pl-style-head:hover{color:var(--dsw-alias-label-primary)}',
      '.pl-style-caret{transition:transform .15s ease;color:var(--dsw-alias-label-tertiary)}',
      '.pl-style-caret-open{transform:rotate(90deg)}',
      '.pl-colorlist{display:flex;flex-direction:column;gap:3px;max-height:168px;overflow-y:auto}',
      '.pl-colorrow{display:flex;align-items:center;gap:7px;font-size:11px}',
      '.pl-colorrow span{flex:1 1 auto;min-width:0;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;color:var(--dsw-alias-label-secondary)}',
      '.pl-swatch{width:26px;height:20px;padding:0;border:1px solid var(--dsw-alias-border-l2);border-radius:5px;background:transparent;cursor:pointer;flex:none}',
      '.pl-swatch::-webkit-color-swatch-wrapper{padding:2px}',
      '.pl-swatch::-webkit-color-swatch{border:none;border-radius:3px}',

      /* 预览 */
      '.pl-preview{display:flex;flex-direction:column;gap:8px;padding:11px 12px;border-bottom:1px solid var(--dsw-alias-border-l1)}',
      '.pl-preview-toolbar{display:flex;align-items:center;gap:5px;flex-wrap:wrap}',
      '.pl-canvas{position:relative;display:flex;align-items:center;justify-content:center;min-height:220px;padding:8px;border:1px solid var(--dsw-alias-border-l1);border-radius:12px;background:repeating-conic-gradient(var(--dsw-alias-bg-layer-2) 0% 25%,var(--dsw-alias-bg-base) 0% 50%) 50%/18px 18px;overflow:hidden}',
      '.pl-canvas img{display:block;max-width:100%;max-height:560px;height:auto;border-radius:6px;background:#fff;box-shadow:0 2px 10px color-mix(in srgb,var(--dsw-alias-label-primary) 12%,transparent)}',
      '.pl-canvas-empty{display:flex;flex-direction:column;align-items:center;gap:6px;color:var(--dsw-alias-label-tertiary);font-size:12px;text-align:center;padding:22px}',
      '.pl-canvas-busy{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;gap:8px;background:color-mix(in srgb,var(--dsw-alias-bg-base) 62%,transparent);backdrop-filter:blur(1px);font-size:12px;color:var(--dsw-alias-label-secondary)}',
      '.pl-spinner{width:14px;height:14px;border-radius:50%;border:2px solid color-mix(in srgb,var(--dsw-alias-brand-primary) 24%,transparent);border-top-color:var(--dsw-alias-brand-primary);animation:pl-spin .8s linear infinite}',
      '@keyframes pl-spin{to{transform:rotate(360deg)}}',
      '.pl-figmeta{display:flex;align-items:center;gap:8px;flex-wrap:wrap;font-size:10.5px;color:var(--dsw-alias-label-tertiary)}',
      '.pl-bullets{margin:0;padding-left:15px;display:flex;flex-direction:column;gap:2px;font-size:11.5px;color:var(--dsw-alias-label-secondary);line-height:1.5}',
      '.pl-bullets li::marker{color:var(--dsw-alias-brand-primary)}',
      '.pl-warns{display:flex;flex-direction:column;gap:2px;font-size:11px;color:var(--dsw-alias-state-warn-primary);line-height:1.45}',

      /* tabs / 表格 / 代码 */
      '.pl-tabs{display:flex;align-items:center;gap:2px;padding:6px 8px 0;border-bottom:1px solid var(--dsw-alias-border-l1);background:var(--dsw-alias-bg-layer-1);overflow-x:auto}',
      '.pl-tab{display:inline-flex;align-items:center;gap:5px;height:28px;padding:0 10px;border:none;border-bottom:2px solid transparent;border-radius:7px 7px 0 0;background:transparent;color:var(--dsw-alias-label-secondary);font:inherit;font-size:11.5px;cursor:pointer;white-space:nowrap}',
      '.pl-tab:hover{color:var(--dsw-alias-label-primary);background:var(--dsw-alias-bg-layer-2)}',
      '.pl-tab-on{color:var(--dsw-alias-brand-primary);border-bottom-color:var(--dsw-alias-brand-primary);font-weight:650}',
      '.pl-tabpane{display:flex;flex-direction:column;gap:9px;padding:10px 12px 18px}',
      '.pl-table-wrap{max-height:340px;overflow:auto;border:1px solid var(--dsw-alias-border-l1);border-radius:9px;background:var(--dsw-alias-bg-base)}',
      '.pl-table{width:100%;border-collapse:separate;border-spacing:0;font-size:11px}',
      '.pl-table th{position:sticky;top:0;z-index:1;padding:5px 8px;border-bottom:1px solid var(--dsw-alias-border-l1);background:var(--dsw-alias-bg-layer-1);color:var(--dsw-alias-label-secondary);font-weight:600;text-align:left;white-space:nowrap}',
      '.pl-table td{padding:3px 8px;border-bottom:1px solid color-mix(in srgb,var(--dsw-alias-border-l1) 52%,transparent);color:var(--dsw-alias-label-primary);white-space:nowrap;max-width:240px;overflow:hidden;text-overflow:ellipsis}',
      '.pl-table tbody tr:hover td{background:color-mix(in srgb,var(--dsw-alias-brand-primary) 4%,transparent)}',
      '.pl-td-num{text-align:right;font-variant-numeric:tabular-nums}',
      '.pl-empty{padding:16px 12px;text-align:center;color:var(--dsw-alias-label-tertiary);font-size:11.5px;line-height:1.6}',
      '.pl-code{max-height:360px;overflow:auto;margin:0;padding:10px 11px;border:1px solid var(--dsw-alias-border-l1);border-radius:9px;background:var(--dsw-alias-bg-layer-2);color:var(--dsw-alias-label-primary);font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:10.5px;line-height:1.55;white-space:pre;tab-size:2}',
      '.pl-statfig img{display:block;max-width:100%;border:1px solid var(--dsw-alias-border-l1);border-radius:8px;background:#fff;margin-top:5px}',
      '.pl-history{display:flex;flex-direction:column;gap:4px}',
      '.pl-history-item{display:flex;align-items:center;gap:8px;padding:6px 8px;border:1px solid var(--dsw-alias-border-l1);border-radius:8px;background:var(--dsw-alias-bg-base);cursor:pointer;text-align:left;font:inherit}',
      '.pl-history-item:hover{border-color:var(--dsw-alias-border-l2);background:var(--dsw-alias-bg-layer-2)}',
      '.pl-history-item img{width:46px;height:34px;object-fit:cover;border-radius:5px;border:1px solid var(--dsw-alias-border-l1);background:#fff}',
      '.pl-history-main{display:flex;flex-direction:column;gap:1px;min-width:0;flex:1 1 auto}',
      '.pl-history-title{font-size:11.5px;font-weight:600;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}',
      '.pl-history-meta{font-size:10px;color:var(--dsw-alias-label-tertiary)}',

      /* 状态提示 */
      '.pl-banner{display:flex;align-items:center;justify-content:space-between;gap:10px;flex:none;padding:6px 14px;border-bottom:1px solid color-mix(in srgb,var(--dsw-alias-state-error-primary) 34%,var(--dsw-alias-border-l1));background:color-mix(in srgb,var(--dsw-alias-state-error-primary) 7%,var(--dsw-alias-bg-layer-1));font-size:11.5px}',
      '.pl-banner-text{flex:1 1 auto;min-width:0;word-break:break-word}',
      '.pl-toast{position:absolute;left:50%;bottom:18px;transform:translateX(-50%);z-index:20;display:flex;align-items:center;gap:8px;max-width:80%;padding:8px 13px;border:1px solid color-mix(in srgb,var(--dsw-alias-brand-primary) 34%,var(--dsw-alias-border-l1));border-radius:10px;background:var(--dsw-alias-bg-overlay);color:var(--dsw-alias-label-primary);font-size:12px;box-shadow:0 6px 24px color-mix(in srgb,var(--dsw-alias-label-primary) 18%,transparent);animation:pl-fade .18s ease}',
      '@keyframes pl-fade{from{opacity:0;transform:translate(-50%,6px)}to{opacity:1;transform:translate(-50%,0)}}',
      '.pl-overlay{position:absolute;inset:0;z-index:30;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:10px;border:2px dashed var(--dsw-alias-brand-primary);border-radius:12px;background:color-mix(in srgb,var(--dsw-alias-bg-base) 88%,transparent);color:var(--dsw-alias-brand-primary);font-size:14px;font-weight:600;pointer-events:none}',
      '.pl-overlay small{font-size:11.5px;font-weight:400;color:var(--dsw-alias-label-secondary)}',
      '.pl-empty-state{display:flex;flex-direction:column;align-items:center;justify-content:center;gap:10px;height:100%;min-height:320px;padding:28px;text-align:center}',
      '.pl-empty-title{font-size:16px;font-weight:650}',
      '.pl-empty-sub{font-size:12px;color:var(--dsw-alias-label-secondary);max-width:520px;line-height:1.65}',
      '.pl-dropzone{display:flex;flex-direction:column;align-items:center;gap:6px;width:min(560px,100%);padding:28px 20px;border:2px dashed var(--dsw-alias-border-l2);border-radius:14px;background:var(--dsw-alias-bg-layer-1);color:var(--dsw-alias-label-secondary);cursor:pointer;transition:border-color .15s ease,background .15s ease}',
      '.pl-dropzone:hover{border-color:color-mix(in srgb,var(--dsw-alias-brand-primary) 55%,var(--dsw-alias-border-l1));background:color-mix(in srgb,var(--dsw-alias-brand-primary) 4%,var(--dsw-alias-bg-layer-1))}',
      '.pl-demo-row{display:flex;gap:8px;flex-wrap:wrap;justify-content:center}',
      '.pl-demo-card{display:flex;flex-direction:column;align-items:flex-start;gap:3px;width:188px;padding:10px 11px;border:1px solid var(--dsw-alias-border-l1);border-radius:10px;background:var(--dsw-alias-bg-layer-1);color:var(--dsw-alias-label-primary);cursor:pointer;text-align:left;font:inherit}',
      '.pl-demo-card:hover{border-color:color-mix(in srgb,var(--dsw-alias-brand-primary) 45%,var(--dsw-alias-border-l1));background:var(--dsw-alias-bg-layer-2)}',
      '.pl-demo-title{font-size:12px;font-weight:650}',
      '.pl-demo-sub{font-size:10.5px;color:var(--dsw-alias-label-tertiary);line-height:1.4}',
      '.pl-kv{display:flex;align-items:center;justify-content:space-between;gap:8px;font-size:11.5px}',
      '.pl-kv span{color:var(--dsw-alias-label-tertiary)}',
      '.pl-kv b{font-weight:600;color:var(--dsw-alias-label-primary);max-width:60%;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}',
    ].join('')

    // ----------------------------------------------------------------------
    // 图标（内联 SVG，currentColor）
    // ----------------------------------------------------------------------
    var ICONS = {
      logo: 'M3 17V7l6 5 6-5v10l6-5v10',
      chart: 'M4 19V5M4 19h16M8 16V9M12 16v-4M16 16v-9',
      data: 'M4 6c0-1.7 3.6-3 8-3s8 1.3 8 3-3.6 3-8 3-8-1.3-8-3zM4 6v12c0 1.7 3.6 3 8 3s8-1.3 8-3V6',
      spark: 'M12 3l2 6 6 2-6 2-2 6-2-6-6-2 6-2z',
      wand: 'M4 20l10-10M14 4l1.5 3.5L19 9l-3.5 1.5L14 14l-1.5-3.5L9 9l3.5-1.5z',
      brush: 'M4 20c3 0 4-2 4-4l9-9 3 3-9 9c0 3-2 4-4 4M6 16h.01',
      download: 'M12 4v11M7 10l5 5 5-5M5 20h14',
      upload: 'M12 20V9M7 14l5-5 5 5M5 4h14',
      refresh: 'M20 12a8 8 0 1 1-2.3-5.6M20 4v5h-5',
      code: 'M8 7l-5 5 5 5M16 7l5 5-5 5',
      stats: 'M4 20V10M10 20V4M16 20v-7M22 20H2',
      check: 'M4 12l5 5L20 6',
      warn: 'M12 3l9 17H3zM12 9v5M12 17h.01',
      close: 'M6 6l12 12M18 6L6 18',
      play: 'M7 4l12 8-12 8z',
      chevron: 'M9 6l6 6-6 6',
      table: 'M3 5h18v14H3zM3 10h18M9 10v9M15 10v9',
      copy: 'M9 9h10v12H9zM5 15V3h10v2',
    }

    function Icon(props) {
      var name = props.name
      var size = props.size || 14
      var path = ICONS[name] || ICONS.chart
      return h('svg', {
        width: size, height: size, viewBox: '0 0 24 24', fill: 'none', stroke: 'currentColor',
        strokeWidth: props.strokeWidth || 1.8, strokeLinecap: 'round', strokeLinejoin: 'round',
        'aria-hidden': 'true',
      }, h('path', { d: path }))
    }

    // ----------------------------------------------------------------------
    // 小部件
    // ----------------------------------------------------------------------
    function Field(props) {
      var children = h('div', { className: 'pl-field' + (props.column ? ' pl-field-col' : '') },
        h('label', { className: 'pl-field-label', title: props.hint || props.label }, props.label),
        childrenToArray(arguments, 2))
      return h('div', { className: 'pl-field-wrap', title: props.hint || '' }, children)
    }
    function childrenToArray(args, start) {
      return Array.prototype.slice.call(args, start)
    }

    function NumberInput(props) {
      var value = props.value
      return h('input', {
        className: 'pl-input', type: 'number', value: isNil(value) ? '' : value,
        min: props.min, max: props.max, step: props.step || 1,
        disabled: props.disabled,
        onChange: function (event) { props.onChange(event.target.value) },
      })
    }

    function Select(props) {
      var options = props.options || []
      return h('select', {
        className: 'pl-input pl-select', value: isNil(props.value) ? '' : String(props.value),
        disabled: props.disabled,
        onChange: function (event) { props.onChange(event.target.value) },
      }, options.map(function (option) {
        var id = isNil(option.id) ? option.value : option.id
        var label = option.label || option.title || String(id)
        return h('option', { key: String(id), value: String(id) }, label)
      }))
    }

    function Toggle(props) {
      return h('label', { className: 'pl-check' },
        h('input', { type: 'checkbox', checked: Boolean(props.value), onChange: function (event) { props.onChange(event.target.checked) } }),
        props.label || '')
    }

    function TextInput(props) {
      return h('input', {
        className: 'pl-input', type: 'text', value: props.value || '', placeholder: props.placeholder || '',
        onChange: function (event) { props.onChange(event.target.value) },
      })
    }

    function RangeInput(props) {
      var value = Array.isArray(props.value) ? props.value : [null, null]
      return h('div', { className: 'pl-range' },
        h('input', { className: 'pl-input', type: 'number', value: isNil(value[0]) ? '' : value[0], placeholder: 'min',
          onChange: function (event) { props.onChange([event.target.value === '' ? null : Number(event.target.value), value[1]]) } }),
        h('span', null, '–'),
        h('input', { className: 'pl-input', type: 'number', value: isNil(value[1]) ? '' : value[1], placeholder: 'max',
          onChange: function (event) { props.onChange([value[0], event.target.value === '' ? null : Number(event.target.value)]) } }))
    }

    function Collapsible(props) {
      var [open, setOpen] = useState(props.defaultOpen !== false)
      return h('div', { className: 'pl-style-group' },
        h('button', { type: 'button', className: 'pl-style-head', onClick: function () { setOpen(!open) } },
          h('span', null, props.title),
          h('span', { className: 'pl-style-caret' + (open ? ' pl-style-caret-open' : '') }, h(Icon, { name: 'chevron', size: 12 }))),
        open ? h('div', { className: 'pl-card-body' }, props.children) : null)
    }

    function Table(props) {
      var table = props.table
      var columns = tableColumns(table)
      var rows = (table && table.rows) || []
      if (!columns.length) return h('div', { className: 'pl-empty' }, '没有数据')
      return h('div', { className: 'pl-table-wrap' },
        h('table', { className: 'pl-table' },
          h('thead', null, h('tr', null, columns.map(function (column) {
            return h('th', { key: column.key, className: column.type === 'num' ? 'pl-td-num' : '' }, column.label || column.key)
          }))),
          h('tbody', null, rows.map(function (row, rowIndex) {
            return h('tr', { key: rowIndex }, columns.map(function (column) {
              return h('td', {
                key: column.key,
                className: column.type === 'num' ? 'pl-td-num' : '',
                title: String(formatCell(column, row)),
              }, formatCell(column, row))
            }))
          }))))
    }

    function Bullets(props) {
      var items = props.items || []
      if (!items.length) return null
      return h('ul', { className: 'pl-bullets' }, items.map(function (item, index) {
        return h('li', { key: index }, String(item))
      }))
    }

    function Empty(props) {
      return h('div', { className: 'pl-empty' }, props.children || props.text)
    }

    // ======================================================================
    // 工作台主体
    // ======================================================================
    function WorkbenchBody() {
      var apiRef = useRef(null)
      if (!apiRef.current) apiRef.current = createApi(null, API)
      var api = apiRef.current

      var [boot, setBoot] = useState(null)
      var [catalog, setCatalog] = useState(null)
      var [dataset, setDataset] = useState(null)
      var [preview, setPreview] = useState(null)
      var [profile, setProfile] = useState(null)
      var [chart, setChart] = useState('box')
      var [params, setParams] = useState({})
      var [style, setStyle] = useState(clone(DEFAULT_STYLE))
      var [roles, setRoles] = useState({})
      var [figure, setFigure] = useState(null)
      var [plotInfo, setPlotInfo] = useState(null)
      var [summary, setSummary] = useState(null)
      var [warnings, setWarnings] = useState([])
      var [code, setCode] = useState(null)
      var [history, setHistory] = useState([])
      var [statId, setStatId] = useState('compare')
      var [statParams, setStatParams] = useState({})
      var [statResult, setStatResult] = useState(null)
      var [statBusy, setStatBusy] = useState(false)
      var [need, setNeed] = useState('')
      var [plan, setPlan] = useState(null)
      var [cleaning, setCleaning] = useState(null)
      var [busy, setBusy] = useState('')
      var [error, setError] = useState('')
      var [toast, setToast] = useState('')
      var [tab, setTab] = useState('preview')
      var [dragOver, setDragOver] = useState(false)
      var [showAllCols, setShowAllCols] = useState(false)
      var [openSections, setOpenSections] = useState(function () { return { font: true, color: true, canvas: false, axis: false, legend: false, stat: false } })
      var [audit, setAudit] = useState(null)
      var [panelMode, setPanelMode] = useState('plot')

      var fileInputRef = useRef(null)
      var plotSeq = useRef(0)
      var plotTimer = useRef(null)
      var pendingPlot = useRef(true)
      var toastTimer = useRef(null)
      var stateRef = useRef({})
      stateRef.current = { dataset: dataset, chart: chart, params: params, style: style, roles: roles }

      function notify(message) {
        setToast(String(message || ''))
        if (toastTimer.current) clearTimeout(toastTimer.current)
        toastTimer.current = setTimeout(function () { setToast('') }, 4200)
      }

      function fail(caught) {
        var message = caught && caught.message ? caught.message : String(caught)
        setError(message)
        setBusy('')
      }

      // ---- 启动 ----
      useEffect(function () {
        var alive = true
        api.health().then(function (payload) {
          if (alive) setBoot(payload)
        }).catch(function (caught) {
          if (alive) setError('工作台引擎连接失败：' + (caught && caught.message ? caught.message : caught))
        })
        api.charts().then(function (payload) {
          if (!alive) return
          setCatalog(payload)
          var charts = payload.charts || []
          var first = charts.length ? charts[0].id : 'box'
          setChart(first)
          var meta = findChart(charts, first)
          setParams(defaultParams(meta))
          setRoles(defaultRolesFor(meta))
          setStyle(mergeStyle(DEFAULT_STYLE, (payload.styles && payload.styles.defaults) || {}))
        }).catch(function (caught) {
          if (alive) setError('图表目录加载失败：' + (caught && caught.message ? caught.message : caught))
        })
        api.state().then(function (payload) {
          if (!alive) return
          if (payload && payload.datasets && payload.datasets.length) {
            adoptDataset({ dataset: payload.datasets[0], preview: null }, true)
          }
        }).catch(function () { /* 忽略：状态只在有数据时才有意义 */ })
        return function () { alive = false }
      }, [])

      // ---- 采纳一个数据集 ----
      function adoptDataset(result, defer) {
        var next = result.dataset
        if (!next) return
        setDataset(next)
        setProfile(null)
        setStatResult(null)
        setPlan(null)
        setCleaning(null)
        setNeed(next.need || '')
        setError('')
        if (result.preview) {
          setPreview(result.preview)
          setHistory((result.preview.history || []).slice())
        } else {
          loadPreview(next.id)
        }
        loadProfile(next.id)
        pickInitialChart(next, result.preview, defer)
      }

      function pickInitialChart(nextDataset, previewPayload, defer) {
        var charts = (catalog && catalog.charts) || []
        var rec = null
        if (profile && profile.recommend && profile.recommend.charts && profile.recommend.charts.length) rec = profile.recommend.charts[0]
        if (!rec && previewPayload && previewPayload.kind === 'matrix') rec = { id: 'heatmap' }
        if (!rec) rec = { id: charts.length ? charts[0].id : 'box' }
        var meta = findChart(charts, rec.id) || charts[0]
        if (!meta) return
        setChart(meta.id)
        setParams(defaultParams(meta))
        var nextRoles = defaultRolesFor(meta)
        if (nextDataset && nextDataset.roles) {
          Object.keys(nextDataset.roles).forEach(function (key) {
            if (nextDataset.roles[key]) nextRoles[key] = nextDataset.roles[key]
          })
        }
        setRoles(nextRoles)
        pendingPlot.current = true
        if (!defer) schedulePlot(120)
      }

      function loadPreview(datasetId) {
        if (!datasetId) return
        api.preview(datasetId).then(function (payload) {
          setPreview(payload)
          if (payload.history) setHistory(payload.history.slice())
          if (payload.need) setNeed(payload.need)
        }).catch(function () { /* 预览失败不致命 */ })
      }

      function loadProfile(datasetId) {
        if (!datasetId) return
        api.profile(datasetId).then(function (payload) { setProfile(payload) })
          .catch(function () { setProfile(null) })
      }

      // ---- 出图（防抖） ----
      function schedulePlot(delay) {
        if (plotTimer.current) clearTimeout(plotTimer.current)
        plotTimer.current = setTimeout(function () { runPlot() }, isNil(delay) ? 320 : delay)
      }

      function runPlot(overrides) {
        var current = stateRef.current
        var ds = (overrides && overrides.dataset) || current.dataset
        if (!ds || !ds.id) return
        var body = buildPlotRequest({
          dataset: ds,
          chart: (overrides && overrides.chart) || current.chart,
          params: (overrides && overrides.params) || current.params,
          style: (overrides && overrides.style) || current.style,
          roles: (overrides && overrides.roles) || current.roles,
        })
        var seq = ++plotSeq.current
        setBusy('plot')
        return api.plot(body).then(function (result) {
          if (seq !== plotSeq.current) return
          setBusy('')
          setError('')
          setFigure(result.figure)
          setPlotInfo(result)
          setSummary(result.summary || null)
          setWarnings(result.warnings || [])
          setCode(result.code || null)
          if (result.dataset) setDataset(result.dataset)
          setHistory(function (previous) {
            var entry = {
              id: result.figure ? result.figure.id : String(Date.now()),
              chart: result.chart,
              chart_title: result.chart_title,
              figure: result.figure,
              at: new Date().toISOString(),
              params: result.params,
              style: result.style,
              roles: result.roles,
            }
            var next = previous.filter(function (item) { return item.id !== entry.id })
            next.push(entry)
            return next.slice(-20)
          })
        }).catch(function (caught) {
          if (seq !== plotSeq.current) return
          setBusy('')
          fail(caught)
        })
      }

      // 参数/样式/角色/图表变化 → 自动重绘
      useEffect(function () {
        if (!dataset || !catalog) return
        if (!pendingPlot.current) return
        pendingPlot.current = false
        schedulePlot(280)
        return function () { if (plotTimer.current) clearTimeout(plotTimer.current) }
      }, [dataset && dataset.id, chart])

      useEffect(function () {
        if (!dataset || !catalog) return
        schedulePlot(420)
        return function () { if (plotTimer.current) clearTimeout(plotTimer.current) }
      }, [JSON.stringify(params), JSON.stringify(style), JSON.stringify(roles)])

      // ---- 数据操作 ----
      function handleOpenFile(file) {
        if (!file) return
        setBusy('upload')
        api.upload(file).then(function (result) {
          setBusy('')
          notify('已导入 ' + (result.dataset ? result.dataset.name : '数据'))
          adoptDataset(result)
        }).catch(function (caught) { setBusy(''); fail(caught) })
      }

      function handleDemo(name) {
        setBusy('demo')
        api.demo(name).then(function (result) {
          setBusy('')
          notify('已载入示例：' + result.dataset.name)
          adoptDataset(result)
        }).catch(function (caught) { setBusy(''); fail(caught) })
      }

      function handleClean() {
        if (!dataset) return
        setBusy('clean')
        api.clean(dataset.id, null, true).then(function (result) {
          setBusy('')
          setCleaning(result)
          if (result.dataset) setDataset(result.dataset)
          notify('清洗完成：应用 ' + (result.applied || []).length + ' 步，现在 ' + result.n_rows + ' 行')
          loadProfile(dataset.id)
          loadPreview(dataset.id)
          runPlot()
        }).catch(function (caught) { setBusy(''); fail(caught) })
      }

      function handleResetData() {
        if (!dataset) return
        setBusy('clean')
        api.resetData(dataset.id).then(function (result) {
          setBusy('')
          setCleaning(null)
          if (result.dataset) setDataset(result.dataset)
          notify('已恢复导入时数据')
          loadProfile(dataset.id)
          loadPreview(dataset.id)
          runPlot()
        }).catch(function (caught) { setBusy(''); fail(caught) })
      }

      function handleNeed(runNow) {
        if (!dataset) return
        var text = String(need || '').trim()
        if (!text) { notify('先写一句需求，例如「按分组画箱线图比较 score」'); return }
        if (!runNow) {
          api.need(dataset.id, text).then(function () {
            notify('需求已记录，Agent 会在对话里看到；也可以直接在这里点「分析并出图」')
          }).catch(function (caught) { fail(caught) })
          return
        }
        setBusy('auto')
        setError('')
        api.auto(dataset.id, text).then(function (result) {
          setBusy('')
          setPlan(result.plan || null)
          setCleaning(result.cleaning || null)
          if (result.dataset) setDataset(result.dataset)
          setFigure(result.figure || null)
          setPlotInfo(result)
          setSummary(result.summary || null)
          setWarnings(result.warnings || [])
          setCode(result.code || null)
          setChart(result.chart)
          setParams(result.params || {})
          setStyle(mergeStyle(DEFAULT_STYLE, result.style || {}))
          setRoles(result.roles || {})
          var firstStat = (result.stats || [])[0]
          setStatResult(firstStat ? { ...firstStat, tables: firstStat.tables || [] } : null)
          setStatId(firstStat ? firstStat.stat : statId)
          notify('已完成：' + result.chart_title + '（' + ((result.plan && result.plan.steps) || []).length + ' 步）')
          setTab('preview')
          loadProfile(dataset.id)
          loadPreview(dataset.id)
        }).catch(function (caught) { setBusy(''); fail(caught) })
      }

      function handleStat() {
        if (!dataset) return
        setStatBusy(true)
        api.stat({ dataset: dataset.id, stat: statId, params: statParams, style: style, roles: roles })
          .then(function (result) { setStatBusy(false); setStatResult(result); setTab('stats') })
          .catch(function (caught) { setStatBusy(false); fail(caught) })
      }

      function handleExport(what) {
        if (!dataset) return
        setBusy('export')
        var body = buildExportRequest({
          dataset: dataset, chart: chart, params: params, style: style, roles: roles,
          stat: statResult ? statResult.stat : null, statParams: statParams,
        }, what)
        api.exportAll(body).then(function (result) {
          setBusy('')
          notify('已导出：' + (result.file || '') + (result.bytes ? '（' + formatBytes(result.bytes) + '）' : ''))
          if (what === 'figure' && result.file) {
            try {
              var link = document.createElement('a')
              link.href = API + '/figure?path=' + encodeURIComponent(result.file)
              link.download = result.files && result.files[0] ? result.files[0].label : 'figure.png'
              document.body.appendChild(link)
              link.click()
              document.body.removeChild(link)
            } catch (error) { /* 下载失败无所谓，用户可看路径 */ }
          }
        }).catch(function (caught) { setBusy(''); fail(caught) })
      }

      function handleAudit() {
        if (!dataset) return
        api.audit(dataset.id).then(function (result) { setAudit(result); setTab('audit') })
          .catch(function (caught) { fail(caught) })
      }

      function handleRestart() {
        setBusy('restart')
        api.restart().then(function (result) {
          setBusy('')
          if (result.ok) { notify('Python 引擎已重启'); api.health().then(setBoot).catch(function () {}) }
          else fail(new Error((result.engine && result.engine.error) || '引擎重启失败'))
        }).catch(function (caught) { setBusy(''); fail(caught) })
      }

      function selectChart(id) {
        if (!catalog) return
        var meta = findChart(catalog.charts, id)
        setChart(id)
        setParams(defaultParams(meta))
        setRoles(function (previous) {
          var next = Object.assign({}, previous)
          var rec = defaultRolesFor(meta)
          Object.keys(rec).forEach(function (key) { if (rec[key]) next[key] = rec[key] })
          return next
        })
        setPanelMode('plot')
      }

      function restoreHistory(entry) {
        if (!entry) return
        if (entry.chart) setChart(entry.chart)
        if (entry.params) setParams(entry.params)
        if (entry.style) setStyle(mergeStyle(DEFAULT_STYLE, entry.style))
        if (entry.roles) setRoles(entry.roles)
        if (entry.figure) setFigure(entry.figure)
        notify('已恢复历史配置：' + (entry.chart_title || entry.chart))
      }

      // ---- 拖拽 ----
      function onDragOver(event) {
        if (event.dataTransfer && Array.prototype.indexOf.call(event.dataTransfer.types || [], 'Files') >= 0) {
          event.preventDefault()
          setDragOver(true)
        }
      }
      function onDragLeave(event) {
        if (event.currentTarget === event.target) setDragOver(false)
      }
      function onDrop(event) {
        event.preventDefault()
        setDragOver(false)
        var files = event.dataTransfer && event.dataTransfer.files
        if (files && files.length) handleOpenFile(files[0])
      }

      var charts = (catalog && catalog.charts) || []
      var statsCatalog = (catalog && catalog.stats) || []
      var meta = findChart(charts, chart)
      var columns = (dataset && dataset.columns) || (preview && preview.columns) || []
      var groupLevels = (styleGroupLevels(dataset, preview, roles))
      var busyLabel = busy === 'plot' ? '正在渲染…' : busy === 'auto' ? '正在分析需求、清洗并出图…' :
        busy === 'clean' ? '正在清洗…' : busy === 'export' ? '正在导出…' : busy === 'upload' ? '正在导入…' :
          busy === 'demo' ? '正在载入示例…' : busy === 'restart' ? '正在重启引擎…' : ''

      return h('div', {
        className: 'pl-root',
        onDragOver: onDragOver, onDragLeave: onDragLeave, onDrop: onDrop,
      },
        h('style', null, CSS),
        h(TopBar, {
          boot: boot, dataset: dataset, busy: busy, status: boot && boot.engine,
          onPickFile: function () { if (fileInputRef.current) fileInputRef.current.click() },
          onDemo: function () { handleDemo('clinical') },
          onRestart: handleRestart,
          onExport: function () { handleExport('bundle') },
          onAudit: handleAudit,
          onReset: function () {
            api.reset().then(function () {
              setDataset(null); setPreview(null); setProfile(null); setFigure(null)
              setSummary(null); setHistory([]); setStatResult(null); setPlan(null); setCleaning(null)
              setNeed(''); setError(''); notify('工作台已清空')
            }).catch(fail)
          },
        }),
        h('input', {
          ref: fileInputRef, type: 'file', style: { display: 'none' },
          accept: '.csv,.tsv,.txt,.xlsx,.xls,.json,.gz,.zip,.tar,.tgz',
          onChange: function (event) {
            var file = event.target.files && event.target.files[0]
            event.target.value = ''
            if (file) handleOpenFile(file)
          },
        }),
        error ? h('div', { className: 'pl-banner' },
          h('div', { className: 'pl-banner-text' }, error),
          h('button', { className: 'pl-btn pl-btn-sm pl-btn-ghost', onClick: function () { setError('') } }, h(Icon, { name: 'close', size: 12 }), '关闭')) : null,
        dataset ? null : h(EmptyState, { boot: boot, onPickFile: function () { if (fileInputRef.current) fileInputRef.current.click() }, onDemo: handleDemo }),
        dataset ? h('div', { className: 'pl-body' },
          h(LeftColumn, {
            dataset: dataset, preview: preview, profile: profile, columns: columns,
            chartMeta: meta, roles: roles, onRolesChange: setRoles, need: need, onNeedChange: setNeed,
            onAuto: function () { handleNeed(true) }, onNeedOnly: function () { handleNeed(false) },
            onClean: handleClean, onResetData: handleResetData, cleaning: cleaning, plan: plan,
            onPickRecommended: selectChart, showAllCols: showAllCols, onToggleCols: function () { setShowAllCols(!showAllCols) },
            busy: busy, onDemo: handleDemo,
          }),
          h(CenterColumn, {
            catalog: catalog, chart: chart, onSelectChart: selectChart, chartMeta: meta,
            params: params, onParamsChange: setParams, style: style, onStyleChange: setStyle,
            groupLevels: groupLevels, columns: columns, roles: roles, busy: busy,
            panelMode: panelMode, onPanelMode: setPanelMode, openSections: openSections,
            onToggleSection: function (key) {
              setOpenSections(function (previous) {
                var next = Object.assign({}, previous)
                next[key] = !next[key]
                return next
              })
            },
            onPlot: function () { runPlot() }, onExport: handleExport,
          }),
          h(RightColumn, {
            dataset: dataset, figure: figure, plotInfo: plotInfo, summary: summary, warnings: warnings,
            busy: busy, busyLabel: busyLabel, tab: tab, onTab: setTab, statsCatalog: statsCatalog,
            statId: statId, onStatId: setStatId, statParams: statParams, onStatParams: setStatParams,
            statResult: statResult, statBusy: statBusy, onRunStat: handleStat, code: code,
            history: history, onRestoreHistory: restoreHistory, onExport: handleExport,
            audit: audit, onAudit: handleAudit, profile: profile, cleaning: cleaning,
            onCopy: function (text, label) {
              copyText(text).then(function () { notify((label || '内容') + '已复制到剪贴板') })
                .catch(function () { notify('复制失败，请手动选择文本') })
            },
            need: need, plan: plan,
          })
        ) : null,
        dragOver ? h('div', { className: 'pl-overlay' },
          h(Icon, { name: 'upload', size: 26 }),
          h('div', null, '松开鼠标导入数据'),
          h('small', null, '支持 CSV / TSV / TXT / XLSX / JSON / GZ / ZIP')) : null,
        toast ? h('div', { className: 'pl-toast' }, h(Icon, { name: 'check', size: 13 }), h('span', null, toast)) : null,
        busyLabel ? h('div', { className: 'pl-canvas-busy', style: { position: 'fixed', left: '50%', top: 12, bottom: 'auto', transform: 'translateX(-50%)', padding: '6px 12px', borderRadius: 8, background: 'var(--dsw-alias-bg-overlay)', border: '1px solid var(--dsw-alias-border-l1)', zIndex: 40 } },
          h('span', { className: 'pl-spinner' }), h('span', null, busyLabel)) : null
      )
    }

    function styleGroupLevels(dataset, preview, roles) {
      var levels = []
      var group = roles && roles.group
      if (group) {
        if (dataset && dataset.groups && dataset.roles && dataset.roles.group === group) levels = dataset.groups.slice(0)
        else if (preview && preview.groups) levels = preview.groups.slice(0)
      }
      if (!levels.length && dataset && dataset.groups) levels = dataset.groups.slice(0)
      if (!levels.length && preview && preview.groups) levels = preview.groups.slice(0)
      return levels.slice(0, 16)
    }

    function copyText(text) {
      if (navigator && navigator.clipboard && navigator.clipboard.writeText) return navigator.clipboard.writeText(String(text))
      return Promise.reject(new Error('clipboard unavailable'))
    }

    // ======================================================================
    // 顶栏
    // ======================================================================
    function TopBar(props) {
      var dataset = props.dataset
      var engine = props.status || {}
      var ready = Boolean(engine.ready)
      var engineBadge = ready
        ? h('span', { className: 'pl-badge pl-badge-ok', title: 'Python ' + (engine.python || '') + '；matplotlib ' +
            ((engine.packages && engine.packages.matplotlib) || '?') },
          h('span', { className: 'pl-dot' }), '引擎 ', h('b', null, engine.version || 'ready'))
        : h('span', { className: 'pl-badge ' + (engine.error ? 'pl-badge-bad' : 'pl-badge-warn'), title: engine.error || '引擎启动中' },
          h('span', { className: 'pl-dot' }), engine.error ? '引擎异常' : '引擎启动中')
      return h('div', { className: 'pl-top' },
        h('div', { className: 'pl-brand' },
          h('span', { className: 'pl-mark' }, h(Icon, { name: 'chart', size: 17 })),
          h('div', { className: 'pl-title-block' },
            h('div', { className: 'pl-title' }, 'PlotLab 绘图工作台'),
            h('div', { className: 'pl-subtitle' }, '拖入数据 · 说需求 · 调字体配色 · 带走图和代码'))),
        dataset ? h('span', { className: 'pl-badge', title: dataset.source || '' },
          h(Icon, { name: 'data', size: 12 }),
          h('b', null, dataset.name),
          '· ' + dataset.n_rows + '×' + dataset.n_cols + ' · ' + (dataset.kind === 'matrix' ? '矩阵' : '长表')) : null,
        engineBadge,
        h('div', { className: 'pl-top-actions' },
          h('button', { className: 'pl-btn pl-btn-primary', onClick: props.onPickFile, disabled: Boolean(props.busy) },
            h(Icon, { name: 'upload', size: 13 }), '导入数据'),
          h('button', { className: 'pl-btn', onClick: props.onDemo, disabled: Boolean(props.busy) }, '示例'),
          h('span', { className: 'pl-sep' }),
          h('button', { className: 'pl-btn', onClick: props.onExport, disabled: !dataset || Boolean(props.busy), title: '导出图片+代码+报告 zip' },
            h(Icon, { name: 'download', size: 13 }), '导出'),
          h('button', { className: 'pl-btn pl-btn-ghost', onClick: props.onAudit, disabled: !dataset, title: '确定性自检' },
            h(Icon, { name: 'check', size: 13 }), '自检'),
          h('button', { className: 'pl-btn pl-btn-ghost', onClick: props.onRestart, title: '重启 Python 引擎（清空内存）' },
            h(Icon, { name: 'refresh', size: 13 })))
      )
    }

    // ======================================================================
    // 空态
    // ======================================================================
    function EmptyState(props) {
      var boot = props.boot
      var ready = boot && boot.engine && boot.engine.ready
      return h('div', { className: 'pl-empty-state' },
        h('div', { className: 'pl-mark', style: { width: 44, height: 44, borderRadius: 13 } }, h(Icon, { name: 'chart', size: 24 })),
        h('div', { className: 'pl-empty-title' }, '把数据拖进来，或者先试一个示例'),
        h('div', { className: 'pl-empty-sub' }, '支持 CSV / TSV / TXT / XLSX / JSON / GZ / ZIP。导入后在左侧写一句需求（例如「按分组画箱线图比较 score，配色用色盲友好」），点「分析并出图」会自动清洗、选图、出图并做统计。'),
        h('button', { className: 'pl-dropzone', onClick: props.onPickFile },
          h(Icon, { name: 'upload', size: 22 }),
          h('div', null, '点击选择文件，或直接把文件拖到页面任意位置'),
          h('small', { style: { color: 'var(--dsw-alias-label-tertiary)' } },
            ready ? '引擎就绪：Python ' + ((boot.engine && boot.engine.version) || '') : '引擎启动中，导入后自动初始化')),
        h('div', { className: 'pl-demo-row' },
          h('button', { className: 'pl-demo-card', onClick: function () { props.onDemo('clinical') } },
            h('span', { className: 'pl-demo-title' }, '临床队列示例'),
            h('span', { className: 'pl-demo-sub' }, '96 例 · 分组/性别/指标/生存时间与事件')),
          h('button', { className: 'pl-demo-card', onClick: function () { props.onDemo('expression') } },
            h('span', { className: 'pl-demo-title' }, '表达矩阵示例'),
            h('span', { className: 'pl-demo-sub' }, '12 基因 × 16 样本 · Tumor/Normal')),
          h('button', { className: 'pl-demo-card', onClick: function () { props.onDemo('dose') } },
            h('span', { className: 'pl-demo-title' }, '剂量反应示例'),
            h('span', { className: 'pl-demo-sub' }, '5 剂量 × 2 基因型 × 4 重复')))
      )
    }
    // ======================================================================
    // 左栏：数据 / 角色 / 需求 / 数据质量
    // ======================================================================
    function LeftColumn(props) {
      var dataset = props.dataset
      var preview = props.preview
      var columns = props.columns || []
      var profile = props.profile
      var chartMeta = props.chartMeta
      var roleSpec = (chartMeta && chartMeta.roles) || {}
      var visibleColumns = props.showAllCols ? columns : columns.slice(0, 12)

      return h('div', { className: 'pl-col pl-left' },
        h('div', { className: 'pl-section' },
          h('div', { className: 'pl-datacard' },
            h('div', { className: 'pl-datacard-name', title: dataset.name }, dataset.name),
            h('div', { className: 'pl-datacard-meta' },
              h('span', null, h('b', null, dataset.n_rows), ' 行'),
              h('span', null, h('b', null, dataset.n_cols), ' 列'),
              h('span', null, dataset.kind === 'matrix' ? '矩阵数据' : '长表数据'),
              dataset.group_col ? h('span', null, '分组 ', h('b', null, dataset.group_col)) : null),
            dataset.source ? h('div', { className: 'pl-source', title: dataset.source }, dataset.source) : null,
            h('div', { className: 'pl-row', style: { marginTop: 2 } },
              h('button', { className: 'pl-btn pl-btn-sm', onClick: props.onToggleCols },
                props.showAllCols ? '收起列' : '查看列（' + columns.length + '）'),
              h('button', { className: 'pl-btn pl-btn-sm pl-btn-ghost', onClick: function () { props.onDemo('expression') } }, '换示例'))),
          visibleColumns.length ? h('div', { className: 'pl-cols' }, visibleColumns.map(function (column) {
            return h('div', { className: 'pl-colrow', key: column.key, title: column.key + '：' + (column.dtype || '') },
              h('span', { className: 'pl-coldot pl-coldot-' + (column.dtype || 'text') }),
              h('span', { className: 'pl-colname' }, column.key),
              h('span', { className: 'pl-tag' + (column.dtype === 'numeric' ? ' pl-tag-num' : column.dtype === 'category' ? ' pl-tag-cat' : '') },
                column.dtype === 'numeric' ? '数值' : column.dtype === 'category' ? '分类' : column.dtype === 'datetime' ? '时间' : '文本'),
              column.missing_pct ? h('span', { className: 'pl-colmeta' }, column.missing_pct + '% 缺') : null)
          })) : null),

        h('div', { className: 'pl-section' },
          h('div', { className: 'pl-section-head' },
            h('div', { className: 'pl-section-title' }, h(Icon, { name: 'data', size: 12 }), '列角色映射'),
            h('span', { className: 'pl-section-note' }, (chartMeta && (chartMeta.title || chartMeta.id)) || '')),
          Object.keys(roleSpec).length ? h('div', { className: 'pl-card-body' },
            Object.keys(roleSpec).map(function (roleKey) {
              var kind = roleSpec[roleKey]
              if (kind === 'none') return null
              var def = null
              ROLE_DEFS.forEach(function (item) { if (item.key === roleKey) def = item })
              var options = columnsForRole(columns, roleKey)
              var value = (props.roles && props.roles[roleKey]) || ''
              return h('div', { className: 'pl-field', key: roleKey, title: def ? def.hint : '' },
                h('label', { className: 'pl-field-label' },
                  (def ? def.label : roleKey),
                  kind === 'required' ? h('span', { style: { color: 'var(--dsw-alias-state-error-primary)', marginLeft: 2 } }, '*') : null),
                h(Select, {
                  value: value,
                  options: [{ id: '', label: '—' }].concat(options.map(function (column) {
                    return { id: column.key, label: column.key + '（' + (column.dtype === 'numeric' ? '数值' : '分类') + '）' }
                  })),
                  onChange: function (next) {
                    var payload = JSON.parse(JSON.stringify(props.roles || {}))
                    if (!next) delete payload[roleKey]
                    else payload[roleKey] = next
                    props.onRolesChange(payload)
                  },
                }))
            })) : h('div', { className: 'pl-hint' }, '这个图表不需要列角色（使用矩阵视图）。')),

        h('div', { className: 'pl-section' },
          h('div', { className: 'pl-need' },
            h('div', { className: 'pl-section-head' },
              h('div', { className: 'pl-section-title' }, h(Icon, { name: 'wand', size: 12 }), 'AI 需求'),
              h('span', { className: 'pl-section-note' }, '自动清洗 + 选图 + 统计')),
            h('textarea', {
              className: 'pl-textarea',
              placeholder: '例如：按 group 分组画箱线图比较 score 并标显著性，配色用色盲友好、标题字号大一点',
              value: props.need,
              onChange: function (event) { props.onNeedChange(event.target.value) },
              onKeyDown: function (event) {
                if (event.key === 'Enter' && (event.metaKey || event.ctrlKey)) { event.preventDefault(); props.onAuto() }
              },
            }),
            h('div', { className: 'pl-row' },
              h('button', { className: 'pl-btn pl-btn-primary', onClick: props.onAuto, disabled: Boolean(props.busy) },
                h(Icon, { name: 'play', size: 12 }), '分析并出图'),
              h('button', { className: 'pl-btn', onClick: props.onNeedOnly, disabled: Boolean(props.busy) }, '交给 Agent'),
              h('span', { className: 'pl-hint', style: { marginLeft: 'auto' } }, '⌘/Ctrl+Enter')),
            props.plan ? h('div', { className: 'pl-plan' },
              h('div', { className: 'pl-plan-steps' }, (props.plan.steps || []).map(function (step, index) {
                return h('span', { className: 'pl-step', key: index }, h('b', null, index + 1), ' ' + (step.label || step.kind))
              })),
              (props.plan.rationale || []).length ? h('ul', { className: 'pl-rationale' }, (props.plan.rationale || []).map(function (line, index) {
                return h('li', { key: index }, String(line))
              })) : null) : null)),

        h('div', { className: 'pl-section', style: { borderBottom: 'none' } },
          h('div', { className: 'pl-section-head' },
            h('div', { className: 'pl-section-title' }, h(Icon, { name: 'spark', size: 12 }), '数据质量'),
            profile ? h('span', { className: 'pl-badge ' + (profile.level === 'good' ? 'pl-badge-ok' : profile.level === 'warn' ? 'pl-badge-warn' : 'pl-badge-bad') },
              profile.score + '/100') : null),
          profile ? h('div', { className: 'pl-card-body' },
            h('div', { className: 'pl-score' },
              h('div', { className: 'pl-score-num' }, profile.score, h('small', null, '/100')),
              h('div', { className: 'pl-score-bar' }, h('span', { style: { width: Math.max(4, profile.score) + '%' } }))),
            (function () {
              var problems = (profile.checks || []).filter(function (check) { return check.level !== 'ok' })
              if (!problems.length) return h('div', { className: 'pl-hint' }, '未发现明显数据问题，可以直接出图。')
              return h('div', { className: 'pl-checks' }, problems.slice(0, 6).map(function (check) {
                return h('div', { className: 'pl-check-row', key: check.id, title: check.fix || '' },
                  h('span', { className: 'pl-check-dot pl-check-' + levelTone(check.level) }),
                  h('span', { className: 'pl-check-text' },
                    h('b', null, check.title),
                    check.detail,
                    check.fix ? h('span', { className: 'pl-fix' }, '　建议：' + check.fix) : null))
              }))
            })(),
            h('div', { className: 'pl-row' },
              h('button', { className: 'pl-btn pl-btn-sm pl-btn-primary', onClick: props.onClean, disabled: Boolean(props.busy) },
                h(Icon, { name: 'wand', size: 12 }), '一键清洗'),
              props.cleaning ? h('span', { className: 'pl-hint' }, '已应用 ' + (props.cleaning.applied || []).length + ' 步') : null),
            props.cleaning && (props.cleaning.applied || []).length
              ? h('div', { className: 'pl-hint' }, (props.cleaning.applied || []).map(function (item) {
                  return (item.op || '') + (item.reason ? '：' + item.reason : '')
                }).join('；'))
              : null,
            profile.recommend && (profile.recommend.charts || []).length
              ? h('div', { className: 'pl-card-body' },
                  h('div', { className: 'pl-gallery-label' }, '推荐图表'),
                  h('div', { className: 'pl-row' }, (profile.recommend.charts || []).slice(0, 5).map(function (item) {
                    return h('button', {
                      className: 'pl-btn pl-btn-sm', key: item.id, title: item.why,
                      onClick: function () { props.onPickRecommended(item.id) },
                    }, item.title)
                  })),
                  profile.recommend.stats && profile.recommend.stats.length
                    ? h('div', { className: 'pl-hint' }, '推荐统计：' + profile.recommend.stats.slice(0, 4).map(function (item) { return item.title }).join('、'))
                    : null)
              : null) : h('div', { className: 'pl-hint' }, '剖析中…'),
          props.cleaning ? null : null)
      )
    }

    // ======================================================================
    // 中栏：图表画廊 / 参数 / 样式
    // ======================================================================
    function CenterColumn(props) {
      var catalog = props.catalog
      var charts = (catalog && catalog.charts) || []
      var meta = props.chartMeta
      var groups = chartGroups(charts)
      return h('div', { className: 'pl-col pl-center' },
        h('div', { className: 'pl-section', style: { paddingBottom: 6 } },
          h('div', { className: 'pl-section-head' },
            h('div', { className: 'pl-section-title' }, h(Icon, { name: 'chart', size: 12 }), '图表类型'),
            h('span', { className: 'pl-section-note' }, charts.length + ' 种 · 点选后自动出图')),
          h('div', { className: 'pl-gallery' }, groups.map(function (group) {
            return h('div', { className: 'pl-gallery-group', key: group.group },
              h('div', { className: 'pl-gallery-label' }, group.group),
              h('div', { className: 'pl-gallery-grid' }, group.charts.map(function (chart) {
                return h('button', {
                  key: chart.id,
                  className: 'pl-chart' + (props.chart === chart.id ? ' pl-chart-on' : ''),
                  title: chart.desc || chart.title,
                  onClick: function () { props.onSelectChart(chart.id) },
                  type: 'button',
                },
                  chartGlyph(chart.id),
                  h('span', null, chart.title))
              })))
          }))),

        meta && (meta.params || []).length ? h('div', { className: 'pl-section' },
          h('div', { className: 'pl-card' },
            h('div', { className: 'pl-card-head' },
              h('div', { className: 'pl-card-title' }, h(Icon, { name: 'play', size: 12 }), '图表参数'),
              h('span', { className: 'pl-section-note' }, meta.title || meta.id)),
            h('div', { className: 'pl-grid2' }, (meta.params || []).map(function (param) {
              return h(StyleField, {
                key: param.key, field: { key: param.key, label: param.label || param.key, type: param.type || 'text', options: param.options, min: param.min, max: param.max, step: param.step, placeholder: param.placeholder },
                value: props.params ? props.params[param.key] : undefined,
                onChange: function (value) {
                  var next = JSON.parse(JSON.stringify(props.params || {}))
                  if (isNil(value)) delete next[param.key]
                  else next[param.key] = value
                  props.onParamsChange(next)
                },
              })
            })))) : null,

        h('div', { className: 'pl-section', style: { borderBottom: 'none' } },
          h('div', { className: 'pl-card' },
            h('div', { className: 'pl-card-head' },
              h('div', { className: 'pl-card-title' }, h(Icon, { name: 'brush', size: 12 }), '样式调整'),
              h('span', { className: 'pl-section-note' }, '改动实时生效')),
            h('div', { className: 'pl-hint' }, '字体/字号/颜色/画布/坐标轴/图例/显著性标注，全部实时重绘预览。'),
            STYLE_GROUPS.map(function (groupKey) {
              var open = Boolean(props.openSections && props.openSections[groupKey])
              var fields = styleFieldsByGroup(groupKey)
              return h('div', { key: groupKey },
                h('button', { type: 'button', className: 'pl-style-head', onClick: function () { props.onToggleSection(groupKey) } },
                  h('span', null, STYLE_GROUP_LABELS[groupKey] || groupKey),
                  h('span', { className: 'pl-style-caret' + (open ? ' pl-style-caret-open' : '') }, h(Icon, { name: 'chevron', size: 12 }))),
                open ? h('div', { className: groupKey === 'color' ? 'pl-card-body' : 'pl-grid2' }, fields.map(function (field) {
                  if (field.type === 'colors') {
                    return h(ColorEditor, {
                      key: field.key,
                      levels: props.groupLevels || [],
                      colors: (props.style && props.style.colors) || {},
                      palette: (props.style && props.style.palette) || 'okabe_ito',
                      onChange: function (next) {
                        var styleNext = JSON.parse(JSON.stringify(props.style || {}))
                        styleNext.colors = next
                        props.onStyleChange(styleNext)
                      },
                    })
                  }
                  return h(StyleField, {
                    key: field.key, field: field,
                    value: props.style ? props.style[field.key] : undefined,
                    onChange: function (value) {
                      var styleNext = JSON.parse(JSON.stringify(props.style || {}))
                      styleNext[field.key] = value
                      props.onStyleChange(styleNext)
                    },
                  })
                })) : null)
            }),
            h('div', { className: 'pl-row', style: { marginTop: 2 } },
              h('button', { className: 'pl-btn pl-btn-sm pl-btn-primary', onClick: props.onPlot, disabled: Boolean(props.busy) }, h(Icon, { name: 'refresh', size: 12 }), '重新渲染'),
              h('button', { className: 'pl-btn pl-btn-sm', onClick: function () { props.onExport('figure') }, disabled: Boolean(props.busy) }, h(Icon, { name: 'download', size: 12 }), '导出 PNG'),
              h('button', { className: 'pl-btn pl-btn-sm', onClick: function () { props.onExport('bundle') }, disabled: Boolean(props.busy) }, '导出打包'))))
      )
    }

    function chartGlyph(id) {
      var paths = {
        bar: 'M5 19V9h3v10zM10.5 19V5h3v14zM16 19v-7h3v7z',
        bar_dot: 'M5 19V10h3v9zM10.5 19V7h3v12zM16 19v-6h3v6zM6 13h1M11 11h1M17 9h1',
        box: 'M5 7h4v10H5zM6 12h2M15 5h4v14h-4zM16 11h2',
        violin: 'M6 5c3 4 3 10 0 14 4-2 4-12 0-14zM18 5c-3 4-3 10 0 14-4-2-4-12 0-14',
        raincloud: 'M5 8c3 2 3 6 0 8 3-1 3-7 0-8zM14 6h5v8h-5zM15 18h.01M18 19h.01',
        dotplot: 'M6 8h.01M10 8h.01M14 8h.01M6 12h.01M10 12h.01M14 12h.01M6 16h.01M10 16h.01',
        stack: 'M5 6h5v12H5zM5 12h5M12 10h7v8h-7zM12 14h7',
        pie: 'M12 3a9 9 0 1 0 9 9h-9zM12 3v9h9',
        scatter: 'M6 17h.01M9 12h.01M12 15h.01M15 8h.01M18 11h.01M8 8h.01',
        bubble: 'M8 15a3 3 0 1 0 6 0 3 3 0 1 0-6 0M15 9a2 2 0 1 0 4 0 2 2 0 1 0-4 0M6 9h.01',
        line: 'M4 17l5-6 4 3 6-8',
        area: 'M4 17l5-6 4 3 6-8v11z',
        heatmap: 'M4 4h6v6H4zM14 4h6v6h-6zM4 14h6v6H4zM14 14h6v6h-6z',
        corr: 'M4 4h6v6H4zM14 4h6v6h-6zM4 14h6v6H4zM14 14h6v6h-6zM7 7h.01M17 17h.01',
        pairplot: 'M4 4h6v6H4zM14 4h6v6h-6zM4 14h6v6H4zM14 14h6v6h-6z',
        hist: 'M4 19V5M4 19h16M7 19v-6h3v6M12 19V8h3v11M17 19v-9h3v9',
        density: 'M3 17c3 0 3-9 6-9s3 5 6 5 3-5 6-5',
        qq: 'M4 19L20 5M6 15h.01M10 12h.01M14 9h.01M17 7h.01',
        roc: 'M4 20V4M4 20h16M4 20c7 0 11-4 13-13',
        km: 'M4 6h4v3h5v4h7M4 6v14h16',
        forest: 'M6 6h10M6 12h6M6 18h13M4 3v18',
        pca: 'M4 18C8 18 12 16 20 6M7 15h.01M12 13h.01M16 9h.01M9 7h.01',
      }
      if (id === 'bar') return h('svg', { width: 22, height: 16, viewBox: '0 0 24 22', fill: 'none', stroke: 'currentColor', strokeWidth: 1.6, strokeLinejoin: 'round' }, h('path', { d: paths.bar }))
      return h('svg', { width: 22, height: 16, viewBox: '0 0 24 22', fill: 'none', stroke: 'currentColor', strokeWidth: 1.6, strokeLinecap: 'round', strokeLinejoin: 'round' }, h('path', { d: paths[id] || paths.chart || 'M4 19V5M4 19h16' }))
    }

    function StyleField(props) {
      var field = props.field || {}
      var value = props.value
      var id = 'pl-field-' + field.key
      var control = null
      if (field.type === 'select') {
        var options = (field.options || []).map(function (option) {
          if (option && typeof option === 'object') return { id: isNil(option.id) ? option.value : option.id, label: option.label || String(isNil(option.id) ? option.value : option.id) }
          return { id: option, label: String(option) }
        })
        control = h(Select, { value: value, options: options, onChange: props.onChange })
      } else if (field.type === 'number') {
        control = h(NumberInput, { value: value, min: field.min, max: field.max, step: field.step, onChange: function (next) { props.onChange(next === '' ? null : Number(next)) } })
      } else if (field.type === 'boolean') {
        control = h(Toggle, { value: value, onChange: props.onChange })
      } else if (field.type === 'range') {
        control = h(RangeInput, { value: value, onChange: props.onChange })
      } else {
        control = h(TextInput, { value: value, placeholder: field.placeholder, onChange: props.onChange })
      }
      return h('div', { className: 'pl-field' + (field.type === 'range' ? ' pl-field-col' : ''), key: field.key },
        h('label', { className: 'pl-field-label', htmlFor: id, title: field.label },
          field.label, field.unit ? h('span', { style: { color: 'var(--dsw-alias-label-tertiary)' } }, '（' + field.unit + '）') : null),
        control)
    }

    function ColorEditor(props) {
      var levels = props.levels || []
      if (!levels.length) {
        return h('div', { className: 'pl-hint' }, '当前图表没有分组/分类颜色可调；选择一个「分组」列后可逐组改色。')
      }
      return h('div', { className: 'pl-colorlist' }, levels.map(function (level) {
        var color = (props.colors && props.colors[level]) || ''
        return h('div', { className: 'pl-colorrow', key: String(level) },
          h('span', { title: String(level) }, String(level)),
          h('input', {
            className: 'pl-swatch', type: 'color',
            value: color || '#4E79A7',
            onChange: function (event) {
              var next = Object.assign({}, props.colors || {})
              next[level] = String(event.target.value).toUpperCase()
              props.onChange(next)
            },
          }),
          color ? h('button', { className: 'pl-btn pl-btn-sm pl-btn-ghost', onClick: function () {
            var next = Object.assign({}, props.colors || {})
            delete next[level]
            props.onChange(next)
          } }, '复位') : null)
      }))
    }

    // ======================================================================
    // 右栏：预览 / 统计 / 代码 / 记录 / 自检
    // ======================================================================
    function RightColumn(props) {
      var dataset = props.dataset
      var tabs = [
        { id: 'preview', label: '预览' },
        { id: 'stats', label: '统计' },
        { id: 'code', label: '代码' },
        { id: 'history', label: '记录' },
        { id: 'audit', label: '自检' },
      ]
      return h('div', { className: 'pl-col pl-right' },
        h('div', { className: 'pl-preview' },
          h('div', { className: 'pl-preview-toolbar' },
            h('div', { className: 'pl-card-title' }, h(Icon, { name: 'chart', size: 12 }),
              props.plotInfo ? (props.plotInfo.chart_title || '预览') : '预览'),
            h('div', { className: 'pl-row', style: { marginLeft: 'auto' } },
              h('button', { className: 'pl-btn pl-btn-sm', onClick: props.onAudit, disabled: !dataset, title: '确定性自检' }, '自检'),
              h('button', { className: 'pl-btn pl-btn-sm', onClick: function () { props.onExport('figure') }, disabled: !props.figure }, h(Icon, { name: 'download', size: 12 }), 'PNG'),
              h('button', { className: 'pl-btn pl-btn-sm', onClick: function () { props.onExport('bundle') }, disabled: !props.figure }, '打包'))),
          h('div', { className: 'pl-canvas' },
            props.figure
              ? h('img', {
                  src: figureUrl(props.figure, props.figure.id),
                  alt: (props.plotInfo && props.plotInfo.chart_title) || 'figure',
                  onClick: function () { window.open(figureUrl(props.figure, props.figure.id), '_blank') },
                  style: { cursor: 'zoom-in' },
                })
              : h('div', { className: 'pl-canvas-empty' },
                  h(Icon, { name: 'chart', size: 26 }),
                  h('div', null, dataset ? '选一个图表，或写一句需求' : '拖入数据开始'),
                  h('div', { className: 'pl-hint' }, '图片 300 dpi，可在页面实时调字体/字号/颜色')),
            props.busy === 'plot' ? h('div', { className: 'pl-canvas-busy' }, h('span', { className: 'pl-spinner' }), h('span', null, '渲染中…')) : null),
          props.figure ? h('div', { className: 'pl-figmeta' },
            props.figure.width && props.figure.height ? h('span', null, props.figure.width + '×' + props.figure.height + ' px') : null,
            props.figure.bytes ? h('span', null, formatBytes(props.figure.bytes)) : null,
            props.plotInfo && props.plotInfo.elapsed_ms ? h('span', null, '渲染 ' + props.plotInfo.elapsed_ms + ' ms') : null,
            props.figure.file ? h('span', { style: { wordBreak: 'break-all' } }, props.figure.file) : null) : null,
          (props.summary && (props.summary.bullets || []).length) ? h(Bullets, { items: props.summary.bullets }) : null,
          (props.warnings || []).length ? h('div', { className: 'pl-warns' }, props.warnings.slice(0, 5).map(function (warning, index) {
            return h('div', { key: index }, '· ' + warning)
          })) : null),

        h('div', { className: 'pl-tabs' }, tabs.map(function (item) {
          return h('button', { key: item.id, className: 'pl-tab' + (props.tab === item.id ? ' pl-tab-on' : ''), onClick: function () { props.onTab(item.id) } },
            item.label,
            item.id === 'stats' && props.statResult ? h('span', { className: 'pl-tag pl-tag-num' }, '1') : null)
        })),

        props.tab === 'preview' ? h('div', { className: 'pl-tabpane' },
          props.plotInfo && props.plotInfo.roles && Object.keys(props.plotInfo.roles).length
            ? h('div', { className: 'pl-card-body' },
                h('div', { className: 'pl-gallery-label' }, '当前映射'),
                h('div', { className: 'pl-row' }, Object.keys(props.plotInfo.roles).map(function (key) {
                  return h('span', { className: 'pl-tag', key: key }, key + ' = ' + props.plotInfo.roles[key])
                })))
            : null,
          props.plotInfo && props.plotInfo.params && Object.keys(props.plotInfo.params).length
            ? h('div', { className: 'pl-card-body' },
                h('div', { className: 'pl-gallery-label' }, '图表参数'),
                h('div', { className: 'pl-hint' }, Object.keys(props.plotInfo.params).map(function (key) {
                  return key + '=' + String(props.plotInfo.params[key])
                }).join('，')))
            : null,
          h('div', { className: 'pl-card-body' },
            h('div', { className: 'pl-gallery-label' }, '样式摘要'),
            h('div', { className: 'pl-hint' }, styleSummary(props.plotInfo && props.plotInfo.style))),
          props.plan ? h('div', { className: 'pl-card-body' },
            h('div', { className: 'pl-gallery-label' }, '分析计划'),
            h('div', { className: 'pl-plan-steps' }, (props.plan.steps || []).map(function (step, index) {
              return h('span', { className: 'pl-step', key: index }, (index + 1) + '. ' + (step.label || step.kind))
            }))) : null
        ) : null,

        props.tab === 'stats' ? h(StatsPanel, {
          catalog: props.statsCatalog, statId: props.statId, onStatId: props.onStatId,
          params: props.statParams, onParams: props.onStatParams, result: props.statResult,
          busy: props.statBusy, onRun: props.onRunStat, onExport: props.onExport, onCopy: props.onCopy,
        }) : null,

        props.tab === 'code' ? h(CodePanel, { code: props.code, onExport: props.onExport, onCopy: props.onCopy }) : null,

        props.tab === 'history' ? h('div', { className: 'pl-tabpane' },
          (props.history || []).length ? h('div', { className: 'pl-history' }, props.history.slice().reverse().map(function (entry, index) {
            return h('button', {
              key: (entry.id || index) + '-' + index,
              className: 'pl-history-item',
              onClick: function () { props.onRestoreHistory(entry) },
              title: '点击恢复这套配置',
            },
              entry.figure ? h('img', { src: figureUrl(entry.figure, entry.figure.id), alt: '' }) : null,
              h('div', { className: 'pl-history-main' },
                h('div', { className: 'pl-history-title' }, entry.chart_title || entry.chart || '图'),
                h('div', { className: 'pl-history-meta' }, formatTime(entry.at) + (entry.need ? ' · ' + entry.need : ''))))
          })) : h(Empty, { text: '还没有出图记录' })) : null,

        props.tab === 'audit' ? h(AuditPanel, { audit: props.audit, onRun: props.onAudit, profile: props.profile }) : null
      )
    }

    function styleSummary(style) {
      if (!style) return '—'
      return '字体 ' + (style.font || '自动') + '；标题 ' + style.title_size + 'pt；轴标题 ' + style.label_size +
        'pt；刻度 ' + style.tick_size + 'pt；配色 ' + style.palette + '；画布 ' + style.fig_width + '×' + style.fig_height +
        ' in；' + style.dpi + ' dpi；图例 ' + style.legend_pos + '；显著性标注 ' + (style.stat_annot ? '开' : '关')
    }

    function StatsPanel(props) {
      var catalog = props.catalog || []
      var meta = findStat(catalog, props.statId)
      var groups = []
      var seen = {}
      catalog.forEach(function (item) {
        var group = item.group || '其他'
        if (!seen[group]) { seen[group] = true; groups.push(group) }
      })
      return h('div', { className: 'pl-tabpane' },
        h('div', { className: 'pl-card' },
          h('div', { className: 'pl-card-head' },
            h('div', { className: 'pl-card-title' }, h(Icon, { name: 'stats', size: 12 }), '统计分析'),
            h('span', { className: 'pl-section-note' }, (meta && meta.title) || '选择方法')),
          h('div', { className: 'pl-grid2' },
            h('div', { className: 'pl-field' },
              h('label', { className: 'pl-field-label' }, '方法'),
              h(Select, {
                value: props.statId,
                options: catalog.map(function (item) {
                  return { id: item.id, label: (item.group ? '[' + item.group + '] ' : '') + (item.title || item.id) }
                }),
                onChange: props.onStatId,
              })),
            h('div', { className: 'pl-row' },
              h('button', { className: 'pl-btn pl-btn-sm pl-btn-primary', onClick: props.onRun, disabled: props.busy },
                h(Icon, { name: 'play', size: 12 }), props.busy ? '运行中…' : '运行分析'))),
          meta && (meta.params || []).length ? h('div', { className: 'pl-grid2' }, (meta.params || []).map(function (param) {
            return h(StyleField, {
              key: param.key,
              field: { key: param.key, label: param.label || param.key, type: param.type || 'text', options: param.options, min: param.min, max: param.max, step: param.step },
              value: (props.params || {})[param.key],
              onChange: function (value) {
                var next = JSON.parse(JSON.stringify(props.params || {}))
                next[param.key] = value
                props.onParams(next)
              },
            })
          })) : null,
          meta && meta.roles ? h('div', { className: 'pl-hint' }, '需要的列：' + Object.keys(meta.roles).filter(function (key) {
            return meta.roles[key] !== 'none'
          }).map(function (key) { return key + (meta.roles[key] === 'required' ? '*' : '') }).join('、') + '（在左侧角色映射里指定）') : null),
        !props.result ? h(Empty, { text: '选一个统计方法点「运行分析」。结果会带上方法、效应量和 p 值，并可导出 CSV。' }) : h('div', null,
          h('div', { className: 'pl-card-body' },
            h('div', { className: 'pl-card-title' }, props.result.title || props.result.stat),
            h('div', { className: 'pl-hint' }, props.result.method || ''),
            h(Bullets, { items: (props.result.summary && props.result.summary.bullets) || [] }),
            props.result.warnings && props.result.warnings.length ? h('div', { className: 'pl-warns' }, props.result.warnings.map(function (w, i) { return h('div', { key: i }, '· ' + w) })) : null),
          (props.result.figures || []).map(function (figure, index) {
            return h('div', { className: 'pl-statfig', key: index },
              h('div', { className: 'pl-gallery-label' }, figure.title || ('统计图 ' + (index + 1))),
              h('img', { src: figureUrl(figure, figure.id), alt: figure.title || 'stat figure' }))
          }),
          (props.result.tables || []).map(function (table, index) {
            return h('div', { className: 'pl-card-body', key: index },
              h('div', { className: 'pl-card-head' },
                h('div', { className: 'pl-gallery-label' }, table.title || ('表 ' + (index + 1))),
                h('span', { className: 'pl-section-note' }, (table.rows || []).length + ' 行')),
              h(Table, { table: table }))
          }),
          h('div', { className: 'pl-row' },
            h('button', { className: 'pl-btn pl-btn-sm', onClick: function () { props.onExport('stats') } }, h(Icon, { name: 'download', size: 12 }), '导出统计表'),
            props.result.report ? h('button', { className: 'pl-btn pl-btn-sm', onClick: function () { props.onCopy(props.result.report, '统计报告') } }, h(Icon, { name: 'copy', size: 12 }), '复制报告片段') : null)))
    }

    function CodePanel(props) {
      var code = props.code || {}
      var [which, setWhich] = useState('python')
      var text = which === 'r' ? (code.r || '') : (code.python || '')
      var lines = text ? String(text).split('\n').length : 0
      return h('div', { className: 'pl-tabpane' },
        h('div', { className: 'pl-preview-toolbar' },
          h('button', { className: 'pl-btn pl-btn-sm' + (which === 'python' ? ' pl-btn-primary' : ''), onClick: function () { setWhich('python') } }, 'Python'),
          h('button', { className: 'pl-btn pl-btn-sm' + (which === 'r' ? ' pl-btn-primary' : ''), onClick: function () { setWhich('r') }, disabled: !code.r }, 'R / ggplot2'),
          h('span', { className: 'pl-hint', style: { marginLeft: 'auto' } }, lines ? lines + ' 行' : '暂无'),
          h('button', { className: 'pl-btn pl-btn-sm', onClick: function () { props.onCopy(text, which === 'r' ? 'R 代码' : 'Python 代码') }, disabled: !text }, h(Icon, { name: 'copy', size: 12 }), '复制'),
          h('button', { className: 'pl-btn pl-btn-sm pl-btn-primary', onClick: function () { props.onExport('code') }, disabled: !text }, h(Icon, { name: 'download', size: 12 }), '导出 .py/.R')),
        text ? h('pre', { className: 'pl-code' }, text) : h(Empty, { text: '出图后会自动生成可复现的关键步骤代码（读取数据 → 清洗 → 作图 → 统计），可直接拷走运行。' }))
    }

    function AuditPanel(props) {
      var audit = props.audit
      if (!audit) {
        return h('div', { className: 'pl-tabpane' },
          h('div', { className: 'pl-card-body' },
            h('div', { className: 'pl-hint' }, '自检会确定性检查：数据就绪、质量分、清洗记录、出图、代码、需求记录，并给出 0-100 分。'),
            h('button', { className: 'pl-btn pl-btn-sm pl-btn-primary', onClick: props.onRun }, h(Icon, { name: 'check', size: 12 }), '运行自检')))
      }
      return h('div', { className: 'pl-tabpane' },
        h('div', { className: 'pl-card-body' },
          h('div', { className: 'pl-score' },
            h('div', { className: 'pl-score-num' }, audit.score, h('small', null, '/100')),
            h('div', { className: 'pl-score-bar' }, h('span', { style: { width: Math.max(4, audit.score) + '%' } }))),
          h('div', { className: 'pl-checks' }, (audit.checks || []).map(function (check) {
            return h('div', { className: 'pl-check-row', key: check.id },
              h('span', { className: 'pl-check-dot pl-check-' + levelTone(check.level) }),
              h('span', { className: 'pl-check-text' }, h('b', null, check.title), check.detail))
          }))),
        h('div', { className: 'pl-row' },
          h('button', { className: 'pl-btn pl-btn-sm', onClick: props.onRun }, h(Icon, { name: 'refresh', size: 12 }), '重新自检')))
    }

    // ======================================================================
    // 错误边界 / 侧边栏图标 / 注册
    // ======================================================================
    function PluginErrorBoundary(props) {
      React.Component.call(this, props)
      this.state = { failed: null }
      this.handleReset = this.handleReset.bind(this)
    }
    PluginErrorBoundary.prototype = Object.create(React.Component.prototype)
    PluginErrorBoundary.prototype.constructor = PluginErrorBoundary
    PluginErrorBoundary.prototype.componentDidCatch = function (error) {
      this.setState({ failed: error })
    }
    PluginErrorBoundary.prototype.handleReset = function () {
      this.setState({ failed: null })
    }
    PluginErrorBoundary.prototype.render = function () {
      if (this.state.failed) {
        return h('div', { className: 'pl-root' },
          h('div', { className: 'pl-empty-state' },
            h('div', { className: 'pl-empty-title' }, 'PlotLab 渲染出错'),
            h('div', { className: 'pl-empty-sub' }, String(this.state.failed && this.state.failed.message ? this.state.failed.message : this.state.failed)),
            h('button', { className: 'pl-btn pl-btn-primary', onClick: this.handleReset }, '重试')))
      }
      return this.props.children
    }

    function SidebarIcon() {
      return h('span', { style: { display: 'inline-flex', alignItems: 'center', gap: 6 } },
        h('span', { className: 'pl-mark', style: { width: 20, height: 20, borderRadius: 6 } }, h(Icon, { name: 'chart', size: 12 })),
        h('span', null, '绘图'))
    }

    var DICT_ZH = { nav: '绘图', title: 'PlotLab 绘图工作台' }
    var DICT_EN = { nav: 'PlotLab', title: 'PlotLab workbench' }

    function apply(ctx) {
      var locale = ctx.locale
      if (locale && typeof locale.register === 'function') {
        var registerDictionaries = function () {
          try {
            return locale.register(NS, { zh: DICT_ZH, en: DICT_EN })
          } catch (error) {
            warn('failed to register locale dictionaries', error)
            return undefined
          }
        }
        try {
          if (typeof ctx.effect === 'function') ctx.effect(registerDictionaries, NS + ': locale')
          else registerDictionaries()
        } catch (error) { warn('failed to install locale effect', error) }
      }

      var slots = ctx.slots
      if (!slots && typeof ctx.get === 'function') {
        try { slots = ctx.get('slots') } catch (error) { slots = null }
      }
      if (!slots || typeof slots.register !== 'function' || typeof slots.inject !== 'function') {
        warn('slots service unavailable; UI not registered')
        return
      }

      var disposers = []
      function keep(handle) {
        if (typeof handle === 'function') disposers.push(handle)
        return handle
      }
      try {
        if (typeof ctx.effect === 'function') {
          ctx.effect(function () {
            return function () {
              while (disposers.length) {
                var dispose = disposers.pop()
                try { dispose() } catch (ignored) { /* ignore */ }
              }
            }
          }, NS + ': slot disposers')
        }
      } catch (error) { warn('failed to install slot cleanup', error) }

      try {
        keep(slots.inject('main', function () {
          return keep(slots.register({ name: 'main', key: 'plotlab', locale: NS }, WorkbenchPage))
        }))
      } catch (error) { warn('failed to register main slot', error) }

      try {
        keep(slots.inject('sidebar.panellist', function () {
          return keep(slots.register({
            name: 'sidebar.panellist',
            id: 'plotlab',
            order: 5,
            label: function () { return DICT_ZH.nav },
            locale: NS,
          }, SidebarIcon))
        }))
      } catch (error) { warn('failed to register sidebar.panellist slot', error) }
    }

    function WorkbenchPage() {
      return h(PluginErrorBoundary, null, h(WorkbenchBody))
    }

    return {
      name: NS,
      inject: ['slots', 'locale'],
      apply: apply,
      __test__: Object.assign({}, PL_TEST, {
        PluginErrorBoundary: PluginErrorBoundary,
        components: {
          WorkbenchPage: WorkbenchPage,
          WorkbenchBody: WorkbenchBody,
          TopBar: TopBar,
          EmptyState: EmptyState,
          LeftColumn: LeftColumn,
          CenterColumn: CenterColumn,
          RightColumn: RightColumn,
          StatsPanel: StatsPanel,
          CodePanel: CodePanel,
          AuditPanel: AuditPanel,
          SidebarIcon: SidebarIcon,
        },
      }),
    }

  }

  // ==========================================================================
  // 注册
  // ==========================================================================
  if (typeof window === 'undefined') {
    globalThis.__PL_TEST__ = PL_TEST
  } else if (window.__ModuleLoader__ && typeof window.__ModuleLoader__.load === 'function') {
    window.__ModuleLoader__.load({ id: NS, factory: factory })
  }
})()
