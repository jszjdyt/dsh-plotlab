"""PlotLab 全格式数据导入。

对外 API（worker 的 `inspect` / `open` 命令与 HTTP / Agent 工具直接消费）：

    FORMATS            # 格式清单（UI 下拉框）
    formats()          # 返回 FORMATS 的副本
    inspect_source(...)  # 只探测：行列数、列摘要、tidy/matrix、候选（sheet/压缩包成员）
    open_source(...)     # 全量读取 → pl_state.build_dataset 生成合法 dataset

支持：CSV / TSV / TXT（自动嗅探分隔符）、XLSX（多 sheet）、JSON（records 列表 /
dict of lists / {"data": [...]}）、zip / tar / tar.gz（内嵌表格，member 选择）、
单文件 .gz、http(s)/ftp URL（下载到临时目录）。
文本编码按 utf-8-sig → utf-8 → gbk → gb18030 → latin-1 依次尝试。

约定（DESIGN.md §2 / §7）：
- 只依赖标准库 + pandas/numpy + pl_common + pl_prep；openpyxl 懒加载；
  pl_state 只在 open_source 内部懒加载，避免模块循环。
- 不碰 registry / engine，不存全局状态；所有探测信息随返回值 / dataset.notes 走。
- matrix 数据约定：首列 = 特征 id，其余列 = 样本；样本表按样本名挂到 sample_info。
"""
from __future__ import annotations

import codecs
import copy
import csv
import gzip
import json
import math
import re
import socket
import tarfile
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import pl_common as C
import pl_prep as P

__all__ = ["FORMATS", "formats", "inspect_source", "open_source"]


# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

_FORMAT_LABELS = {
    "auto": "自动识别",
    "csv": "CSV（逗号分隔）",
    "tsv": "TSV（制表符分隔）",
    "txt": "TXT 文本",
    "xlsx": "Excel 工作簿",
    "json": "JSON",
    "archive": "压缩包",
    "gz": "gzip 单文件",
}

FORMATS: list[dict] = [
    {"id": "auto", "label": _FORMAT_LABELS["auto"], "extensions": [],
     "desc": "按扩展名与文件内容自动判断格式"},
    {"id": "csv", "label": _FORMAT_LABELS["csv"], "extensions": [".csv"],
     "desc": "逗号分隔；支持引号内逗号、千分位与 <0.001 之类的数值写法"},
    {"id": "tsv", "label": _FORMAT_LABELS["tsv"], "extensions": [".tsv", ".tab"],
     "desc": "制表符分隔"},
    {"id": "txt", "label": _FORMAT_LABELS["txt"], "extensions": [".txt", ".text", ".dat"],
     "desc": "自动嗅探 , ; 制表符 | 分隔符"},
    {"id": "xlsx", "label": _FORMAT_LABELS["xlsx"], "extensions": [".xlsx", ".xlsm"],
     "desc": "多 sheet 默认第一个，可用 sheet 指定；.xls 需另存为 .xlsx"},
    {"id": "json", "label": _FORMAT_LABELS["json"], "extensions": [".json", ".jsonl"],
     "desc": "records 列表 / dict of lists / {\"data\": [...]}；嵌套值转 JSON 字符串"},
    {"id": "archive", "label": _FORMAT_LABELS["archive"],
     "extensions": [".zip", ".tar", ".tar.gz", ".tgz"],
     "desc": "zip / tar / tar.gz；多个表格文件用 member 指定"},
    {"id": "gz", "label": _FORMAT_LABELS["gz"], "extensions": [".gz", ".gzip"],
     "desc": "如 expr.csv.gz，自动解压后按内层文件格式读取"},
]

_FORMAT_ALIASES = {
    "auto": "auto", "automatic": "auto", "detect": "auto",
    "csv": "csv", "comma": "csv", "commas": "csv", "comma-separated": "csv",
    "tsv": "tsv", "tab": "tsv", "tabs": "tsv", "tab-separated": "tsv",
    "txt": "txt", "text": "txt", "plain": "txt", "dat": "txt",
    "xlsx": "xlsx", "xlsm": "xlsx", "xls": "xlsx", "excel": "xlsx",
    "json": "json", "jsonl": "json", "ndjson": "json",
    "archive": "archive", "zip": "archive", "tar": "archive", "tgz": "archive",
    "tar.gz": "archive", "tar-gz": "archive",
    "gz": "gz", "gzip": "gz",
}

_ARCHIVE_SUFFIXES = (
    ".zip", ".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tbz", ".tbz2", ".tar.xz", ".txz",
)
_XLSX_SUFFIXES = (".xlsx", ".xlsm", ".xls")
_JSON_SUFFIXES = (".json", ".jsonl", ".ndjson")
_TEXT_SUFFIXES = (".txt", ".text", ".dat", ".matrix", ".counts", ".values", ".tsv2")

_ENCODINGS = ("utf-8-sig", "utf-8", "gbk", "gb18030", "latin-1")
_GROUP_KEYS = ("group", "group_name", "分组", "组别", "condition", "class", "type",
               "arm", "cohort", "treatment", "status", "label")
_SAMPLE_KEYS = ("sample", "sample_id", "sampleid", "samples", "sample_name",
                "id", "name", "样本", "样本名")

_HEAD_BYTES = 64 * 1024
_DEFAULT_MAX_ROWS = 200_000
_INSPECT_FULL_BYTES = 12 * 1024 * 1024
_INSPECT_ROWS = 5_000
_INSPECT_TOTAL_CAP = 2_000_000
_MAX_CELL_CHARS = 200
_MAX_DOWNLOAD_BYTES = 4 * 1024 * 1024 * 1024
_MAX_UNPACK_BYTES = 4 * 1024 * 1024 * 1024
_URL_TIMEOUT = 60
_UA = "dsh-plotlab/1.0 (plot workbench)"

_GZIP_MAGIC = b"\x1f\x8b"
_ZIP_MAGIC = b"PK\x03\x04"
_OLE_MAGIC = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"


# ---------------------------------------------------------------------------
# 格式清单
# ---------------------------------------------------------------------------


def formats() -> list:
    """返回支持的导入格式清单（UI 下拉框用，深拷贝避免外部改动）。"""
    return copy.deepcopy(FORMATS)


def _format_label(fmt: str) -> str:
    return _FORMAT_LABELS.get(str(fmt or ""), str(fmt or ""))


def _normalize_format(value: Any) -> str:
    """把 data_format 参数规整成内部 id；auto/空 → auto，未知值 bad_request。"""
    if value is None:
        return "auto"
    text = str(value).strip().lower().lstrip(".")
    if text in ("", "auto"):
        return "auto"
    fmt = _FORMAT_ALIASES.get(text)
    if fmt:
        return fmt
    raise C.bad_request(
        f"未知的 data_format：{value}；可选：{'、'.join(item['id'] for item in FORMATS)}"
    )


def _format_from_name(name: str) -> str | None:
    """按文件名推断表格格式（压缩包成员 / 目录扫描用）；不认识的返回 None。"""
    lower = str(name or "").lower()
    base = lower
    if base.endswith((".gz", ".gzip")):
        base = base[:-3] if base.endswith(".gz") else base[:-5]
    if base.endswith(_XLSX_SUFFIXES):
        return "xlsx"
    if base.endswith(_JSON_SUFFIXES):
        return "json"
    if base.endswith(".csv"):
        return "csv"
    if base.endswith((".tsv", ".tab")) or base.endswith(".tsv2"):
        return "tsv"
    if base.endswith(_TEXT_SUFFIXES):
        return "txt"
    return None


# ---------------------------------------------------------------------------
# 参数规整
# ---------------------------------------------------------------------------


def _norm_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return bool(value)
    text = str(value).strip().lower()
    if text in ("", "none", "null"):
        return default
    if text in ("1", "true", "yes", "y", "on"):
        return True
    if text in ("0", "false", "no", "n", "off"):
        return False
    return default


def _norm_int(value: Any, default: int | None = None) -> int | None:
    if value is None or value == "":
        return default
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return default


def _norm_kind(value: Any) -> str | None:
    text = str(value or "").strip().lower()
    return text if text in ("tidy", "matrix") else None


def _resolve_max_rows(value: Any) -> int:
    number = _norm_int(value, None)
    if number is None or number <= 0:
        return _DEFAULT_MAX_ROWS
    return number


def _norm_encoding(value: Any) -> str | None:
    """编码别名归一 + codecs 校验（未知编码 bad_request）。"""
    if value is None or str(value).strip() == "":
        return None
    text = str(value).strip().lower().replace("_", "-")
    aliases = {
        "utf8": "utf-8", "utf-8-sig": "utf-8-sig", "utf8-sig": "utf-8-sig",
        "gb2312": "gbk", "gb-2312": "gbk", "cp936": "gbk", "gbk": "gbk",
        "gb18030": "gb18030", "gb-18030": "gb18030",
        "latin1": "latin-1", "latin-1": "latin-1", "iso-8859-1": "latin-1",
        "utf-16": "utf-16", "utf16": "utf-16",
    }
    encoding = aliases.get(text, text)
    try:
        codecs.lookup(encoding)
    except LookupError:
        raise C.bad_request(
            f"未知编码：{value}；常用：utf-8、gbk、gb18030、latin-1"
        )
    return encoding


def _encoding_label(encoding: str | None) -> str:
    label = str(encoding or "").lower()
    return {
        "utf-8-sig": "UTF-8（含 BOM）",
        "utf-8": "UTF-8",
        "gbk": "GBK",
        "gb18030": "GB18030",
        "latin-1": "Latin-1",
        "utf-16": "UTF-16",
    }.get(label, str(encoding or "未知"))


def _encoding_candidates(explicit: str | None, detected: str | None) -> list[str]:
    out: list[str] = []
    for encoding in (explicit, detected):
        if encoding and encoding not in out:
            out.append(encoding)
    for encoding in _ENCODINGS:
        if encoding not in out:
            out.append(encoding)
    return out


def _decode_probe(data: bytes, encoding: str) -> str:
    """解码文件头；若正好截断在多字节字符中间，去掉末尾 1-4 字节再试。"""
    try:
        return data.decode(encoding)
    except UnicodeDecodeError:
        if len(data) > 8:
            for cut in (1, 2, 3, 4):
                try:
                    return data[:-cut].decode(encoding)
                except UnicodeDecodeError:
                    continue
        raise


def _detect_text_encoding(head: bytes) -> str:
    if not head:
        return "utf-8"
    if head.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig"
    if head.startswith(b"\xff\xfe") or head.startswith(b"\xfe\xff"):
        return "utf-16"
    for encoding in ("utf-8", "gbk", "gb18030", "latin-1"):
        try:
            _decode_probe(head, encoding)
            return encoding
        except (UnicodeDecodeError, LookupError):
            continue
    return "latin-1"


def _read_bytes_head(path: Path, size: int = _HEAD_BYTES) -> bytes:
    try:
        with open(path, "rb") as handle:
            return handle.read(size)
    except OSError as error:
        raise C.bad_request(f"无法读取文件：{path.name}（{error}）")


# ---------------------------------------------------------------------------
# 路径 / 命名小工具
# ---------------------------------------------------------------------------


def _looks_like_url(text: Any) -> bool:
    return bool(re.match(r"^(?:https?|ftp)://", str(text or "").strip(), re.I))


def _display_stem(name: str) -> str:
    """文件名去扩展名/压缩后缀：expr.csv.gz → expr。"""
    text = Path(str(name or "")).name
    changed = True
    while changed and text:
        changed = False
        lower = text.lower()
        for suffix in (".gz", ".gzip", ".bz2", ".xz", ".zip", ".tar", ".tgz",
                       ".csv", ".tsv", ".tab", ".txt", ".text", ".xlsx", ".xlsm",
                       ".xls", ".json", ".jsonl", ".ndjson", ".dat", ".matrix", ".counts"):
            if lower.endswith(suffix) and len(text) > len(suffix):
                text = text[: -len(suffix)]
                changed = True
                break
    return text.strip() or "dataset"


def _safe_filename(name: str) -> str:
    text = re.sub(r"[^\w.\-()\u4e00-\u9fff]+", "_", str(name or ""), flags=re.UNICODE)
    text = text.strip("._ ")
    return text[:120] or "download"


def _unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem, suffix = path.stem, path.suffix
    for index in range(1, 1000):
        candidate = path.with_name(f"{stem}-{index}{suffix}")
        if not candidate.exists():
            return candidate
    return path.with_name(f"{stem}-{len(str(path))}{suffix}")


def _normalize_rel(name: Any) -> str:
    text = str(name or "").replace("\\", "/").strip()
    while text.startswith("./"):
        text = text[2:]
    return text


def _safe_target(root: Path, relative: str) -> Path:
    """压缩包成员解压目标路径（拒绝 ../ 与绝对路径）。"""
    rel = _normalize_rel(relative)
    if not rel or rel.startswith("/"):
        raise C.bad_request(f"压缩包成员路径不合法：{relative}")
    parts = [part for part in rel.split("/") if part not in ("", ".")]
    if not parts or any(part == ".." for part in parts):
        raise C.bad_request(f"压缩包成员路径不合法：{relative}")
    root_resolved = root.resolve()
    target = root_resolved.joinpath(*parts)
    try:
        target = target.resolve()
    except OSError:
        pass
    if target != root_resolved and root_resolved not in target.parents:
        raise C.bad_request(f"压缩包成员路径越界：{relative}")
    return target


def _copy_stream(source, sink, limit: int = _MAX_UNPACK_BYTES) -> int:
    written = 0
    while True:
        chunk = source.read(1024 * 1024)
        if not chunk:
            break
        written += len(chunk)
        if written > limit:
            raise C.bad_request(f"解压后文件超过 {limit // (1024 ** 3)}GB 上限")
        sink.write(chunk)
    return written


# ---------------------------------------------------------------------------
# 工作目录 / URL 下载
# ---------------------------------------------------------------------------


def _session_dirs(workdir: Any) -> dict:
    """下载/解压临时目录：优先 worker 的 workdir，否则系统临时目录。"""
    base: Path | None = None
    if workdir not in (None, ""):
        try:
            base = Path(str(workdir)).expanduser()
            base.mkdir(parents=True, exist_ok=True)
            base = base / "plotlab-io"
            base.mkdir(parents=True, exist_ok=True)
        except OSError:
            base = None
    if base is None:
        base = Path(tempfile.mkdtemp(prefix="plotlab-io-"))
    dirs = {"base": base}
    for key in ("downloads", "tmp", "unpack"):
        folder = base / key
        try:
            folder.mkdir(parents=True, exist_ok=True)
        except OSError:
            folder = Path(tempfile.mkdtemp(prefix=f"plotlab-{key}-"))
        dirs[key] = folder
    return dirs


def _download_url(url: str, dirs: dict, notes: list[str]) -> tuple[Path, str]:
    """urllib 流式下载到本地（60s 超时，失败给可读中文错误）。"""
    text = str(url or "").strip()
    parsed = urllib.parse.urlparse(text)
    if parsed.scheme.lower() not in ("http", "https", "ftp"):
        raise C.bad_request(f"URL 只支持 http/https/ftp：{text}")
    filename = _safe_filename(Path(urllib.parse.unquote(parsed.path or "")).name or "download")
    request = urllib.request.Request(text, headers={"User-Agent": _UA})
    target: Path | None = None
    written = 0
    try:
        with urllib.request.urlopen(request, timeout=_URL_TIMEOUT) as response:
            disposition = response.headers.get("Content-Disposition") or ""
            match = re.search(r"filename\*?=(?:UTF-8''|\")?([^\";]+)", disposition, re.I)
            if match:
                filename = _safe_filename(urllib.parse.unquote(match.group(1).strip()))
            target = _unique_path(dirs["downloads"] / filename)
            with open(target, "wb") as sink:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    written += len(chunk)
                    if written > _MAX_DOWNLOAD_BYTES:
                        raise C.bad_request(
                            f"下载文件超过 {_MAX_DOWNLOAD_BYTES // (1024 ** 3)}GB 上限：{text}"
                        )
                    sink.write(chunk)
    except C.EngineError:
        if target is not None and target.exists():
            target.unlink(missing_ok=True)
        raise
    except urllib.error.HTTPError as error:
        raise C.bad_request(f"下载失败（HTTP {error.code}）：{text}")
    except urllib.error.URLError as error:
        reason = getattr(error, "reason", error)
        if isinstance(reason, (socket.timeout, TimeoutError)):
            raise C.bad_request(f"下载超时（{_URL_TIMEOUT} 秒）：{text}")
        raise C.bad_request(f"无法访问 URL（{reason}），请检查网络或地址：{text}")
    except (socket.timeout, TimeoutError):
        raise C.bad_request(f"下载超时（{_URL_TIMEOUT} 秒）：{text}")
    except OSError as error:
        raise C.bad_request(f"下载失败：{error}（{text}）")
    if written == 0:
        if target is not None and target.exists():
            target.unlink(missing_ok=True)
        raise C.bad_request(f"下载的内容为空：{text}")
    notes.append(f"已从 URL 下载：{text} → {target}")
    return target, filename


def _pick_directory_file(directory: Path, notes: list[str]) -> Path:
    """path 指到目录时：目录里唯一可识别的数据文件直接用，多个则 ambiguous。"""
    try:
        entries = sorted(
            (item for item in directory.iterdir()
             if item.is_file() and not item.name.startswith(".") and item.stat().st_size > 0),
            key=lambda item: item.name.lower(),
        )
    except OSError as error:
        raise C.bad_request(f"无法读取目录：{directory}（{error}）")
    recognized: list[tuple[Path, str]] = []
    for item in entries:
        fmt = _format_from_name(item.name)
        if fmt is None:
            lower = item.name.lower()
            if lower.endswith((".gz", ".gzip")):
                fmt = "gz"
            elif any(lower.endswith(suffix) for suffix in _ARCHIVE_SUFFIXES):
                fmt = "archive"
        if fmt:
            recognized.append((item, fmt))
    if len(recognized) == 1:
        notes.append(f"目录中自动选中唯一数据文件「{recognized[0][0].name}」")
        return recognized[0][0]
    if len(recognized) > 1:
        candidates = [
            {"name": item.name, "id": item.name, "label": item.name, "member": item.name,
             "kind": fmt, "format": fmt, "size": int(item.stat().st_size)}
            for item, fmt in recognized
        ]
        raise C.EngineError(
            "ambiguous",
            f"目录里有 {len(candidates)} 个可识别的数据文件，请指定具体文件：{directory}",
            candidates=candidates,
        )
    raise C.bad_request(f"目录里没有可识别的数据文件：{directory}")


def _resolve_input(path: Any, url: Any, name: Any, workdir: Any,
                   notes: list[str]) -> dict:
    """path/url → 本地绝对路径 + 展示名 + 会话临时目录。"""
    dirs = _session_dirs(workdir)
    name_arg = str(name).strip() if name not in (None, "") else ""
    if isinstance(url, str) and url.strip():
        target, downloaded_name = _download_url(url.strip(), dirs, notes)
        return {
            "path": target,
            "source": str(target.resolve()),
            "name": name_arg or _display_stem(downloaded_name),
            "url": url.strip(),
            "dirs": dirs,
            "downloaded": True,
        }
    if path is None or str(path).strip() == "":
        raise C.bad_request("缺少数据源：请提供 path 或 url")
    raw = str(path).strip()
    local = Path(raw).expanduser()
    try:
        local = local.resolve()
    except OSError:
        pass
    if not local.exists():
        raise C.not_found(f"路径不存在：{raw}")
    if local.is_dir():
        local = _pick_directory_file(local, notes)
    if not local.is_file():
        raise C.bad_request(f"不是普通文件：{local}")
    return {
        "path": local,
        "source": str(local),
        "name": name_arg or _display_stem(local.name),
        "url": None,
        "dirs": dirs,
        "downloaded": False,
    }


# ---------------------------------------------------------------------------
# 格式探测
# ---------------------------------------------------------------------------


def _zip_is_xlsx(path: Path) -> bool:
    try:
        with zipfile.ZipFile(path) as archive:
            names = set(archive.namelist())
    except (zipfile.BadZipFile, OSError):
        return False
    return "xl/workbook.xml" in names or "xl/sharedStrings.xml" in names


def _detect_format(path: Path, data_format: Any, names: list[Any] | None = None) -> str:
    """确定导入格式：显式 data_format 优先，否则扩展名 → 文件内容。"""
    forced = _normalize_format(data_format)
    if forced != "auto":
        return forced
    candidates = [str(item) for item in (names or []) if item not in (None, "")]
    candidates.append(path.name)
    for name in candidates:
        lower = name.lower()
        if any(lower.endswith(suffix) for suffix in _ARCHIVE_SUFFIXES):
            return "archive"
        if lower.endswith((".gz", ".gzip")):
            return "gz"
        if lower.endswith(_XLSX_SUFFIXES):
            return "xlsx"
        if lower.endswith(_JSON_SUFFIXES):
            return "json"
        if lower.endswith(".csv"):
            return "csv"
        if lower.endswith((".tsv", ".tab", ".tsv2")):
            return "tsv"
        if lower.endswith(_TEXT_SUFFIXES):
            return "txt"
    head = _read_bytes_head(path, _HEAD_BYTES)
    if not head.strip():
        raise C.bad_request(f"文件是空的：{path.name}")
    if head.startswith(_GZIP_MAGIC):
        return "gz"
    if head.startswith(_ZIP_MAGIC):
        return "xlsx" if _zip_is_xlsx(path) else "archive"
    if head.startswith(_OLE_MAGIC):
        return "xlsx"  # 旧版 .xls，读取时给明确的转换提示
    if b"\x00" in head:
        raise C.bad_request(
            f"无法识别文件格式（疑似二进制文件）：{path.name}；"
            "请用 data_format 指定 csv/tsv/txt/xlsx/json/archive/gz"
        )
    stripped = head.decode("utf-8-sig", errors="ignore").lstrip()
    if stripped.startswith("{") or stripped.startswith("["):
        return "json"
    try:
        text = _decode_probe(head, _detect_text_encoding(head))
    except UnicodeDecodeError:
        raise C.bad_request(f"无法识别文件格式：{path.name}；请用 data_format 显式指定")
    tabs = text.count("\t")
    commas = text.count(",")
    if tabs and tabs >= commas:
        return "tsv"
    if commas:
        return "csv"
    return "txt"


# ---------------------------------------------------------------------------
# 分隔符
# ---------------------------------------------------------------------------


def _normalize_delimiter(value: Any) -> str | None:
    if value is None or isinstance(value, (bool, int, float, np.integer, np.floating)):
        return None
    text = str(value)
    if text == "":
        return None
    low = text.strip().lower()
    if low in ("auto", "detect", "sniff"):
        return "auto"
    if low in ("\\t", "tab", "tabs", "tsv"):
        return "\t"
    if low in (",", "comma", "commas"):
        return ","
    if low in (";", "semicolon", "semi"):
        return ";"
    if low in ("|", "pipe", "pipes"):
        return "|"
    if low in ("space", "whitespace", "blank"):
        return "whitespace"
    if len(text) == 1:
        return text
    raise C.bad_request(f"delimiter 参数不合法：{value!r}（可用 , ; 制表符 | 或 auto）")


def _delimiter_label(delimiter: str | None) -> str:
    return {
        ",": "逗号",
        "\t": "制表符",
        ";": "分号",
        "|": "竖线",
        "whitespace": "空白",
    }.get(str(delimiter or ""), str(delimiter or "未识别"))


def _split_quoted(line: str, delimiter: str) -> list[str]:
    if delimiter == "whitespace":
        return line.split()
    try:
        return next(csv.reader([line], delimiter=delimiter, quotechar='"'))
    except (csv.Error, StopIteration):
        return line.split(delimiter)


def _delimiter_consistency(lines: list[str], delimiter: str) -> float:
    counts = []
    for line in lines:
        fields = _split_quoted(line, delimiter)
        counts.append(len(fields))
    if not counts:
        return 0.0
    mode = max(set(counts), key=counts.count)
    if mode < 2:
        return 0.0
    return counts.count(mode) / len(counts)


def _count_delimiter_guess(lines: list[str]) -> str | None:
    best: str | None = None
    best_score = 0.0
    for candidate in (",", "\t", ";", "|"):
        consistency = _delimiter_consistency(lines, candidate)
        if consistency < 0.7:
            continue
        counts = [len(_split_quoted(line, candidate)) for line in lines]
        mode = max(set(counts), key=counts.count)
        score = mode * consistency
        if score > best_score:
            best_score, best = score, candidate
    return best


def _sniff_delimiter(head_text: str, fmt: str, delimiter: Any,
                     notes: list[str]) -> str:
    explicit = _normalize_delimiter(delimiter)
    lines = [line for line in head_text.splitlines() if line.strip()][:60]
    if explicit and explicit != "auto":
        notes.append(f"分隔符（指定）：{_delimiter_label(explicit)}")
        return explicit
    if explicit is None:
        if fmt == "csv":
            notes.append("分隔符：逗号")
            return ","
        if fmt == "tsv":
            notes.append("分隔符：制表符")
            return "\t"
    sample = "\n".join(lines[:50])
    if sample:
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
            best = dialect.delimiter
            if _delimiter_consistency(lines, best) >= 0.7:
                notes.append(f"自动嗅探分隔符：{_delimiter_label(best)}")
                return best
        except csv.Error:
            pass
    guess = _count_delimiter_guess(lines)
    if guess:
        notes.append(f"自动嗅探分隔符：{_delimiter_label(guess)}")
        return guess
    notes.append("未检测到明显分隔符，按单列读取")
    return ","


def _single_column_hint(head_text: str, used_delimiter: str) -> str | None:
    """只解析出 1 列时，看看是否其实是别的分隔符。"""
    lines = [line for line in head_text.splitlines() if line.strip()][:30]
    if len(lines) < 2:
        return None
    for candidate in (",", "\t", ";", "|"):
        if candidate == used_delimiter:
            continue
        if _delimiter_consistency(lines, candidate) >= 0.8:
            return candidate
    return None


# ---------------------------------------------------------------------------
# 压缩包 / gzip
# ---------------------------------------------------------------------------


def _candidate_priority(name: str) -> int:
    base = Path(_normalize_rel(name)).name.lower()
    if any(token in base for token in ("readme", "license", "licence", "changelog",
                                       "description", "notes", "gpl")):
        return 1
    if any(token in base for token in ("matrix", "counts", "tpm", "fpkm", "expr",
                                       "expression", "data", "table", "values",
                                       "samples", "clinical", "metadata", "meta")):
        return -1
    return 0


def _list_archive(path: Path) -> list[dict]:
    """列出压缩包里的表格候选（不解压整包）。"""
    entries: list[dict] = []

    def add(name: str, size: int) -> None:
        rel = _normalize_rel(name)
        if not rel or rel.startswith("/") or any(part == ".." for part in rel.split("/")):
            return
        parts = rel.split("/")
        if any(part.startswith(".") for part in parts) or "__MACOSX" in parts:
            return
        fmt = _format_from_name(rel)
        if fmt is None or int(size) <= 0:
            return
        entries.append({
            "name": rel, "id": rel, "label": rel, "member": rel,
            "kind": fmt, "format": fmt, "size": int(size),
        })

    try:
        if zipfile.is_zipfile(path):
            with zipfile.ZipFile(path) as archive:
                for info in archive.infolist():
                    if info.is_dir():
                        continue
                    add(info.filename, info.file_size)
        elif tarfile.is_tarfile(path):
            with tarfile.open(path, "r:*") as archive:
                for member in archive.getmembers():
                    if not member.isfile():
                        continue
                    add(member.name, member.size)
        else:
            raise C.bad_request(f"不是有效的 zip / tar 压缩包：{path.name}")
    except C.EngineError:
        raise
    except (zipfile.BadZipFile, tarfile.TarError) as error:
        raise C.bad_request(f"压缩包损坏或格式无法识别（{path.name}）：{error}")
    entries.sort(key=lambda item: (_candidate_priority(item["name"]), item["name"].lower()))
    return entries


def _match_candidate(candidates: list[dict], member: Any) -> dict | None:
    text = _normalize_rel(member)
    if not text:
        return None
    base = Path(text).name
    for item in candidates:
        name = item["name"]
        if name == text or Path(name).name == base:
            return item
        if name.endswith(text) or text.endswith(name):
            return item
    return None


def _extract_member(path: Path, member: str, dirs: dict) -> Path:
    """把压缩包里的一个成员解压到临时目录，返回本地路径。"""
    rel = _normalize_rel(member)
    root = Path(tempfile.mkdtemp(prefix="unpack-", dir=dirs["unpack"]))
    target = _safe_target(root, rel)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        if zipfile.is_zipfile(path):
            with zipfile.ZipFile(path) as archive:
                info = next(
                    (item for item in archive.infolist()
                     if not item.is_dir() and _normalize_rel(item.filename) == rel),
                    None,
                )
                if info is None:
                    raise C.not_found(f"压缩包里没有文件：{member}")
                with archive.open(info) as source, open(target, "wb") as sink:
                    _copy_stream(source, sink)
        else:
            with tarfile.open(path, "r:*") as archive:
                member_obj = next(
                    (item for item in archive.getmembers()
                     if item.isfile() and _normalize_rel(item.name) == rel),
                    None,
                )
                if member_obj is None:
                    raise C.not_found(f"压缩包里没有文件：{member}")
                source = archive.extractfile(member_obj)
                if source is None:
                    raise C.bad_request(f"压缩包成员无法读取：{member}")
                with source, open(target, "wb") as sink:
                    _copy_stream(source, sink)
    except C.EngineError:
        raise
    except (zipfile.BadZipFile, tarfile.TarError, OSError) as error:
        raise C.bad_request(f"解压失败（{path.name} / {member}）：{error}")
    return target


def _gunzip(path: Path, dirs: dict) -> Path:
    """单文件 gzip 解压到临时目录，返回内层文件路径。"""
    inner_name = path.name
    lower = inner_name.lower()
    if lower.endswith(".gzip"):
        inner_name = inner_name[:-5]
    elif lower.endswith(".gz"):
        inner_name = inner_name[:-3]
    inner_name = Path(inner_name or "data").name or "data"
    out_dir = Path(tempfile.mkdtemp(prefix="gz-", dir=dirs["tmp"]))
    target = out_dir / inner_name
    try:
        with gzip.open(path, "rb") as source, open(target, "wb") as sink:
            _copy_stream(source, sink)
    except C.EngineError:
        raise
    except (gzip.BadGzipFile, OSError, EOFError) as error:
        raise C.bad_request(f"不是有效的 gzip 文件：{path.name}（{error}）")
    return target


# ---------------------------------------------------------------------------
# 通用读取调度
# ---------------------------------------------------------------------------


def _load_table(path: Path, fmt: str, options: dict, dirs: dict, notes: list[str],
                warnings: list[str], mode: str, depth: int = 0) -> dict:
    """按格式读成 DataFrame；处理 gz / 压缩包嵌套。返回 dict 携带格式元数据。"""
    if depth > 3:
        raise C.bad_request("压缩/解压嵌套层数过多，无法导入")

    # 扩展名说是文本，但文件本身是 gzip（如把 data.csv 改名成 data.txt.gz 又选错格式）
    if fmt not in ("archive", "gz"):
        head = _read_bytes_head(path, 4)
        if head.startswith(_GZIP_MAGIC):
            fmt = "gz"

    if fmt == "gz":
        inner = _gunzip(path, dirs)
        inner_fmt = _detect_format(inner, None)
        notes.append(f"gzip 已解压：{path.name} → {inner.name}")
        loaded = _load_table(inner, inner_fmt, options, dirs, notes, warnings,
                             mode, depth + 1)
        loaded["gzip"] = path.name
        return loaded

    if fmt == "archive":
        candidates = _list_archive(path)
        if not candidates:
            raise C.bad_request(
                f"压缩包里没有找到可识别的表格文件（支持 .csv/.tsv/.txt/.xlsx/.json）：{path.name}"
            )
        member = options.get("member")
        selected: dict | None = None
        if member not in (None, ""):
            selected = _match_candidate(candidates, member)
            if selected is None:
                names = "、".join(item["name"] for item in candidates[:10])
                raise C.not_found(f"压缩包里没有「{member}」；可用文件：{names}")
        elif len(candidates) == 1:
            selected = candidates[0]
        elif mode == "inspect":
            selected = candidates[0]
            notes.append(
                f"压缩包内共 {len(candidates)} 个候选表格，inspect 只解析第一个"
                f"「{selected['name']}」；导入时用 member 指定"
            )
        else:
            raise C.EngineError(
                "ambiguous",
                f"压缩包里找到 {len(candidates)} 个候选表格，请用 member 指定要导入哪一个",
                candidates=candidates,
            )
        inner_path = _extract_member(path, selected["name"], dirs)
        inner_fmt = selected.get("format") or _format_from_name(selected["name"]) or "txt"
        loaded = _load_table(inner_path, inner_fmt, options, dirs, notes, warnings,
                             mode, depth + 1)
        loaded["candidates"] = candidates
        loaded["archive"] = path.name
        loaded["member"] = selected["name"]
        notes.append(
            f"来自压缩包 {path.name}：{selected['name']}（共 {len(candidates)} 个候选）"
        )
        if loaded.get("sheet") and len(candidates) == 1:
            notes.append(f"压缩包内 {selected['name']} 读取 sheet「{loaded['sheet']}」")
        return loaded

    if fmt == "xlsx":
        loaded = _read_excel(path, options, mode, notes, warnings)
        loaded["candidates"] = loaded.get("candidates") or []
        return loaded
    if fmt == "json":
        return _read_json_file(path, options, mode, notes, warnings)
    if fmt in ("csv", "tsv", "txt"):
        return _read_delimited(path, fmt, options, mode, notes, warnings)

    raise C.unsupported(
        f"不支持的导入格式：{fmt}；可选：{'、'.join(item['id'] for item in FORMATS)}"
    )


# ---------------------------------------------------------------------------
# 文本（csv / tsv / txt）
# ---------------------------------------------------------------------------


def _read_csv_frame(path: Path, encoding: str, delimiter: str, header: bool,
                    nrows: int | None) -> tuple[pd.DataFrame, bool]:
    """读一次分隔文本；返回 (DataFrame, 是否跳过了坏行)。"""
    kwargs: dict[str, Any] = {
        "encoding": encoding,
        "skip_blank_lines": True,
        "header": 0 if header else None,
    }
    if nrows is not None:
        kwargs["nrows"] = max(1, int(nrows))
    if delimiter == "whitespace":
        kwargs["sep"] = r"\s+"
        kwargs["engine"] = "python"
    else:
        kwargs["sep"] = delimiter
        kwargs["low_memory"] = False

    def finalize(frame: pd.DataFrame, skipped: bool) -> tuple[pd.DataFrame, bool]:
        if not header:  # header=False 时用 col_1..col_n，而非 pandas 的整数列名
            frame.columns = [f"col_{index + 1}" for index in range(frame.shape[1])]
        return frame, skipped

    try:
        return finalize(pd.read_csv(str(path), **kwargs), False)
    except C.EngineError:
        raise
    except UnicodeDecodeError:
        raise  # 交给上层换编码重试
    except LookupError:
        raise
    except pd.errors.EmptyDataError:
        raise C.bad_request(f"文件里没有数据行：{path.name}")
    except pd.errors.ParserError as first_error:
        retry = dict(kwargs)
        retry["on_bad_lines"] = "skip"
        if delimiter == "whitespace":
            retry["engine"] = "python"
        else:
            retry.pop("low_memory", None)
        try:
            return finalize(pd.read_csv(str(path), **retry), True)
        except C.EngineError:
            raise
        except UnicodeDecodeError:
            raise
        except pd.errors.EmptyDataError:
            raise C.bad_request(f"文件里没有数据行：{path.name}")
        except pd.errors.ParserError:
            raise C.bad_request(
                f"无法解析表格（{path.name}）：{first_error}；"
                "请确认分隔符（可用 delimiter 指定）或 data_format"
            )
        except Exception as error:  # noqa: BLE001
            raise C.bad_request(f"无法解析表格（{path.name}）：{error}")
    except ValueError as error:
        message = str(error)
        if "lines" in message.lower() or "header" in message.lower():
            raise C.bad_request(f"文件里没有数据行：{path.name}（{message}）")
        raise C.bad_request(f"无法解析表格（{path.name}）：{message}")
    except Exception as error:  # noqa: BLE001 - 兜底成可读错误
        raise C.bad_request(f"无法解析表格（{path.name}）：{error}")


def _count_text_rows(path: Path, encoding: str, header: bool) -> int:
    """逐行统计非空行数（大文件 inspect 用；引号内换行会被多算，已随 notes 说明）。"""
    count = 0
    try:
        with open(path, "r", encoding=encoding, errors="replace", newline="") as handle:
            for line in handle:
                if line.strip():
                    count += 1
    except OSError:
        return -1
    return max(0, count - (1 if header else 0))


def _normalize_columns(frame: pd.DataFrame, notes: list[str] | None = None) -> pd.DataFrame:
    """列名去空白、空表头补名、重复列加后缀；全空的 Unnamed 列丢掉。"""
    frame = frame.copy()
    keep: list[int] = []
    labels: list[str] = []
    for index, column in enumerate(frame.columns):
        name = str(column).strip()
        unnamed = re.fullmatch(r"unnamed:\s*\d+", name.lower()) is not None
        if unnamed:
            try:
                empty = bool(frame.iloc[:, index].isna().all())
            except (IndexError, ValueError):
                empty = False
            if empty:
                if notes is not None:
                    notes.append("已忽略 1 个空表头列")
                continue
            name = f"col_{index + 1}"
        if not name:
            name = f"col_{index + 1}"
        keep.append(index)
        labels.append(name)
    if len(keep) != frame.shape[1]:
        frame = frame.iloc[:, keep]
    seen: dict[str, int] = {}
    deduped: list[str] = []
    for label in labels:
        if label in seen:
            seen[label] += 1
            deduped.append(f"{label}.{seen[label]}")
        else:
            seen[label] = 1
            deduped.append(label)
    frame.columns = deduped
    return frame


def _drop_all_nan_rows(frame: pd.DataFrame, warnings: list[str]) -> pd.DataFrame:
    if frame.shape[1] == 0:
        return frame
    mask = frame.isna().all(axis=1)
    count = int(mask.sum())
    if count:
        frame = frame.loc[~mask].reset_index(drop=True)
        warnings.append(f"已丢弃 {count} 行完全空白的行")
    return frame


def _apply_index_col(frame: pd.DataFrame, index_col: Any, notes: list[str]) -> pd.DataFrame:
    """index_col 指定的列放回首列（dataset 不保留 DataFrame index 语义）。"""
    if index_col is None or index_col == "":
        return frame
    value = index_col
    if isinstance(value, str) and value.strip().isdigit():
        value = int(value.strip())
    column_name: str | None = None
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        index = int(value)
        if index < 0:
            index += frame.shape[1]
        if not 0 <= index < frame.shape[1]:
            raise C.bad_request(
                f"index_col={index_col} 超出列范围（共 {frame.shape[1]} 列）"
            )
        column_name = str(frame.columns[index])
    else:
        text = str(value).strip()
        for column in frame.columns:
            if str(column) == text:
                column_name = str(column)
                break
        if column_name is None:
            raise C.bad_request(
                f"index_col={index_col} 不在表头里；可用列："
                f"{'、'.join(str(item) for item in frame.columns[:10])}"
            )
    if column_name not in list(map(str, frame.columns)):
        raise C.bad_request(f"index_col={index_col} 不在表头里")
    frame = frame.set_index(column_name).reset_index()
    notes.append(f"index_col 已设为「{column_name}」，仍保留为首列")
    return frame


def _replace_inf(frame: pd.DataFrame, warnings: list[str]) -> pd.DataFrame:
    numeric_columns = [
        column for column in frame.columns
        if pd.api.types.is_numeric_dtype(frame[column])
    ]
    if not numeric_columns:
        return frame
    total = 0
    for column in numeric_columns:
        values = np.asarray(frame[column], dtype=float)
        total += int(np.isinf(values).sum())
    if total:
        frame = frame.copy()
        frame[numeric_columns] = frame[numeric_columns].replace([np.inf, -np.inf], np.nan)
        warnings.append(f"有 {total} 个 ±inf 值已置为缺失")
    return frame


def _read_delimited(path: Path, fmt: str, options: dict, mode: str,
                    notes: list[str], warnings: list[str]) -> dict:
    try:
        size = int(path.stat().st_size)
    except OSError:
        size = 0
    head_raw = _read_bytes_head(path, _HEAD_BYTES)
    if not head_raw.strip():
        raise C.bad_request(f"文件是空的：{path.name}")
    explicit = _norm_encoding(options.get("encoding"))
    detected = _detect_text_encoding(head_raw)
    encodings = _encoding_candidates(explicit, detected)
    head_text = None
    for encoding in encodings:
        try:
            head_text = _decode_probe(head_raw, encoding)
            break
        except (UnicodeDecodeError, LookupError):
            continue
    if head_text is None:
        raise C.bad_request(
            f"无法解码文件（尝试了 {'、'.join(encodings)}）：{path.name}；"
            "请用 encoding 指定正确编码"
        )
    delimiter = _sniff_delimiter(head_text, fmt, options.get("delimiter"), notes)
    header = _norm_bool(options.get("header"), True)
    max_rows = _resolve_max_rows(options.get("max_rows"))
    if mode == "open":
        nrows: int | None = max_rows + 1
    elif size > _INSPECT_FULL_BYTES:
        nrows = _INSPECT_ROWS
    else:
        nrows = None
    frame: pd.DataFrame | None = None
    used_encoding: str | None = None
    skipped = False
    for encoding in encodings:
        try:
            frame, skipped = _read_csv_frame(path, encoding, delimiter, header, nrows)
            used_encoding = encoding
            break
        except UnicodeDecodeError:
            continue
        except LookupError:
            continue
    if frame is None or used_encoding is None:
        raise C.bad_request(
            f"无法解码文件（尝试了 {'、'.join(encodings)}）：{path.name}；"
            "请用 encoding 指定正确编码"
        )
    if skipped:
        warnings.append(
            f"{path.name} 有无法解析的行，已跳过（请检查分隔符/引号，或用 delimiter 指定）"
        )
    if explicit and used_encoding != explicit:
        notes.append(
            f"指定编码 {_encoding_label(explicit)} 读取失败，"
            f"实际按 {_encoding_label(used_encoding)} 读取"
        )
    notes.append(f"编码：{_encoding_label(used_encoding)}")
    frame = _normalize_columns(frame, notes)
    if frame.shape[1] == 0:
        raise C.bad_request(f"文件里没有数据列：{path.name}")
    if frame.shape[1] == 1:
        hint = _single_column_hint(head_text, delimiter)
        if hint:
            warnings.append(
                f"只解析出 1 列，可能分隔符不对（疑似{_delimiter_label(hint)}）；"
                "可用 delimiter 参数指定"
            )
    frame = _drop_all_nan_rows(frame, warnings)
    frame = _apply_index_col(frame, options.get("index_col"), notes)
    frame = _replace_inf(frame, warnings)
    if frame.shape[0] == 0:
        raise C.bad_request(f"文件里没有数据行：{path.name}（只有表头或全为空行）")
    if mode == "open" and frame.shape[0] > max_rows:
        frame = frame.iloc[:max_rows].copy()
        warnings.append(
            f"文件行数超过 max_rows={max_rows}，只读取前 {max_rows} 行（可用 max_rows 调整）"
        )
    total_rows = int(frame.shape[0])
    if mode == "inspect" and nrows is not None:
        counted = _count_text_rows(path, used_encoding, header)
        if counted >= 0:
            total_rows = int(counted)
            notes.append(
                f"大文件：总行数逐行统计为 {counted}；列摘要基于前 {int(frame.shape[0])} 行"
            )
    return {
        "df": frame,
        "format": fmt,
        "encoding": used_encoding,
        "delimiter": delimiter,
        "total_rows": total_rows,
        "candidates": [],
        "sheet": None,
        "member": None,
    }


# ---------------------------------------------------------------------------
# Excel
# ---------------------------------------------------------------------------


def _open_workbook(path: Path):
    try:
        import openpyxl
    except ImportError:
        raise C.unsupported("缺少 openpyxl，无法读取 Excel；请另存为 CSV/TSV 后导入")
    try:
        return openpyxl.load_workbook(str(path), read_only=True, data_only=True)
    except Exception as error:  # noqa: BLE001 - openpyxl 错误类型较杂
        raise C.bad_request(f"无法读取 Excel 文件（{path.name}）：{error}")


def _sheet_candidates(sheet_names: list[str]) -> list[dict]:
    return [
        {"name": name, "id": name, "label": name, "kind": "sheet", "format": "xlsx"}
        for name in sheet_names
    ]


def _resolve_sheet_name(sheet_names: list[str], requested: Any) -> str:
    if isinstance(requested, (bool,)):
        raise C.bad_request(f"sheet 参数不合法：{requested!r}")
    if isinstance(requested, (int, np.integer)):
        index = int(requested)
        if index < 0:
            index += len(sheet_names)
        if not 0 <= index < len(sheet_names):
            raise C.not_found(
                f"sheet 序号 {requested} 越界：共 {len(sheet_names)} 个 sheet"
                f"（{'、'.join(sheet_names[:10])}）"
            )
        return sheet_names[index]
    text = str(requested).strip()
    if text in sheet_names:
        return text
    for name in sheet_names:
        if name.casefold() == text.casefold():
            return name
    if text.isdigit():
        return _resolve_sheet_name(sheet_names, int(text))
    raise C.not_found(
        f"工作表不存在：{requested}；可用：{'、'.join(sheet_names[:10])}"
    )


def _blank_cell(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    return False


def _sheet_values(ws, store_limit: int | None, stop_early: bool,
                  max_scan: int | None = None) -> tuple[list[tuple], int | None, bool]:
    """读一个 sheet 的非空行；stop_early=True 时读够 store_limit 就停（返回值 total=None）。"""
    rows: list[tuple] = []
    total = 0
    for row in ws.iter_rows(values_only=True):
        if row is None:
            continue
        if all(_blank_cell(cell) for cell in row):
            continue
        total += 1
        if store_limit is None or len(rows) < store_limit:
            rows.append(tuple(row))
        if stop_early and store_limit is not None and total >= store_limit:
            return rows, None, True
        if max_scan is not None and total >= max_scan:
            return rows, total, True
    return rows, total, False


def _header_label(value: Any, index: int) -> str:
    if value is None:
        return f"col_{index + 1}"
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if not math.isnan(number) and number.is_integer():
            return str(int(number))
        return str(value)
    text = str(value).strip()
    return text or f"col_{index + 1}"


def _cell_or_nan(value: Any) -> Any:
    if value is None:
        return np.nan
    if isinstance(value, str) and value.strip() == "":
        return np.nan
    return value


def _frame_from_grid(rows: list[tuple], header: bool) -> pd.DataFrame:
    if not rows:
        raise C.bad_request("文件里没有数据行")
    width = max(len(row) for row in rows)
    if width <= 0:
        raise C.bad_request("文件里没有数据列")
    if header:
        head_row = list(rows[0]) + [None] * (width - len(rows[0]))
        columns = [_header_label(value, index) for index, value in enumerate(head_row[:width])]
        body = rows[1:]
    else:
        columns = [f"col_{index + 1}" for index in range(width)]
        body = rows
    data = [
        [_cell_or_nan(cell) for cell in (list(row) + [None] * (width - len(row)))]
        for row in body
    ]
    return pd.DataFrame(data, columns=columns)


def _read_excel(path: Path, options: dict, mode: str, notes: list[str],
                warnings: list[str]) -> dict:
    head = _read_bytes_head(path, 8)
    if head.startswith(_OLE_MAGIC):
        raise C.bad_request(
            f"这是旧版 Excel（.xls）文件，openpyxl 无法读取：{path.name}；"
            "请另存为 .xlsx 或 CSV 后再导入"
        )
    workbook = _open_workbook(path)
    try:
        sheet_names = [str(name) for name in workbook.sheetnames]
        if not sheet_names:
            raise C.bad_request(f"Excel 文件里没有工作表：{path.name}")
        candidates = _sheet_candidates(sheet_names)
        header = _norm_bool(options.get("header"), True)
        max_rows = _resolve_max_rows(options.get("max_rows"))
        requested = options.get("sheet")
        target = _resolve_sheet_name(sheet_names, requested) if requested not in (None, "") else None
        if mode == "open":
            store_limit: int | None = max_rows + 2
            stop_early = True
            max_scan: int | None = None
        else:
            store_limit = _INSPECT_ROWS + 1
            stop_early = False
            max_scan = _INSPECT_TOTAL_CAP
        order = [target] if target else sheet_names
        chosen: str | None = None
        rows: list[tuple] = []
        total: int | None = None
        scan_truncated = False
        skipped_sheets: list[str] = []
        for position, name in enumerate(order):
            try:
                rows, total, scan_truncated = _sheet_values(
                    workbook[name], store_limit, stop_early, max_scan
                )
            except C.EngineError:
                raise
            except Exception as error:  # noqa: BLE001 - openpyxl 错误类型较杂
                raise C.bad_request(f"读取 Excel 工作表「{name}」失败：{error}")
            data_rows = len(rows) - (1 if header else 0) if rows else 0
            if data_rows <= 0:
                skipped_sheets.append(name)
                if target is None and position + 1 < len(order):
                    continue
                if target is not None:
                    break
                rows = []
                break
            chosen = name
            break
        if not rows or chosen is None:
            if target:
                raise C.bad_request(f"工作表「{target}」里没有数据行：{path.name}")
            raise C.bad_request(f"Excel 文件里没有数据行：{path.name}")
        if skipped_sheets:
            notes.append(f"已跳过 {len(skipped_sheets)} 个没有数据行的 sheet")
        if target is None:
            if len(sheet_names) > 1:
                notes.append(
                    f"Excel 共 {len(sheet_names)} 个 sheet（{'、'.join(sheet_names[:6])}），"
                    f"读取「{chosen}」；可用 sheet 参数指定"
                )
            else:
                notes.append(f"Excel 读取 sheet「{chosen}」")
            if chosen != sheet_names[0]:
                notes.append(f"首个 sheet 为空，已改读「{chosen}」")
        else:
            notes.append(f"Excel 读取 sheet「{chosen}」")
        try:
            frame = _frame_from_grid(rows, header)
            frame = _normalize_columns(frame, notes)
            if frame.shape[1] == 0:
                raise C.bad_request(f"Excel 工作表「{chosen}」里没有数据列：{path.name}")
            frame = _drop_all_nan_rows(frame, warnings)
            frame = _apply_index_col(frame, options.get("index_col"), notes)
            frame = _replace_inf(frame, warnings)
            if frame.shape[0] == 0:
                raise C.bad_request(f"Excel 工作表「{chosen}」里没有数据行：{path.name}")
            if mode == "open" and frame.shape[0] > max_rows:
                frame = frame.iloc[:max_rows].copy()
                warnings.append(
                    f"Excel 行数超过 max_rows={max_rows}，只读取前 {max_rows} 行（可用 max_rows 调整）"
                )
        except C.EngineError:
            raise
        except Exception as error:  # noqa: BLE001 - 兜底成可读错误
            raise C.bad_request(f"解析 Excel 工作表「{chosen}」失败：{error}")
        total_rows = int(frame.shape[0])
        if mode == "inspect":
            if scan_truncated:
                warnings.append(
                    f"Excel 行数超过统计上限 {_INSPECT_TOTAL_CAP}，n_rows 为下限"
                )
            if total is not None:
                total_rows = max(0, int(total) - (1 if header else 0))
        return {
            "df": frame,
            "format": "xlsx",
            "encoding": None,
            "delimiter": None,
            "total_rows": total_rows,
            "candidates": candidates,
            "sheet": chosen,
            "member": None,
        }
    finally:
        try:
            workbook.close()
        except Exception:  # noqa: BLE001
            pass


# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------


def _try_ndjson(text: str) -> list | None:
    objects: list = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            objects.append(json.loads(stripped))
        except json.JSONDecodeError:
            return None
    return objects or None


def _jsonify_cells(frame: pd.DataFrame, notes: list[str]) -> pd.DataFrame:
    """嵌套 dict/list 值转成 JSON 字符串保留。"""
    converted_total = 0
    converted_columns = 0
    for column in list(frame.columns):
        series = frame[column]
        if series.dtype != object:
            continue
        mask = series.map(lambda value: isinstance(value, (dict, list, tuple)))
        count = int(mask.sum())
        if not count:
            continue
        frame[column] = series.map(
            lambda value: json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
            if isinstance(value, (dict, list, tuple)) else value
        )
        converted_total += count
        converted_columns += 1
    if converted_total:
        notes.append(
            f"JSON 嵌套字段已转成 JSON 字符串：{converted_total} 个单元格"
            f"（{converted_columns} 列）"
        )
    return frame


def _frame_from_json(data: Any, source_name: str, notes: list[str],
                     warnings: list[str]) -> pd.DataFrame:
    """JSON 三种结构 → DataFrame：records 列表 / dict of lists / {"data":[...]}。"""
    columns_hint: list[str] | None = None
    if isinstance(data, dict) and isinstance(data.get("data"), list):
        payload = data["data"]
        hint = data.get("columns") or data.get("header")
        if isinstance(hint, list) and all(not isinstance(item, (dict, list)) for item in hint):
            columns_hint = [str(item) for item in hint]
        data = payload
        notes.append("使用 JSON 中的 data 字段作为数据表")
    if isinstance(data, list):
        if not data:
            raise C.bad_request(f"JSON 数组为空，文件里没有数据行：{source_name}")
        if all(isinstance(item, dict) for item in data):
            frame = pd.DataFrame(data)
        elif all(isinstance(item, (list, tuple)) for item in data):
            width = max(len(item) for item in data)
            if width == 0:
                raise C.bad_request(f"JSON 里没有数据列：{source_name}")
            columns = columns_hint or [f"col_{index + 1}" for index in range(width)]
            if len(columns) < width:
                columns = columns + [f"col_{index + 1}" for index in range(len(columns), width)]
            padded = [list(item) + [None] * (width - len(item)) for item in data]
            frame = pd.DataFrame(padded, columns=columns[:width])
        elif all(not isinstance(item, (dict, list, tuple)) for item in data):
            frame = pd.DataFrame({"value": data})
        else:
            raise C.bad_request(
                "JSON 数组元素类型混杂，无法转成表格（需要 records 列表或二维数组）"
            )
    elif isinstance(data, dict):
        if not data:
            raise C.bad_request(f"JSON 对象为空，文件里没有数据行：{source_name}")
        values = list(data.values())
        if all(isinstance(item, (list, tuple)) for item in values):
            lengths = {len(item) for item in values}
            if len(lengths) > 1:
                warnings.append("JSON dict of lists 各列长度不一致，已用缺失值补齐")
            rows_count = max(lengths) if lengths else 0
            if rows_count == 0:
                raise C.bad_request(f"JSON 里没有数据行：{source_name}")
            padded = {
                str(key): list(value) + [None] * (rows_count - len(value))
                for key, value in data.items()
            }
            frame = pd.DataFrame(padded)
        elif all(isinstance(item, dict) for item in values):
            frame = pd.DataFrame.from_dict(data, orient="index")
            frame.insert(0, "id", [str(key) for key in frame.index])
            frame = frame.reset_index(drop=True)
            notes.append("JSON 对象按 key→记录 读取（key 作为 id 列）")
        elif all(not isinstance(item, (dict, list, tuple)) for item in values):
            frame = pd.DataFrame([{str(key): value for key, value in data.items()}])
            notes.append("JSON 是单个对象，按 1 行读取")
        else:
            raise C.bad_request(
                "JSON 对象结构无法转成表格（需要 dict of lists 或 records）"
            )
    else:
        raise C.bad_request(
            f"JSON 内容不是表格（需要 records 列表、dict of lists 或 data 数组）：{source_name}"
        )
    if frame.shape[1] == 0:
        raise C.bad_request(f"JSON 里没有数据列：{source_name}")
    return _jsonify_cells(frame, notes)


def _read_json_file(path: Path, options: dict, mode: str, notes: list[str],
                    warnings: list[str]) -> dict:
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise C.bad_request(f"无法读取文件：{path.name}（{error}）")
    if not raw.strip():
        raise C.bad_request(f"文件是空的：{path.name}")
    explicit = _norm_encoding(options.get("encoding"))
    detected = _detect_text_encoding(raw[:_HEAD_BYTES])
    text: str | None = None
    used_encoding: str | None = None
    for encoding in _encoding_candidates(explicit, detected):
        try:
            text = raw.decode(encoding)
            used_encoding = encoding
            break
        except UnicodeDecodeError:
            continue
        except LookupError:
            continue
    if text is None or used_encoding is None:
        raise C.bad_request(f"无法解码 JSON 文件：{path.name}；请用 encoding 指定编码")
    notes.append(f"编码：{_encoding_label(used_encoding)}")
    data: Any = None
    try:
        data = json.loads(text)
    except json.JSONDecodeError as error:
        data = _try_ndjson(text)
        if data is None:
            raise C.bad_request(
                f"JSON 解析失败（第 {error.lineno} 行第 {error.colno} 列）：{error.msg}；"
                '支持 records 列表、dict of lists 或 {"data": [...]}'
            )
        notes.append("检测到 JSON Lines（每行一个 JSON），已按行解析")
    try:
        frame = _frame_from_json(data, path.name, notes, warnings)
    except C.EngineError:
        raise
    except Exception as error:  # noqa: BLE001 - 兜底成可读错误
        raise C.bad_request(f"JSON 转成表格失败（{path.name}）：{error}")
    frame = _normalize_columns(frame, notes)
    if frame.shape[1] == 0:
        raise C.bad_request(f"文件里没有数据列：{path.name}")
    frame = _drop_all_nan_rows(frame, warnings)
    frame = _replace_inf(frame, warnings)
    if frame.shape[0] == 0:
        raise C.bad_request(f"文件里没有数据行：{path.name}")
    max_rows = _resolve_max_rows(options.get("max_rows"))
    total_rows = int(frame.shape[0])
    if mode == "open" and total_rows > max_rows:
        frame = frame.iloc[:max_rows].copy()
        warnings.append(
            f"文件行数超过 max_rows={max_rows}，只读取前 {max_rows} 行（可用 max_rows 调整）"
        )
    return {
        "df": frame,
        "format": "json",
        "encoding": used_encoding,
        "delimiter": None,
        "total_rows": total_rows if mode == "inspect" else int(frame.shape[0]),
        "candidates": [],
        "sheet": None,
        "member": None,
    }


# ---------------------------------------------------------------------------
# 转置 / 矩阵规整
# ---------------------------------------------------------------------------


def _as_sample_name(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if math.isnan(number):
            return ""
        if number.is_integer():
            return str(int(number))
        return str(number)
    return str(value).strip()


def _norm_sample_key(value: Any) -> str:
    return _as_sample_name(value).casefold()


def _clean_sample_names(samples: list[Any], warnings: list[str]) -> list[str]:
    out: list[str] = []
    seen: dict[str, int] = {}
    renamed = 0
    for index, raw in enumerate(samples):
        name = _as_sample_name(raw) or f"sample_{index + 1}"
        if name in seen:
            seen[name] += 1
            name = f"{name}_{seen[name]}"
            renamed += 1
        else:
            seen[name] = 1
        out.append(name)
    if renamed:
        warnings.append(f"有 {renamed} 个重复/空样本名，已自动加后缀")
    return out


def _transpose_frame(frame: pd.DataFrame, notes: list[str], warnings: list[str],
                     mode: str = "open") -> pd.DataFrame:
    """transpose=true：样本×特征宽表 → matrix（首列特征 id，其余列样本）。"""
    if frame.shape[1] < 2:
        raise C.bad_request("只有 1 列数据，无法按 transpose=true 转置（需要首列 + 至少 1 个数据列）")
    first_column = str(frame.columns[0])
    samples = _clean_sample_names(frame.iloc[:, 0].tolist(), warnings)
    features = [str(column).strip() for column in frame.columns[1:]]
    data = frame.iloc[:, 1:]
    numeric = data.apply(P.to_numeric)
    lost = int((data.notna() & numeric.isna()).sum().sum())
    if lost and mode == "open":
        warnings.append(f"转置时 {lost} 个非数值单元格已置为缺失")
    matrix = numeric.to_numpy(dtype=float)
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise C.bad_request("转置失败：数据为空")
    result = pd.DataFrame(matrix.T, columns=samples)
    feature_name = "feature"
    sample_set = set(samples)
    while feature_name in sample_set:
        feature_name += "_id"
    result.insert(0, feature_name, features)
    notes.append(
        f"已按 transpose=true 转置：{len(features)} 特征 × {len(samples)} 样本"
        f"（原首列「{first_column}」作样本名，原表头作特征名）"
    )
    return result


def _group_name_like(name: Any) -> bool:
    """列名是否像分组列（group/condition/class/分组…）。"""
    text = str(name or "").strip().lower()
    return any(hint in text for hint in P.GROUP_NAME_HINTS)


_STRONG_FEATURE_HINTS = (
    "gene", "probe", "symbol", "feature", "metabolite", "protein", "peptide",
    "transcript", "mrna", "ensg", "ensp", "基因", "探针", "特征", "代谢", "蛋白", "转录",
)


def _feature_name_strong(name: Any) -> bool:
    """首列列名是否明确是特征 ID（gene/probe/symbol…，比 id/name 更强）。"""
    text = str(name or "").lower()
    return any(hint in text for hint in _STRONG_FEATURE_HINTS)


def _feature_name_like(name: Any) -> bool:
    """首列列名是否像特征 ID（gene/probe/symbol/id/name/中文基因等）。"""
    text = str(name or "").lower()
    if any(hint in text for hint in P.FEATURE_HINTS) or text.endswith(("id", "name")):
        return True
    return any(hint in text for hint in ("基因", "特征", "探针", "代谢物", "蛋白", "转录本", "位点"))


def _matrix_column_profile(frame: pd.DataFrame, feature: str) -> tuple[list[str], list[str]]:
    """把除首列外的列分成「数值型」与「非数值型」；非数值列不强行数值化。"""
    numeric_like: list[str] = []
    non_numeric: list[str] = []
    for column in frame.columns:
        if str(column) == str(feature):
            continue
        series = frame[column]
        if pd.api.types.is_numeric_dtype(series) or P.infer_dtype(series) == "numeric":
            numeric_like.append(str(column))
        else:
            non_numeric.append(str(column))
    return numeric_like, non_numeric


def _coerce_matrix_numeric(frame: pd.DataFrame, feature: str, warnings: list[str],
                           notes: list[str], mode: str = "open") -> list[str]:
    """matrix 数据的样本列必须是数值（pl_prep 的宽表视图只认 numeric dtype）。

    只转换「看起来是数值」的列（千分位 / <0.001 / 空白等装饰可解析），
    文本/类别列原样保留，避免把分组列误清空。返回被判为非数值的列名。
    """
    numeric_like, non_numeric = _matrix_column_profile(frame, feature)
    converted = 0
    lost = 0
    for column in numeric_like:
        series = frame[column]
        if pd.api.types.is_numeric_dtype(series):
            continue
        numeric = P.to_numeric(series)
        lost += int((series.notna() & numeric.isna()).sum())
        frame[column] = numeric.astype(float)
        converted += 1
    if converted and mode == "open":
        if lost:
            warnings.append(f"矩阵数值化：{lost} 个非数值单元格已置为缺失")
        else:
            notes.append(f"矩阵数值化：{converted} 个文本数值列已转为数值")
    return non_numeric


def _stringify_feature(frame: pd.DataFrame, feature: str, notes: list[str],
                       mode: str = "open") -> None:
    """数值型特征 id 转文本，避免 12345 显示成 12345.0。"""
    series = frame[feature]
    if not pd.api.types.is_numeric_dtype(series):
        return

    def fmt(value: Any) -> str:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return ""
        try:
            number = float(value)
        except (TypeError, ValueError):
            return str(value)
        if number.is_integer():
            return str(int(number))
        return str(value)

    frame[feature] = series.map(fmt)
    if mode == "open":
        notes.append(f"特征列「{feature}」为数值 ID，已转文本保留原样")


def _prepare_frame(frame: pd.DataFrame, kind_hint: str | None, feature_col: Any,
                   transpose: Any, notes: list[str], warnings: list[str],
                   mode: str = "open") -> tuple[pd.DataFrame, str, str | None]:
    """转置 + tidy/matrix 判定 + matrix 数值规整；返回 (frame, kind, feature_col)。"""
    transposed = _norm_bool(transpose, False)
    if transposed:
        frame = _transpose_frame(frame, notes, warnings, mode=mode)
        if not kind_hint:
            kind_hint = "matrix"
    columns = [str(column) for column in frame.columns]
    feature: str | None = None
    feature_locked = False
    if feature_col not in (None, ""):
        text = str(feature_col).strip()
        if text in columns:
            feature = text
            feature_locked = True
            if kind_hint is None:
                kind_hint = "matrix"
            elif kind_hint == "tidy":
                notes.append("数据按 tidy 处理，feature_col 不生效")
        else:
            warnings.append(
                f"feature_col=「{feature_col}」不在数据列里，已忽略"
                f"（可用列：{'、'.join(columns[:8])}）"
            )
    detected_kind, detected_feature = P.infer_kind_and_feature_col(frame, kind_hint)
    feature = feature or detected_feature or (columns[0] if columns else None)
    if detected_kind == "matrix" and feature:
        numeric_like, non_numeric = _matrix_column_profile(frame, feature)
        user_locked = bool(kind_hint) or transposed or feature_locked
        group_columns = [column for column in non_numeric if _group_name_like(column)]
        plausible = True
        reason = ""
        if non_numeric and _feature_name_strong(feature):
            pass  # 明确是基因/探针/特征矩阵：非数值列当作注释列，不推翻判定
        elif group_columns:
            plausible = False
            reason = f"检测到分组列「{'、'.join(group_columns[:3])}」"
        elif non_numeric and not _feature_name_like(feature):
            ratio = len(non_numeric) / max(1, len(numeric_like) + len(non_numeric))
            if ratio >= 0.25:
                plausible = False
                reason = f"首列「{feature}」不像特征 ID 且有 {len(non_numeric)} 个文本列"
        if not plausible and not user_locked:
            notes.append(f"{reason}，按 tidy 数据导入（如确实是矩阵，请指定 kind=matrix）")
            detected_kind = "tidy"
            feature = None
        else:
            skipped = _coerce_matrix_numeric(frame, feature, warnings, notes, mode=mode)
            if skipped and mode == "open":
                warnings.append(
                    f"matrix 数据里有 {len(skipped)} 个非数值列"
                    f"（{'、'.join(skipped[:5])}）不属于样本列，宽表视图会忽略；"
                    "如果是 tidy 数据请用 kind=tidy 导入"
                )
            _stringify_feature(frame, feature, notes, mode=mode)
    return frame, detected_kind, feature


# ---------------------------------------------------------------------------
# 长文本单元格
# ---------------------------------------------------------------------------


def _truncate_long_cells(frame: pd.DataFrame, limit: int = _MAX_CELL_CHARS) -> int:
    total = 0
    for column in list(frame.columns):
        series = frame[column]
        if series.dtype != object:
            continue
        mask = series.map(lambda value: isinstance(value, str) and len(value) > limit)
        count = int(mask.sum())
        if not count:
            continue
        frame[column] = series.map(
            lambda value: value[:limit] if isinstance(value, str) and len(value) > limit else value
        )
        total += count
    return total


def _count_long_cells(frame: pd.DataFrame, limit: int = _MAX_CELL_CHARS) -> int:
    total = 0
    for column in list(frame.columns):
        series = frame[column]
        if series.dtype != object:
            continue
        total += int(series.map(lambda value: isinstance(value, str) and len(value) > limit).sum())
    return total


# ---------------------------------------------------------------------------
# 样本表
# ---------------------------------------------------------------------------


def _sample_info_from_frame(frame: pd.DataFrame, notes: list[str],
                            warnings: list[str]) -> pd.DataFrame | None:
    """样本表 → 标准 sample_info（sample 列 + 其余列；有分组线索时补 group 列）。"""
    if frame is None or frame.empty:
        warnings.append("样本表为空，已忽略")
        return None
    columns = [str(column) for column in frame.columns]
    sample_key: str | None = None
    for column in columns:
        if column.strip().casefold() in _SAMPLE_KEYS:
            sample_key = column
            break
    if sample_key is None:
        sample_key = columns[0]
        notes.append(f"样本表没有 sample 列，使用首列「{sample_key}」作为样本名")
    extras = [column for column in columns if column != sample_key]
    out = pd.DataFrame({"sample": frame[sample_key].map(_as_sample_name).to_numpy()})
    extra_names: list[str] = []
    for extra in extras:
        name = str(extra).strip() or f"col_{len(extra_names) + 2}"
        while name in out.columns:
            name = f"{name}_2"
        out[name] = frame[extra].to_numpy()
        extra_names.append(name)
    empty_mask = out["sample"].map(lambda value: str(value).strip() == "")
    if int(empty_mask.sum()):
        out = out.loc[~empty_mask].reset_index(drop=True)
        warnings.append("样本表里有空样本名，已跳过这些行")
    if out.empty:
        warnings.append("样本表没有有效样本行，已忽略")
        return None
    duplicated = out["sample"].duplicated()
    if int(duplicated.sum()):
        out = out.loc[~duplicated].reset_index(drop=True)
        warnings.append("样本表里有重复样本名，只保留第一次出现的行")
    lower_names = {str(column).casefold(): str(column) for column in out.columns}
    has_group = "group" in lower_names
    if not has_group:
        group_column = None
        for name in out.columns:
            if str(name).casefold() in _GROUP_KEYS:
                group_column = str(name)
                break
        if group_column is None and extra_names:
            second = out[extra_names[0]]
            if (not pd.api.types.is_numeric_dtype(second)
                    or int(second.nunique(dropna=True)) <= 20):
                group_column = extra_names[0]
        if group_column is not None:
            out.insert(1, "group", out[group_column].to_numpy())
            notes.append(f"样本表第二列「{group_column}」按分组列处理（同时保留原列）")
    notes.append(
        f"样本表：{len(out)} 行，列 "
        f"{'、'.join(str(column) for column in out.columns[:6])}"
    )
    return out


def _read_sample_sheet(value: Any, dirs: dict) -> tuple[pd.DataFrame | None, list[str], list[str]]:
    """读样本表（CSV/TSV/XLSX/JSON，可 URL），返回 (sample_info, notes, warnings)。"""
    if value is None:
        return None, [], []
    if isinstance(value, pd.DataFrame):
        notes: list[str] = []
        warnings: list[str] = []
        info = _sample_info_from_frame(value.copy(), notes, warnings)
        return info, notes, warnings
    if isinstance(value, (dict, list)):
        notes = []
        warnings = []
        try:
            frame = _frame_from_json(value, "sample_sheet", notes, warnings)
        except C.EngineError as error:
            raise C.bad_request(f"样本表解析失败：{error.message}")
        info = _sample_info_from_frame(frame, notes, warnings)
        return info, notes, warnings
    text = str(value).strip()
    if not text:
        return None, [], []
    notes = []
    warnings = []
    if _looks_like_url(text):
        local, downloaded_name = _download_url(text, dirs, notes)
        source_name = downloaded_name
    else:
        local = Path(text).expanduser()
        try:
            local = local.resolve()
        except OSError:
            pass
        if not local.exists():
            raise C.bad_request(f"样本表不存在：{text}")
        if not local.is_file():
            raise C.bad_request(f"样本表不是普通文件：{text}")
        source_name = local.name
    fmt = _detect_format(local, None)
    if fmt == "archive":
        raise C.bad_request("样本表不支持压缩包：请提供 CSV/TSV/XLSX/JSON 文件")
    if fmt == "gz":
        local = _gunzip(local, dirs)
        fmt = _detect_format(local, None)
        notes.append(f"样本表 gzip 已解压：{local.name}")
    options = {"header": True, "max_rows": _DEFAULT_MAX_ROWS}
    try:
        if fmt == "xlsx":
            loaded = _read_excel(local, options, "open", notes, warnings)
        elif fmt == "json":
            loaded = _read_json_file(local, options, "open", notes, warnings)
        elif fmt in ("csv", "tsv", "txt"):
            loaded = _read_delimited(local, fmt, options, "open", notes, warnings)
        else:
            raise C.bad_request(f"样本表格式不支持：{source_name}")
    except C.EngineError as error:
        raise C.bad_request(f"样本表读取失败（{source_name}）：{error.message}")
    info = _sample_info_from_frame(loaded["df"], notes, warnings)
    return info, notes, warnings


def _align_sample_info(info: pd.DataFrame, samples: list[str],
                       warnings: list[str]) -> pd.DataFrame:
    """按数据里的样本列顺序重排样本表；缺的样本补缺失行并提示。"""
    if "sample" not in info.columns:
        warnings.append("样本表没有 sample 列，无法按样本名匹配")
        return info
    keys = info["sample"].map(_norm_sample_key)
    lookup: dict[str, int] = {}
    for index, key in enumerate(keys):
        if key and key not in lookup:
            lookup[key] = index
    rows: list[dict] = []
    missing: list[str] = []
    for sample in samples:
        key = _norm_sample_key(sample)
        if key in lookup:
            row = info.iloc[lookup[key]].to_dict()
            row["sample"] = str(sample)
            rows.append(row)
        else:
            missing.append(str(sample))
            row = {column: np.nan for column in info.columns}
            row["sample"] = str(sample)
            rows.append(row)
    sample_keys = {_norm_sample_key(sample) for sample in samples}
    unmatched = [key for key in keys.tolist() if key and key not in sample_keys]
    if missing:
        warnings.append(
            f"数据里有 {len(missing)} 个样本在样本表中没有记录："
            f"{'、'.join(missing[:5])}" + ("…" if len(missing) > 5 else "")
        )
    if unmatched:
        warnings.append(
            f"样本表里有 {len(unmatched)} 个样本在数据里找不到："
            f"{'、'.join(unmatched[:5])}" + ("…" if len(unmatched) > 5 else "")
        )
    return pd.DataFrame(rows, columns=list(info.columns))


# ---------------------------------------------------------------------------
# inspect
# ---------------------------------------------------------------------------


def inspect_source(path: Any = None, url: Any = None, name: Any = None,
                   data_format: Any = None, delimiter: Any = None,
                   encoding: Any = None, sheet: Any = None, header: Any = True,
                   max_rows: Any = None, **kwargs: Any) -> dict:
    """探测数据源：格式、行列数、tidy/matrix、列摘要、候选（sheet/压缩包成员）。

    不构建 dataset、不写全局状态；大文件只抽样做列摘要，行数逐行统计。
    """
    member = kwargs.get("member")
    workdir = kwargs.get("workdir")
    index_col = kwargs.get("index_col")
    kind_hint = _norm_kind(kwargs.get("kind"))
    feature_col = kwargs.get("feature_col")
    transpose = kwargs.get("transpose", False)
    notes: list[str] = []
    warnings: list[str] = []
    resolved = _resolve_input(path, url, name, workdir, notes)
    fmt = _detect_format(
        resolved["path"], data_format,
        names=[resolved["name"], resolved["path"].name],
    )
    options = {
        "encoding": encoding,
        "delimiter": delimiter,
        "header": header,
        "sheet": sheet,
        "index_col": index_col,
        "max_rows": max_rows,
        "member": member,
    }
    loaded = _load_table(resolved["path"], fmt, options, resolved["dirs"],
                         notes, warnings, mode="inspect")
    frame: pd.DataFrame = loaded["df"]
    total_rows = loaded.get("total_rows")
    if total_rows is None:
        total_rows = int(frame.shape[0])
    sample = frame.head(_INSPECT_ROWS) if frame.shape[0] > _INSPECT_ROWS else frame
    prepared, detected_kind, detected_feature = _prepare_frame(
        sample, kind_hint, feature_col, transpose, notes, warnings, mode="inspect"
    )
    transposed = _norm_bool(transpose, False)
    if transposed:
        # 转置后：行 = 特征（原表头），列 = 样本（原数据行）
        report_rows = int(prepared.shape[0])
        report_cols = (
            int(prepared.shape[1]) if frame.shape[0] <= _INSPECT_ROWS
            else max(int(total_rows), int(prepared.shape[1]))
        )
        notes.append("行列数按转置后的结构统计")
    else:
        report_rows = int(total_rows)
        report_cols = int(frame.shape[1])
        if prepared.shape[1] != frame.shape[1]:
            notes.append("列结构在探测阶段发生变化，请以导入结果为准")
    columns = P.column_summaries(prepared)
    long_count = _count_long_cells(prepared)
    if long_count:
        warnings.append(
            f"样例中有 {long_count} 个单元格文本超过 {_MAX_CELL_CHARS} 字符，导入时会截断"
        )
    if sample.shape[0] < total_rows and not transposed:
        notes.append(f"列摘要基于前 {int(sample.shape[0])} 行；总行数 {int(total_rows)}")
    notes.insert(0, f"格式：{_format_label(str(loaded.get('format') or fmt))}")
    size = 0
    try:
        size = int(resolved["path"].stat().st_size)
    except OSError:
        size = 0
    return {
        "format": str(loaded.get("format") or fmt),
        "n_rows": report_rows,
        "n_cols": report_cols,
        "kind": detected_kind,
        "columns": C.jsonable(columns),
        "candidates": C.jsonable(loaded.get("candidates") or []),
        "notes": C.jsonable(notes),
        "warnings": C.jsonable(warnings),
        "size": size,
        "name": resolved["name"],
        "encoding": loaded.get("encoding"),
        "delimiter": loaded.get("delimiter"),
        "sheet": loaded.get("sheet"),
        "feature_col": detected_feature if detected_kind == "matrix" else None,
    }


# ---------------------------------------------------------------------------
# open
# ---------------------------------------------------------------------------


def open_source(path: Any = None, url: Any = None, name: Any = None,
                data_format: Any = None, delimiter: Any = None,
                encoding: Any = None, sheet: Any = None, header: Any = True,
                index_col: Any = None, kind: Any = None, feature_col: Any = None,
                transpose: Any = False, sample_sheet: Any = None,
                max_rows: Any = None, workdir: Any = None, **kwargs: Any) -> dict:
    """读取数据源并生成 dataset dict（pl_state.build_dataset），返回导入元数据。

    返回 {"dataset", "format", "notes", "warnings", "candidates"}。
    matrix 数据会把样本表按样本名挂到 dataset["sample_info"]。
    """
    member = kwargs.get("member")
    kind_hint = _norm_kind(kind)
    notes: list[str] = []
    warnings: list[str] = []
    resolved = _resolve_input(path, url, name, workdir, notes)
    fmt = _detect_format(
        resolved["path"], data_format,
        names=[resolved["name"], resolved["path"].name],
    )
    options = {
        "encoding": encoding,
        "delimiter": delimiter,
        "header": header,
        "sheet": sheet,
        "index_col": index_col,
        "max_rows": max_rows,
        "member": member,
    }
    loaded = _load_table(resolved["path"], fmt, options, resolved["dirs"],
                         notes, warnings, mode="open")
    frame: pd.DataFrame = loaded["df"]
    if frame is None or frame.shape[0] == 0:
        raise C.bad_request("文件里没有数据行")
    frame, detected_kind, detected_feature = _prepare_frame(
        frame, kind_hint, feature_col, transpose, notes, warnings, mode="open"
    )
    if frame.shape[0] == 0:
        raise C.bad_request("文件里没有数据行")
    truncated_cells = _truncate_long_cells(frame)
    if truncated_cells:
        warnings.append(
            f"有 {truncated_cells} 个单元格文本超过 {_MAX_CELL_CHARS} 字符，已截断"
        )
    # 样本表：先读出并处理，再随 build_dataset 一起挂上
    sample_info, sample_notes, sample_warnings = _read_sample_sheet(
        sample_sheet, resolved["dirs"]
    )
    notes.extend(sample_notes)
    warnings.extend(sample_warnings)
    info_has_group = False
    if sample_info is not None and not sample_info.empty:
        info_has_group = (
            "group" in sample_info.columns
            and bool(sample_info["group"].notna().any())
        )
        if detected_kind == "matrix":
            samples = [
                str(column) for column in frame.columns
                if str(column) != str(detected_feature)
            ]
            sample_info = _align_sample_info(sample_info, samples, warnings)
            notes.append(
                f"已挂载样本表：{len(sample_info)} 行"
                f"（列：{'、'.join(str(column) for column in sample_info.columns[:6])}）"
            )
        else:
            notes.append(
                f"已挂载样本表：{len(sample_info)} 行"
                "（tidy 数据只保存在 sample_info，不并入数据表）"
            )
    if sample_info is not None and isinstance(sample_info, pd.DataFrame) and sample_info.empty:
        sample_info = None

    dataset_name = (
        str(name).strip() if name not in (None, "") and str(name).strip()
        else resolved["name"]
    )
    notes.insert(0, f"格式：{_format_label(str(loaded.get('format') or fmt))}")
    import pl_state as S  # 懒加载：避免 io ↔ state 顶层循环

    dataset = S.build_dataset(
        frame,
        dataset_name,
        kind=detected_kind,
        source=resolved["source"],
        feature_col=detected_feature,
        sample_info=sample_info,
        notes=notes,
    )
    dataset["warnings"] = list(dataset.get("warnings") or []) + warnings
    # matrix：角色指向 sample_info 的 group 列（与内置表达矩阵示例一致）
    if detected_kind == "matrix":
        roles = {key: None for key in P.ROLE_KEYS}
        roles["group"] = "group" if info_has_group else None
        dataset["roles"] = roles
        if info_has_group and isinstance(sample_info, pd.DataFrame):
            levels = [
                str(value) for value in pd.unique(sample_info["group"].dropna().astype(str))
            ][:60]
            dataset["group_col"] = "group"
            dataset["groups"] = levels
        else:
            dataset["group_col"] = None
            dataset["groups"] = None
    # 给 pl_codegen / 页面复用的导入元数据（不改动 DESIGN §2 的既有键）
    dataset["import"] = C.jsonable({
        "format": str(loaded.get("format") or fmt),
        "encoding": loaded.get("encoding"),
        "delimiter": loaded.get("delimiter"),
        "sheet": loaded.get("sheet"),
        "header": _norm_bool(header, True),
        "index_col": str(index_col) if index_col not in (None, "") else None,
        "member": loaded.get("member"),
        "archive": loaded.get("archive"),
        "gzip": loaded.get("gzip"),
        "transpose": _norm_bool(transpose, False),
        "sample_sheet": (
            str(sample_sheet) if isinstance(sample_sheet, str)
            else ("inline" if sample_sheet is not None else None)
        ),
        "url": resolved.get("url"),
        "source": resolved["source"],
        "name": dataset_name,
        "max_rows": _resolve_max_rows(max_rows),
        "kind": detected_kind,
    })
    if loaded.get("sheet"):
        dataset["sheet"] = str(loaded["sheet"])  # pl_codegen 读取用
    return {
        "dataset": dataset,
        "format": str(loaded.get("format") or fmt),
        "notes": notes,
        "warnings": warnings,
        "candidates": loaded.get("candidates") or [],
    }
