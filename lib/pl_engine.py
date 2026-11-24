#!/usr/bin/env python3
"""PlotLab Python worker（长驻进程，JSON-lines over stdio）。

启动：python3 -u lib/pl_engine.py --workdir <abs_dir>   （cwd 建议 lib/）
协议：
  请求  {"id":"1","cmd":"ping","args":{}}
  成功  {"id":"1","ok":true,"result":{...}}
  失败  {"id":"1","ok":false,"error":{"code","message","trace"?}}
  进度  {"id":"1","event":"progress","message":"...","pct":42}

stdout 只允许协议行；所有日志走 stderr。
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import traceback

LIB_DIR = os.path.dirname(os.path.abspath(__file__))
if LIB_DIR not in sys.path:
    sys.path.insert(0, LIB_DIR)


def _configure_stdio() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass


def emit(payload: dict) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
    sys.stdout.flush()


def log(message: str) -> None:
    try:
        sys.stderr.write(str(message) + "\n")
        sys.stderr.flush()
    except Exception:  # noqa: BLE001
        pass


def main() -> int:
    parser = argparse.ArgumentParser(description="PlotLab worker")
    parser.add_argument("--workdir", required=True, help="工作目录（图/上传/导出）")
    args = parser.parse_args()

    _configure_stdio()
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.makedirs(args.workdir, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", os.path.join(args.workdir, "mpl-cache"))
    os.environ.setdefault("MPLAMBIENT", "1")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

    try:
        import matplotlib
        matplotlib.use("Agg")
    except Exception as error:  # noqa: BLE001
        log(f"[plotlab] matplotlib 不可用：{error}")

    import pl_common as C

    try:
        from pl_registry import WorkerState
        state = WorkerState(args.workdir)
    except Exception as error:  # noqa: BLE001
        emit({"id": "boot", "ok": False, "error": {
            "code": "internal",
            "message": f"引擎初始化失败：{error}",
            "trace": traceback.format_exc()[-4000:],
        }})
        return 1

    def progress(message: str, pct: float | None = None) -> None:
        callback = getattr(state, "progress_cb", None)
        if callable(callback):
            try:
                callback(str(message), pct)
            except Exception:  # noqa: BLE001
                pass

    state.progress_cb = None  # 每条请求处理时设置

    def graceful_exit(signum, frame):  # noqa: ARG001
        log(f"[plotlab] 收到信号 {signum}，退出")
        try:
            sys.stdout.flush()
        except Exception:  # noqa: BLE001
            pass
        os._exit(0)

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, graceful_exit)
        except Exception:  # noqa: BLE001
            pass

    log(f"[plotlab] worker 启动 workdir={args.workdir}")
    for line in sys.stdin:
        text = str(line).strip()
        if not text:
            continue
        try:
            message = json.loads(text)
        except json.JSONDecodeError as error:
            log(f"[plotlab] 非 JSON 请求：{error}")
            emit({"id": None, "ok": False, "error": {"code": "bad_request", "message": f"请求 JSON 解析失败：{error}"}})
            continue
        if not isinstance(message, dict):
            continue
        request_id = message.get("id")
        command = str(message.get("cmd") or "")
        payload = message.get("args") or {}
        if not isinstance(payload, dict):
            payload = {}
        if command == "shutdown":
            emit({"id": request_id, "ok": True, "result": {"bye": True}})
            break

        def _progress_event(text_value: str, pct_value: float | None = None, _rid=request_id) -> None:
            try:
                emit({"id": _rid, "event": "progress", "message": str(text_value),
                      **({"pct": float(pct_value)} if pct_value is not None else {})})
            except Exception:  # noqa: BLE001
                pass

        state.progress_cb = _progress_event
        try:
            result = state.handle(command, payload)
            emit({"id": request_id, "ok": True, "result": C.jsonable(result)})
        except C.EngineError as error:
            emit({"id": request_id, "ok": False, "error": {
                "code": error.code, "message": error.message,
                **({"details": C.jsonable(error.details)} if error.details else {}),
            }})
        except (BrokenPipeError, KeyboardInterrupt):
            log("[plotlab] 管道断开，退出")
            break
        except MemoryError:
            emit({"id": request_id, "ok": False, "error": {
                "code": "internal", "message": "内存不足：请先过滤/抽样数据再操作",
            }})
        except SystemExit:
            break
        except Exception as error:  # noqa: BLE001 - 任何异常都要回一条协议错误
            trace = traceback.format_exc()
            log(f"[plotlab] {command} 失败：{error}\n{trace}")
            emit({"id": request_id, "ok": False, "error": {
                "code": "internal",
                "message": f"{type(error).__name__}: {error}",
                "trace": trace[-4000:],
            }})
        finally:
            state.progress_cb = None
    log("[plotlab] worker 退出")
    return 0


if __name__ == "__main__":
    sys.exit(main())
