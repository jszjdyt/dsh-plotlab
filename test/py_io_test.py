#!/usr/bin/env python3
"""pl_io 自测：CSV/TSV/GBK/XLSX/JSON/matrix/sample_sheet/gzip/zip/空文件/transpose。

离线、确定性、秒级；逐项 print PASS，全部通过后 print ALL PASS 并 exit 0。
运行：PYTHONPATH=lib python3 test/py_io_test.py
"""
from __future__ import annotations

import gzip
import json
import os
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lib"))

import pandas as pd  # noqa: E402

import pl_io as IO  # noqa: E402
from pl_common import EngineError  # noqa: E402


def check(label: str, fn) -> None:
    fn()
    print(f"PASS {label}")


def expect_error(code: str, fn, label: str = "") -> EngineError:
    try:
        fn()
    except EngineError as exc:
        assert exc.code == code, f"{label}: 期望错误码 {code}，实际 {exc.code}（{exc.message}）"
        return exc
    raise AssertionError(f"{label}: 期望抛出 EngineError({code})，但没有抛错")


def columns_of(payload: dict) -> dict[str, dict]:
    return {str(item["key"]): item for item in payload["columns"]}


def main() -> int:
    tempdir = Path(tempfile.mkdtemp(prefix="plotlab-io-test-"))
    try:
        _run(tempdir)
    finally:
        shutil.rmtree(tempdir, ignore_errors=True)
    print("ALL PASS")
    return 0


def _run(root: Path) -> None:
    # ---------------------------------------------------------------- 1. tidy CSV
    def test_tidy_csv() -> None:
        path = root / "tidy.csv"
        path.write_text(
            "样品,分组,年龄,表达量\n"
            'A,Control,45,"1,234.5"\n'
            "B,Treatment,52,<0.001\n"
            "C,Control,,67\n"
            'D,Treatment,61,"2,000"\n',
            encoding="utf-8",
        )
        info = IO.inspect_source(path=str(path))
        assert info["format"] == "csv", info
        assert info["n_rows"] == 4 and info["n_cols"] == 4, info
        assert info["kind"] == "tidy", info
        assert info["size"] == path.stat().st_size
        cols = columns_of(info)
        assert list(cols) == ["样品", "分组", "年龄", "表达量"], list(cols)
        assert cols["年龄"]["dtype"] == "numeric", cols["年龄"]
        assert cols["年龄"]["missing"] == 1, cols["年龄"]
        assert cols["表达量"]["dtype"] == "numeric", cols["表达量"]
        out = IO.open_source(path=str(path))
        dataset = out["dataset"]
        assert out["format"] == "csv", out
        assert dataset["df"].shape == (4, 4), dataset["df"].shape
        assert list(dataset["df"].columns) == ["样品", "分组", "年龄", "表达量"]
        assert dataset["name"] == "tidy", dataset["name"]
        assert dataset["source"] == str(path.resolve()), dataset["source"]
        assert dataset["kind"] == "tidy", dataset["kind"]
        assert dataset["warnings"] == [], dataset["warnings"]
        truncated = IO.open_source(path=str(path), max_rows=2)["dataset"]
        assert truncated["df"].shape == (2, 4), truncated["df"].shape
        assert any("max_rows" in warning for warning in truncated["warnings"])

    check("1 tidy CSV（中文列名/千分位/空值/<0.001）", test_tidy_csv)

    # ---------------------------------------------------------------- 2. TSV
    def test_tsv() -> None:
        path = root / "expr.tsv"
        path.write_text(
            "group\tcontrol\ttreated\n"
            "A\t1\t2\n"
            "B\t3\t4\n"
            "C\t5\t6\n",
            encoding="utf-8",
        )
        info = IO.inspect_source(path=str(path))
        assert info["format"] == "tsv", info
        assert info["n_rows"] == 3 and info["n_cols"] == 3, info
        assert info["delimiter"] == "\t", info
        out = IO.open_source(path=str(path))
        assert out["dataset"]["df"].shape == (3, 3), out["dataset"]["df"].shape
        assert list(out["dataset"]["df"].columns) == ["group", "control", "treated"]

    check("2 TSV（制表符分隔）", test_tsv)

    # ---------------------------------------------------------------- 3. GBK
    def test_gbk() -> None:
        path = root / "gbk.csv"
        text = "基因,分组,表达量\nIL6,对照,1.2\nTNF,处理,3.4\n"
        path.write_bytes(text.encode("gbk"))
        info = IO.inspect_source(path=str(path))
        cols = columns_of(info)
        assert "基因" in cols and "分组" in cols, list(cols)
        assert any("GBK" in note for note in info["notes"]), info["notes"]
        out = IO.open_source(path=str(path))
        frame = out["dataset"]["df"]
        assert list(frame.columns) == ["基因", "分组", "表达量"], list(frame.columns)
        assert not frame.astype(str).apply(lambda s: s.str.contains("�")).any().any()

    check("3 GBK 编码 CSV 不乱码", test_gbk)

    # ---------------------------------------------------------------- 4. XLSX
    def test_xlsx() -> None:
        from openpyxl import Workbook

        path = root / "multi.xlsx"
        workbook = Workbook()
        sheet1 = workbook.active
        sheet1.title = "Sheet1"
        sheet1.append(["sample", "value"])
        for index in range(3):
            sheet1.append([f"S{index + 1}", index + 1.5])
        sheet2 = workbook.create_sheet("数据2")
        sheet2.append(["x", "y"])
        for index in range(2):
            sheet2.append([index, index * 10])
        workbook.save(path)

        info = IO.inspect_source(path=str(path))
        assert info["format"] == "xlsx", info
        names = [item["name"] for item in info["candidates"]]
        assert names == ["Sheet1", "数据2"], names
        assert info["n_rows"] == 3 and info["n_cols"] == 2, info
        assert info["sheet"] == "Sheet1", info
        out = IO.open_source(path=str(path), sheet="数据2")
        frame = out["dataset"]["df"]
        assert list(frame.columns) == ["x", "y"], list(frame.columns)
        assert frame.shape == (2, 2), frame.shape
        assert out["dataset"]["sheet"] == "数据2"
        assert out["dataset"]["import"]["sheet"] == "数据2"
        default = IO.open_source(path=str(path))["dataset"]
        assert list(default["df"].columns) == ["sample", "value"], list(default["df"].columns)
        # 指定不存在的 sheet → not_found
        expect_error("not_found", lambda: IO.open_source(path=str(path), sheet="不存在"),
                     "xlsx 不存在的 sheet")

    check("4 XLSX 多 sheet（candidates / sheet 指定）", test_xlsx)

    # ---------------------------------------------------------------- 5. JSON
    def test_json() -> None:
        path = root / "records.json"
        payload = [
            {"sample": "S1", "group": "A", "value": 1.5, "meta": {"batch": 1}},
            {"sample": "S2", "group": "B", "value": 2.5, "meta": {"batch": 2}},
            {"sample": "S3", "group": "A", "value": 3.5, "meta": {"batch": 1}},
        ]
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        info = IO.inspect_source(path=str(path))
        assert info["format"] == "json", info
        assert info["n_rows"] == 3 and info["n_cols"] == 4, info
        dataset = IO.open_source(path=str(path))["dataset"]
        frame = dataset["df"]
        assert frame.shape == (3, 4), frame.shape
        assert isinstance(frame.loc[0, "meta"], str) and "batch" in frame.loc[0, "meta"]
        # {"data": [...]} 结构
        wrapped = root / "wrapped.json"
        wrapped.write_text(
            json.dumps({"data": [{"a": 1, "b": 2}, {"a": 3, "b": 4}]}), encoding="utf-8"
        )
        wrapped_dataset = IO.open_source(path=str(wrapped))["dataset"]
        assert wrapped_dataset["df"].shape == (2, 2), wrapped_dataset["df"].shape
        # dict of lists
        dict_lists = root / "dict_lists.json"
        dict_lists.write_text(
            json.dumps({"a": [1, 2, 3], "b": ["x", "y", "z"]}), encoding="utf-8"
        )
        dict_dataset = IO.open_source(path=str(dict_lists))["dataset"]
        assert dict_dataset["df"].shape == (3, 2), dict_dataset["df"].shape

    check("5 JSON（records / data 包装 / dict of lists）", test_json)

    # ---------------------------------------------------------------- 6. matrix
    def test_matrix() -> None:
        path = root / "matrix.csv"
        path.write_text(
            "gene,s1,s2,s3,s4\n"
            "IL6,1.2,2.3,3.4,4.5\n"
            "TNF,2.0,2.1,2.2,2.3\n"
            "CD8A,5.1,4.2,3.3,2.4\n"
            "ACTB,9.9,9.8,9.7,9.6\n",
            encoding="utf-8",
        )
        info = IO.inspect_source(path=str(path))
        assert info["kind"] == "matrix", info
        assert info["feature_col"] == "gene", info
        assert info["n_rows"] == 4 and info["n_cols"] == 5, info
        out = IO.open_source(path=str(path))
        dataset = out["dataset"]
        assert dataset["kind"] == "matrix", dataset["kind"]
        assert dataset["feature_col"] == "gene", dataset["feature_col"]
        assert dataset["df"].shape == (4, 5), dataset["df"].shape
        assert dataset["roles"].get("group") is None, dataset["roles"]
        for sample in ("s1", "s2", "s3", "s4"):
            assert pd.api.types.is_numeric_dtype(dataset["df"][sample]), sample

    check("6 matrix CSV（首列 gene + 4 样本）", test_matrix)

    # ---------------------------------------------------------------- 7. sample sheet
    def test_sample_sheet() -> None:
        matrix_path = root / "matrix2.csv"
        matrix_path.write_text(
            "gene,s1,s2,s3,s4\n"
            "IL6,1,2,3,4\n"
            "TNF,2,3,4,5\n"
            "CD8A,3,4,5,6\n",
            encoding="utf-8",
        )
        sheet_path = root / "samples.csv"
        sheet_path.write_text(
            "sample,group,batch\n"
            "s1,Tumor,1\n"
            "s2,Normal,1\n"
            "s3,Tumor,2\n"
            "s4,Normal,2\n",
            encoding="utf-8",
        )
        out = IO.open_source(path=str(matrix_path), sample_sheet=str(sheet_path))
        dataset = out["dataset"]
        info = dataset["sample_info"]
        assert info is not None and isinstance(info, pd.DataFrame), info
        assert list(info.columns)[:2] == ["sample", "group"], list(info.columns)
        assert list(info["sample"]) == ["s1", "s2", "s3", "s4"], list(info["sample"])
        mapping = dict(zip(info["sample"], info["group"]))
        assert mapping == {"s1": "Tumor", "s2": "Normal", "s3": "Tumor", "s4": "Normal"}, mapping
        assert dataset["roles"].get("group") == "group", dataset["roles"]
        # 缺失样本：样本表少一行 → 补缺失 + warnings
        partial = root / "samples_partial.csv"
        partial.write_text("sample,group\ns1,Tumor\ns2,Normal\n", encoding="utf-8")
        partial_out = IO.open_source(path=str(matrix_path), sample_sheet=str(partial))["dataset"]
        partial_info = partial_out["sample_info"]
        assert len(partial_info) == 4, len(partial_info)
        assert partial_info["group"].isna().sum() == 2, partial_info
        assert any("样本表" in warning for warning in partial_out["warnings"]), partial_out["warnings"]

    check("7 matrix + sample_sheet（group 正确）", test_sample_sheet)

    # ---------------------------------------------------------------- 8. gzip
    def test_gzip() -> None:
        path = root / "gzipped.csv.gz"
        text = "sample,value\nA,1\nB,2\nC,3\n"
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            handle.write(text)
        info = IO.inspect_source(path=str(path))
        assert info["format"] == "csv", info
        assert info["n_rows"] == 3 and info["n_cols"] == 2, info
        assert any("gzip" in note for note in info["notes"]), info["notes"]
        dataset = IO.open_source(path=str(path))["dataset"]
        assert dataset["df"].shape == (3, 2), dataset["df"].shape
        assert dataset["import"]["gzip"] == "gzipped.csv.gz", dataset["import"]

    check("8 gzip 单文件", test_gzip)

    # ---------------------------------------------------------------- 9. zip
    def test_zip() -> None:
        path = root / "pack.zip"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("a.csv", "x,y\n1,2\n3,4\n")
            archive.writestr("b.tsv", "u\tv\n5\t6\n7\t8\n9\t10\n")
            archive.writestr("README.txt", "说明文件\n")
        info = IO.inspect_source(path=str(path))
        names = [item["name"] for item in info["candidates"]]
        assert "a.csv" in names and "b.tsv" in names, names
        assert names[0] == "a.csv", names
        assert info["n_rows"] == 2 and info["n_cols"] == 2, info
        error = expect_error("ambiguous", lambda: IO.open_source(path=str(path)), "zip 多文件")
        assert len(error.details.get("candidates") or []) == 3, error.details
        selected = IO.open_source(path=str(path), member="b.tsv")
        assert selected["format"] == "tsv", selected
        assert selected["dataset"]["df"].shape == (3, 2), selected["dataset"]["df"].shape
        # 只写 basename 也能匹配
        base_selected = IO.open_source(path=str(path), member="a.csv")
        assert base_selected["dataset"]["df"].shape == (2, 2), base_selected["dataset"]["df"].shape
        # 单一表格文件的压缩包直接读
        single = root / "single.zip"
        with zipfile.ZipFile(single, "w") as archive:
            archive.writestr("only.csv", "m,n\n1,2\n")
        single_dataset = IO.open_source(path=str(single))["dataset"]
        assert single_dataset["df"].shape == (1, 2), single_dataset["df"].shape

    check("9 zip（多文件 candidates / member 选择）", test_zip)

    # ---------------------------------------------------------------- 10. empty
    def test_empty() -> None:
        path = root / "empty.csv"
        path.write_text("a,b\n", encoding="utf-8")
        inspect_error = expect_error("bad_request", lambda: IO.inspect_source(path=str(path)),
                                     "空 CSV inspect")
        assert "数据行" in inspect_error.message, inspect_error.message
        expect_error("bad_request", lambda: IO.open_source(path=str(path)), "空 CSV open")
        blank = root / "blank.csv"
        blank.write_text("\n\n\n", encoding="utf-8")
        expect_error("bad_request", lambda: IO.open_source(path=str(blank)), "全空 CSV")
        expect_error("not_found", lambda: IO.open_source(path=str(root / "nope.csv")), "路径不存在")

    check("10 0 行 CSV → bad_request", test_empty)

    # ---------------------------------------------------------------- 11. transpose
    def test_transpose() -> None:
        path = root / "wide.csv"
        path.write_text(
            "sample,IL6,TNF\n"
            "S1,1,2\n"
            "S2,3,4\n"
            "S3,5,6\n",
            encoding="utf-8",
        )
        info = IO.inspect_source(path=str(path), transpose=True)
        assert info["kind"] == "matrix", info
        assert info["n_rows"] == 2 and info["n_cols"] == 4, info
        out = IO.open_source(path=str(path), transpose=True)
        dataset = out["dataset"]
        frame = dataset["df"]
        assert frame.shape == (2, 4), frame.shape
        assert list(frame.columns) == ["feature", "S1", "S2", "S3"], list(frame.columns)
        assert list(frame["feature"]) == ["IL6", "TNF"], list(frame["feature"])
        assert dataset["kind"] == "matrix", dataset["kind"]
        assert dataset["feature_col"] == "feature", dataset["feature_col"]
        assert dataset["import"]["transpose"] is True
        assert any("transpose" in note for note in dataset["notes"]), dataset["notes"]

    check("11 transpose=true 转置（样本×特征→特征×样本）", test_transpose)

    # ---------------------------------------------------------------- 12. extras
    def test_extras() -> None:
        # 长文本单元格截断
        long_path = root / "long.csv"
        long_path.write_text(f"id,note\n1,{'x' * 260}\n2,ok\n", encoding="utf-8")
        dataset = IO.open_source(path=str(long_path))["dataset"]
        assert len(dataset["df"].loc[0, "note"]) == 200, len(dataset["df"].loc[0, "note"])
        assert any("截断" in warning for warning in dataset["warnings"]), dataset["warnings"]
        # txt 自动嗅探分号
        txt_path = root / "semi.txt"
        txt_path.write_text("a;b;c\n1;2;3\n4;5;6\n", encoding="utf-8")
        txt_dataset = IO.open_source(path=str(txt_path))["dataset"]
        assert txt_dataset["df"].shape == (2, 3), txt_dataset["df"].shape
        # matrix 特征列数值 ID 转文本
        num_matrix = root / "probe.csv"
        num_matrix.write_text("probe,s1,s2,s3\n101,1,2,3\n102,4,5,6\n103,7,8,9\n",
                              encoding="utf-8")
        probe = IO.open_source(path=str(num_matrix))["dataset"]
        assert probe["kind"] == "matrix", probe["kind"]
        assert list(probe["df"]["probe"]) == ["101", "102", "103"], list(probe["df"]["probe"])
        # formats()
        ids = [item["id"] for item in IO.formats()]
        assert ids == ["auto", "csv", "tsv", "txt", "xlsx", "json", "archive", "gz"], ids

    check("12 附加：长文本截断 / txt 嗅探 / 数值特征 ID / formats()", test_extras)

    # ---------------------------------------------------------------- 13. extras 2
    def test_extras2() -> None:
        import tarfile

        # tar.gz：多成员 → candidates + member 选择
        members = {"one.csv": b"a,b\n1,2\n", "two.tsv": b"c\td\n3\t4\n5\t6\n"}
        tgz = root / "pack.tar.gz"
        payload_dir = root / "tar_parts"
        payload_dir.mkdir(exist_ok=True)
        with tarfile.open(tgz, "w:gz") as archive:
            for name, body in members.items():
                part = payload_dir / name
                part.write_bytes(body)
                archive.add(part, arcname=name)
        info = IO.inspect_source(path=str(tgz))
        names = [item["name"] for item in info["candidates"]]
        assert names == ["one.csv", "two.tsv"], names
        expect_error("ambiguous", lambda: IO.open_source(path=str(tgz)), "tar.gz 多文件")
        selected = IO.open_source(path=str(tgz), member="two.tsv")
        assert selected["format"] == "tsv", selected
        assert selected["dataset"]["df"].shape == (2, 2), selected["dataset"]["df"].shape

        # header=False：列名 col_1..col_n
        plain = root / "noheader.csv"
        plain.write_text("A,1,2\nB,3,4\nC,5,6\n", encoding="utf-8")
        no_header = IO.open_source(path=str(plain), header=False)["dataset"]
        assert list(no_header["df"].columns) == ["col_1", "col_2", "col_3"], list(no_header["df"].columns)
        assert no_header["df"].shape == (3, 3), no_header["df"].shape

        # 小 tidy 表（首列样本 ID + 类别列 + 数值列）不能被误判为 matrix 而清空类别列
        tidy = root / "small_tidy.csv"
        tidy.write_text(
            "id,group,age,value\n"
            "S1,Control,41,1.2\n"
            "S2,Control,44,2.3\n"
            "S3,Treated,51,3.4\n"
            "S4,Treated,49,4.5\n"
            "S5,Control,38,5.6\n",
            encoding="utf-8",
        )
        out = IO.open_source(path=str(tidy))
        dataset = out["dataset"]
        assert dataset["kind"] == "tidy", dataset["kind"]
        assert dataset["df"]["group"].notna().all(), dataset["df"]["group"]
        assert set(dataset["df"]["group"]) == {"Control", "Treated"}
        assert any("tidy" in note for note in dataset["notes"]), dataset["notes"]

    check("13 附加：tar.gz / header=False / 小 tidy 不误判 matrix", test_extras2)


if __name__ == "__main__":
    raise SystemExit(main())
