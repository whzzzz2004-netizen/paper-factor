import json
import os
import re
from functools import lru_cache
from pathlib import Path
from xml.etree import ElementTree as ET
from zipfile import ZipFile


DEFAULT_FACTOR_SCHEMA_XLSX = Path("/mnt/c/Users/Administrator/Desktop/因子汇总.xlsx")
FACTOR_SCHEMA_ENV = "RDAGENT_FACTOR_FIELD_SCHEMA_XLSX"


def _column_index(cell_ref: str) -> int:
    letters = "".join(ch for ch in cell_ref if ch.isalpha())
    index = 0
    for ch in letters:
        index = index * 26 + ord(ch.upper()) - ord("A") + 1
    return max(index - 1, 0)


def _cell_text(cell: ET.Element, shared_strings: list[str], ns: dict[str, str]) -> str:
    if cell.attrib.get("t") == "inlineStr":
        return "".join(t.text or "" for t in cell.findall(".//a:t", ns)).strip()
    value = cell.find("a:v", ns)
    if value is None or value.text is None:
        return ""
    if cell.attrib.get("t") == "s":
        return shared_strings[int(value.text)].strip()
    return value.text.strip()


def _read_xlsx_rows(path: Path, sheet_name: str) -> list[list[str]]:
    ns = {
        "a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
        "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
        "rel": "http://schemas.openxmlformats.org/package/2006/relationships",
    }
    with ZipFile(path) as archive:
        names = set(archive.namelist())
        shared_strings: list[str] = []
        if "xl/sharedStrings.xml" in names:
            shared_root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            for item in shared_root.findall("a:si", ns):
                shared_strings.append("".join(t.text or "" for t in item.findall(".//a:t", ns)))

        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        rels = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        relmap = {rel.attrib["Id"]: rel.attrib["Target"] for rel in rels}
        sheet_path = ""
        for sheet in workbook.findall("a:sheets/a:sheet", ns):
            if sheet.attrib.get("name") == sheet_name:
                rid = sheet.attrib["{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"]
                target = relmap[rid]
                sheet_path = "xl/" + target.lstrip("/") if not target.startswith("xl/") else target
                break
        if not sheet_path:
            return []

        root = ET.fromstring(archive.read(sheet_path))
        rows: list[list[str]] = []
        for row in root.findall("a:sheetData/a:row", ns):
            values: list[str] = []
            for cell in row.findall("a:c", ns):
                index = _column_index(cell.attrib.get("r", "A1"))
                while len(values) <= index:
                    values.append("")
                values[index] = _cell_text(cell, shared_strings, ns)
            rows.append(values)
        return rows


def _candidate_schema_paths(base_dir: Path | None) -> list[Path]:
    paths: list[Path] = []
    env_path = os.environ.get(FACTOR_SCHEMA_ENV, "").strip()
    if env_path:
        paths.append(Path(env_path))
    if base_dir is not None:
        paths.extend([base_dir / "factor_field_schema.json", base_dir / "factor_field_schema.xlsx", base_dir / "因子汇总.xlsx"])
    paths.append(DEFAULT_FACTOR_SCHEMA_XLSX)
    return paths


def _load_schema_json(path: Path) -> dict[str, dict[str, str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return {}
    schema: dict[str, dict[str, str]] = {}
    for key, value in payload.items():
        if isinstance(value, dict):
            column = str(key) if str(key).startswith("$") else f"${key}"
            schema[column] = {str(k): str(v) for k, v in value.items() if v is not None}
    return schema


def _load_schema_xlsx(path: Path) -> dict[str, dict[str, str]]:
    rows = _read_xlsx_rows(path, "基本面因子")
    if len(rows) < 2:
        return {}
    header = {name: idx for idx, name in enumerate(rows[0])}
    schema: dict[str, dict[str, str]] = {}
    for row in rows[1:]:
        factor_name = row[header.get("因子名", -1)].strip() if header.get("因子名", -1) < len(row) else ""
        if not factor_name:
            continue
        column = factor_name if factor_name.startswith("$") else f"${factor_name}"
        short_name = row[header.get("因子简称", -1)].strip() if header.get("因子简称", -1) < len(row) else ""
        formula = row[header.get("计算公式", -1)].strip() if header.get("计算公式", -1) < len(row) else ""
        source = row[header.get("数据来源", -1)].strip() if header.get("数据来源", -1) < len(row) else ""
        english_name = (
            row[header.get("数据英文名（如有）", -1)].strip()
            if header.get("数据英文名（如有）", -1) < len(row)
            else ""
        )
        note = row[header.get("备注", -1)].strip() if header.get("备注", -1) < len(row) else ""
        schema[column] = {
            "factor_name": factor_name,
            "short_name": short_name,
            "formula": formula,
            "source": source,
            "english_name": english_name,
            "note": note,
            "frequency": "daily",
        }
    return schema


@lru_cache(maxsize=8)
def _load_factor_field_schema_cached(base_dir_text: str) -> dict[str, dict[str, str]]:
    base_dir = Path(base_dir_text) if base_dir_text else None
    for path in _candidate_schema_paths(base_dir):
        if not path.exists():
            continue
        try:
            if path.suffix.lower() == ".json":
                schema = _load_schema_json(path)
            elif path.suffix.lower() == ".xlsx":
                schema = _load_schema_xlsx(path)
            else:
                continue
        except Exception:
            continue
        if schema:
            return schema
    return {}


def load_factor_field_schema(base_dir: Path | None = None) -> dict[str, dict[str, str]]:
    return _load_factor_field_schema_cached(str(base_dir or ""))


def filter_field_schema(
    schema: dict[str, dict[str, str]],
    available_columns: list[str] | tuple[str, ...] | set[str],
) -> dict[str, dict[str, str]]:
    available = {str(column) for column in available_columns}
    return {column: value for column, value in schema.items() if column in available}


def format_field_schema_for_prompt(schema: dict[str, dict[str, str]], max_items: int | None = None) -> str:
    lines: list[str] = []
    for column in sorted(schema, key=lambda x: [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", x)]):
        item = schema[column]
        parts = [column]
        if item.get("short_name"):
            parts.append(item["short_name"])
        if item.get("formula"):
            parts.append(f"计算公式: {item['formula']}")
        if item.get("english_name"):
            parts.append(f"英文名: {item['english_name']}")
        if item.get("source"):
            parts.append(f"来源: {item['source']}")
        if item.get("note"):
            parts.append(f"备注: {item['note']}")
        lines.append(" = ".join(parts))
        if max_items is not None and len(lines) >= max_items:
            break
    return "\n".join(lines)
