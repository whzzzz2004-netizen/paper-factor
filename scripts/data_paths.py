"""统一解析数据根目录（paper-factor-data）。

两台机器的数据盘不同：
  - 当前开发机：D 盘 → /mnt/d/paper-factor-data
  - 另一台电脑：G 盘 → /mnt/g/paper-factor-data（Windows 为 G:\\paper-factor-data）

优先级：
  1. 环境变量 PAPER_FACTOR_DATA_ROOT（显式指定，最可靠）
  2. 自动探测：D 盘存在数据仓库则用 D，否则用 G
  3. 默认 D 盘

当前 Windows 机器直接使用 D:\\paper-factor-data；另一台电脑若没有 D 盘数据，
会自动回退到 G:\\paper-factor-data。
"""

from __future__ import annotations

import os
from pathlib import Path


def _candidates() -> tuple[Path, Path]:
    """当前 Windows 机器优先 D 盘；D 盘没有数据时回退 G 盘。"""
    if os.name == "nt":
        return Path(r"D:\paper-factor-data"), Path(r"G:\paper-factor-data")
    return Path("/mnt/d/paper-factor-data"), Path("/mnt/g/paper-factor-data")


def resolve_data_root() -> Path:
    env = os.environ.get("PAPER_FACTOR_DATA_ROOT", "").strip()
    if env:
        return Path(env).expanduser()

    first, second = _candidates()
    for candidate in (first, second):
        if (candidate / "数据仓库").is_dir() or (candidate / "schema.json").is_file():
            return candidate
    return first


DATA_ROOT = resolve_data_root()
