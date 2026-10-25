"""统一解析数据根目录（paper-factor-data）。

两台机器的数据盘不同：
  - 当前开发机：D 盘 → /mnt/d/paper-factor-data
  - 另一台电脑：G 盘 → /mnt/g/paper-factor-data（Windows 为 G:\\paper-factor-data）

优先级：
  1. 环境变量 PAPER_FACTOR_DATA_ROOT（显式指定，最可靠）
  2. 自动探测：G 盘存在数据仓库则用 G，否则用 D
  3. 默认 D 盘

这样另一台电脑 git pull 后无需改任何路径，脚本会自动落到 G 盘。
"""

from __future__ import annotations

import os
from pathlib import Path


def _candidates() -> tuple[Path, Path]:
    if os.name == "nt":
        return Path(r"G:\paper-factor-data"), Path(r"D:\paper-factor-data")
    return Path("/mnt/g/paper-factor-data"), Path("/mnt/d/paper-factor-data")


def resolve_data_root() -> Path:
    env = os.environ.get("PAPER_FACTOR_DATA_ROOT", "").strip()
    if env:
        return Path(env).expanduser()

    g_drive, d_drive = _candidates()
    for candidate in (g_drive, d_drive):
        if (candidate / "数据仓库").is_dir() or (candidate / "schema.json").is_file():
            return candidate
    return d_drive


DATA_ROOT = resolve_data_root()
