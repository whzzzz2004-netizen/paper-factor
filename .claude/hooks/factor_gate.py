#!/usr/bin/env python3
"""决定 factor 专用钩子当前是否生效。

运行 `/factor` 时由 skill 创建哨兵文件；平时文件不存在 → 钩子一律放行。
这样日常改代码时不受任何限制，只有跑 /factor 时才拦截探索类命令。
"""

import os
import time

SENTINEL = "/tmp/factor_hooks_on"

# 崩溃遗留的哨兵最多影响 24 小时，之后自动失效
MAX_AGE_SECONDS = 24 * 3600


def active() -> bool:
    """哨兵存在且未过期 → 钩子生效。任何异常都按「不生效」处理（fail-open）。"""
    try:
        st = os.stat(SENTINEL)
    except OSError:
        return False
    return (time.time() - st.st_mtime) < MAX_AGE_SECONDS


def enable() -> None:
    try:
        with open(SENTINEL, "w") as fh:
            fh.write(str(int(time.time())))
    except OSError:
        pass


def disable() -> None:
    try:
        os.remove(SENTINEL)
    except OSError:
        pass


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "on":
        enable()
        print("factor 钩子已启用")
    elif len(sys.argv) > 1 and sys.argv[1] == "off":
        disable()
        print("factor 钩子已停用")
    else:
        print("active" if active() else "inactive")
