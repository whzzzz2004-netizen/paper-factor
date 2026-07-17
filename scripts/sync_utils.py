#!/usr/bin/env python3
"""
远程同步工具：通过 CIFS 挂载直接读写远程 E 盘（优先），
smbclient 作为后备。

用法:
  from scripts.sync_utils import (
      upload_file, upload_tree, ensure_remote_writable,
      REMOTE_BASE_TEST, REMOTE_BASE_FULL
  )

  上传单个文件:
  >>> upload_file(local_path, remote_relative_path)
"""

import os
import subprocess
import tempfile
import shutil
from pathlib import Path

# ── 远程配置 ──────────────────────────────────────────────
SMB_HOST = "192.168.1.13"
SMB_SHARE = "E"  # 对应 E: 盘的共享名
SMB_USER = "pc"
SMB_PASS = "123456"

# CIFS 挂载点（与本机 .bashrc 保持一致）
CIFS_MOUNT = Path("/mnt/remote_e")

# 远程根路径
REMOTE_ROOT = "paper_factors"

# 测试集输出 → 远程 文献因子/
REMOTE_BASE_TEST = f"{REMOTE_ROOT}\\文献因子"

# 全量输出 → 远程 文献因子_全量/
REMOTE_BASE_FULL = f"{REMOTE_ROOT}\\文献因子_全量"

# 每日增量输出 → 远程 文献因子_每日更新/
REMOTE_BASE_DAILY = f"{REMOTE_ROOT}\\文献因子_每日更新"


def _smb(args: list[str], timeout: int = 60) -> bool:
    """执行 smbclient 命令，返回成功/失败。"""
    cmd = [
        "smbclient", f"//{SMB_HOST}/{SMB_SHARE}",
        "-U", f"{SMB_USER}%{SMB_PASS}",
        "-c", " ; ".join(args),
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if r.returncode != 0:
            print(f"  ⚠️ smbclient 错误: {r.stderr.strip() or r.stdout.strip()}")
        return r.returncode == 0
    except subprocess.TimeoutExpired:
        print("  ⚠️ smbclient 超时")
        return False
    except FileNotFoundError:
        print("  ⚠️ smbclient 未安装，请执行: conda install -c conda-forge smbclient")
        return False


def _ensure_cifs_mounted() -> bool:
    """确保 CIFS 挂载可用。"""
    if CIFS_MOUNT.exists() and any(CIFS_MOUNT.iterdir()):
        return True
    # 尝试挂载
    CIFS_MOUNT.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(
        ["sudo", "mount", "-t", "cifs", f"//{SMB_HOST}/{SMB_SHARE}", str(CIFS_MOUNT),
         "-o", f"user={SMB_USER},password={SMB_PASS},uid={os.getuid()},gid={os.getgid()},file_mode=0644,dir_mode=0755,iocharset=utf8,noperm"],
        capture_output=True, text=True, timeout=30,
    )
    if r.returncode != 0:
        print(f"  ⚠️ CIFS 挂载失败: {r.stderr.strip()}")
        return False
    return True


def _remote_to_cifs_path(remote_path: str) -> Path:
    """将远程相对路径（paper_factors\\xxx\\yyy）转为 CIFS 挂载路径。"""
    # 替换反斜杠为正斜杠，拼接挂载点
    rel = remote_path.replace("\\", "/")
    return CIFS_MOUNT / rel


def ensure_remote_writable() -> bool:
    """测试远程 E 盘是否可写（优先 CIFS，后备 smbclient）。"""
    if _ensure_cifs_mounted():
        test_file = CIFS_MOUNT / f"_write_test_{os.getpid()}.txt"
        try:
            test_file.write_text("test")
            test_file.unlink()
            print("  ✅ 远程E盘可写（CIFS）")
            return True
        except OSError as e:
            print(f"  ⚠️ CIFS 写入测试失败: {e}")
            # 降级到 smbclient

    test_file = Path(tempfile.mktemp(suffix=".txt", prefix="smb_test_"))
    test_file.write_text("test")
    test_name = f"_smb_test_{os.getpid()}.txt"
    r = _smb([f"put {test_file} {REMOTE_ROOT}\\{test_name}", f"rm {REMOTE_ROOT}\\{test_name}"])
    test_file.unlink(missing_ok=True)
    if r:
        print("  ✅ 远程E盘可写（SMB）")
    else:
        print("  ⚠️ 远程E盘不可写")
    return r


def _ensure_remote_dir_cifs(remote_dir: str) -> bool:
    """通过 CIFS 递归创建目录。"""
    target = _remote_to_cifs_path(remote_dir)
    target.mkdir(parents=True, exist_ok=True)
    return True


def _ensure_remote_dir_smb(remote_dir: str) -> bool:
    """通过 smbclient 递归创建目录（后备）。"""
    parts = remote_dir.replace("\\", "/").split("/")
    for i in range(1, len(parts) + 1):
        sub = "\\".join(parts[:i])
        _smb([f'mkdir "{sub}"'], timeout=10)
    return True


def upload_file(local_path: Path, remote_path: str) -> bool:
    """上传单个文件到远程（优先 CIFS，后备 smbclient）。

    Args:
        local_path: 本地文件路径
        remote_path: 远程相对路径（如 paper_factors\\文献因子\\xxx\\factor.parquet）
    """
    if not local_path.exists():
        print(f"  ⚠️ 本地文件不存在: {local_path}")
        return False

    # ── 方案 A: CIFS 挂载（推荐，无空格/中文路径问题）──
    if _ensure_cifs_mounted():
        parts = remote_path.replace("/", "\\").split("\\")
        remote_dir = "\\".join(parts[:-1])
        remote_file = parts[-1]
        _ensure_remote_dir_cifs(remote_dir)
        dst = _remote_to_cifs_path(remote_dir) / remote_file
        try:
            shutil.copy2(local_path, dst)
            sz = local_path.stat().st_size
            print(f"  ✅ 已上传 {remote_path} ({sz//1024}KB)")
            return True
        except OSError as e:
            print(f"  ⚠️ CIFS 写入失败，降级到 smbclient: {e}")

    # ── 方案 B: smbclient（后备）──
    parts = remote_path.replace("/", "\\").split("\\")
    remote_dir = "\\".join(parts[:-1])
    remote_file = parts[-1]
    _ensure_remote_dir_smb(remote_dir)

    tmp = Path(tempfile.mktemp(suffix=local_path.suffix))
    shutil.copy2(local_path, tmp)
    try:
        result = _smb([f'put {tmp} "{remote_dir}\\{remote_file}"'], timeout=120)
    finally:
        tmp.unlink(missing_ok=True)
    if result:
        sz = local_path.stat().st_size
        print(f"  ✅ 已上传 {remote_path} ({sz//1024}KB)")
    return result


def upload_tree(local_dir: Path, remote_prefix: str) -> int:
    """上传整个目录树到远程。

    Returns:
        成功上传的文件数
    """
    if not local_dir.is_dir():
        print(f"  ⚠️ 本地目录不存在: {local_dir}")
        return 0

    count = 0
    for root, dirs, files in os.walk(local_dir):
        for f in files:
            local_file = Path(root) / f
            rel = local_file.relative_to(local_dir)
            remote_path = f"{remote_prefix}\\{rel}".replace("/", "\\")
            if upload_file(local_file, remote_path):
                count += 1
    return count


def ensure_remote_mounted() -> bool:
    """兼容旧接口，直接调用 ensure_remote_writable。"""
    return ensure_remote_writable()
