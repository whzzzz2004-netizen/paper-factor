import os
from typing import Optional


from pydantic_settings import SettingsConfigDict

from rdagent.utils.env import CondaConf, Env, LocalEnv

try:
    from scripts.data_paths import DATA_ROOT
except ImportError:
    from pathlib import Path as _Path
    _env_root = os.environ.get("PAPER_FACTOR_DATA_ROOT", "").strip()
    _g_root = _Path("/mnt/g/paper-factor-data")
    _d_root = _Path("/mnt/d/paper-factor-data")
    DATA_ROOT = _Path(_env_root) if _env_root else (_g_root if (_g_root / "数据仓库").exists() else _d_root)


class FactorCoSTEERSettings:
    model_config = SettingsConfigDict(env_prefix="FACTOR_CoSTEER_")

    data_folder: str = str(DATA_ROOT / "数据仓库" / "行情数据" / "日线" / "测试")
    """Path to the folder containing financial data (default is fundamental data in Qlib)"""

    data_folder_debug: str = str(DATA_ROOT / "数据仓库" / "行情数据" / "日线" / "测试")
    """Path to the folder containing partial financial data (for debugging)"""

    simple_background: bool = False
    """Whether to use simple background information for code feedback"""

    file_based_execution_timeout: int = 3600
    """Timeout in seconds for each factor implementation execution"""

    select_method: str = "random"
    """Method for the selection of factors implementation"""

    python_bin: str = "python"
    """Path to the Python binary"""

    execution_backend: str = "local"
    """Execution backend for factor code: conda/docker/local/auto"""

    execution_conda_env_name: str = "torch-pip"
    """Dedicated conda env used when execution_backend is conda or auto falls back to conda."""

    min_abs_ic: float = 0.005
    """Minimum absolute cross-sectional IC required for a factor to be accepted into the pool"""

    enable_filelock: bool = False
    filelock_path: str = ""


def get_factor_env(
    conf_type: Optional[str] = None,
    extra_volumes: dict = {},
    running_timeout_period: int = 600,
    enable_cache: Optional[bool] = None,
) -> Env:
    conf = FactorCoSTEERSettings()
    if hasattr(conf, "python_bin"):
        env = LocalEnv(conf=(CondaConf(conda_env_name=os.environ.get("CONDA_DEFAULT_ENV"))))
    env.conf.extra_volumes = extra_volumes.copy()
    env.conf.running_timeout_period = running_timeout_period
    if enable_cache is not None:
        env.conf.enable_cache = enable_cache
    env.prepare()
    return env


FACTOR_COSTEER_SETTINGS = FactorCoSTEERSettings()
