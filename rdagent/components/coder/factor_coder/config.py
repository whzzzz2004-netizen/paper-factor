import os
from typing import Optional

from pydantic_settings import SettingsConfigDict

from rdagent.components.coder.CoSTEER.config import CoSTEERSettings
from rdagent.utils.env import CondaConf, Env, LocalEnv


class FactorCoSTEERSettings(CoSTEERSettings):
    model_config = SettingsConfigDict(env_prefix="FACTOR_CoSTEER_")

    data_folder: str = "git_ignore_folder/factor_implementation_source_data"
    """Path to the folder containing financial data (default is fundamental data in Qlib)"""

    data_folder_debug: str = "git_ignore_folder/factor_implementation_source_data_debug"
    """Path to the folder containing partial financial data (for debugging)"""

    simple_background: bool = False
    """Whether to use simple background information for code feedback"""

    file_based_execution_timeout: int = 3600
    """Timeout in seconds for each factor implementation execution"""

    select_method: str = "random"
    """Method for the selection of factors implementation"""

    python_bin: str = "python"
    """Path to the Python binary"""

    execution_backend: str = "conda"
    """Execution backend for factor code: conda/docker/local/auto"""

    execution_conda_env_name: str = "torch-pip"
    """Dedicated conda env used when execution_backend is conda or auto falls back to conda."""

    docker_image: str = "local_factor_exec:latest"
    """Docker image used when execution_backend is docker."""

    min_abs_ic: float = 0.005
    """Minimum absolute cross-sectional IC required for a factor to be accepted into the pool"""


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
