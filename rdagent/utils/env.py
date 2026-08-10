"""
The motivation of the utils is for environment management

Tries to create uniform environment for the agent to run;
- All the code and data is expected included in one folder
"""

# TODO: move the scenario specific docker env into other folders.

import contextlib
import json
import os
import pickle
import re
import select
import shutil
import subprocess
import time
import uuid
import zipfile
from abc import abstractmethod
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Generic,
    Mapping,
    Optional,
    TypeVar,
    cast,
)

from pydantic import BaseModel, model_validator
from pydantic_settings import SettingsConfigDict
from rich import print
from rich.console import Console
from rich.rule import Rule
from rich.table import Table

from rdagent.core.conf import ExtendedBaseSettings
from rdagent.core.experiment import RD_AGENT_SETTINGS
from rdagent.core.utils import cache_with_pickle
from rdagent.log import rdagent_logger as logger
from rdagent.oai.llm_utils import md5_hash
from rdagent.utils import filter_redundant_text
from rdagent.utils.fmt import shrink_text

CacheKeyFunc = Callable[[str | Path], list[list[str]]]


# Normalize all bind paths in volumes to absolute paths using the workspace (working_dir).
def normalize_volumes(vols: dict[str, str | dict[str, str]], working_dir: str) -> dict:
    abs_vols: dict[str, str | dict[str, str]] = {}

    def to_abs(path: str) -> str:
        # Converts a relative path to an absolute path using the workspace (working_dir).
        return os.path.abspath(os.path.join(working_dir, path)) if not os.path.isabs(path) else path

    for lp, vinfo in vols.items():
        # Support both:
        # 1. {'host_path': {'bind': 'container_path', ...}}
        # 2. {'host_path': 'container_path'}
        if isinstance(vinfo, dict):
            # abs_vols = cast(dict[str, dict[str, str]], abs_vols)
            vinfo = vinfo.copy()
            vinfo["bind"] = to_abs(vinfo["bind"])
            abs_vols[lp] = vinfo
        else:
            # abs_vols = cast(dict[str, str], abs_vols)
            abs_vols[lp] = to_abs(vinfo)
    return abs_vols


class EnvConf(ExtendedBaseSettings):
    default_entry: str
    env_dict: dict = {}
    extra_volumes: dict = {}
    running_timeout_period: int | None = 3600  # 10 minutes

    """it is a function to calculating hash keys"""

    def get_workspace_content_for_hash(self, local_path: str | Path) -> list[list[str]]:
        """Get content of key files in workspace for cache hash calculation.

        Scans .py, .csv, and .yaml files.
        """
        # we must add the information of data (beyond code) into the key.
        # Otherwise, all commands operating on data will become invalid (e.g. rm -r submission.csv)
        # So we recursively walk in the folder and add the sorted relative filename list as part of the key.
        # data_key = []
        # for path in Path(local_path).rglob("*"):
        #     p = str(path.relative_to(Path(local_path)))
        #     if p.startswith("__pycache__"):
        #         continue
        #     data_key.append(p)
        # data_key = sorted(data_key)
        local_path = Path(local_path)
        return [
            [str(path.relative_to(local_path)), path.read_text()]
            for path in sorted(
                list(local_path.rglob("*.py")) + list(local_path.rglob("*.csv")) + list(local_path.rglob("*.yaml"))
            )
        ]

    redirect_stdout_to_file: bool = False
    # helper settings to support transparent;
    enable_cache: bool = True
    retry_count: int = 5  # retry count for the docker run
    retry_wait_seconds: int = 10  # retry wait seconds for the docker run

    model_config = SettingsConfigDict(
        # TODO: add prefix ....
        env_parse_none_str="None",  # Nthis is the key to accept `RUNNING_TIMEOUT_PERIOD=None`
    )


ASpecificEnvConf = TypeVar("ASpecificEnvConf", bound=EnvConf)


@dataclass
class EnvResult:
    """
    The result of running the environment.
    It contains the stdout, the exit code, and the running time in seconds.
    """

    full_stdout: str
    exit_code: int
    running_time: float
    stored_full_stdout_to_truncated_stdout: Dict[str, str]

    def __init__(self, stdout: str, exit_code: int, running_time: float):
        self.full_stdout = stdout
        self.exit_code = exit_code
        self.running_time = running_time
        self.stored_full_stdout_to_truncated_stdout = {}

    def update_stdout(self, stdout: str) -> None:
        self.full_stdout = stdout

    @property
    def stdout(self) -> str:
        if self.full_stdout not in self.stored_full_stdout_to_truncated_stdout:
            truncated: str = self._get_truncated_stdout(self.full_stdout)
            self.stored_full_stdout_to_truncated_stdout[self.full_stdout] = truncated
        return self.stored_full_stdout_to_truncated_stdout[self.full_stdout]

    def hash_full_stdout(self, full_stdout: str) -> str:
        return md5_hash(full_stdout)

    @cache_with_pickle(hash_full_stdout)
    def _get_truncated_stdout(self, full_stdout: str) -> str:
        return shrink_text(
            filter_redundant_text(full_stdout),
            context_lines=RD_AGENT_SETTINGS.stdout_context_len,
            line_len=RD_AGENT_SETTINGS.stdout_line_len,
        )


class Env(Generic[ASpecificEnvConf]):
    """
    We use BaseModel as the setting due to the features it provides
    - It provides base typing and checking features.
    - loading and dumping the information will be easier: for example, we can use package like `pydantic-yaml`
    """

    conf: ASpecificEnvConf  # different env have different conf.

    def __init__(self, conf: ASpecificEnvConf):
        self.conf = conf

    def zip_a_folder_into_a_file(self, folder_path: str, zip_file_path: str) -> None:
        """
        Zip a folder into a file, use zipfile instead of subprocess
        """
        with zipfile.ZipFile(zip_file_path, "w") as z:
            for root, _, files in os.walk(folder_path):
                for file in files:
                    z.write(
                        os.path.join(root, file),
                        os.path.relpath(os.path.join(root, file), folder_path),
                    )

    def unzip_a_file_into_a_folder(
        self, zip_file_path: str, folder_path: str, files_to_extract: list[str] | None = None
    ) -> None:
        """
        Unzip a file into a folder, use zipfile instead of subprocess
        """
        if files_to_extract is None:
            # Clear folder_path before extracting
            if os.path.exists(folder_path):
                shutil.rmtree(folder_path)
            os.makedirs(folder_path)

        with zipfile.ZipFile(zip_file_path, "r") as z:
            if files_to_extract is not None:
                for file_name in files_to_extract:
                    try:
                        z.extract(file_name, folder_path)
                    except KeyError:
                        logger.warning(f"File {file_name} not found in cache zip.")
            else:
                z.extractall(folder_path)

    @abstractmethod
    def prepare(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        """
        Prepare for the environment based on it's configure
        """

    def check_output(
        self,
        entry: str | None = None,
        local_path: str = ".",
        env: dict | None = None,
        running_extra_volume: Mapping = MappingProxyType({}),
        cache_key_extra_func: CacheKeyFunc | None = None,
        cache_files_to_extract: list[str] | None = None,
    ) -> str:
        result = self.run(
            entry=entry,
            local_path=local_path,
            env=env,
            running_extra_volume=running_extra_volume,
            cache_key_extra_func=cache_key_extra_func,
            cache_files_to_extract=cache_files_to_extract,
        )
        return result.stdout

    def __run_with_retry(
        self,
        entry: str | None = None,
        local_path: str = ".",
        env: dict | None = None,
        running_extra_volume: Mapping = MappingProxyType({}),
    ) -> EnvResult:
        for retry_index in range(self.conf.retry_count + 1):
            try:
                start = time.time()
                log_output, return_code = self._run(
                    entry,
                    local_path,
                    env,
                    running_extra_volume=running_extra_volume,
                )
                end = time.time()
                logger.info(f"Running time: {end - start} seconds")
                if self.conf.running_timeout_period is not None and end - start + 1 >= self.conf.running_timeout_period:
                    logger.warning(
                        f"The running time exceeds {self.conf.running_timeout_period} seconds, so the process is killed."
                    )
                    log_output += f"\n\nThe running time exceeds {self.conf.running_timeout_period} seconds, so the process is killed."
                return EnvResult(log_output, return_code, end - start)
            except Exception as e:
                if retry_index == self.conf.retry_count:
                    raise
                logger.warning(
                    f"Error while running the container: {e}, current try index: {retry_index + 1}, {self.conf.retry_count - retry_index - 1} retries left."
                )
                time.sleep(self.conf.retry_wait_seconds)
        raise RuntimeError  # for passing CI

    def run(
        self,
        entry: str | None = None,
        local_path: str = ".",
        env: dict | None = None,
        running_extra_volume: Mapping = MappingProxyType({}),
        cache_key_extra_func: CacheKeyFunc | None = None,
        cache_files_to_extract: list[str] | None = None,
    ) -> EnvResult:
        """
        Run the folder under the environment and return the stdout, exit code, and running time.

        Parameters
        ----------
        entry : str | None
            We may we the entry point when we run it.
            For example, we may have different entries when we run and summarize the project.
        local_path : str | None
            the local path (to project, mainly for code) will be mounted into the docker
            Here are some examples for a None local path
            - for example, run docker for updating the data in the extra_volumes.
            - simply run the image. The results are produced by output or network
        env : dict | None
            Run the code with your specific environment.
        running_extra_volume : Mapping
            Extra volumes to mount during execution.
        cache_key_extra_func : CacheKeyFunc | None
            Optional function to calculate extra information for cache key calculation
        cache_files_to_extract : list[str] | None
            Optional list of files to extract from cache zip. If None, extract all.

        Returns
        -------
            EnvResult: An object containing the stdout, the exit code, and the running time in seconds.
        """
        _env = self.conf.env_dict.copy()
        if env:
            _env.update(env)
        env = _env

        if entry is None:
            entry = self.conf.default_entry

        if "|" in entry:
            logger.warning(
                "You are using a command with a shell pipeline (i.e., '|'). "
                "The exit code ($exit_code) will reflect the result of "
                "the last command in the pipeline.",
            )

        if self.conf.redirect_stdout_to_file:
            log_file_name = md5_hash(entry)[:8] + ".log"
            log_file = Path(local_path) / f"{log_file_name}"
            log_file_relative_path = log_file.relative_to(Path(local_path))
            entry = f"{entry} > {log_file_relative_path} 2>&1"

        if self.conf.running_timeout_period is None:
            timeout_cmd = entry
        else:
            timeout_cmd = f"timeout --kill-after=10 {self.conf.running_timeout_period} {entry}"
        entry_add_timeout = (
            f"/bin/sh -c '"  # start of the sh command
            + f"{timeout_cmd}; entry_exit_code=$?; "
            + "exit $entry_exit_code"
            + "'"  # end of the sh command
        )

        if self.conf.enable_cache:
            result = self.cached_run(
                entry_add_timeout,
                local_path,
                env,
                running_extra_volume,
                cache_key_extra_func,
                cache_files_to_extract,
            )
        else:
            result = self.__run_with_retry(
                entry_add_timeout,
                local_path,
                env,
                running_extra_volume,
            )
        if self.conf.redirect_stdout_to_file:
            stdout = log_file.read_text(errors="replace")
            log_file.unlink(missing_ok=True)
            result.update_stdout(stdout)
        if str(Path(local_path).resolve()) in result.stdout:
            result.update_stdout(result.stdout.replace(str(Path(local_path).resolve()), "<WORKSPACE_PATH>"))

        return result

    def cached_run(
        self,
        entry: str | None = None,
        local_path: str = ".",
        env: dict | None = None,
        running_extra_volume: Mapping = MappingProxyType({}),
        cache_key_extra_func: CacheKeyFunc | None = None,
        cache_files_to_extract: list[str] | None = None,
    ) -> EnvResult:
        """
        Run the folder under the environment.
        Will cache the output and the folder diff for next round of running.
        Use the python codes and the parameters(entry, running_extra_volume) as key to hash the input.
        """
        target_folder = Path(RD_AGENT_SETTINGS.pickle_cache_folder_path_str) / f"utils.env.run"
        target_folder.mkdir(parents=True, exist_ok=True)

        if cache_key_extra_func is not None:
            cache_key_extra = cache_key_extra_func(local_path)
        else:
            cache_key_extra = self.conf.get_workspace_content_for_hash(local_path)

        key = md5_hash(
            json.dumps(cache_key_extra)
            + json.dumps({"entry": entry, "running_extra_volume": dict(running_extra_volume)})
            + json.dumps({"extra_volumes": self.conf.extra_volumes})
            # + json.dumps(data_key)
        )
        if Path(target_folder / f"{key}.pkl").exists() and Path(target_folder / f"{key}.zip").exists():
            with open(target_folder / f"{key}.pkl", "rb") as f:
                ret = pickle.load(f)
            self.unzip_a_file_into_a_folder(str(target_folder / f"{key}.zip"), local_path, cache_files_to_extract)
        else:
            ret = self.__run_with_retry(entry, local_path, env, running_extra_volume)
            with open(target_folder / f"{key}.pkl", "wb") as f:
                pickle.dump(ret, f)
            self.zip_a_folder_into_a_file(local_path, str(target_folder / f"{key}.zip"))
        return cast(EnvResult, ret)

    @abstractmethod
    def _run(
        self,
        entry: str | None,
        local_path: str = ".",
        env: dict | None = None,
        running_extra_volume: Mapping = MappingProxyType({}),
        **kwargs: Any,
    ) -> tuple[str, int]:
        """
        Execute the specified entry point within the given environment and local path.

        Parameters
        ----------
        entry : str | None
            The entry point to execute. If None, defaults to the configured entry.
        local_path : str
            The local directory path where the execution should occur.
        env : dict | None
            Environment variables to set during execution.
        kwargs : dict
            Additional keyword arguments for execution customization.

        Returns
        -------
        tuple[str, int]
            A tuple containing the standard output and the exit code.
        """
        pass

    def dump_python_code_run_and_get_results(
        self,
        code: str,
        dump_file_names: list[str],
        local_path: str,
        env: dict | None = None,
        running_extra_volume: Mapping = MappingProxyType({}),
        code_dump_file_py_name: Optional[str] = None,
    ) -> tuple[str, list]:
        """
        Dump the code into the local path and run the code.
        """
        random_file_name = f"{uuid.uuid4()}.py" if code_dump_file_py_name is None else f"{code_dump_file_py_name}.py"
        with open(os.path.join(local_path, random_file_name), "w") as f:
            f.write(code)
        entry = f"python {random_file_name}"
        log_output = self.check_output(entry, local_path, env, running_extra_volume=dict(running_extra_volume))
        results = []
        os.remove(os.path.join(local_path, random_file_name))
        for name in dump_file_names:
            if os.path.exists(os.path.join(local_path, f"{name}")):
                results.append(pickle.load(open(os.path.join(local_path, f"{name}"), "rb")))
                os.remove(os.path.join(local_path, f"{name}"))
            else:
                return log_output, []
        return log_output, results

    def refresh_env(self) -> None:
        """Refresh the environment, e.g., pull the latest docker image. rebuild the conda env."""
        pass


# class EnvWithCache
#

## Local Environment -----


class LocalConf(EnvConf):
    bin_path: str = ""
    """path like <path1>:<path2>:<path3>, which will be prepend to bin path."""

    retry_count: int = 0  # retry count for; run `retry_count + 1` times
    live_output: bool = True


ASpecificLocalConf = TypeVar("ASpecificLocalConf", bound=LocalConf)


class LocalEnv(Env[ASpecificLocalConf]):
    """
    Sometimes local environment may be more convenient for testing
    """

    def prepare(self) -> None: ...

    def _run(
        self,
        entry: str | None = None,
        local_path: str | None = None,
        env: dict | None = None,
        running_extra_volume: Mapping = MappingProxyType({}),
        **kwargs: dict,
    ) -> tuple[str, int]:

        # Handle volume links
        volumes = {}
        if self.conf.extra_volumes is not None:
            for lp, rp in self.conf.extra_volumes.items():
                volumes[lp] = rp["bind"] if isinstance(rp, dict) else rp
            cache_path = "/tmp/sample" if "/sample/" in "".join(self.conf.extra_volumes.keys()) else "/tmp/full"
            Path(cache_path).mkdir(parents=True, exist_ok=True)
            volumes[cache_path] = "/tmp/cache"
        for lp, rp in running_extra_volume.items():
            volumes[lp] = rp

        assert local_path is not None, "local_path should not be None"
        volumes = normalize_volumes(volumes, local_path)

        @contextlib.contextmanager
        def _symlink_ctx(vol_map: Mapping[str, str]) -> Generator[None, None, None]:
            created_links: list[Path] = []
            try:
                for real, link in vol_map.items():
                    link_path = Path(link)
                    real_path = Path(real)
                    if not link_path.parent.exists():
                        link_path.parent.mkdir(parents=True, exist_ok=True)
                    if link_path.exists() or link_path.is_symlink():
                        link_path.unlink()
                    link_path.symlink_to(real_path)
                    created_links.append(link_path)
                yield
            finally:
                for p in created_links:
                    try:
                        if p.is_symlink() or p.exists():
                            p.unlink()
                    except FileNotFoundError:
                        pass

        with _symlink_ctx(volumes):
            # Setup environment
            if env is None:
                env = {}

            # Auto-propagate CUDA_VISIBLE_DEVICES for proper GPU isolation
            if "CUDA_VISIBLE_DEVICES" in os.environ and "CUDA_VISIBLE_DEVICES" not in env:
                env["CUDA_VISIBLE_DEVICES"] = os.environ["CUDA_VISIBLE_DEVICES"]

            path = [
                *self.conf.bin_path.split(":"),
                "/bin/",
                "/usr/bin/",
                *env.get("PATH", "").split(":"),
            ]
            env["PATH"] = ":".join(path)

            if entry is None:
                entry = self.conf.default_entry

            print(Rule("[bold green]LocalEnv Logs Begin[/bold green]", style="dark_orange"))
            table = Table(title="Run Info", show_header=False)
            table.add_column("Key", style="bold cyan")
            table.add_column("Value", style="bold magenta")
            table.add_row("Entry", entry)
            table.add_row("Local Path", local_path or "")
            table.add_row("Env", "\n".join(f"{k}:{v}" for k, v in env.items()))
            table.add_row("Volumes", "\n".join(f"{k}:\n  {v}" for k, v in volumes.items()))
            print(table)

            cwd = Path(local_path).resolve() if local_path else None
            env = {k: str(v) if isinstance(v, int) else v for k, v in env.items()}

            process = subprocess.Popen(
                entry,
                cwd=cwd,
                env={**os.environ, **env},
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                shell=True,
                bufsize=1,
                universal_newlines=True,
            )

            # Setup polling
            if process.stdout is None or process.stderr is None:
                raise RuntimeError("The subprocess did not correctly create stdout/stderr pipes")

            if self.conf.live_output:
                stdout_fd = process.stdout.fileno()
                stderr_fd = process.stderr.fileno()

                poller = select.poll()
                poller.register(stdout_fd, select.POLLIN)
                poller.register(stderr_fd, select.POLLIN)

                combined_output = ""
                while True:
                    if process.poll() is not None:
                        break
                    events = poller.poll(100)
                    for fd, event in events:
                        if event & select.POLLIN:
                            if fd == stdout_fd:
                                while True:
                                    output = process.stdout.readline()
                                    if output == "":
                                        break
                                    Console().print(output.strip(), markup=False)
                                    combined_output += output
                            elif fd == stderr_fd:
                                while True:
                                    error = process.stderr.readline()
                                    if error == "":
                                        break
                                    Console().print(error.strip(), markup=False)
                                    combined_output += error

                # Capture any final output
                remaining_output, remaining_error = process.communicate()
                if remaining_output:
                    Console().print(remaining_output.strip(), markup=False)
                    combined_output += remaining_output
                if remaining_error:
                    Console().print(remaining_error.strip(), markup=False)
                    combined_output += remaining_error
            else:
                # Sacrifice real-time output to avoid possible standard I/O hangs
                out, err = process.communicate()
                Console().print(out, end="", markup=False)
                Console().print(err, end="", markup=False)
                combined_output = out + err

            return_code = process.returncode
            print(Rule("[bold green]LocalEnv Logs End[/bold green]", style="dark_orange"))

            return combined_output, return_code


class CondaConf(LocalConf):
    conda_env_name: str
    default_entry: str = "python main.py"

    @model_validator(mode="after")
    def change_bin_path(self, **data: Any) -> "CondaConf":
        self._update_bin_path()
        return self

    def _update_bin_path(self) -> None:
        """Update bin_path by querying the conda environment's PATH.

        This is called during initialization and can be called again after prepare()
        to ensure bin_path is set correctly even if the conda env was just created.
        """
        conda_path_result = subprocess.run(
            f"conda run -n {self.conda_env_name} --no-capture-output env | grep '^PATH='",
            capture_output=True,
            text=True,
            shell=True,
        )
        self.bin_path = conda_path_result.stdout.strip().split("=")[1] if conda_path_result.returncode == 0 else ""


class QlibCondaConf(CondaConf):
    conda_env_name: str = "rdagent4qlib"
    enable_cache: bool = False
    default_entry: str = "qrun conf.yaml"
    # extra_volumes: dict = {str(Path("~/.qlib/").expanduser().resolve().absolute()): "/root/.qlib/"}


class QlibCondaEnv(LocalEnv[QlibCondaConf]):
    def prepare(self) -> None:
        """Prepare the conda environment if not already created."""
        try:
            envs = subprocess.run("conda env list", capture_output=True, text=True, shell=True)
            if self.conf.conda_env_name not in envs.stdout:
                print(f"[yellow]Conda env '{self.conf.conda_env_name}' not found, creating...[/yellow]")
                subprocess.check_call(
                    f"conda create -y -n {self.conf.conda_env_name} python=3.10",
                    shell=True,
                )
                subprocess.check_call(
                    f"conda run -n {self.conf.conda_env_name} pip install --upgrade pip cython",
                    shell=True,
                )
                subprocess.check_call(
                    f"conda run -n {self.conf.conda_env_name} pip install git+https://github.com/microsoft/qlib.git@2fb9380b342556ddb50a4b24e4fe8655d548b2b8",
                    shell=True,
                )
                subprocess.check_call(
                    f"conda run -n {self.conf.conda_env_name} pip install catboost xgboost tables torch",
                    shell=True,
                )

        except Exception as e:
            print(f"[red]Failed to prepare conda env: {e}[/red]")
