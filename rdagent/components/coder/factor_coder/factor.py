from __future__ import annotations

import hashlib
import json
import os
import site
import subprocess
import textwrap
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Tuple, Union

import pandas as pd
import docker  # type: ignore[import-untyped]
from filelock import FileLock

from rdagent.components.coder.CoSTEER.task import CoSTEERTask
from rdagent.components.coder.factor_coder.config import FACTOR_COSTEER_SETTINGS
from rdagent.core.exception import CodeFormatError, CustomRuntimeError, NoOutputError
from rdagent.core.experiment import Experiment, FBWorkspace
from rdagent.core.utils import cache_with_pickle
from rdagent.oai.llm_utils import md5_hash
from rdagent.scenarios.qlib.developer.factor_dashboard import refresh_factor_dashboard
from rdagent.utils.env import DockerConf, DockerEnv


class FactorTask(CoSTEERTask):
    # TODO:  generalized the attributes into the Task
    # - factor_* -> *
    def __init__(
        self,
        factor_name,
        factor_description,
        factor_formulation,
        *args,
        variables: dict = {},
        resource: str = None,
        factor_implementation: bool = False,
        **kwargs,
    ) -> None:
        self.factor_name = (
            factor_name  # TODO: remove it in the later version. Keep it only for pickle version compatibility
        )
        self.factor_formulation = factor_formulation
        self.variables = variables
        self.factor_resources = resource
        self.factor_implementation = factor_implementation
        super().__init__(name=factor_name, description=factor_description, *args, **kwargs)

    @property
    def factor_description(self):
        """for compatibility"""
        return self.description

    def get_task_information(self):
        return f"""factor_name: {self.factor_name}
factor_description: {self.factor_description}
factor_formulation: {self.factor_formulation}
variables: {str(self.variables)}"""

    def get_task_brief_information(self):
        return f"""factor_name: {self.factor_name}
factor_description: {self.factor_description}
factor_formulation: {self.factor_formulation}
variables: {str(self.variables)}"""

    def get_task_information_and_implementation_result(self):
        return {
            "factor_name": self.factor_name,
            "factor_description": self.factor_description,
            "factor_formulation": self.factor_formulation,
            "variables": str(self.variables),
            "factor_implementation": str(self.factor_implementation),
        }

    @staticmethod
    def from_dict(dict):
        return FactorTask(**dict)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}[{self.factor_name}]>"


class FactorDockerConf(DockerConf):
    build_from_dockerfile: bool = True
    dockerfile_folder_path: Path = Path(__file__).parent / "docker"
    image: str = FACTOR_COSTEER_SETTINGS.docker_image
    mount_path: str = "/workspace/factor_workspace"
    default_entry: str = "python _rdagent_factor_launcher.py"
    enable_cache: bool = False
    shm_size: str | None = "16g"
    mem_limit: str | None = "48g"
    save_logs_to_file: bool = True
    terminal_tail_lines: int = 20


class FactorDockerEnv(DockerEnv):
    def __init__(self, conf: DockerConf | None = None):
        super().__init__(conf or FactorDockerConf())

    def prepare(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        force_build = os.environ.get("FACTOR_CoSTEER_FORCE_DOCKER_BUILD", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "y",
            "on",
        }
        if not force_build:
            try:
                docker.from_env().images.get(self.conf.image)
                return
            except docker.errors.ImageNotFound:
                pass
        super().prepare(*args, **kwargs)


def _conda_env_exists(env_name: str) -> bool:
    result = subprocess.run(
        f"conda env list | grep -q '^{env_name} '",
        shell=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def _docker_daemon_available() -> bool:
    try:
        client = docker.from_env()
        client.ping()
        return True
    except Exception:
        return False


class FactorFBWorkspace(FBWorkspace):
    """
    This class is used to implement a factor by writing the code to a file.
    Input data and output factor value are also written to files.
    """

    # TODO: (Xiao) think raising errors may get better information for processing
    FB_EXEC_SUCCESS = "Execution succeeded without error."
    FB_CODE_NOT_SET = "code is not set."
    FB_EXECUTION_SUCCEEDED = "Execution succeeded without error."
    FB_OUTPUT_FILE_NOT_FOUND = "\nExpected output file not found."
    FB_OUTPUT_FILE_FOUND = "\nExpected output file found."
    EXPORTED_PARQUET_DIR = Path.cwd() / "git_ignore_folder" / "factor_outputs"
    LEADERBOARD_PATH = EXPORTED_PARQUET_DIR / "leaderboard.csv"
    EXECUTION_LAUNCHER = "_rdagent_factor_launcher.py"

    def __init__(
        self,
        *args,
        raise_exception: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.raise_exception = raise_exception

    def hash_func(self, data_type: str = "Debug") -> str:
        return (
            md5_hash(data_type + self.file_dict["factor.py"])
            if ("factor.py" in self.file_dict and not self.raise_exception)
            else None
        )

    @staticmethod
    def _sanitize_factor_name(name: str) -> str:
        return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in name).strip("_") or "factor"

    def _resolve_export_dir(self, review_metadata: dict[str, Any] | None = None) -> Path:
        review_metadata = review_metadata or {}
        if review_metadata.get("source_type") == "literature_report":
            report_title = str(review_metadata.get("source_report_title") or "unknown_report")
            return self.EXPORTED_PARQUET_DIR / "literature_reports" / self._sanitize_factor_name(report_title)
        return self.EXPORTED_PARQUET_DIR

    def _clear_rejected_marker(self, factor_name: str, review_metadata: dict[str, Any] | None = None) -> None:
        review_metadata = review_metadata or {}
        if review_metadata.get("source_type") != "literature_report":
            return
        report_dir = self._resolve_export_dir(review_metadata)
        reason_path = report_dir / f"SKIPPED__{self._sanitize_factor_name(factor_name)}.md"
        if reason_path.exists():
            reason_path.unlink()

        summary_path = report_dir / "_SKIPPED_FACTORS.md"
        if not summary_path.exists():
            return
        lines = summary_path.read_text(encoding="utf-8").splitlines()
        kept_lines = [line for line in lines if not line.startswith(f"- `{factor_name}`：")]
        has_remaining_skip = any(line.startswith("- `") for line in kept_lines)
        if has_remaining_skip:
            summary_path.write_text("\n".join(kept_lines).rstrip() + "\n", encoding="utf-8")
        else:
            summary_path.unlink()

    @staticmethod
    def _hash_factor_dataframe(df: pd.DataFrame) -> str:
        hashed = pd.util.hash_pandas_object(df, index=True).values
        return hashlib.md5(hashed.tobytes()).hexdigest()

    @staticmethod
    def _env_flag(name: str, default: bool = False) -> bool:
        value = os.environ.get(name)
        if value is None:
            return default
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}

    @staticmethod
    def _infer_time_granularity(df: pd.DataFrame) -> str:
        if df is None or df.empty or "datetime" not in df.index.names:
            return "unknown"
        dt_index = pd.to_datetime(df.index.get_level_values("datetime"))
        diffs = dt_index.to_series().diff().dropna()
        positive_diffs = diffs[diffs > pd.Timedelta(0)].unique()
        if len(positive_diffs) == 0:
            return "unknown"
        min_step = min(positive_diffs)
        if min_step <= pd.Timedelta(minutes=1):
            return "minute"
        if min_step >= pd.Timedelta(days=1):
            return "daily"
        return str(min_step)

    @staticmethod
    def _infer_factor_tags(task: FactorTask | None, extra_tags: list[str] | None = None) -> list[str]:
        content = " ".join(
            [
                getattr(task, "factor_name", "") or "",
                getattr(task, "factor_description", "") or "",
                getattr(task, "factor_formulation", "") or "",
                str(getattr(task, "variables", {}) or {}),
            ]
        ).lower()
        tags: set[str] = set(extra_tags or [])
        keyword_to_tag = {
            "momentum": "momentum",
            "reversal": "reversal",
            "rev_": "reversal",
            "volatility": "volatility",
            "range": "range",
            "volume": "volume",
            "liquidity": "liquidity",
            "spread": "liquidity",
            "vwap": "vwap",
            "minute": "minute_input",
            "intraday": "minute_input",
            "microstructure": "microstructure",
            "gap": "gap",
            "price-volume": "price_volume",
            "correlation": "correlation",
            "acceleration": "acceleration",
            "trend": "trend",
        }
        for keyword, tag in keyword_to_tag.items():
            if keyword in content:
                tags.add(tag)
        if "minute_pv.h5" in content:
            tags.add("minute_input")
        if "daily_pv.h5" in content:
            tags.add("daily_input")
        return sorted(tags)

    @staticmethod
    def _compact_logic_summary(text: str | None, limit: int = 160) -> str | None:
        if text is None:
            return None
        compact = " ".join(str(text).split())
        if len(compact) <= limit:
            return compact
        return compact[: limit - 3].rstrip() + "..."

    def _write_factor_metadata(
        self,
        factor_name: str,
        latest_path: Path,
        df: pd.DataFrame,
        factor_hash: str,
        review_metadata: dict[str, Any] | None = None,
    ) -> None:
        metadata_path = latest_path.with_suffix(".meta.json")
        code_path = latest_path.with_suffix(".code.py")
        task = self.target_task if isinstance(self.target_task, FactorTask) else None
        metadata = {
            "factor_name": factor_name,
            "display_name": str(df.columns[0]),
            "factor_description": task.factor_description if task is not None else None,
            "factor_formulation": task.factor_formulation if task is not None else None,
            "variables": task.variables if task is not None else None,
            "hash": factor_hash,
            "rows": len(df),
            "non_null": int(df.iloc[:, 0].notna().sum()),
            "time_granularity": self._infer_time_granularity(df),
            "logic_summary": (
                task.factor_description if task is not None else "No factor description recorded."
            ),
            "tags": self._infer_factor_tags(task, extra_tags=(review_metadata or {}).get("tags")),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "workspace_path": str(self.workspace_path),
            "latest_path": str(latest_path),
            "metadata_path": str(metadata_path),
            "code_path": str(code_path),
        }
        if review_metadata:
            metadata.update(review_metadata)
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    def _write_factor_code_snapshot(self, latest_path: Path) -> Path | None:
        code_path = latest_path.with_suffix(".code.py")
        code = self.file_dict.get("factor.py")
        if code is None:
            workspace_code_path = self.workspace_path / "factor.py"
            if workspace_code_path.exists():
                code = workspace_code_path.read_text(encoding="utf-8", errors="replace")
        if code is None:
            return None
        code_path.write_text(code, encoding="utf-8")
        return code_path

    def _update_factor_manifest(
        self,
        factor_name: str,
        latest_path: Path,
        df: pd.DataFrame,
        factor_hash: str,
        review_metadata: dict[str, Any] | None = None,
    ) -> None:
        manifest_path = self.EXPORTED_PARQUET_DIR / "manifest.csv"
        metadata_path = latest_path.with_suffix(".meta.json")
        code_path = latest_path.with_suffix(".code.py")
        task = self.target_task if isinstance(self.target_task, FactorTask) else None
        tags = self._infer_factor_tags(
            task,
            extra_tags=(review_metadata or {}).get("tags"),
        )
        row = pd.DataFrame(
            [
                {
                    "factor_name": factor_name,
                    "display_name": str(df.columns[0]),
                    "hash": factor_hash,
                    "rows": len(df),
                    "non_null": int(df.iloc[:, 0].notna().sum()),
                    "time_granularity": self._infer_time_granularity(df),
                    "accepted": bool((review_metadata or {}).get("accepted", False)),
                    "ic_score": (review_metadata or {}).get("ic_score"),
                    "factor_description": task.factor_description if task is not None else None,
                    "factor_formulation": task.factor_formulation if task is not None else None,
                    "variables": json.dumps(task.variables, ensure_ascii=False)
                    if task is not None and task.variables is not None
                    else None,
                    "logic_summary": (review_metadata or {}).get("logic_summary")
                    or (
                        task.factor_description if task is not None else ""
                    ),
                    "tags": json.dumps(tags, ensure_ascii=False),
                    "source_type": (review_metadata or {}).get("source_type", "agent_generated"),
                    "source_report_title": (review_metadata or {}).get("source_report_title"),
                    "source_report_path": (review_metadata or {}).get("source_report_path"),
                    "review_notes": (review_metadata or {}).get("review_notes"),
                    "latest_path": str(latest_path),
                    "metadata_path": str(metadata_path),
                    "code_path": str(code_path) if code_path.exists() else None,
                    "workspace_path": str(self.workspace_path),
                    "updated_at": datetime.now().isoformat(timespec="seconds"),
                }
            ]
        )
        if manifest_path.exists():
            manifest = pd.read_csv(manifest_path)
            same_factor = manifest["factor_name"].astype(str) == factor_name
            source_report_path = (review_metadata or {}).get("source_report_path")
            if source_report_path and "source_report_path" in manifest.columns:
                same_report = (
                    manifest["source_report_path"].fillna("").astype(str).map(lambda p: str(Path(p).resolve()) if p else "")
                    == str(Path(source_report_path).resolve())
                )
                manifest = manifest[~(same_factor & same_report)]
            else:
                manifest = manifest[~same_factor]
            manifest = pd.concat([manifest, row], ignore_index=True)
        else:
            manifest = row
        manifest.sort_values("factor_name").to_csv(manifest_path, index=False)
        self._clear_rejected_marker(factor_name, review_metadata)
        self._update_factor_leaderboard(manifest)

    def _update_factor_leaderboard(self, manifest: pd.DataFrame) -> None:
        if manifest.empty:
            return
        leaderboard = manifest.copy()
        if "accepted" not in leaderboard.columns:
            leaderboard["accepted"] = False
        leaderboard["accepted"] = leaderboard["accepted"].fillna(False).astype(bool)
        if "ic_score" in leaderboard.columns:
            leaderboard["ic_score"] = pd.to_numeric(leaderboard["ic_score"], errors="coerce")
        else:
            leaderboard["ic_score"] = pd.NA
        if "updated_at" in leaderboard.columns:
            leaderboard["updated_at"] = pd.to_datetime(leaderboard["updated_at"], errors="coerce")
        if "logic_summary" in leaderboard.columns:
            leaderboard["logic_summary"] = leaderboard["logic_summary"].apply(self._compact_logic_summary)
        preferred_columns = [
            "rank",
            "factor_name",
            "ic_score",
            "logic_summary",
            "tags",
        ]
        leaderboard = leaderboard.sort_values(
            by=["accepted", "ic_score", "updated_at", "factor_name"],
            ascending=[False, False, False, True],
            na_position="last",
        ).reset_index(drop=True)
        leaderboard.insert(0, "rank", leaderboard.index + 1)
        existing_columns = [column for column in preferred_columns if column in leaderboard.columns]
        leaderboard = leaderboard[existing_columns]
        leaderboard.to_csv(self.LEADERBOARD_PATH, index=False)

    def _export_factor_dataframe(self, df: pd.DataFrame, review_metadata: dict[str, Any] | None = None) -> None:
        if df is None or df.empty:
            return

        self.EXPORTED_PARQUET_DIR.mkdir(parents=True, exist_ok=True)
        export_dir = self._resolve_export_dir(review_metadata)
        export_dir.mkdir(parents=True, exist_ok=True)
        factor_name = self._sanitize_factor_name(str(df.columns[0]))
        latest_path = export_dir / f"{factor_name}.parquet"
        current_hash = self._hash_factor_dataframe(df)

        if latest_path.exists():
            try:
                existing_df = pd.read_parquet(latest_path)
                if self._hash_factor_dataframe(existing_df) == current_hash:
                    self._write_factor_code_snapshot(latest_path)
                    self._write_factor_metadata(factor_name, latest_path, df, current_hash, review_metadata)
                    self._update_factor_manifest(factor_name, latest_path, df, current_hash, review_metadata)
                    refresh_factor_dashboard()
                    return
            except Exception:
                # If the previous parquet cannot be read, overwrite it with the current successful output.
                pass

        df.to_parquet(latest_path, engine="pyarrow")
        self._write_factor_code_snapshot(latest_path)
        self._write_factor_metadata(factor_name, latest_path, df, current_hash, review_metadata)
        self._update_factor_manifest(factor_name, latest_path, df, current_hash, review_metadata)
        refresh_factor_dashboard()

        if self._env_flag("FACTOR_EXPORT_KEEP_SNAPSHOTS"):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            snapshot_path = export_dir / f"{timestamp}__{factor_name}.parquet"
            df.to_parquet(snapshot_path, engine="pyarrow")

    def export_reviewed_factor(
        self,
        df: pd.DataFrame,
        *,
        accepted: bool,
        logic_summary: str | None = None,
        tags: list[str] | None = None,
        review_notes: str | None = None,
        **extra_review_metadata: Any,
    ) -> None:
        review_metadata = {
            "accepted": accepted,
            "logic_summary": logic_summary,
            "tags": tags or [],
            "review_notes": review_notes,
            "source_type": "agent_generated",
        }
        review_metadata.update(extra_review_metadata)
        self._export_factor_dataframe(df, review_metadata=review_metadata)

    @classmethod
    def _build_shared_data_launcher(cls, source_data_path: Path, code_path: Path) -> str:
        source_data_path = source_data_path.resolve() if source_data_path.is_absolute() else source_data_path
        code_path = code_path.resolve() if code_path.is_absolute() else code_path
        return textwrap.dedent(
            f"""
            import builtins
            import os
            import runpy
            from pathlib import Path

            import pandas as pd

            DATA_DIR = Path({str(source_data_path)!r})
            os.environ["FACTOR_DATA_DIR"] = str(DATA_DIR)
            os.environ["RDAGENT_FACTOR_DATA_DIR"] = str(DATA_DIR)

            def _resolve_data_path(path_like):
                if path_like is None:
                    return path_like
                try:
                    candidate = Path(path_like)
                except TypeError:
                    return path_like
                if candidate.is_absolute() or candidate.exists():
                    return path_like
                fallback = DATA_DIR / candidate
                if fallback.exists():
                    return str(fallback)
                return path_like

            _orig_read_hdf = pd.read_hdf
            _orig_read_pickle = pd.read_pickle
            _orig_read_csv = pd.read_csv
            _orig_read_parquet = pd.read_parquet
            _orig_open = builtins.open

            pd.read_hdf = lambda path_or_buf, *args, **kwargs: _orig_read_hdf(
                _resolve_data_path(path_or_buf), *args, **kwargs
            )
            pd.read_pickle = lambda filepath_or_buffer, *args, **kwargs: _orig_read_pickle(
                _resolve_data_path(filepath_or_buffer), *args, **kwargs
            )
            pd.read_csv = lambda filepath_or_buffer, *args, **kwargs: _orig_read_csv(
                _resolve_data_path(filepath_or_buffer), *args, **kwargs
            )
            pd.read_parquet = lambda path, *args, **kwargs: _orig_read_parquet(
                _resolve_data_path(path), *args, **kwargs
            )
            builtins.open = lambda file, *args, **kwargs: _orig_open(_resolve_data_path(file), *args, **kwargs)

            runpy.run_path({str(code_path)!r}, run_name="__main__")
            """
        ).strip() + "\n"

    @staticmethod
    def _resolve_execution_backend() -> str:
        backend = str(FACTOR_COSTEER_SETTINGS.execution_backend).strip().lower()
        if backend != "auto":
            return backend
        if _docker_daemon_available():
            return "docker"
        if _conda_env_exists(FACTOR_COSTEER_SETTINGS.execution_conda_env_name):
            return "conda"
        return "local"

    @staticmethod
    def _python_command_for_backend() -> str:
        backend = FactorFBWorkspace._resolve_execution_backend()
        if backend == "conda":
            env_name = FACTOR_COSTEER_SETTINGS.execution_conda_env_name
            return f"conda run -n {env_name} python"
        return FACTOR_COSTEER_SETTINGS.python_bin

    @staticmethod
    def _sanitize_execution_feedback(raw_feedback: str, execution_code_path: Path) -> str:
        feedback = (
            raw_feedback.replace(str(execution_code_path.parent.absolute()), r"/path/to")
            .replace(str(site.getsitepackages()[0]), r"/path/to/site-packages")
        )
        if len(feedback) > 2000:
            feedback = feedback[:1000] + "....hidden long error message...." + feedback[-1000:]
        return feedback

    def _execute_locally(
        self,
        execution_code_path: Path,
        source_data_path: Path,
    ) -> tuple[bool, str]:
        command = f"{self._python_command_for_backend()} {execution_code_path.name}"
        completed = subprocess.run(
            command,
            shell=True,
            cwd=self.workspace_path,
            env={
                **os.environ,
                "FACTOR_DATA_DIR": str(source_data_path.resolve()),
                "RDAGENT_FACTOR_DATA_DIR": str(source_data_path.resolve()),
            },
            stderr=subprocess.STDOUT,
            stdout=subprocess.PIPE,
            text=True,
            timeout=FACTOR_COSTEER_SETTINGS.file_based_execution_timeout,
        )
        return completed.returncode == 0, self._sanitize_execution_feedback(completed.stdout or "", execution_code_path)

    def _execute_in_docker(
        self,
        execution_code_path: Path,
        source_data_path: Path,
    ) -> tuple[bool, str]:
        docker_env = FactorDockerEnv()
        docker_env.prepare()
        result = docker_env.run(
            local_path=str(self.workspace_path),
            entry=f"python {execution_code_path.name}",
            env={
                "FACTOR_DATA_DIR": "/workspace/factor_data",
                "RDAGENT_FACTOR_DATA_DIR": "/workspace/factor_data",
            },
            running_extra_volume={
                str(source_data_path.resolve()): {
                    "bind": "/workspace/factor_data",
                    "mode": "ro",
                }
            },
        )
        return result.exit_code == 0, self._sanitize_execution_feedback(result.full_stdout or "", execution_code_path)

    @cache_with_pickle(hash_func)
    def execute(self, data_type: str = "Debug") -> Tuple[str, pd.DataFrame]:
        """
        execute the implementation and get the factor value by the following steps:
        1. make the directory in workspace path
        2. write the code to the file in the workspace path
        3. expose the shared source data directory to the execution process
        if call_factor_py is True:
            4. execute the code
        else:
            4. generate a script from template to import the factor.py dump get the factor value to result.h5
        5. read the factor value from the output file in the workspace path folder
        returns the execution feedback as a string and the factor value as a pandas dataframe


        Regarding the cache mechanism:
        1. We will store the function's return value to ensure it behaves as expected.
        - The cached information will include a tuple with the following: (execution_feedback, executed_factor_value_dataframe, Optional[Exception])

        """
        self.before_execute()
        if self.file_dict is None or "factor.py" not in self.file_dict:
            if self.raise_exception:
                raise CodeFormatError(self.FB_CODE_NOT_SET)
            else:
                return self.FB_CODE_NOT_SET, None
        with FileLock(self.workspace_path / "execution.lock"):
            backend = self._resolve_execution_backend()
            if self.target_task.version == 1:
                source_data_path = (
                    Path(
                        FACTOR_COSTEER_SETTINGS.data_folder_debug,
                    )
                    if data_type == "Debug"  # FIXME: (yx) don't think we should use a debug tag for this.
                    else Path(
                        FACTOR_COSTEER_SETTINGS.data_folder,
                    )
                )
            elif self.target_task.version == 2:
                raise CustomRuntimeError("Only paper_factor factor tasks (version=1) are supported in this package.")

            source_data_path.mkdir(exist_ok=True, parents=True)
            code_path = self.workspace_path / f"factor.py"

            execution_feedback = self.FB_EXECUTION_SUCCEEDED
            execution_success = False
            execution_error = None

            if self.target_task.version == 1:
                launcher_data_path = source_data_path.resolve()
                launcher_code_path = code_path
                if backend == "docker":
                    launcher_data_path = Path("/workspace/factor_data")
                    launcher_code_path = Path("factor.py")
                execution_code_path = self.workspace_path / self.EXECUTION_LAUNCHER
                execution_code_path.write_text(
                    self._build_shared_data_launcher(
                        source_data_path=launcher_data_path,
                        code_path=launcher_code_path,
                    ),
                    encoding="utf-8",
                )
            elif self.target_task.version == 2:
                execution_code_path = self.workspace_path / f"{uuid.uuid4()}.py"
                execution_code_path.write_text((Path(__file__).parent / "factor_execution_template.txt").read_text())

            try:
                if backend == "docker":
                    execution_success, execution_feedback = self._execute_in_docker(
                        execution_code_path=execution_code_path,
                        source_data_path=source_data_path,
                    )
                elif backend in {"local", "conda"}:
                    execution_success, execution_feedback = self._execute_locally(
                        execution_code_path=execution_code_path,
                        source_data_path=source_data_path,
                    )
                else:
                    raise RuntimeError(f"Unsupported factor execution backend: {backend}")

                if not execution_success:
                    if self.raise_exception:
                        raise CustomRuntimeError(execution_feedback)
                    execution_error = CustomRuntimeError(execution_feedback)
            except subprocess.TimeoutExpired:
                execution_feedback += (
                    f"Execution timeout error and the timeout is set to "
                    f"{FACTOR_COSTEER_SETTINGS.file_based_execution_timeout} seconds."
                )
                if self.raise_exception:
                    raise CustomRuntimeError(execution_feedback)
                execution_error = CustomRuntimeError(execution_feedback)
            except Exception as e:
                if isinstance(e, CustomRuntimeError):
                    raise
                execution_feedback = str(e)
                if self.raise_exception:
                    raise CustomRuntimeError(execution_feedback) from e
                execution_error = CustomRuntimeError(execution_feedback)

            workspace_output_file_path = self.workspace_path / "result.h5"
            if workspace_output_file_path.exists() and execution_success:
                try:
                    executed_factor_value_dataframe = pd.read_hdf(workspace_output_file_path)
                    execution_feedback += self.FB_OUTPUT_FILE_FOUND
                except Exception as e:
                    execution_feedback += f"Error found when reading hdf file: {e}"[:1000]
                    executed_factor_value_dataframe = None
            else:
                execution_feedback += self.FB_OUTPUT_FILE_NOT_FOUND
                executed_factor_value_dataframe = None
                if self.raise_exception:
                    raise NoOutputError(execution_feedback)
                else:
                    execution_error = NoOutputError(execution_feedback)

        return execution_feedback, executed_factor_value_dataframe

    def __str__(self) -> str:
        # NOTE:
        # If the code cache works, the workspace will be None.
        return f"File Factor[{self.target_task.factor_name}]: {self.workspace_path}"

    def __repr__(self) -> str:
        return self.__str__()

    @staticmethod
    def from_folder(task: FactorTask, path: Union[str, Path], **kwargs):
        path = Path(path)
        code_dict = {}
        for file_path in path.iterdir():
            if file_path.suffix == ".py":
                code_dict[file_path.name] = file_path.read_text()
        return FactorFBWorkspace(target_task=task, code_dict=code_dict, **kwargs)


FactorExperiment = Experiment
FeatureExperiment = Experiment
