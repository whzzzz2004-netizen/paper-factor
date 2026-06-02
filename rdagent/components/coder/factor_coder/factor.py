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
    # factor_type: "single_stock" | "cross_section" | "deep_learning"
    FACTOR_TYPE_SINGLE = "single_stock"
    FACTOR_TYPE_CROSS = "cross_section"
    FACTOR_TYPE_DL = "deep_learning"

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
        factor_type: str = "single_stock",
        **kwargs,
    ) -> None:
        self.factor_name = (
            factor_name  # TODO: remove it in the later version. Keep it only for pickle version compatibility
        )
        self.factor_formulation = factor_formulation
        self.variables = variables
        self.factor_resources = resource
        self.factor_implementation = factor_implementation
        self.factor_type = factor_type
        super().__init__(name=factor_name, description=factor_description, *args, **kwargs)

    @property
    def factor_description(self):
        """for compatibility"""
        return self.description

    def get_task_information(self):
        return f"""factor_name: {self.factor_name}
factor_type: {getattr(self, 'factor_type', 'single_stock')}
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
    EXECUTION_LAUNCHER = "_rdagent_factor_launcher.py"

    # 日线框架代码模板
    DAILY_FRAMEWORK_TEMPLATE = """import pandas as pd
import numpy as np
import sys, json, os
from pathlib import Path
from joblib import Parallel, delayed
from tqdm.auto import tqdm

DATA_DIR = Path(os.environ.get("FACTOR_DATA_DIR") or os.environ.get("RDAGENT_FACTOR_DATA_DIR") or ".")
STOCK_DATA_DIR = DATA_DIR / "stock_data" / "daily"
STOCK_LIST = json.load(open(STOCK_DATA_DIR / "stock_list.json"))
TRADE_DATES = json.load(open(STOCK_DATA_DIR / "trade_dates.json"))

def load_stock(stock):
    return pd.read_parquet(STOCK_DATA_DIR / f"{{stock}}.parquet")

{user_code}

def _compute_stock(stock):
    df = load_stock(stock)
    results = []
    for td_str in TRADE_DATES:
        td = pd.Timestamp(td_str)
        sub = df[df.index <= td]
        if sub.empty:
            continue
        r = calc_factor_single_stock(sub, td)
        if r:
            results.append({{"datetime": td_str, "instrument": stock, **r}})
    return results

if __name__ == '__main__':
    print("计算因子...")
    all_records = []
    for stock_results in Parallel(n_jobs=10, backend="loky")(
        delayed(_compute_stock)(s) for s in tqdm(STOCK_LIST, desc="按股票并行")
    ):
        all_records.extend(stock_results)
    long_df = pd.DataFrame(all_records)
    long_df["datetime"] = pd.to_datetime(long_df["datetime"])
    factor_name = [c for c in long_df.columns if c not in ("datetime", "instrument")][0]
    wide = long_df.pivot(index="datetime", columns="instrument", values=factor_name)
    wide = wide.sort_index().sort_index(axis=1)
    wide.index.name = "Date"
    wide.columns.name = "Code"
    wide = wide.replace([np.inf, -np.inf], np.nan)
    wide.attrs["factor_name"] = factor_name
    wide.to_parquet("result.parquet")
    print(f"完成，共 {{wide.shape[0]}} 天 x {{wide.shape[1]}}, 只股票")
"""

    # 分钟线框架代码模板（按日期文件，每天一个 parquet）
    MINUTE_FRAMEWORK_TEMPLATE = """import pandas as pd
import numpy as np
import sys, json, os
from pathlib import Path
from joblib import Parallel, delayed
from tqdm.auto import tqdm

DATA_DIR = Path(os.environ.get("FACTOR_DATA_DIR") or os.environ.get("RDAGENT_FACTOR_DATA_DIR") or ".")
MINUTE_BY_DATE_DIR = DATA_DIR / "stock_data" / "minute_by_date"
STOCK_LIST = json.load(open(DATA_DIR / "stock_data" / "minute" / "stock_list.json"))
TRADE_DATES = json.load(open(MINUTE_BY_DATE_DIR / "trade_dates.json"))

def load_day(td):
    return pd.read_parquet(MINUTE_BY_DATE_DIR / f"{{td}}.parquet")

{user_code}

def _compute_day(td):
    day_all = load_day(td)
    results = []
    for stock in day_all.index.get_level_values("instrument").unique():
        day_df = day_all.xs(stock, level="instrument")
        if day_df.empty:
            continue
        r = calc_factors_one_day(day_df)
        if r:
            results.append({{"datetime": td, "instrument": stock, **r}})
    return results

if __name__ == '__main__':
    print("计算分钟线因子...")
    all_records = []
    for day_results in Parallel(n_jobs=10, backend="loky")(
        delayed(_compute_day)(td) for td in tqdm(TRADE_DATES, desc="按日期并行")
    ):
        all_records.extend(day_results)
    long_df = pd.DataFrame(all_records)
    long_df["datetime"] = pd.to_datetime(long_df["datetime"])
    factor_name = [c for c in long_df.columns if c not in ("datetime", "instrument")][0]
    wide = long_df.pivot(index="datetime", columns="instrument", values=factor_name)
    wide = wide.sort_index().sort_index(axis=1)
    wide.index.name = "Date"
    wide.columns.name = "Code"
    wide = wide.replace([np.inf, -np.inf], np.nan)
    wide.attrs["factor_name"] = factor_name
    wide.to_parquet("result.parquet")
    print(f"完成，共 {{wide.shape[0]}} 天 x {{wide.shape[1]}}, 只股票")
"""

    def __init__(
        self,
        *args,
        raise_exception: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.raise_exception = raise_exception

    # 截面因子框架代码模板
    CROSS_SECTION_FRAMEWORK_TEMPLATE = """import pandas as pd
import numpy as np
import sys, json, os
from pathlib import Path

DATA_DIR = Path(os.environ.get("FACTOR_DATA_DIR") or os.environ.get("RDAGENT_FACTOR_DATA_DIR") or ".")
STOCK_DATA_DIR = DATA_DIR / "stock_data" / "daily"
STOCK_LIST = json.load(open(STOCK_DATA_DIR / "stock_list.json"))
TRADE_DATES = json.load(open(STOCK_DATA_DIR / "trade_dates.json"))

def load_stock(stock):
    return pd.read_parquet(STOCK_DATA_DIR / f"{{stock}}.parquet")

{user_code}

if __name__ == '__main__':
    print("计算截面因子...")
    all_records = []
    for td_str in tqdm(TRADE_DATES, desc="按日期计算"):
        td = pd.Timestamp(td_str)
        all_data = {{}}
        for stock in STOCK_LIST:
            df = load_stock(stock)
            sub = df[df.index <= td]
            if not sub.empty:
                all_data[stock] = sub
        if not all_data:
            continue
        result = calc_factor_cross_section(all_data, td)
        for stock, fdict in result.items():
            if fdict:
                all_records.append({{"datetime": td_str, "instrument": stock, **fdict}})
    long_df = pd.DataFrame(all_records)
    long_df["datetime"] = pd.to_datetime(long_df["datetime"])
    factor_name = [c for c in long_df.columns if c not in ("datetime", "instrument")][0]
    wide = long_df.pivot(index="datetime", columns="instrument", values=factor_name)
    wide = wide.sort_index().sort_index(axis=1)
    wide.index.name = "Date"
    wide.columns.name = "Code"
    wide = wide.replace([np.inf, -np.inf], np.nan)
    wide.attrs["factor_name"] = factor_name
    wide.to_parquet("result.parquet")
    print(f"完成，共 {{wide.shape[0]}} 天 x {{wide.shape[1]}}, 只股票")
"""

    # 深度学习因子框架代码模板
    DEEP_LEARNING_FRAMEWORK_TEMPLATE = """import pandas as pd
import numpy as np
import sys, json, os
from pathlib import Path
from tqdm.auto import tqdm

DATA_DIR = Path(os.environ.get("FACTOR_DATA_DIR") or os.environ.get("RDAGENT_FACTOR_DATA_DIR") or ".")
STOCK_DATA_DIR = DATA_DIR / "stock_data" / "daily"
STOCK_LIST = json.load(open(STOCK_DATA_DIR / "stock_list.json"))
TRADE_DATES = json.load(open(STOCK_DATA_DIR / "trade_dates.json"))

def load_stock(stock):
    return pd.read_parquet(STOCK_DATA_DIR / f"{{stock}}.parquet")

{user_code}

if __name__ == '__main__':
    print("计算深度学习因子...")
    # 预加载所有股票数据
    all_data = {{}}
    for stock in tqdm(STOCK_LIST, desc="加载数据"):
        all_data[stock] = load_stock(stock)

    all_records = []
    for td_str in tqdm(TRADE_DATES, desc="按日期计算"):
        td = pd.Timestamp(td_str)
        # 训练模型（只用截至 trade_date 的数据）
        data_for_train = {{}}
        for stock, df in all_data.items():
            sub = df[df.index <= td]
            if not sub.empty:
                data_for_train[stock] = sub
        if not data_for_train:
            continue
        model = train_model(data_for_train, td)
        # 逐股票推理
        for stock, df in data_for_train.items():
            r = predict(model, df, td)
            if r:
                all_records.append({{"datetime": td_str, "instrument": stock, **r}})
    long_df = pd.DataFrame(all_records)
    long_df["datetime"] = pd.to_datetime(long_df["datetime"])
    factor_name = [c for c in long_df.columns if c not in ("datetime", "instrument")][0]
    wide = long_df.pivot(index="datetime", columns="instrument", values=factor_name)
    wide = wide.sort_index().sort_index(axis=1)
    wide.index.name = "Date"
    wide.columns.name = "Code"
    wide = wide.replace([np.inf, -np.inf], np.nan)
    wide.attrs["factor_name"] = factor_name
    wide.to_parquet("result.parquet")
    print(f"完成，共 {{wide.shape[0]}} 天 x {{wide.shape[1]}}, 只股票")
"""

    def inject_files(self, *args, **kwargs):
        """Override to wrap AI-generated code with framework if needed."""
        # Call parent inject_files first
        super().inject_files(*args, **kwargs)
        # Check if factor.py needs framework wrapping
        if "factor.py" in self.file_dict:
            code = self.file_dict["factor.py"]
            # 按代码内容检测模板类型
            if "def train_model" in code and "def predict" in code:
                wrapped = self.DEEP_LEARNING_FRAMEWORK_TEMPLATE.format(user_code=code)
            elif "calc_factor_cross_section" in code:
                wrapped = self.CROSS_SECTION_FRAMEWORK_TEMPLATE.format(user_code=code)
            elif "calc_factors_one_day" in code:
                wrapped = self.MINUTE_FRAMEWORK_TEMPLATE.format(user_code=code)
            else:
                wrapped = self.DAILY_FRAMEWORK_TEMPLATE.format(user_code=code)
            self.file_dict["factor.py"] = wrapped
            (self.workspace_path / "factor.py").write_text(wrapped)

    def hash_func(self, data_type: str = "Debug") -> str:
        if "factor.py" not in self.file_dict or self.raise_exception:
            return None
        # Include data file mtimes so cache invalidates when data changes
        data_folder = Path(FACTOR_COSTEER_SETTINGS.data_folder_debug if data_type == "Debug" else FACTOR_COSTEER_SETTINGS.data_folder)
        data_sig = ""
        if data_folder.exists():
            for f in sorted(data_folder.iterdir()):
                data_sig += f"{f.name}:{f.stat().st_mtime_ns}"
        return md5_hash(data_type + self.file_dict["factor.py"] + data_sig)

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
        if df is None or df.empty:
            return "unknown"
        # 处理宽表格式 (Date index) 和旧格式 (MultiIndex with datetime)
        if df.index.name == "Date":
            dt_index = pd.to_datetime(df.index)
        elif "datetime" in df.index.names:
            dt_index = pd.to_datetime(df.index.get_level_values("datetime"))
        else:
            return "unknown"
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
        if "minute_pv" in content or '/minute"' in content or "/minute'" in content:
            tags.add("minute_input")
        if "daily_pv" in content or '/daily"' in content or "/daily'" in content:
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
            "display_name": factor_name,
            "factor_description": task.factor_description if task is not None else None,
            "factor_formulation": task.factor_formulation if task is not None else None,
            "variables": task.variables if task is not None else None,
            "hash": factor_hash,
            "rows": len(df),
            "non_null": int(df.stack().notna().sum()) if df.index.name == "Date" and df.columns.name == "Code" else int(df.iloc[:, 0].notna().sum()),
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

    def _export_factor_dataframe(self, df: pd.DataFrame, review_metadata: dict[str, Any] | None = None) -> None:
        if df is None or df.empty:
            return

        self.EXPORTED_PARQUET_DIR.mkdir(parents=True, exist_ok=True)
        export_dir = self._resolve_export_dir(review_metadata)
        export_dir.mkdir(parents=True, exist_ok=True)
        # 从 attrs 获取因子名（宽表格式），或从列名获取（旧格式）
        if "factor_name" in df.attrs:
            factor_name = self._sanitize_factor_name(df.attrs["factor_name"])
        elif df.index.name == "Date" and df.columns.name == "Code":
            # 宽表但没有 attrs，尝试从任务获取
            factor_name = self._sanitize_factor_name(self.target_task.factor_name if self.target_task else "unknown")
        else:
            factor_name = self._sanitize_factor_name(str(df.columns[0]))
        latest_path = export_dir / f"{factor_name}.parquet"
        current_hash = self._hash_factor_dataframe(df)

        if latest_path.exists():
            try:
                existing_df = pd.read_parquet(latest_path)
                if self._hash_factor_dataframe(existing_df) == current_hash:
                    self._write_factor_code_snapshot(latest_path)
                    self._write_factor_metadata(factor_name, latest_path, df, current_hash, review_metadata)
                    self._clear_rejected_marker(factor_name, review_metadata)
                    refresh_factor_dashboard()
                    return
            except Exception:
                # If the previous parquet cannot be read, overwrite it with the current successful output.
                pass

        # 统一 size：reindex 到完整的日期×股票，NaN 填充
        if df.index.name == "Date" and df.columns.name == "Code":
            full_dates = pd.read_json(
                Path(FACTOR_COSTEER_SETTINGS.data_folder) / "stock_data" / "daily" / "trade_dates.json",
                typ="series"
            )
            full_dates = pd.to_datetime(full_dates).sort_values()
            full_stocks = json.loads(
                (Path(FACTOR_COSTEER_SETTINGS.data_folder) / "stock_data" / "daily" / "stock_list.json").read_text()
            )
            full_stocks = sorted(full_stocks)
            df = df.reindex(index=full_dates, columns=full_stocks)
            df.index.name = "Date"
            df.columns.name = "Code"
            # 重新计算 hash
            current_hash = self._hash_factor_dataframe(df)

        df.to_parquet(latest_path, engine="pyarrow")
        self._write_factor_code_snapshot(latest_path)
        self._write_factor_metadata(factor_name, latest_path, df, current_hash, review_metadata)
        self._clear_rejected_marker(factor_name, review_metadata)
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
                "HDF5_USE_FILE_LOCKING": "FALSE",
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
            4. generate a script from template to import the factor.py dump get the factor value to result.parquet
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

            workspace_output_file_path = self.workspace_path / "result.parquet"
            if workspace_output_file_path.exists() and execution_success:
                try:
                    executed_factor_value_dataframe = pd.read_parquet(workspace_output_file_path)
                    execution_feedback += self.FB_OUTPUT_FILE_FOUND
                except Exception as e:
                    execution_feedback += f"Error found when reading parquet file: {e}"[:1000]
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
