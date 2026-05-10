import json
import os
import re
from pathlib import Path
from typing import Any

import pandas as pd

from rdagent.core.experiment import FBWorkspace
from rdagent.log import rdagent_logger as logger
from rdagent.utils.env import QlibCondaConf, QlibCondaEnv, QTDockerEnv


class QlibFBWorkspace(FBWorkspace):
    def __init__(self, template_folder_path: Path, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.inject_code_from_folder(template_folder_path)

    def _safe_write_text(self, path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8", errors="replace")

    def _collect_qlib_artifact_status(self, qlib_config_name: str) -> dict[str, Any]:
        artifact_paths = {
            "qlib_config": self.workspace_path / qlib_config_name,
            "qlib_res": self.workspace_path / "qlib_res.csv",
            "ret": self.workspace_path / "ret.pkl",
            "pred": self.workspace_path / "pred.pkl",
            "combined_factors": self.workspace_path / "combined_factors_df.parquet",
            "portfolio_report": self.workspace_path / "portfolio_analysis" / "report_normal_1day.pkl",
            "portfolio_positions": self.workspace_path / "portfolio_analysis" / "positions_normal_1day.pkl",
        }
        mlflow_runs = []
        mlruns_dir = self.workspace_path / "mlruns"
        if mlruns_dir.exists():
            for meta_path in sorted(mlruns_dir.glob("*/*/meta.yaml"), key=lambda p: p.stat().st_mtime, reverse=True)[:5]:
                mlflow_runs.append(str(meta_path.relative_to(self.workspace_path)))

        return {
            "workspace": str(self.workspace_path),
            "qlib_config_name": qlib_config_name,
            "artifacts": {name: {"path": str(path), "exists": path.exists()} for name, path in artifact_paths.items()},
            "recent_mlflow_run_meta": mlflow_runs,
        }

    def _write_qlib_status(self, qlib_config_name: str) -> Path:
        status_path = self.workspace_path / f"qlib_status_{Path(qlib_config_name).stem}.json"
        status = self._collect_qlib_artifact_status(qlib_config_name)
        self._safe_write_text(status_path, json.dumps(status, indent=2, ensure_ascii=False))
        logger.info(f"Qlib artifact status saved to {status_path}")
        return status_path

    def _missing_result_message(self, qlib_config_name: str, status_path: Path) -> str:
        return (
            "Qlib execution finished but expected backtest result files were not found. "
            f"Workspace: {self.workspace_path}. Config: {qlib_config_name}. "
            f"Diagnostic status: {status_path}. "
            "Check qrun/read_exp_res logs in the same workspace for the exact failure point."
        )

    def execute(self, qlib_config_name: str = "conf.yaml", run_env: dict = {}, *args, **kwargs) -> str:
        env_type = os.environ.get("QLIB_EXECUTION_ENV_TYPE", "conda").strip().lower()
        if env_type == "docker":
            qtde = QTDockerEnv()
        elif env_type == "conda":
            qtde = QlibCondaEnv(conf=QlibCondaConf())
        else:
            logger.error(f"Unknown env_type: {env_type}")
            return None, "Unknown environment type"
        qtde.prepare()

        # Run the Qlib backtest
        qrun_result = qtde.run(
            local_path=str(self.workspace_path),
            entry=f"qrun {qlib_config_name}",
            env=run_env,
        )
        execute_qlib_log = qrun_result.stdout
        qrun_log_path = self.workspace_path / f"qrun_{Path(qlib_config_name).stem}.log"
        self._safe_write_text(qrun_log_path, qrun_result.full_stdout)
        logger.info(
            f"Qlib qrun log saved to {qrun_log_path}; exit_code={qrun_result.exit_code}; "
            f"running_time={qrun_result.running_time:.2f}s"
        )
        logger.log_object(execute_qlib_log, tag="Qlib_execute_log")

        read_result = qtde.run(
            local_path=str(self.workspace_path),
            entry="python read_exp_res.py",
            env=run_env,
        )
        execute_log = read_result.stdout
        read_log_path = self.workspace_path / f"read_exp_res_{Path(qlib_config_name).stem}.log"
        self._safe_write_text(read_log_path, read_result.full_stdout)
        logger.info(
            f"Qlib result-reader log saved to {read_log_path}; exit_code={read_result.exit_code}; "
            f"running_time={read_result.running_time:.2f}s"
        )

        status_path = self._write_qlib_status(qlib_config_name)

        quantitative_backtesting_chart_path = self.workspace_path / "ret.pkl"
        if quantitative_backtesting_chart_path.exists():
            ret_df = pd.read_pickle(quantitative_backtesting_chart_path)
            logger.log_object(ret_df, tag="Quantitative Backtesting Chart")
        else:
            message = self._missing_result_message(qlib_config_name, status_path)
            logger.error(message)
            return None, "\n".join([execute_qlib_log, execute_log, message])

        qlib_res_path = self.workspace_path / "qlib_res.csv"
        if qlib_res_path.exists():
            # Here, we ensure that the qlib experiment has run successfully before extracting information from execute_qlib_log using regex; otherwise, we keep the original experiment stdout.
            pattern = r"(Epoch\d+: train -[0-9\.]+, valid -[0-9\.]+|best score: -[0-9\.]+ @ \d+ epoch)"
            matches = re.findall(pattern, execute_qlib_log)
            execute_qlib_log = "\n".join(matches)
            return pd.read_csv(qlib_res_path, index_col=0).iloc[:, 0], execute_qlib_log
        else:
            message = self._missing_result_message(qlib_config_name, status_path)
            logger.error(f"File {qlib_res_path} does not exist. {message}")
            return None, "\n".join([execute_qlib_log, execute_log, message])
