import io
import json
import math
import os
from pathlib import Path
from abc import abstractmethod
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from rdagent.components.coder.factor_coder.config import FACTOR_COSTEER_SETTINGS
from rdagent.components.coder.factor_coder.factor import FactorTask
from rdagent.core.experiment import Task, Workspace
from rdagent.oai.llm_conf import LLM_SETTINGS
from rdagent.oai.llm_utils import APIBackend
from rdagent.utils.agent.tpl import T


class FactorEvaluator:
    """Although the init method is same to Evaluator, but we want to emphasize they are different"""

    def __init__(self, scen=None) -> None:
        self.scen = scen

    @abstractmethod
    def evaluate(
        self,
        target_task: Task,
        implementation: Workspace,
        gt_implementation: Workspace,
        **kwargs,
    ) -> Tuple[str, object]:
        """You can get the dataframe by

        .. code-block:: python

            _, gen_df = implementation.execute()
            _, gt_df = gt_implementation.execute()

        Returns
        -------
        Tuple[str, object]
            - str: the text-based description of the evaluation result
            - object: a comparable metric (bool, integer, float ...) None for evaluator with only text-based result

        """
        raise NotImplementedError("Please implement the `evaluator` method")

    def _get_df(self, gt_implementation: Workspace, implementation: Workspace):
        if gt_implementation is not None:
            _, gt_df = gt_implementation.execute()
            if isinstance(gt_df, pd.Series):
                gt_df = gt_df.to_frame("gt_factor")
            if isinstance(gt_df, pd.DataFrame):
                gt_df = gt_df.sort_index()
        else:
            gt_df = None

        _, gen_df = implementation.execute()
        if isinstance(gen_df, pd.Series):
            gen_df = gen_df.to_frame("source_factor")
        if isinstance(gen_df, pd.DataFrame):
            gen_df = gen_df.sort_index()
        return gt_df, gen_df

    def __str__(self) -> str:
        return self.__class__.__name__


class FactorCodeEvaluator(FactorEvaluator):
    def evaluate(
        self,
        target_task: FactorTask,
        implementation: Workspace,
        execution_feedback: str,
        value_feedback: str = "",
        gt_implementation: Workspace = None,
        **kwargs,
    ):
        factor_information = target_task.get_task_information()
        code = implementation.all_codes
        system_prompt = T(".prompts:evaluator_code_feedback_v1_system").r(
            scenario=(
                self.scen.get_scenario_all_desc(
                    target_task,
                    filtered_tag="feature",
                    simple_background=FACTOR_COSTEER_SETTINGS.simple_background,
                )
                if self.scen is not None
                else "No scenario description."
            )
        )

        execution_feedback_to_render = execution_feedback
        for _ in range(10):  # 10 times to split the content is enough
            user_prompt = T(".prompts:evaluator_code_feedback_v1_user").r(
                factor_information=factor_information,
                code=code,
                execution_feedback=execution_feedback_to_render,
                value_feedback=value_feedback,
                gt_code=gt_implementation.code if gt_implementation else None,
            )
            if (
                APIBackend().build_messages_and_calculate_token(
                    user_prompt=user_prompt,
                    system_prompt=system_prompt,
                )
                > APIBackend().chat_token_limit
            ):
                execution_feedback_to_render = execution_feedback_to_render[len(execution_feedback_to_render) // 2 :]
            else:
                break
        critic_response = APIBackend().build_messages_and_create_chat_completion(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            json_mode=False,
        )

        return critic_response, None


class FactorInfEvaluator(FactorEvaluator):
    def evaluate(
        self,
        implementation: Workspace,
        gt_implementation: Workspace,
    ) -> Tuple[str, object]:
        _, gen_df = self._get_df(gt_implementation, implementation)
        if gen_df is None:
            return (
                "The source dataframe is None. Please check the implementation.",
                False,
            )
        INF_count = gen_df.isin([float("inf"), -float("inf")]).sum().sum()
        if INF_count == 0:
            return "The source dataframe does not have any infinite values.", True
        else:
            return (
                f"The source dataframe has {INF_count} infinite values. Please check the implementation.",
                False,
            )


class FactorSingleColumnEvaluator(FactorEvaluator):
    def evaluate(
        self,
        implementation: Workspace,
        gt_implementation: Workspace,
    ) -> Tuple[str, object]:
        _, gen_df = self._get_df(gt_implementation, implementation)
        if gen_df is None:
            return (
                "The source dataframe is None. Please check the implementation.",
                False,
            )
        if len(gen_df.columns) == 1:
            return "The source dataframe has only one column which is correct.", True
        else:
            return (
                "The source dataframe has more than one column. Please check the implementation. We only evaluate the first column.",
                False,
            )


class FactorNumericValueEvaluator(FactorEvaluator):
    def evaluate(
        self,
        implementation: Workspace,
        gt_implementation: Workspace,
    ) -> Tuple[str, object]:
        _, gen_df = self._get_df(gt_implementation, implementation)
        if gen_df is None:
            return "The source dataframe is None. Please check the implementation.", False
        if gen_df.shape[1] == 0:
            return "The source dataframe has no factor column.", False

        factor = gen_df.iloc[:, 0]
        non_null_count = int(factor.notna().sum())
        if non_null_count == 0:
            return "The factor column has no non-null values. All values are NaN — this is a critical failure. The code must produce valid numeric factor values.", False

        numeric_factor = pd.to_numeric(factor, errors="coerce")
        invalid_count = int(factor.notna().sum() - numeric_factor.notna().sum())
        if invalid_count == 0:
            return "The factor column is numeric or can be safely converted to numeric values.", True
        return (
            "The factor column contains non-numeric values. Factor outputs must be numeric; "
            f"found {invalid_count} non-null value(s) that cannot be converted to numeric. "
            "Categorical stages must be encoded as numbers before saving.",
            False,
        )


class FactorOutputFormatEvaluator(FactorEvaluator):
    @staticmethod
    def _paper_fast_enabled() -> bool:
        return os.environ.get("RDAGENT_PAPER_FACTOR_FAST", "").strip().lower() in {"1", "true", "yes", "on"}

    def evaluate(
        self,
        implementation: Workspace,
        gt_implementation: Workspace,
    ) -> Tuple[str, object]:
        gt_df, gen_df = self._get_df(gt_implementation, implementation)
        if gen_df is None:
            return (
                "The source dataframe is None. Skip the evaluation of the output format.",
                False,
            )
        if self._paper_fast_enabled():
            if not isinstance(gen_df.index, pd.MultiIndex):
                return "Output format check failed: result index is not a MultiIndex.", False
            index_names = list(gen_df.index.names)
            if index_names != ["datetime", "instrument"]:
                return (
                    "Output format check failed: result index must be named ['datetime', 'instrument'], "
                    f"but got {index_names}.",
                    False,
                )
            if gen_df.shape[1] != 1:
                return f"Output format check failed: result must have exactly one factor column, got {gen_df.shape[1]}.", False
            return "Output format check passed by deterministic paper-factor fast check.", True
        buffer = io.StringIO()
        gen_df.info(buf=buffer)
        gen_df_info_str = f"The user is currently working on a feature related task.\nThe output dataframe info is:\n{buffer.getvalue()}"
        system_prompt = T(".prompts:evaluator_output_format_system").r(
            scenario=(
                self.scen.get_scenario_all_desc(implementation.target_task, filtered_tag="feature")
                if self.scen is not None
                else "No scenario description."
            )
        )

        # TODO: with retry_context(retry_n=3, except_list=[KeyError]):
        max_attempts = 3
        attempts = 0
        final_evaluation_dict = None

        while attempts < max_attempts:
            try:
                api = APIBackend() if attempts == 0 else APIBackend(use_chat_cache=False)
                resp = api.build_messages_and_create_chat_completion(
                    user_prompt=gen_df_info_str,
                    system_prompt=system_prompt,
                    json_mode=True,
                    json_target_type=Dict[str, str | bool | int],
                )
                resp_dict = json.loads(resp)
                resp_dict["output_format_decision"] = str(resp_dict["output_format_decision"]).lower() in ["true", "1"]

                return (
                    str(resp_dict["output_format_feedback"]),
                    resp_dict["output_format_decision"],
                )
            except (KeyError, json.JSONDecodeError) as e:
                attempts += 1
                if attempts >= max_attempts:
                    raise KeyError(
                        "Wrong JSON Response or missing 'output_format_decision' or 'output_format_feedback' key after multiple attempts."
                    ) from e

        return "Failed to evaluate output format after multiple attempts.", False


class FactorDatetimeDailyEvaluator(FactorEvaluator):
    def evaluate(
        self,
        implementation: Workspace,
        gt_implementation: Workspace,
    ) -> Tuple[str | object]:
        _, gen_df = self._get_df(gt_implementation, implementation)
        if gen_df is None:
            return "The source dataframe is None. Skip the evaluation of the datetime format.", False

        if "datetime" not in gen_df.index.names:
            return "The source dataframe does not have a datetime index. Please check the implementation.", False

        try:
            pd.to_datetime(gen_df.index.get_level_values("datetime"))
        except Exception:
            return (
                f"The source dataframe has a datetime index but it is not in the correct format (maybe a regular string or other objects). Please check the implementation.\n The head of the output dataframe is: \n{gen_df.head()}",
                False,
            )

        datetime_index = pd.to_datetime(gen_df.index.get_level_values("datetime"))
        time_diff = datetime_index.to_series().diff().dropna()
        positive_diff = time_diff[time_diff > pd.Timedelta(0)].unique()

        if len(positive_diff) == 0:
            return "The generated dataframe has a valid datetime index, but its granularity cannot be inferred.", True

        min_step = min(positive_diff)
        if min_step <= pd.Timedelta(minutes=1):
            return (
                "The generated dataframe is minute-level. This pipeline expects daily factor outputs even when the input data is minute-level. "
                "Aggregate the minute data into one daily factor value per instrument.",
                False,
            )
        if min_step >= pd.Timedelta(days=1):
            return "The generated dataframe is daily-level, which is correct.", True
        return (
            f"The generated dataframe uses an unsupported datetime granularity ({min_step}). "
            "Please output one daily factor value per instrument.",
            False,
        )


def _get_daily_label_from_data_folder(data_folder: Path) -> pd.Series:
    daily_path = data_folder / "daily_pv.h5"
    df = pd.read_hdf(daily_path, key="data").sort_index()
    close = pd.to_numeric(df["$close"], errors="coerce")
    label = close.groupby(level="instrument").pct_change().groupby(level="instrument").shift(-1)
    label.name = "label_next_return"
    return label


def _mean_cross_sectional_ic(factor_df: pd.DataFrame, label: pd.Series) -> float:
    if factor_df is None or factor_df.empty:
        return float("nan")
    factor_series = pd.to_numeric(factor_df.iloc[:, 0], errors="coerce")
    factor_series.name = "factor"
    merged = pd.concat([factor_series, label], axis=1, join="inner").dropna()
    if merged.empty:
        return float("nan")
    daily_ic = merged.groupby(level="datetime").apply(lambda x: x["factor"].corr(x["label_next_return"]))
    daily_ic = pd.to_numeric(daily_ic, errors="coerce").dropna()
    if daily_ic.empty:
        return float("nan")
    return float(daily_ic.mean())


def _merge_factor_and_label(factor_df: pd.DataFrame, label: pd.Series) -> pd.DataFrame:
    factor_series = pd.to_numeric(factor_df.iloc[:, 0], errors="coerce")
    factor_series.name = "factor"
    merged = pd.concat([factor_series, label], axis=1, join="inner").dropna()
    return merged


def _safe_float(x: float | None) -> float | None:
    if x is None:
        return None
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return None
    if isinstance(xf, float) and (math.isnan(xf) or math.isinf(xf)):
        return None
    return xf


def _compute_raw_factor_value_quality(gen_df: pd.DataFrame) -> dict[str, Any]:
    """Cheap distributional checks on the generated factor column (before label merge)."""
    out: dict[str, Any] = {}
    if gen_df is None or gen_df.empty or gen_df.shape[1] < 1:
        out["factor_quality_error"] = "empty_or_missing_factor_column"
        return out
    s = pd.to_numeric(gen_df.iloc[:, 0], errors="coerce")
    n = int(len(s))
    nn = int(s.notna().sum())
    out["factor_non_null"] = nn
    out["factor_nan_ratio"] = float(1.0 - nn / n) if n else 1.0
    clean = s.dropna()
    if clean.empty:
        out["factor_std"] = None
        out["factor_unique_pct"] = None
        out["factor_quality_warnings"] = ["all_nan_values"]
        return out
    std = float(clean.std(ddof=1)) if len(clean) > 1 else 0.0
    nuniq = int(clean.nunique())
    out["factor_std"] = _safe_float(std)
    out["factor_unique_pct"] = float(nuniq / len(clean)) if len(clean) else 0.0
    warnings: list[str] = []
    if out["factor_nan_ratio"] > 0.5:
        warnings.append(f"high_nan_ratio_{out['factor_nan_ratio']:.1%}")
    if std < 1e-10:
        warnings.append("near_constant_std")
    if out["factor_unique_pct"] < 0.01:
        warnings.append(f"low_unique_value_fraction_{out['factor_unique_pct']:.4%}")
    out["factor_quality_warnings"] = warnings or None
    return out


def _compute_extended_metrics(
    merged: pd.DataFrame,
    *,
    top_frac: float = 0.2,
    bottom_frac: float = 0.2,
    periods_per_year: int = 252,
) -> dict[str, Any]:
    """Cross-sectional IC/RankIC, long-short quintiles, turnover, drawdown (label = forward return)."""
    out: dict[str, Any] = {"subuniverse_ic": None, "subuniverse_note": "not configured (set FACTOR_EVAL_SUBUNIVERSE_COLUMN if needed)"}
    if merged is None or merged.empty or len(merged) < 10:
        out["error"] = "insufficient_merged_rows"
        return out

    if "datetime" not in merged.index.names:
        out["error"] = "missing_datetime_index_level"
        return out

    dates = merged.index.get_level_values("datetime").unique().sort_values()
    out["n_calendar_days"] = int(len(dates))
    out["n_observations"] = int(len(merged))

    daily_ic_pearson: list[float] = []
    daily_ic_rank: list[float] = []
    daily_ls: list[float] = []
    turnover_churn: list[float] = []
    prev_top: set[Any] | None = None

    for dt in dates:
        try:
            slab = merged.loc[dt]
        except KeyError:
            continue
        if isinstance(slab, pd.Series):
            continue
        g = slab.dropna(subset=["factor", "label_next_return"])
        if len(g) < 3:
            continue

        ic_p = g["factor"].corr(g["label_next_return"])
        daily_ic_pearson.append(float(ic_p) if pd.notna(ic_p) else float("nan"))

        ic_r = g["factor"].corr(g["label_next_return"], method="spearman")
        daily_ic_rank.append(float(ic_r) if pd.notna(ic_r) else float("nan"))

        if len(g) >= 10:
            rnk = g["factor"].rank(pct=True, method="average")
            top_mask = rnk >= (1.0 - top_frac)
            bot_mask = rnk <= bottom_frac
            top_g = g.loc[top_mask, "label_next_return"]
            bot_g = g.loc[bot_mask, "label_next_return"]
            if len(top_g) > 0 and len(bot_g) > 0:
                daily_ls.append(float(top_g.mean() - bot_g.mean()))
            else:
                daily_ls.append(float("nan"))

            top_idx = set(g.index[top_mask])
            if prev_top is not None and len(top_idx | prev_top) > 0:
                churn = len(top_idx.symmetric_difference(prev_top)) / float(len(top_idx | prev_top))
                turnover_churn.append(float(churn))
            prev_top = top_idx
        else:
            daily_ls.append(float("nan"))

    s_ic_p = pd.Series(daily_ic_pearson).replace([np.inf, -np.inf], np.nan).dropna()
    s_ic_r = pd.Series(daily_ic_rank).replace([np.inf, -np.inf], np.nan).dropna()
    s_ls = pd.Series(daily_ls).replace([np.inf, -np.inf], np.nan).dropna()

    ic_mean = float(s_ic_p.mean()) if not s_ic_p.empty else float("nan")
    ic_std = float(s_ic_p.std(ddof=1)) if len(s_ic_p) > 1 else float("nan")
    icir = float(ic_mean / ic_std) if ic_std and not math.isnan(ic_std) and ic_std > 0 else float("nan")

    rank_ic_mean = float(s_ic_r.mean()) if not s_ic_r.empty else float("nan")
    rank_ic_std = float(s_ic_r.std(ddof=1)) if len(s_ic_r) > 1 else float("nan")
    rank_icir = (
        float(rank_ic_mean / rank_ic_std)
        if rank_ic_std and not math.isnan(rank_ic_std) and rank_ic_std > 0
        else float("nan")
    )

    out["ic_mean_pearson"] = _safe_float(ic_mean)
    out["ic_std_pearson"] = _safe_float(ic_std)
    out["icir_pearson"] = _safe_float(icir)
    out["ic_days"] = int(len(s_ic_p))
    out["ic_positive_hit_rate"] = (
        float((s_ic_p > 0).mean()) if not s_ic_p.empty else None
    )

    out["rank_ic_mean"] = _safe_float(rank_ic_mean)
    out["rank_ic_std"] = _safe_float(rank_ic_std)
    out["rank_icir"] = _safe_float(rank_icir)
    out["rank_ic_days"] = int(len(s_ic_r))

    ls_mean = float(s_ls.mean()) if not s_ls.empty else float("nan")
    ls_std = float(s_ls.std(ddof=1)) if len(s_ls) > 1 else float("nan")
    ls_sharpe = (
        float(ls_mean / ls_std * math.sqrt(periods_per_year))
        if ls_std and not math.isnan(ls_std) and ls_std > 0
        else float("nan")
    )
    out["ls_daily_mean"] = _safe_float(ls_mean)
    out["ls_daily_std"] = _safe_float(ls_std)
    out["ls_sharpe_annualized"] = _safe_float(ls_sharpe)
    out["ls_trading_days"] = int(len(s_ls))

    if not s_ls.empty:
        r = s_ls.fillna(0.0)
        wealth = float(np.prod(np.asarray(1.0 + r)) - 1.0)
        out["ls_cumulative_return_compound"] = _safe_float(wealth)
        n = len(r)
        if n > 0:
            ann = float((1.0 + wealth) ** (periods_per_year / n) - 1.0) if wealth > -1 else float("nan")
            out["ls_annualized_return_approx"] = _safe_float(ann)
        cum = (1.0 + r).cumprod()
        roll_max = cum.cummax()
        dd_series = cum / roll_max - 1.0
        mdd = float(dd_series.min()) if len(dd_series) else float("nan")
        out["ls_max_drawdown"] = _safe_float(mdd)
    else:
        out["ls_cumulative_return_compound"] = None
        out["ls_annualized_return_approx"] = None
        out["ls_max_drawdown"] = None

    if turnover_churn:
        out["top_bottom_turnover_mean"] = float(np.nanmean(turnover_churn))
        out["top_bottom_turnover_note"] = (
            f"mean symmetric churn of top {top_frac:.0%} vs prior day (equal-weight proxy)"
        )
    else:
        out["top_bottom_turnover_mean"] = None
        out["top_bottom_turnover_note"] = "insufficient consecutive days for turnover"

    return out


def _format_extended_metrics_feedback(metrics: dict[str, Any], *, threshold: float) -> str:
    """Human-readable block appended to evaluation feedback."""
    if metrics.get("error"):
        extra = ""
        if metrics.get("factor_non_null") is not None or metrics.get("factor_nan_ratio") is not None:
            extra = (
                f"\nFactor value quality (generated column): non_null={metrics.get('factor_non_null')}, "
                f"nan_ratio={metrics.get('factor_nan_ratio')}, std={metrics.get('factor_std')}, "
                f"unique_pct={metrics.get('factor_unique_pct')}, warnings={metrics.get('factor_quality_warnings')}"
            )
        return f"Extended metrics unavailable ({metrics.get('error')}).{extra}"

    lines = [
        "--- Extended evaluation (in-sample, forward return label from data folder) ---",
        f"IC (Pearson, daily CS mean): {metrics.get('ic_mean_pearson')}",
        f"IC std (daily): {metrics.get('ic_std_pearson')}",
        f"ICIR (IC mean / IC std): {metrics.get('icir_pearson')}",
        f"IC positive day hit rate: {metrics.get('ic_positive_hit_rate')}",
        f"Rank IC (Spearman, daily CS mean): {metrics.get('rank_ic_mean')}",
        f"Rank IC std: {metrics.get('rank_ic_std')}",
        f"Rank ICIR: {metrics.get('rank_icir')}",
        f"Long-short daily mean (top {metrics.get('top_frac', 0.2):.0%} long vs bottom {metrics.get('bottom_frac', 0.2):.0%} short by CS rank): {metrics.get('ls_daily_mean')}",
        f"Long-short Sharpe (annualized, ~sqrt({252})): {metrics.get('ls_sharpe_annualized')}",
        f"Long-short cumulative return (compound daily LS): {metrics.get('ls_cumulative_return_compound')}",
        f"Long-short approx annualized return: {metrics.get('ls_annualized_return_approx')}",
        f"Long-short max drawdown on compounded LS curve: {metrics.get('ls_max_drawdown')}",
        f"Top cohort turnover (mean): {metrics.get('top_bottom_turnover_mean')} ({metrics.get('top_bottom_turnover_note', '')})",
        f"Subuniverse IC: {metrics.get('subuniverse_ic')} ({metrics.get('subuniverse_note', '')})",
        f"Factor value quality: non_null={metrics.get('factor_non_null')}, nan_ratio={metrics.get('factor_nan_ratio')}, "
        f"std={metrics.get('factor_std')}, unique_pct={metrics.get('factor_unique_pct')}, "
        f"warnings={metrics.get('factor_quality_warnings')}",
        f"(Threshold check still uses |IC_pearson_mean| >= {threshold})",
    ]
    return "\n".join(lines)


def evaluate_factor_metrics_bundle(
    implementation: Workspace | None,
    *,
    data_type: str = "Debug",
    gen_df: pd.DataFrame | None = None,
) -> tuple[str, float | None, dict[str, Any]]:
    """
    Full cross-sectional metrics + long-short proxy portfolio stats.
    Primary scalar for legacy gates: IC Pearson daily mean (same as historical IC).
    """
    if gen_df is None:
        if implementation is None:
            return "IC evaluation failed: no workspace and no dataframe.", None, {}
        try:
            _, gen_df = implementation.execute(data_type)
        except Exception as exc:  # noqa: BLE001
            return f"IC evaluation failed because factor execution raised an exception: {exc}", None, {}
    if gen_df is None or gen_df.empty:
        return "IC evaluation skipped because no valid factor dataframe was generated.", None, {}

    raw_quality = _compute_raw_factor_value_quality(gen_df)

    data_folder = (
        Path(FACTOR_COSTEER_SETTINGS.data_folder_debug)
        if data_type == "Debug"
        else Path(FACTOR_COSTEER_SETTINGS.data_folder)
    )
    try:
        label = _get_daily_label_from_data_folder(data_folder)
        merged = _merge_factor_and_label(gen_df, label)
        metrics = _compute_extended_metrics(merged)
        metrics.update(raw_quality)
        metrics["top_frac"] = 0.2
        metrics["bottom_frac"] = 0.2
        ic_mean = metrics.get("ic_mean_pearson")
    except Exception as exc:  # noqa: BLE001
        merged_metrics = dict(raw_quality)
        merged_metrics["error"] = f"label_merge_failed: {exc}"
        qual = (
            f" (Generated factor column: non_null={raw_quality.get('factor_non_null')}, "
            f"nan_ratio={raw_quality.get('factor_nan_ratio')}, warnings={raw_quality.get('factor_quality_warnings')})"
        )
        return (
            f"IC evaluation failed while preparing labels or computing correlation: {exc}{qual}",
            None,
            merged_metrics,
        )

    threshold = FACTOR_COSTEER_SETTINGS.min_abs_ic

    if metrics.get("error") or ic_mean is None:
        base = (
            "IC evaluation could not produce a valid value. The factor may have too few aligned daily observations "
            "or no cross-sectional variation."
        )
        return base + "\n" + _format_extended_metrics_feedback(metrics, threshold=threshold), None, metrics

    ic_scalar = float(ic_mean)
    ext_block = _format_extended_metrics_feedback(metrics, threshold=threshold)

    if abs(ic_scalar) >= threshold:
        base = (
            f"The factor has mean daily IC={ic_scalar:.6f}, which passes the minimum absolute IC threshold {threshold:.4f}."
        )
        return base + "\n" + ext_block, ic_scalar, metrics
    if os.environ.get("RDAGENT_PAPER_FACTOR_SKIP_LOW_IC_REPAIR", "").strip().lower() in {"1", "true", "yes", "on"}:
        base = (
            f"The factor has mean daily IC={ic_scalar:.6f}, which is below the minimum absolute IC threshold {threshold:.4f}. "
            "For paper-factor reproduction, treat this as a terminal rejected reproduction instead of revising the formula."
        )
        return base + "\n" + ext_block, ic_scalar, metrics
    base = (
        f"The factor has mean daily IC={ic_scalar:.6f}, which is below the minimum absolute IC threshold {threshold:.4f}. "
        "Revise the factor logic or aggregation method."
    )
    return base + "\n" + ext_block, ic_scalar, metrics


def evaluate_factor_ic_from_workspace(
    implementation: Workspace,
    *,
    data_type: str = "Debug",
    gen_df: pd.DataFrame | None = None,
) -> tuple[str, float | None]:
    feedback, ic, _metrics = evaluate_factor_metrics_bundle(
        implementation,
        data_type=data_type,
        gen_df=gen_df,
    )
    return feedback, ic


class FactorICEvaluator(FactorEvaluator):
    def evaluate(
        self,
        implementation: Workspace,
        gt_implementation: Workspace = None,
        data_type: str = "Debug",
    ) -> Tuple[str, object]:
        feedback, ic = evaluate_factor_ic_from_workspace(implementation, data_type=data_type)
        if ic is None:
            return feedback, False
        return feedback, abs(ic) >= FACTOR_COSTEER_SETTINGS.min_abs_ic


class FactorRowCountEvaluator(FactorEvaluator):
    def evaluate(
        self,
        implementation: Workspace,
        gt_implementation: Workspace,
    ) -> Tuple[str, object]:
        gt_df, gen_df = self._get_df(gt_implementation, implementation)
        if gen_df is None:
            return (
                "The source dataframe is None. Please check the implementation.",
                False,
            )
        ratio = min(len(gen_df), len(gt_df)) / max(len(gen_df), len(gt_df))
        return (
            (
                f"The ratio of rows count in the source dataframe to the ground truth dataframe is {ratio:.2f}. "
                + "Please verify the implementation. "
                if ratio <= 0.99
                else ""
            ),
            ratio,
        )


class FactorIndexEvaluator(FactorEvaluator):
    def evaluate(
        self,
        implementation: Workspace,
        gt_implementation: Workspace,
    ) -> Tuple[str, object]:
        gt_df, gen_df = self._get_df(gt_implementation, implementation)
        if gen_df is None:
            return (
                "The source dataframe is None. Please check the implementation.",
                False,
            )
        gen_index_set, gt_index_set = set(gen_df.index), set(gt_df.index)
        similarity = len(gen_index_set.intersection(gt_index_set)) / len(gen_index_set.union(gt_index_set))
        return (
            (
                f"The source dataframe and the ground truth dataframe have different index with a similarity of {similarity:.2%}. The similarity is calculated by the number of shared indices divided by the union indices. "
                + "Please check the implementation."
                if similarity <= 0.99
                else ""
            ),
            similarity,
        )


class FactorMissingValuesEvaluator(FactorEvaluator):
    def evaluate(
        self,
        implementation: Workspace,
        gt_implementation: Workspace,
    ) -> Tuple[str, object]:
        gt_df, gen_df = self._get_df(gt_implementation, implementation)
        if gen_df is None:
            return (
                "The source dataframe is None. Please check the implementation.",
                False,
            )
        if gen_df.isna().sum().sum() == gt_df.isna().sum().sum():
            return "Both dataframes have the same missing values.", True
        else:
            return (
                f"The dataframes do not have the same missing values. The source dataframe has {gen_df.isna().sum().sum()} missing values, while the ground truth dataframe has {gt_df.isna().sum().sum()} missing values. Please check the implementation.",
                False,
            )


class FactorEqualValueRatioEvaluator(FactorEvaluator):
    def evaluate(
        self,
        implementation: Workspace,
        gt_implementation: Workspace,
    ) -> Tuple[str, object]:
        gt_df, gen_df = self._get_df(gt_implementation, implementation)
        if gen_df is None:
            return (
                "The source dataframe is None. Please check the implementation.",
                -1,
            )
        try:
            close_values = gen_df.sub(gt_df).abs().lt(1e-6)
            result_int = close_values.astype(int)
            pos_num = result_int.sum().sum()
            acc_rate = pos_num / close_values.size
        except:
            close_values = gen_df
        if close_values.all().iloc[0]:
            return (
                "All values in the dataframes are equal within the tolerance of 1e-6.",
                acc_rate,
            )
        else:
            return (
                "Some values differ by more than the tolerance of 1e-6. Check for rounding errors or differences in the calculation methods.",
                acc_rate,
            )


class FactorCorrelationEvaluator(FactorEvaluator):
    def __init__(self, hard_check: bool, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.hard_check = hard_check

    def evaluate(
        self,
        implementation: Workspace,
        gt_implementation: Workspace,
    ) -> Tuple[str, object]:
        gt_df, gen_df = self._get_df(gt_implementation, implementation)
        if gen_df is None:
            return (
                "The source dataframe is None. Please check the implementation.",
                False,
            )
        concat_df = pd.concat([gen_df, gt_df], axis=1)
        concat_df.columns = ["source", "gt"]
        ic = concat_df.groupby("datetime").apply(lambda df: df["source"].corr(df["gt"])).dropna().mean()
        ric = (
            concat_df.groupby("datetime")
            .apply(lambda df: df["source"].corr(df["gt"], method="spearman"))
            .dropna()
            .mean()
        )

        if self.hard_check:
            if ic > 0.99 and ric > 0.99:
                return (
                    f"The dataframes are highly correlated. The ic is {ic:.6f} and the rankic is {ric:.6f}.",
                    True,
                )
            else:
                return (
                    f"The dataframes are not sufficiently high correlated. The ic is {ic:.6f} and the rankic is {ric:.6f}. Investigate the factors that might be causing the discrepancies and ensure that the logic of the factor calculation is consistent.",
                    False,
                )
        else:
            return f"The ic is ({ic:.6f}) and the rankic is ({ric:.6f}).", ic


class FactorValueEvaluator(FactorEvaluator):
    def evaluate(
        self,
        implementation: Workspace,
        gt_implementation: Workspace,
        version: int = 1,  # 1 for qlib factors and 2 for kaggle factors
        **kwargs,
    ) -> Tuple:
        conclusions = []

        # Initialize result variables
        row_result = 0
        index_result = 0
        output_format_result = None
        equal_value_ratio_result = 0
        high_correlation_result = False
        row_result = None

        # Check if both dataframe has only one columns Mute this since factor task might generate more than one columns now
        if version == 1:
            feedback_str, _ = FactorSingleColumnEvaluator(self.scen).evaluate(implementation, gt_implementation)
            conclusions.append(feedback_str)
        elif version == 2:
            input_shape = self.scen.input_shape
            _, gen_df = self._get_df(gt_implementation, implementation)
            if gen_df.shape[-1] > input_shape[-1]:
                conclusions.append(
                    "Output dataframe has more columns than input feature which is not acceptable in feature processing tasks. Please check the implementation to avoid generating too many columns. Consider this implementation as a failure."
                )

        feedback_str, numeric_check_result = FactorNumericValueEvaluator(self.scen).evaluate(
            implementation, gt_implementation
        )
        conclusions.append(feedback_str)

        feedback_str, inf_evaluate_res = FactorInfEvaluator(self.scen).evaluate(implementation, gt_implementation)
        conclusions.append(feedback_str)

        if version == 1:
            feedback_str, ic_check_result = FactorICEvaluator(self.scen).evaluate(
                implementation,
                gt_implementation,
                data_type="All",
            )
            conclusions.append(feedback_str)
        else:
            ic_check_result = None

        # Check if the index of the dataframe is ("datetime", "instrument")
        feedback_str, _ = FactorOutputFormatEvaluator(self.scen).evaluate(implementation, gt_implementation)
        conclusions.append(feedback_str)
        if version == 1:
            feedback_str, daily_check_result = FactorDatetimeDailyEvaluator(self.scen).evaluate(
                implementation, gt_implementation
            )
            conclusions.append(feedback_str)
        else:
            daily_check_result = None

        # Check dataframe format
        if gt_implementation is not None:
            feedback_str, row_result = FactorRowCountEvaluator(self.scen).evaluate(implementation, gt_implementation)
            conclusions.append(feedback_str)

            feedback_str, index_result = FactorIndexEvaluator(self.scen).evaluate(implementation, gt_implementation)
            conclusions.append(feedback_str)

            feedback_str, output_format_result = FactorMissingValuesEvaluator(self.scen).evaluate(
                implementation, gt_implementation
            )
            conclusions.append(feedback_str)

            feedback_str, equal_value_ratio_result = FactorEqualValueRatioEvaluator(self.scen).evaluate(
                implementation, gt_implementation
            )
            conclusions.append(feedback_str)

            if index_result > 0.99:
                feedback_str, high_correlation_result = FactorCorrelationEvaluator(
                    hard_check=True, scen=self.scen
                ).evaluate(implementation, gt_implementation)
            else:
                high_correlation_result = False
                feedback_str = "The source dataframe and the ground truth dataframe have different index. Give up comparing the values and correlation because it's useless"
            conclusions.append(feedback_str)

        # Combine all conclusions into a single string
        conclusion_str = "\n".join(conclusions)

        if gt_implementation is not None and (equal_value_ratio_result > 0.99) or high_correlation_result:
            decision_from_value_check = True
        elif (
            row_result is not None
            and row_result <= 0.99
            or output_format_result is False
            or daily_check_result is False
            or numeric_check_result is False
            or inf_evaluate_res is False
            or ic_check_result is False
        ):
            decision_from_value_check = False
        else:
            decision_from_value_check = None
        return conclusion_str, decision_from_value_check


class FactorFinalDecisionEvaluator(FactorEvaluator):
    def evaluate(
        self,
        target_task: FactorTask,
        execution_feedback: str,
        value_feedback: str,
        code_feedback: str,
        **kwargs,
    ) -> Tuple:
        system_prompt = T(".prompts:evaluator_final_decision_v1_system").r(
            scenario=(
                self.scen.get_scenario_all_desc(target_task, filtered_tag="feature")
                if self.scen is not None
                else "No scenario description."
            )
        )
        execution_feedback_to_render = execution_feedback

        for _ in range(10):  # 10 times to split the content is enough
            user_prompt = T(".prompts:evaluator_final_decision_v1_user").r(
                factor_information=target_task.get_task_information(),
                execution_feedback=execution_feedback_to_render,
                code_feedback=code_feedback,
                value_feedback=(
                    value_feedback
                    if value_feedback is not None
                    else "No Ground Truth Value provided, so no evaluation on value is performed."
                ),
            )
            if (
                APIBackend().build_messages_and_calculate_token(
                    user_prompt=user_prompt,
                    system_prompt=system_prompt,
                )
                > APIBackend().chat_token_limit
            ):
                execution_feedback_to_render = execution_feedback_to_render[len(execution_feedback_to_render) // 2 :]
            else:
                break

        # TODO:  with retry_context(retry_n=3, except_list=[KeyError]):
        final_evaluation_dict = None
        attempts = 0
        max_attempts = 3

        while attempts < max_attempts:
            try:
                api = APIBackend() if attempts == 0 else APIBackend(use_chat_cache=False)
                final_evaluation_dict = json.loads(
                    api.build_messages_and_create_chat_completion(
                        user_prompt=user_prompt,
                        system_prompt=system_prompt,
                        json_mode=True,
                        seed=attempts,  # in case of useless retrying when cache enabled.
                        json_target_type=Dict[str, str | bool | int],
                    ),
                )
                final_decision = final_evaluation_dict["final_decision"]
                final_feedback = final_evaluation_dict["final_feedback"]

                final_decision = str(final_decision).lower() in ["true", "1"]
                return final_decision, final_feedback

            except json.JSONDecodeError as e:
                raise ValueError("Failed to decode JSON response from API.") from e
            except KeyError as e:
                attempts += 1
                if attempts >= max_attempts:
                    raise KeyError(
                        "Response from API is missing 'final_decision' or 'final_feedback' key after multiple attempts."
                    ) from e

        return None, None
