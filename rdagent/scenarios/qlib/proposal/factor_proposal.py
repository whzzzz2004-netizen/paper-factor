import json
import re
from pathlib import Path
from typing import List, Tuple

from rdagent.components.coder.factor_coder.factor import FactorExperiment, FactorTask
from rdagent.components.proposal import FactorHypothesis2Experiment, FactorHypothesisGen
from rdagent.core.proposal import Hypothesis, Scenario, Trace
from rdagent.scenarios.qlib.experiment.factor_experiment import QlibFactorExperiment
from rdagent.scenarios.qlib.knowledge_router import (
    build_factor_generation_knowledge_summary,
)
from rdagent.utils.agent.tpl import T

QlibFactorHypothesis = Hypothesis

FACTOR_OUTPUT_DIR = Path.cwd() / "git_ignore_folder" / "factor_outputs"


def _strip_snapshot_prefix(name: str) -> str:
    return re.sub(r"^\d{8}_\d{6}__", "", name)


def _normalize_factor_name_for_dedup(name: str) -> str:
    normalized = _strip_snapshot_prefix(name).lower()
    normalized = normalized.replace("momentum", "mom").replace("reversal", "rev")
    normalized = normalized.replace("volume", "vol").replace("day", "d")
    normalized = re.sub(r"[^a-z0-9]+", "", normalized)

    match = re.search(r"(mom|rev)(\d+)d?", normalized)
    if match:
        return f"{match.group(1)}{match.group(2)}d"

    match = re.search(r"(\d+)d?(mom|rev)", normalized)
    if match:
        return f"{match.group(2)}{match.group(1)}d"

    return normalized


def _exported_factor_names(limit: int = 80) -> list[str]:
    if not FACTOR_OUTPUT_DIR.exists():
        return []
    names = {
        _strip_snapshot_prefix(path.stem)
        for path in FACTOR_OUTPUT_DIR.glob("*.parquet")
        if not path.stem.startswith(".")
    }
    return sorted(names)[:limit]


class QlibFactorHypothesisGen(FactorHypothesisGen):
    def __init__(self, scen: Scenario) -> Tuple[dict, bool]:
        super().__init__(scen)

    def prepare_context(self, trace: Trace) -> Tuple[dict, bool]:
        hypothesis_and_feedback = (
            T("scenarios.qlib.prompts:hypothesis_and_feedback").r(
                trace=trace,
            )
            if len(trace.hist) > 0
            else "No previous hypothesis and feedback available since it's the first round."
        )
        last_hypothesis_and_feedback = (
            T("scenarios.qlib.prompts:last_hypothesis_and_feedback").r(
                experiment=trace.hist[-1][0], feedback=trace.hist[-1][1]
            )
            if len(trace.hist) > 0
            else "No previous hypothesis and feedback available since it's the first round."
        )

        exported_names = _exported_factor_names()
        existing_factor_hint = (
            "\nAlready exported factor names: "
            + ", ".join(exported_names)
            + ". Do not propose the same factor again, including renamed aliases such as MOM_10D vs Momentum_10D."
            if exported_names
            else ""
        )
        knowledge_brief = build_factor_generation_knowledge_summary()

        context_dict = {
            "hypothesis_and_feedback": hypothesis_and_feedback,
            "last_hypothesis_and_feedback": last_hypothesis_and_feedback,
            "RAG": (
                "Try the easiest and fastest factors to experiment with from various perspectives first."
                if len(trace.hist) < 15
                else "Now, you need to try factors that can achieve high IC (e.g., machine learning-based factors)."
            )
            + existing_factor_hint
            + "\n\nKnowledge snippets you should actually use for factor generation:\n"
            + knowledge_brief,
            "hypothesis_output_format": T("scenarios.qlib.prompts:factor_hypothesis_output_format").r(),
            "hypothesis_specification": T("scenarios.qlib.prompts:factor_hypothesis_specification").r(),
        }
        return context_dict, True

    def convert_response(self, response: str) -> Hypothesis:
        response_dict = json.loads(response)
        hypothesis = QlibFactorHypothesis(
            hypothesis=response_dict.get("hypothesis"),
            reason=response_dict.get("reason"),
            concise_reason=response_dict.get("concise_reason"),
            concise_observation=response_dict.get("concise_observation"),
            concise_justification=response_dict.get("concise_justification"),
            concise_knowledge=response_dict.get("concise_knowledge"),
        )
        return hypothesis


class QlibFactorHypothesis2Experiment(FactorHypothesis2Experiment):
    def prepare_context(self, hypothesis: Hypothesis, trace: Trace) -> Tuple[dict | bool]:
        if trace.scen.__class__.__name__ == "QlibQuantScenario":
            scenario = trace.scen.get_scenario_all_desc(action="factor")
        else:
            scenario = trace.scen.get_scenario_all_desc()

        experiment_output_format = T("scenarios.qlib.prompts:factor_experiment_output_format").r()

        if len(trace.hist) == 0:
            hypothesis_and_feedback = "No previous hypothesis and feedback available since it's the first round."
        else:
            specific_trace = Trace(trace.scen)
            for i in range(len(trace.hist) - 1, -1, -1):
                if not hasattr(trace.hist[i][0].hypothesis, "action") or trace.hist[i][0].hypothesis.action == "factor":
                    specific_trace.hist.insert(0, trace.hist[i])
            if len(specific_trace.hist) > 0:
                specific_trace.hist.reverse()
                hypothesis_and_feedback = T("scenarios.qlib.prompts:hypothesis_and_feedback").r(
                    trace=specific_trace,
                )
            else:
                hypothesis_and_feedback = "No previous hypothesis and feedback available."

        exported_names = _exported_factor_names()
        duplicate_guard = (
            "Existing exported factor names that must not be recreated: "
            + ", ".join(exported_names)
            + ". Treat renamed aliases as duplicates; for example, MOM_10D, Momentum_10D, and 10_day_momentum are the same direction."
            if exported_names
            else None
        )
        knowledge_brief = build_factor_generation_knowledge_summary()

        return {
            "target_hypothesis": str(hypothesis),
            "scenario": scenario,
            "hypothesis_and_feedback": hypothesis_and_feedback,
            "experiment_output_format": experiment_output_format,
            "target_list": exported_names,
            "RAG": (duplicate_guard or "")
            + "\n\nKnowledge snippets you should actually use when expanding the factor hypothesis:\n"
            + knowledge_brief,
        }, True

    def convert_response(self, response: str, hypothesis: Hypothesis, trace: Trace) -> FactorExperiment:
        response_dict = json.loads(response)
        tasks = []

        for factor_name in response_dict:
            description = response_dict[factor_name]["description"]
            formulation = response_dict[factor_name]["formulation"]
            variables = response_dict[factor_name]["variables"]
            tasks.append(
                FactorTask(
                    factor_name=factor_name,
                    factor_description=description,
                    factor_formulation=formulation,
                    variables=variables,
                )
            )

        exp = QlibFactorExperiment(tasks, hypothesis=hypothesis)
        exp.based_experiments = [QlibFactorExperiment(sub_tasks=[])] + [
            t[0] for t in trace.hist if t[1] and isinstance(t[0], FactorExperiment)
        ]

        unique_tasks = []
        known_factor_keys = {_normalize_factor_name_for_dedup(name) for name in _exported_factor_names()}
        for task in tasks:
            duplicate = False
            if _normalize_factor_name_for_dedup(task.factor_name) in known_factor_keys:
                duplicate = True
            for based_exp in exp.based_experiments:
                if based_exp.__class__.__name__ == "QlibModelExperiment":
                    continue
                for sub_task in based_exp.sub_tasks:
                    if task.factor_name == sub_task.factor_name:
                        duplicate = True
                        break
                if duplicate:
                    break
            if not duplicate:
                unique_tasks.append(task)

        exp.tasks = unique_tasks
        return exp
