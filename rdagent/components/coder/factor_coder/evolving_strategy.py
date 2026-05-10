from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Dict, Generator

from rdagent.components.coder.CoSTEER.evaluators import CoSTEERSingleFeedback
from rdagent.components.coder.CoSTEER.evolving_strategy import (
    MultiProcessEvolvingStrategy,
)
from rdagent.components.coder.CoSTEER.knowledge_management import (
    CoSTEERQueriedKnowledge,
    CoSTEERQueriedKnowledgeV2,
)
from rdagent.components.coder.factor_coder.config import FACTOR_COSTEER_SETTINGS
from rdagent.components.coder.factor_coder.factor import FactorFBWorkspace, FactorTask
from rdagent.core.experiment import FBWorkspace
from rdagent.oai.llm_conf import LLM_SETTINGS
from rdagent.oai.llm_utils import APIBackend
from rdagent.utils.agent.tpl import T

# Load the factor code template
FACTOR_TEMPLATE_PATH = Path(__file__).parent / "factor_template.py"
FACTOR_CODE_TEMPLATE = FACTOR_TEMPLATE_PATH.read_text() if FACTOR_TEMPLATE_PATH.exists() else None


def get_factor_template() -> str:
    """Load the factor code template from file."""
    template_path = Path(__file__).parent / "factor_template.py"
    if template_path.exists():
        return template_path.read_text()
    return ""  # Return empty string if template doesn't exist


class FactorMultiProcessEvolvingStrategy(MultiProcessEvolvingStrategy):
    MAX_CONSECUTIVE_OUTPUT_WITHOUT_ACCEPT = int(os.environ.get("RDAGENT_FACTOR_MAX_CONSECUTIVE_OUTPUT_WITHOUT_ACCEPT", "5"))

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.num_loop = 0
        self.haveSelected = False

    @classmethod
    def _has_output_but_not_accepted(cls, feedback: CoSTEERSingleFeedback | None) -> bool:
        if feedback is None:
            return False
        value_generated_flag = getattr(feedback, "value_generated_flag", False)
        final_decision = getattr(feedback, "final_decision", None)
        return bool(value_generated_flag) and final_decision is False

    @classmethod
    def _mark_stalled_tasks_from_trace(cls, evo, queried_knowledge, evolving_trace) -> None:
        if queried_knowledge is None or not evolving_trace:
            return

        for task_index, target_task in enumerate(evo.sub_tasks):
            task_info = target_task.get_task_information()
            consecutive_failures = 0
            for step in reversed(evolving_trace):
                fb = None
                if step.feedback is not None and task_index < len(step.feedback):
                    fb = step.feedback[task_index]
                if cls._has_output_but_not_accepted(fb):
                    consecutive_failures += 1
                    if consecutive_failures >= cls.MAX_CONSECUTIVE_OUTPUT_WITHOUT_ACCEPT:
                        if cls.MAX_CONSECUTIVE_OUTPUT_WITHOUT_ACCEPT <= 0:
                            break
                        queried_knowledge.failed_task_info_set.add(task_info)
                        break
                else:
                    break

    def evolve_iter(
        self,
        *,
        evo,
        queried_knowledge: CoSTEERQueriedKnowledge | None = None,
        evolving_trace=[],
        **kwargs,
    ) -> Generator:
        self._mark_stalled_tasks_from_trace(evo, queried_knowledge, evolving_trace)
        yield from super().evolve_iter(
            evo=evo,
            queried_knowledge=queried_knowledge,
            evolving_trace=evolving_trace,
            **kwargs,
        )

    def error_summary(
        self,
        target_task: FactorTask,
        queried_former_failed_knowledge_to_render: list,
        queried_similar_error_knowledge_to_render: list,
    ) -> str:
        error_summary_system_prompt = T(".prompts:evolving_strategy_error_summary_v2_system").r(
            scenario=self.scen.get_scenario_all_desc(target_task),
            factor_information_str=target_task.get_task_information(),
            code_and_feedback=queried_former_failed_knowledge_to_render[-1].get_implementation_and_feedback_str(),
        )
        for _ in range(10):  # max attempt to reduce the length of error_summary_user_prompt
            error_summary_user_prompt = T(".prompts:evolving_strategy_error_summary_v2_user").r(
                queried_similar_error_knowledge=queried_similar_error_knowledge_to_render,
            )
            if (
                APIBackend().build_messages_and_calculate_token(
                    user_prompt=error_summary_user_prompt, system_prompt=error_summary_system_prompt
                )
                < APIBackend().chat_token_limit
            ):
                break
            elif len(queried_similar_error_knowledge_to_render) > 0:
                queried_similar_error_knowledge_to_render = queried_similar_error_knowledge_to_render[:-1]
        error_summary_critics = APIBackend(
            use_chat_cache=FACTOR_COSTEER_SETTINGS.coder_use_cache
        ).build_messages_and_create_chat_completion(
            user_prompt=error_summary_user_prompt, system_prompt=error_summary_system_prompt, json_mode=False
        )
        return error_summary_critics

    def implement_one_task(
        self,
        target_task: FactorTask,
        queried_knowledge: CoSTEERQueriedKnowledge,
        workspace: FBWorkspace | None = None,
        prev_task_feedback: CoSTEERSingleFeedback | None = None,
    ) -> str:
        def _extract_code(candidate: str) -> str:
            candidate = candidate.strip()
            if not candidate:
                raise ValueError("Empty model response.")

            try:
                parsed = json.loads(candidate)
            except json.JSONDecodeError:
                parsed = None

            if isinstance(parsed, dict):
                code_value = parsed.get("code")
                if isinstance(code_value, str) and code_value.strip():
                    return code_value

            match = re.search(r"```python(.*?)```", candidate, re.DOTALL | re.IGNORECASE)
            if match:
                return _extract_code(match.group(1))

            generic_match = re.search(r"```(?:[a-zA-Z0-9_+-]*)?\n(.*?)```", candidate, re.DOTALL)
            if generic_match:
                return _extract_code(generic_match.group(1))

            looks_like_python = any(
                token in candidate
                for token in (
                    "import pandas",
                    "import numpy",
                    "import torch",
                    "from pathlib import Path",
                    "def calculate_",
                    "to_hdf(",
                )
            )
            if looks_like_python:
                return candidate

            raise ValueError("Unable to extract executable Python code from model response.")

        target_factor_task_information = target_task.get_task_information()

        queried_similar_successful_knowledge = []

        if isinstance(queried_knowledge, CoSTEERQueriedKnowledgeV2):
            queried_similar_error_knowledge = (
                queried_knowledge.task_to_similar_error_successful_knowledge[target_factor_task_information]
                if queried_knowledge is not None
                else {}
            )  # A dict, {{error_type:[[error_imp_knowledge, success_imp_knowledge],...]},...}
        else:
            queried_similar_error_knowledge = {}

        queried_former_failed_knowledge = (
            queried_knowledge.task_to_former_failed_traces[target_factor_task_information][0]
            if queried_knowledge is not None
            else []
        )

        queried_former_failed_knowledge_to_render = queried_former_failed_knowledge

        latest_attempt_to_latest_successful_execution = queried_knowledge.task_to_former_failed_traces[
            target_factor_task_information
        ][1]
        system_prompt = T(".prompts:evolving_strategy_factor_implementation_v1_system").r(
            scenario=self.scen.get_scenario_all_desc(target_task, filtered_tag="feature"),
            queried_former_failed_knowledge=queried_former_failed_knowledge_to_render,
        )
        queried_similar_successful_knowledge_to_render = queried_similar_successful_knowledge
        queried_similar_error_knowledge_to_render = queried_similar_error_knowledge
        # 动态地防止prompt超长
        for _ in range(10):  # max attempt to reduce the length of user_prompt
            # 总结error（可选）
            if (
                isinstance(queried_knowledge, CoSTEERQueriedKnowledgeV2)
                and FACTOR_COSTEER_SETTINGS.v2_error_summary
                and len(queried_similar_error_knowledge_to_render) != 0
                and len(queried_former_failed_knowledge_to_render) != 0
            ):
                error_summary_critics = self.error_summary(
                    target_task,
                    queried_former_failed_knowledge_to_render,
                    queried_similar_error_knowledge_to_render,
                )
            else:
                error_summary_critics = None
            # 构建user_prompt。开始写代码
            user_prompt = T(".prompts:evolving_strategy_factor_implementation_v2_user").r(
                factor_information_str=target_factor_task_information,
                prev_task_feedback=str(prev_task_feedback) if prev_task_feedback is not None else None,
                queried_similar_successful_knowledge=queried_similar_successful_knowledge_to_render,
                queried_similar_error_knowledge=queried_similar_error_knowledge_to_render,
                error_summary_critics=error_summary_critics,
                latest_attempt_to_latest_successful_execution=latest_attempt_to_latest_successful_execution,
            )
            if (
                APIBackend().build_messages_and_calculate_token(user_prompt=user_prompt, system_prompt=system_prompt)
                < APIBackend().chat_token_limit
            ):
                break
            elif len(queried_former_failed_knowledge_to_render) > 1:
                queried_former_failed_knowledge_to_render = queried_former_failed_knowledge_to_render[1:]
            elif len(queried_similar_error_knowledge_to_render) > 0:
                queried_similar_error_knowledge_to_render = queried_similar_error_knowledge_to_render[:-1]
        for _ in range(10):
            try:
                response = APIBackend(
                    use_chat_cache=FACTOR_COSTEER_SETTINGS.coder_use_cache
                ).build_messages_and_create_chat_completion(
                    user_prompt=user_prompt,
                    system_prompt=system_prompt,
                    # Do not force backend-level JSON parsing here.
                    # DeepSeek frequently returns valid code with non-standard JSON wrappers,
                    # and strict json_mode may fail before our local fallbacks can run.
                    json_mode=False,
                )
                code = _extract_code(response)

                if not isinstance(code, str) or not code.strip():
                    raise ValueError("Empty code extracted from model response.")

                return code

            except (json.decoder.JSONDecodeError, KeyError, ValueError):
                pass
        else:
            return ""  # return empty code if failed to get code after 10 attempts

    def assign_code_list_to_evo(self, code_list, evo):
        for index in range(len(evo.sub_tasks)):
            if code_list[index] is None:
                continue
            if evo.sub_workspace_list[index] is None:
                evo.sub_workspace_list[index] = FactorFBWorkspace(target_task=evo.sub_tasks[index])
            # Since the `implement_one_task` method is not standardized and the `code_list` has both `str` and `dict` data types,
            # we ended up getting an `TypeError` here, so we chose to fix the problem temporarily with this dirty method.
            if isinstance(code_list[index], dict):
                evo.sub_workspace_list[index].inject_files(**code_list[index])
            else:
                evo.sub_workspace_list[index].inject_files(**{"factor.py": code_list[index]})
        return evo
