from copy import deepcopy
from datetime import datetime

from rdagent.components.coder.CoSTEER import CoSTEER
from rdagent.components.coder.CoSTEER.evaluators import CoSTEERMultiEvaluator
from rdagent.components.coder.CoSTEER.evolvable_subjects import EvolvingItem
from rdagent.components.coder.factor_coder.config import FACTOR_COSTEER_SETTINGS
from rdagent.components.coder.factor_coder.evaluators import FactorEvaluatorForCoder
from rdagent.components.coder.factor_coder.evolving_strategy import (
    FactorMultiProcessEvolvingStrategy,
)
from rdagent.core.evolving_agent import RAGEvoAgent
from rdagent.core.experiment import Experiment
from rdagent.core.scenario import Scenario
from rdagent.log import rdagent_logger as logger
from rdagent.oai.backend.base import RD_Agent_TIMER_wrapper


class FactorCoSTEER(CoSTEER):
    def __init__(
        self,
        scen: Scenario,
        *args,
        **kwargs,
    ) -> None:
        setting = FACTOR_COSTEER_SETTINGS
        eva = CoSTEERMultiEvaluator(FactorEvaluatorForCoder(scen=scen), scen=scen)
        es = FactorMultiProcessEvolvingStrategy(scen=scen, settings=FACTOR_COSTEER_SETTINGS)

        super().__init__(*args, settings=setting, eva=eva, es=es, evolving_version=2, scen=scen, **kwargs)

    def develop(self, exp: Experiment) -> Experiment:
        # Override to break early when all tasks succeed on first try.
        max_seconds = self.get_develop_max_seconds()
        evo_exp = EvolvingItem.from_experiment(exp)

        evolve_agent = RAGEvoAgent[EvolvingItem](
            max_loop=self.max_loop,
            evolving_strategy=self.evolving_strategy,
            rag=self.rag,
            with_knowledge=self.with_knowledge,
            knowledge_self_gen=self.knowledge_self_gen,
            enable_filelock=self.settings.enable_filelock,
            filelock_path=self.settings.filelock_path,
            stop_eval_chain_on_fail=self.stop_eval_chain_on_fail,
        )

        start_datetime = datetime.now()
        fallback_evo_exp = None
        fallback_evo_fb = None
        reached_max_seconds = False

        evo_fb = None
        for evo_exp in evolve_agent.multistep_evolve(evo_exp, self.evaluator):
            evo_fb = self._get_last_fb(evolve_agent)
            update_fallback = self.should_use_new_evo(base_fb=fallback_evo_fb, new_fb=evo_fb)
            if update_fallback:
                fallback_evo_exp = deepcopy(evo_exp)
                fallback_evo_fb = deepcopy(evo_fb)
                fallback_evo_exp.create_ws_ckp()

            logger.log_object(evo_exp.sub_workspace_list, tag="evolving code")

            # Early exit: if all tasks are acceptable, no need to iterate further
            if evo_fb is not None and evo_fb.is_acceptable():
                logger.info("All tasks accepted on first success, skipping remaining iterations.")
                break

            if max_seconds is not None and (datetime.now() - start_datetime).total_seconds() > max_seconds:
                logger.info(f"Reached max time limit {max_seconds} seconds, stop evolving")
                reached_max_seconds = True
                break
            if RD_Agent_TIMER_wrapper.timer.started and RD_Agent_TIMER_wrapper.timer.is_timeout():
                logger.info("Global timer is timeout, stop evolving")
                break

        try:
            if fallback_evo_exp is not None:
                evo_exp = fallback_evo_exp
                evo_exp.recover_ws_ckp()
                evo_fb = fallback_evo_fb
            assert evo_fb is not None
            evo_exp = self._exp_postprocess_by_feedback(evo_exp, evo_fb)
        except Exception as e:
            if hasattr(e, 'caused_by_timeout'):
                e.caused_by_timeout = reached_max_seconds
            raise

        exp.sub_workspace_list = evo_exp.sub_workspace_list
        exp.experiment_workspace = evo_exp.experiment_workspace
        exp._evolving_trace = evolve_agent.evolving_trace

        try:
            evolving_trace = getattr(exp, "_evolving_trace", None)
            if evolving_trace:
                es = evolving_trace[-1]
                exp.prop_dev_feedback = es.feedback
        except Exception:
            pass
        return exp
