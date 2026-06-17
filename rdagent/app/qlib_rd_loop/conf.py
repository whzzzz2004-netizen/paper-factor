from typing import Optional

from pydantic_settings import SettingsConfigDict

from rdagent.components.workflow.conf import BasePropSetting


class FactorBasePropSetting(BasePropSetting):
    model_config = SettingsConfigDict(env_prefix="QLIB_FACTOR_", protected_namespaces=())

    # 1) override base settings
    scen: str = "rdagent.scenarios.qlib.experiment.factor_experiment.QlibFactorScenario"
    """Scenario class for Qlib Factor"""

    coder: str = "rdagent.components.coder.factor_coder.FactorCoSTEER"
    """Coder class"""

    evolving_n: int = 10
    """Number of evolutions"""

    train_start: str = "2018-01-01"
    """Start date of the training segment"""

    train_end: str = "2022-12-31"
    """End date of the training segment"""

    valid_start: str = "2023-01-01"
    """Start date of the validation segment"""

    valid_end: str = "2023-12-31"
    """End date of the validation segment"""

    test_start: str = "2024-01-01"
    """Start date of the test / backtest segment"""

    test_end: Optional[str] = "2026-12-31"
    """End date of the test / backtest segment"""


class FactorFromReportPropSetting(FactorBasePropSetting):
    # 1) override the scen attribute
    scen: str = "rdagent.scenarios.qlib.experiment.factor_from_report_experiment.QlibFactorFromReportScenario"
    """Scenario class for Qlib Factor from Report"""

    # 2) data source:
    factor_data_dir: str = "git_ignore_folder/factor_implementation_source_data_1000"
    """Path to factor source data directory (use factor_implementation_source_data for full dataset)"""

    # 3) sub task specific:
    report_result_json_file_path: str = "git_ignore_folder/report_list.json"
    """Path to the JSON file listing research reports for factor extraction"""

    max_factors_per_exp: int = 15
    """Maximum number of factors implemented per experiment"""

    report_limit: int = 20
    """Maximum number of reports to process"""


FACTOR_PROP_SETTING = FactorBasePropSetting()
FACTOR_FROM_REPORT_PROP_SETTING = FactorFromReportPropSetting()
