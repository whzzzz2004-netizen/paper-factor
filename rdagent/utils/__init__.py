"""
This is some common utils functions.
it is not binding to the scenarios or framework (So it is not placed in rdagent.core.utils)
"""

# TODO: merge the common utils in `rdagent.core.utils` into this folder
# TODO: split the utils in this module into different modules in the future.

import hashlib
import importlib
import json
import re
import sys
from pathlib import Path
from types import ModuleType
from typing import Union

import regex  # type: ignore[import-untyped]

from rdagent.log import rdagent_logger as logger
from rdagent.oai.llm_conf import LLM_SETTINGS

# Default timeout (in seconds) for all regex operations
REGEX_TIMEOUT = 120.0


def get_module_by_module_path(module_path: Union[str, ModuleType]) -> ModuleType:
    """Load module from path like a/b/c/d.py or a.b.c.d

    :param module_path:
    :return:
    :raises: ModuleNotFoundError
    """
    if module_path is None:
        raise ModuleNotFoundError("None is passed in as parameters as module_path")

    if isinstance(module_path, ModuleType):
        module = module_path
    else:
        if module_path.endswith(".py"):
            module_name = re.sub("^[^a-zA-Z_]+", "", re.sub("[^0-9a-zA-Z_]", "", module_path[:-3].replace("/", "_")))
            module_spec = importlib.util.spec_from_file_location(module_name, module_path)
            if module_spec is None:
                raise ModuleNotFoundError(f"Cannot find module at {module_path}")
            module = importlib.util.module_from_spec(module_spec)
            sys.modules[module_name] = module
            if module_spec.loader is not None:
                module_spec.loader.exec_module(module)
            else:
                raise ModuleNotFoundError(f"Cannot load module at {module_path}")
        else:
            module = importlib.import_module(module_path)
    return module


def convert2bool(value: Union[str, bool]) -> bool:
    """
    Motivation: the return value of LLM is not stable. Try to convert the value into bool
    """
    # TODO: if we have more similar functions, we can build a library to converting unstable LLM response to stable results.
    if isinstance(value, str):
        v = value.lower().strip()
        if v in ["true", "yes", "ok"]:
            return True
        if v in ["false", "no"]:
            return False
        raise ValueError(f"Can not convert {value} to bool")
    elif isinstance(value, bool):
        return value
    else:
        raise ValueError(f"Unknown value type {value} to bool")


def try_regex_sub(pattern: str, text: str, replace_with: str = "", flags: int = 0) -> str:
    """
    Try to sub a regex pattern against a text string.
    """
    try:
        text = regex.sub(pattern, replace_with, text, timeout=REGEX_TIMEOUT, flags=flags)
    except TimeoutError:
        logger.warning(f"Pattern '{pattern}' timed out after {REGEX_TIMEOUT} seconds; skipping it.")
    except Exception as e:
        logger.warning(f"Pattern '{pattern}' raised an error: {e}; skipping it.")
    return text


def filter_with_time_limit(regex_patterns: Union[str, list[str]], text: str) -> str:
    """
    Apply one or more regex patterns to filter `text`, using a timeout for each substitution.
    If `regex_patterns` is a list, they are applied sequentially; if a single string, only that pattern is applied.
    """
    if not isinstance(regex_patterns, list):
        regex_patterns = [regex_patterns]
    for pattern in regex_patterns:
        text = try_regex_sub(pattern, text)
    return text


def filter_redundant_text(stdout: str) -> str:
    """
    Filter out progress bars and other redundant patterns from stdout using regex-based trimming.
    NOTE: This function uses rdagent.utils.agent.tpl.T which has been removed.
          Kept as stub for backward compatibility.
    """
    return stdout


def remove_path_info_from_str(base_path: Path, target_string: str) -> str:
    """
    Remove the absolute path from the target string
    """
    target_string = re.sub(str(base_path), "...", target_string)
    target_string = re.sub(str(base_path.absolute()), "...", target_string)
    return target_string


def md5_hash(input_string: str) -> str:
    hash_md5 = hashlib.md5(usedforsecurity=False)
    input_bytes = input_string.encode("utf-8")
    hash_md5.update(input_bytes)
    return hash_md5.hexdigest()
