import copyreg
import json
import socket
import traceback
from typing import Any, Literal, Optional, Type, TypedDict, Union, cast

import numpy as np
import requests
from litellm import (
    completion,
    completion_cost,
    embedding,
    get_model_info,
    supports_function_calling,
    supports_response_schema,
    token_counter,
)
from litellm.exceptions import BadRequestError, Timeout
from pydantic import BaseModel

from rdagent.log import LogColors
from rdagent.log import rdagent_logger as logger
from rdagent.oai.backend.base import APIBackend
from rdagent.oai.llm_conf import LLMSettings


# NOTE: Patching! Otherwise, the exception will call the constructor and with following error:
# `BadRequestError.__init__() missing 2 required positional arguments: 'model' and 'llm_provider'`
def _reduce_no_init(exc: Exception) -> tuple:
    cls = exc.__class__
    return (cls.__new__, (cls,), exc.__dict__)


# suppose you want to apply this to MyError
for cls in [BadRequestError, Timeout]:
    copyreg.pickle(cls, _reduce_no_init)


class LiteLLMSettings(LLMSettings):

    class Config:
        env_prefix = "LITELLM_"
        """Use `LITELLM_` as prefix for environment variables"""

    # Placeholder for LiteLLM specific settings, so far it's empty


LITELLM_SETTINGS = LiteLLMSettings()
ACC_COST = 0.0


class LiteLLMAPIBackend(APIBackend):
    """LiteLLM implementation of APIBackend interface"""

    _has_logged_settings: bool = False
    _response_schema_warned_models: set[str] = set()

    @staticmethod
    def _uses_openai_compatible_custom_endpoint() -> bool:
        api_base = LITELLM_SETTINGS.chat_openai_base_url or LITELLM_SETTINGS.openai_api_base or ""
        return "compatible-mode" in api_base

    @staticmethod
    def _approx_token_count_from_messages(messages: list[dict[str, Any]]) -> int:
        text = "\n".join(str(message.get("content", "")) for message in messages)
        return max(1, len(text) // 4)

    @staticmethod
    def _safe_supports_response_schema(model: str) -> bool:
        if LiteLLMAPIBackend._uses_openai_compatible_custom_endpoint():
            return False
        try:
            return bool(supports_response_schema(model=model))
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                f"LiteLLM could not infer response-schema support for model {model}: {exc}. "
                "Fallback to response_format disabled."
            )
            return False

    @staticmethod
    def _compose_provider_model(model: str, custom_llm_provider: str | None) -> str:
        if custom_llm_provider:
            return f"{custom_llm_provider}/{model}"
        return model

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if not self.__class__._has_logged_settings:
            logger.info(f"{LITELLM_SETTINGS}")
            logger.log_object(LITELLM_SETTINGS.model_dump(), tag="LITELLM_SETTINGS")
            self.__class__._has_logged_settings = True
        super().__init__(*args, **kwargs)

    def _calculate_token_from_messages(self, messages: list[dict[str, Any]]) -> int:
        """
        Calculate the token count from messages
        """
        if self._uses_openai_compatible_custom_endpoint():
            num_tokens = self._approx_token_count_from_messages(messages)
            logger.info(
                f"{LogColors.CYAN}Token count (approx for compatible endpoint): {LogColors.END} {num_tokens}",
                tag="debug_litellm_token",
            )
            return num_tokens
        try:
            num_tokens = token_counter(
                model=LITELLM_SETTINGS.chat_model,
                messages=messages,
            )
        except Exception as exc:  # noqa: BLE001
            num_tokens = self._approx_token_count_from_messages(messages)
            logger.warning(
                f"LiteLLM token counting failed for model {LITELLM_SETTINGS.chat_model}: {exc}. "
                f"Fallback approximate token count={num_tokens}."
            )
        logger.info(f"{LogColors.CYAN}Token count: {LogColors.END} {num_tokens}", tag="debug_litellm_token")
        return num_tokens

    def _create_embedding_inner_function(self, input_content_list: list[str]) -> list[list[float]]:
        """
        Call the embedding function
        """
        model_name = LITELLM_SETTINGS.embedding_model
        api_base = LITELLM_SETTINGS.embedding_openai_base_url or LITELLM_SETTINGS.openai_api_base or None
        api_key = LITELLM_SETTINGS.embedding_openai_api_key or LITELLM_SETTINGS.openai_api_key or None
        logger.info(f"{LogColors.GREEN}Using emb model{LogColors.END} {model_name}", tag="debug_litellm_emb")
        if LITELLM_SETTINGS.log_llm_chat_content:
            logger.info(
                f"{LogColors.MAGENTA}Creating embedding{LogColors.END} for: {input_content_list}",
                tag="debug_litellm_emb",
            )
        batch_size = self._get_embedding_batch_size(model_name=model_name, api_base=api_base)
        content_to_embedding_dict = {}
        for batch in [input_content_list[i : i + batch_size] for i in range(0, len(input_content_list), batch_size)]:
            if api_base and "dashscope.aliyuncs.com" in api_base and model_name.startswith("openai/"):
                response_list = self._create_dashscope_embedding(batch, model_name, api_base, api_key)
            else:
                response = embedding(
                    model=model_name,
                    input=batch,
                    # DashScope's OpenAI-compatible embedding endpoint rejects the
                    # implicit/default encoding format from the OpenAI SDK path.
                    encoding_format="float",
                    api_base=api_base,
                    api_key=api_key,
                )
                response_list = [data["embedding"] for data in response.data]
            for idx, content in enumerate(batch):
                content_to_embedding_dict[content] = response_list[idx]
        return [content_to_embedding_dict[content] for content in input_content_list]

    @staticmethod
    def _get_embedding_batch_size(model_name: str, api_base: str | None) -> int:
        default_batch_size = max(1, LITELLM_SETTINGS.embedding_max_str_num)
        normalized_model_name = model_name.split("/", 1)[-1] if "/" in model_name else model_name
        if api_base and "dashscope.aliyuncs.com" in api_base:
            # DashScope's text-embedding-v3/v4 OpenAI-compatible endpoint only
            # accepts up to 10 strings per request.
            if normalized_model_name in {"text-embedding-v3", "text-embedding-v4"}:
                return min(default_batch_size, 10)
        return default_batch_size

    @staticmethod
    def _trim_dashscope_embedding_inputs(input_content_list: list[str], max_chars: int) -> list[str]:
        trimmed = []
        for content in input_content_list:
            if len(content) > max_chars:
                logger.warning(
                    f"DashScope embedding input is too long ({len(content)} chars); truncating to {max_chars} chars."
                )
                trimmed.append(content[:max_chars])
            else:
                trimmed.append(content)
        return trimmed

    @staticmethod
    def _create_dashscope_embedding(
        input_content_list: list[str],
        model_name: str,
        api_base: str,
        api_key: str | None,
    ) -> list[list[float]]:
        if not api_key:
            raise ValueError("DashScope embedding requires an API key.")
        if not api_key.isascii():
            non_ascii_positions = [idx for idx, char in enumerate(api_key) if ord(char) > 127][:5]
            raise ValueError(
                "DashScope embedding API key contains non-ASCII characters. "
                f"Please replace placeholder text with a real key. non_ascii_positions={non_ascii_positions}"
            )

        host = api_base.split("//", 1)[-1].split("/", 1)[0]
        try:
            socket.getaddrinfo(host, 443)
        except OSError as exc:
            raise RuntimeError(f"DashScope DNS resolution failed for {host}: {exc}") from exc

        endpoint = api_base.rstrip("/") + "/embeddings"
        model = model_name.split("/", 1)[1]
        max_chars = min(LITELLM_SETTINGS.embedding_max_length, 4096)
        request_input_content_list = LiteLLMAPIBackend._trim_dashscope_embedding_inputs(
            input_content_list,
            max_chars=max_chars,
        )
        session = requests.Session()
        session.trust_env = False

        def post_embedding(contents: list[str]) -> requests.Response:
            payload = {
                "model": model,
                "input": contents,
                "encoding_format": "float",
            }
            try:
                return session.post(
                    endpoint,
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=60,
                )
            except UnicodeEncodeError as exc:
                raise RuntimeError(
                    "DashScope request failed while encoding HTTP headers. "
                    f"api_key_ascii={api_key.isascii()}, api_key_prefix={api_key[:6]!r}, "
                    f"traceback={traceback.format_exc(limit=3)}"
                ) from exc

        response = post_embedding(request_input_content_list)
        if response.status_code == 400:
            logger.warning(
                "DashScope embedding returned 400; retrying once with 2048-char truncated inputs. "
                f"response={response.text[:500]}"
            )
            request_input_content_list = LiteLLMAPIBackend._trim_dashscope_embedding_inputs(
                input_content_list,
                max_chars=2048,
            )
            response = post_embedding(request_input_content_list)
        if response.status_code == 400:
            logger.warning(
                "DashScope embedding still returned 400; retrying once with 512-char truncated inputs. "
                f"response={response.text[:500]}"
            )
            request_input_content_list = LiteLLMAPIBackend._trim_dashscope_embedding_inputs(
                input_content_list,
                max_chars=512,
            )
            response = post_embedding(request_input_content_list)
        response.raise_for_status()
        body = response.json()
        return [data["embedding"] for data in body["data"]]

    class CompleteKwargs(TypedDict):
        model: str
        temperature: float
        max_tokens: int | None
        reasoning_effort: Literal["low", "medium", "high"] | None
        api_base: str | None
        api_key: str | None
        custom_llm_provider: str | None

    @staticmethod
    def _resolve_chat_connection_settings(model: str) -> tuple[str, str | None, str | None, str | None]:
        api_base = LITELLM_SETTINGS.chat_openai_base_url or LITELLM_SETTINGS.openai_api_base or None
        api_key = LITELLM_SETTINGS.chat_openai_api_key or LITELLM_SETTINGS.openai_api_key or None
        custom_llm_provider = None
        resolved_model = model

        if "/" in model:
            provider, candidate_model = model.split("/", 1)
            if provider.strip() and candidate_model.strip():
                custom_llm_provider = provider.strip()
                resolved_model = candidate_model.strip()

        if custom_llm_provider is None and api_base and "dashscope.aliyuncs.com/compatible-mode" in api_base:
            # DashScope here exposes an OpenAI-compatible chat/completions API.
            custom_llm_provider = "openai"

        return resolved_model, api_base, api_key, custom_llm_provider

    def get_complete_kwargs(self) -> CompleteKwargs:
        """
        return several key settings for completion
        getting these values from settings makes it easier to adapt to backend calls in agent systems.
        """
        # Call LiteLLM completion
        model = LITELLM_SETTINGS.chat_model
        temperature = LITELLM_SETTINGS.chat_temperature
        max_tokens = LITELLM_SETTINGS.chat_max_tokens
        reasoning_effort = LITELLM_SETTINGS.reasoning_effort

        if LITELLM_SETTINGS.chat_model_map:
            for t, mc in LITELLM_SETTINGS.chat_model_map.items():
                if t in logger._tag:
                    model = mc["model"]
                    if "temperature" in mc:
                        temperature = float(mc["temperature"])
                    if "max_tokens" in mc:
                        max_tokens = int(mc["max_tokens"])
                    if "reasoning_effort" in mc:
                        if mc["reasoning_effort"] in ["low", "medium", "high"]:
                            reasoning_effort = cast(Literal["low", "medium", "high"], mc["reasoning_effort"])
                        else:
                            reasoning_effort = None
                    break
        model, api_base, api_key, custom_llm_provider = self._resolve_chat_connection_settings(model)
        return self.CompleteKwargs(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
            api_base=api_base,
            api_key=api_key,
            custom_llm_provider=custom_llm_provider,
        )

    def _create_chat_completion_inner_function(  # type: ignore[no-untyped-def] # noqa: C901, PLR0912, PLR0915
        self,
        messages: list[dict[str, Any]],
        response_format: Optional[Union[dict, Type[BaseModel]]] = None,
        *args,
        **kwargs,
    ) -> tuple[str, str | None]:
        """
        Call the chat completion function
        """

        response_schema_probe_model, _, _, _ = self._resolve_chat_connection_settings(LITELLM_SETTINGS.chat_model)
        _, _, _, response_schema_probe_provider = self._resolve_chat_connection_settings(LITELLM_SETTINGS.chat_model)
        response_schema_probe_model = self._compose_provider_model(
            response_schema_probe_model,
            response_schema_probe_provider,
        )
        # DeepSeek models usually do not support OpenAI response_schema in LiteLLM.
        if response_format and (
            response_schema_probe_provider == "deepseek"
            or not self._safe_supports_response_schema(response_schema_probe_model)
        ):
            # Deepseek will enter this branch
            if LITELLM_SETTINGS.chat_model not in self._response_schema_warned_models:
                logger.warning(
                    f"{LogColors.YELLOW}Model {LITELLM_SETTINGS.chat_model} does not support response schema, ignoring response_format argument.{LogColors.END}",
                    tag="llm_messages",
                )
                self._response_schema_warned_models.add(LITELLM_SETTINGS.chat_model)
            response_format = None

        if response_format:
            kwargs["response_format"] = response_format

        if LITELLM_SETTINGS.log_llm_chat_content:
            logger.info(self._build_log_messages(messages), tag="llm_messages")

        complete_kwargs = self.get_complete_kwargs()
        model = complete_kwargs["model"]
        model_for_stats = self._compose_provider_model(model, complete_kwargs.get("custom_llm_provider"))

        response = completion(
            messages=messages,
            stream=LITELLM_SETTINGS.chat_stream,
            max_retries=0,
            **complete_kwargs,
            **kwargs,
        )
        if LITELLM_SETTINGS.log_llm_chat_content:
            logger.info(f"{LogColors.GREEN}Using chat model{LogColors.END} {model}", tag="llm_messages")

        if LITELLM_SETTINGS.chat_stream:
            if LITELLM_SETTINGS.log_llm_chat_content:
                logger.info(f"{LogColors.BLUE}assistant:{LogColors.END}", tag="llm_messages")
            content = ""
            finish_reason = None
            for message in response:
                if message["choices"][0]["finish_reason"]:
                    finish_reason = message["choices"][0]["finish_reason"]
                if "content" in message["choices"][0]["delta"]:
                    chunk = (
                        message["choices"][0]["delta"]["content"] or ""
                    )  # when finish_reason is "stop", content is None
                    content += chunk
                    if LITELLM_SETTINGS.log_llm_chat_content:
                        logger.info(LogColors.CYAN + chunk + LogColors.END, raw=True, tag="llm_messages")
            if LITELLM_SETTINGS.log_llm_chat_content:
                logger.info("\n", raw=True, tag="llm_messages")
        else:
            content = str(response.choices[0].message.content)
            finish_reason = response.choices[0].finish_reason
            finish_reason_str = (
                f"({LogColors.RED}Finish reason: {finish_reason}{LogColors.END})"
                if finish_reason and finish_reason != "stop"
                else ""
            )
            if LITELLM_SETTINGS.log_llm_chat_content:
                logger.info(
                    f"{LogColors.BLUE}assistant:{LogColors.END} {finish_reason_str}\n{content}", tag="llm_messages"
                )

        global ACC_COST
        custom_provider = complete_kwargs.get("custom_llm_provider")
        if self._uses_openai_compatible_custom_endpoint() or custom_provider == "deepseek":
            cost = np.nan
            prompt_tokens = self._approx_token_count_from_messages(messages)
            completion_tokens = max(1, len(content) // 4) if content else 0
        else:
            try:
                cost = completion_cost(model=model_for_stats, messages=messages, completion=content)
            except Exception as e:
                logger.warning(f"Cost calculation failed for model {model_for_stats}: {e}. Skip cost statistics.")
                cost = np.nan
            else:
                ACC_COST += cost
                if LITELLM_SETTINGS.log_llm_chat_content:
                    logger.info(
                        f"Current Cost: ${float(cost):.10f}; Accumulated Cost: ${float(ACC_COST):.10f}; {finish_reason=}",
                    )
            try:
                prompt_tokens = token_counter(model=model_for_stats, messages=messages)
                completion_tokens = token_counter(model=model_for_stats, text=content)
            except ValueError as e:
                logger.warning(f"Token counting failed for model {model_for_stats}: {e}. Skip token statistics.")
                prompt_tokens = 0
                completion_tokens = 0
        logger.log_object(
            {
                "model": model,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "cost": cost,
                "accumulated_cost": ACC_COST,
            },
            tag="token_cost",
        )
        return content, finish_reason

    def supports_response_schema(self) -> bool:
        """
        Check if the backend supports function calling
        """
        probe_model, _, _, probe_provider = self._resolve_chat_connection_settings(LITELLM_SETTINGS.chat_model)
        probe_model = self._compose_provider_model(probe_model, probe_provider)
        return self._safe_supports_response_schema(probe_model) and LITELLM_SETTINGS.enable_response_schema

    @property
    def chat_token_limit(self) -> int:
        """Suggest an input token limit, ensuring enough space in the context window for the maximum output tokens."""
        if self._uses_openai_compatible_custom_endpoint():
            return super().chat_token_limit
        try:
            model_info = get_model_info(LITELLM_SETTINGS.chat_model)
            if model_info is None:
                return super().chat_token_limit

            max_input = model_info.get("max_input_tokens")
            max_output = model_info.get("max_output_tokens")

            if max_input is None or max_output is None:
                return super().chat_token_limit

            max_input_tokens = max_input - max_output
            return max_input_tokens
        except Exception as e:
            return super().chat_token_limit
