from __future__ import annotations

from typing import Any, AsyncGenerator, Dict, List, Optional
import os
import asyncio
import json
import time
from pathlib import Path

import openai

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class OpenAICompatibleAdapter:
    """Unified adapter for chat-completions and responses/codex-style endpoints."""

    def __init__(self, async_client, sync_client, model_name: str, **kwargs):
        self.async_client = async_client
        self.sync_client = sync_client
        self.model_name = model_name
        self.config = kwargs
        self.api_style = str(
            kwargs.get("api_style")
            or os.environ.get("TOOLWEAVER_LLM_API_STYLE")
            or "chat"
        ).strip().lower()
        debug_raw_flag = os.environ.get("DYNACALL_DEBUG_RAW_LLM", "")
        self.debug_raw = str(debug_raw_flag).strip().lower() in {
            "1",
            "true",
            "yes",
        }

    def _dump_raw_response(self, mode: str, prompt: str, response: Any) -> None:
        if not self.debug_raw:
            return
        try:
            payload = {
                "ts": time.time(),
                "model": self.model_name,
                "mode": mode,
                "prompt_preview": str(prompt)[:1200],
            }
            dumped = getattr(response, "model_dump", None)
            if callable(dumped):
                payload["response"] = dumped()
            else:
                payload["response"] = str(response)

            exps = PROJECT_ROOT / "exps"
            exps.mkdir(parents=True, exist_ok=True)
            out_path = exps / "raw_llm_responses.jsonl"
            with out_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
        except Exception:
            pass

    def _normalize_stop(self, stop=None):
        if self.model_name == "gpt-5.4-nano":
            return None
        return stop

    def _temperature(self) -> float:
        try:
            return float(self.config.get("temperature", 0.0))
        except Exception:
            return 0.0

    def _chat_request_kwargs(self, messages, stop=None, stream=False):
        kwargs: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "stream": stream,
        }
        temp = self._temperature()
        if temp is not None:
            kwargs["temperature"] = temp
        normalized_stop = self._normalize_stop(stop)
        if normalized_stop is not None:
            kwargs["stop"] = normalized_stop
        if self.model_name == "qwen/qwen3.6-plus:free":
            kwargs["extra_body"] = {"reasoning": {"enabled": False}}
        return kwargs

    def _callbacks(self, callbacks=None) -> List[Any]:
        if callbacks is None:
            return []
        if isinstance(callbacks, (list, tuple)):
            return [callback for callback in callbacks if callback is not None]
        return [callbacks]

    def _extract_usage(self, response: Any) -> Dict[str, int]:
        usage = getattr(response, "usage", None)
        if usage is None and isinstance(response, dict):
            usage = response.get("usage")
        if usage is None:
            return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

        def get_any(obj: Any, *names: str) -> int:
            for name in names:
                if isinstance(obj, dict):
                    value = obj.get(name)
                else:
                    value = getattr(obj, name, None)
                if value is not None:
                    try:
                        return int(value)
                    except Exception:
                        return 0
            return 0

        input_tokens = get_any(usage, "input_tokens", "prompt_tokens")
        output_tokens = get_any(usage, "output_tokens", "completion_tokens")
        total_tokens = get_any(usage, "total_tokens")
        if not total_tokens:
            total_tokens = input_tokens + output_tokens
        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
        }

    def _record_usage(self, callbacks=None, usage: Optional[Dict[str, int]] = None, elapsed: float = 0.0) -> None:
        usage = usage or {}
        for callback in self._callbacks(callbacks):
            record_usage = getattr(callback, "record_usage", None)
            if callable(record_usage):
                record_usage(
                    input_tokens=usage.get("input_tokens", 0),
                    output_tokens=usage.get("output_tokens", 0),
                    total_tokens=usage.get("total_tokens", 0),
                    elapsed=elapsed,
                )

    def _responses_request_kwargs(self, text: str, stop=None, stream=False):
        kwargs: Dict[str, Any] = {
            "model": self.model_name,
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": text,
                        }
                    ],
                }
            ],
            "stream": stream,
        }
        if self.model_name == "qwen/qwen3.6-plus:free":
            kwargs["extra_body"] = {"reasoning": {"enabled": False}}
        return kwargs

    def _extract_chat_text(self, response: Any) -> str:
        try:
            return response.choices[0].message.content or ""
        except Exception:
            return ""

    def _extract_completion_text(self, response: Any) -> str:
        try:
            return response.choices[0].text or ""
        except Exception:
            return ""

    def _extract_responses_text(self, response: Any) -> str:
        direct = getattr(response, "output_text", None)
        if isinstance(direct, str) and direct:
            return direct
        output = getattr(response, "output", None) or []
        texts: List[str] = []
        for item in output:
            content = getattr(item, "content", None) or []
            for part in content:
                part_text = getattr(part, "text", None)
                if isinstance(part_text, str) and part_text:
                    texts.append(part_text)
        if texts:
            return "".join(texts)
        dumped = getattr(response, "model_dump", None)
        if callable(dumped):
            data = dumped()
            if isinstance(data, dict):
                if isinstance(data.get("output_text"), str) and data.get("output_text"):
                    return data["output_text"]
                for item in data.get("output", []) or []:
                    for part in item.get("content", []) or []:
                        if isinstance(part, dict) and isinstance(part.get("text"), str):
                            texts.append(part["text"])
        return "".join(texts)

    def _response_api_preference(self) -> List[str]:
        if self.api_style in {"chat", "chat_completions", "chat-completions"}:
            return ["chat"]
        if self.api_style in {"responses", "codex", "response", "responses_api"}:
            return ["responses"]
        return ["chat", "responses"]

    def _is_chat_model(self) -> bool:
        name = self.model_name.lower()
        chat_markers = [
            "gpt-",
            "chatgpt",
            "qwen",
            "claude",
            "gemini",
            "glm-",
            "minimax",
            "deepseek",
            "llama",
            "codex",
        ]
        return any(marker in name for marker in chat_markers)

    def _predict_sync_via_chat(self, text: str, stop=None, callbacks=None) -> str:
        start = time.time()
        response = self.sync_client.chat.completions.create(
            **self._chat_request_kwargs(
                [{"role": "user", "content": text}], stop=stop, stream=False
            )
        )
        self._record_usage(callbacks, self._extract_usage(response), time.time() - start)
        self._dump_raw_response("sync_chat", text, response)
        return self._extract_chat_text(response)

    def _predict_sync_via_responses(self, text: str, stop=None, callbacks=None) -> str:
        start = time.time()
        response = self.sync_client.responses.create(
            **self._responses_request_kwargs(text, stop=stop, stream=False)
        )
        self._record_usage(callbacks, self._extract_usage(response), time.time() - start)
        self._dump_raw_response("sync_responses", text, response)
        return self._extract_responses_text(response)

    async def _predict_async_via_chat(self, text: str, stop=None, callbacks=None) -> str:
        start = time.time()
        response = await self.async_client.chat.completions.create(
            **self._chat_request_kwargs(
                [{"role": "user", "content": text}], stop=stop, stream=False
            )
        )
        self._record_usage(callbacks, self._extract_usage(response), time.time() - start)
        self._dump_raw_response("async_chat", text, response)
        return self._extract_chat_text(response)

    async def _predict_async_via_responses(self, text: str, stop=None, callbacks=None) -> str:
        start = time.time()
        response = await self.async_client.responses.create(
            **self._responses_request_kwargs(text, stop=stop, stream=False)
        )
        self._record_usage(callbacks, self._extract_usage(response), time.time() - start)
        self._dump_raw_response("async_responses", text, response)
        extracted = self._extract_responses_text(response)
        if extracted:
            return extracted
        tokens: List[str] = []
        stream = await self.async_client.responses.create(
            **self._responses_request_kwargs(text, stop=stop, stream=True)
        )
        async for event in stream:
            event_type = getattr(event, "type", "")
            if event_type in {"response.output_text.delta", "response.output_text.annotation.added"}:
                delta = getattr(event, "delta", None)
                if isinstance(delta, str) and delta:
                    tokens.append(delta)
            elif event_type == "response.output_text.done":
                done_text = getattr(event, "text", None)
                if isinstance(done_text, str) and done_text and not tokens:
                    tokens.append(done_text)
        return "".join(tokens)

    def _predict_sync_auto(self, text: str, stop=None, callbacks=None) -> str:
        errors: List[str] = []
        for mode in self._response_api_preference():
            try:
                if mode == "chat":
                    return self._predict_sync_via_chat(text, stop=stop, callbacks=callbacks)
                return self._predict_sync_via_responses(text, stop=stop, callbacks=callbacks)
            except Exception as exc:
                errors.append(f"{mode}: {exc}")
        return f"Error: {' | '.join(errors)}"

    async def _predict_async_auto(self, text: str, stop=None, callbacks=None) -> str:
        errors: List[str] = []
        for mode in self._response_api_preference():
            try:
                if mode == "chat":
                    return await self._predict_async_via_chat(text, stop=stop, callbacks=callbacks)
                return await self._predict_async_via_responses(text, stop=stop, callbacks=callbacks)
            except Exception as exc:
                errors.append(f"{mode}: {exc}")
        return f"Error: {' | '.join(errors)}"

    def predict(self, text, callbacks=None, stop=None):
        try:
            if self._is_chat_model():
                return self._predict_sync_auto(text, stop=stop, callbacks=callbacks)
            start = time.time()
            response = self.sync_client.completions.create(
                model=self.model_name,
                prompt=text,
                temperature=self._temperature(),
                stop=self._normalize_stop(stop),
                stream=False,
            )
            self._record_usage(callbacks, self._extract_usage(response), time.time() - start)
            return self._extract_completion_text(response)
        except Exception as e:
            return f"Error: {str(e)}"

    async def apredict(self, text, callbacks=None, stop=None):
        try:
            if self._is_chat_model():
                return await self._predict_async_auto(text, stop=stop, callbacks=callbacks)
            start = time.time()
            response = await self.async_client.completions.create(
                model=self.model_name,
                prompt=text,
                temperature=self._temperature(),
                stop=self._normalize_stop(stop),
                stream=False,
            )
            self._record_usage(callbacks, self._extract_usage(response), time.time() - start)
            return self._extract_completion_text(response)
        except Exception as e:
            return f"Error: {str(e)}"

    def _call(self, messages, callbacks=None, stop=None):
        try:
            start = time.time()
            response = self.sync_client.chat.completions.create(
                **self._chat_request_kwargs(messages, stop=stop, stream=False)
            )
            self._record_usage(callbacks, self._extract_usage(response), time.time() - start)
            return self._extract_chat_text(response)
        except Exception as e:
            return f"Error: {str(e)}"

    async def _call_async(self, messages, callbacks=None, stop=None):
        try:
            start = time.time()
            response = await self.async_client.chat.completions.create(
                **self._chat_request_kwargs(messages, stop=stop, stream=False)
            )
            self._record_usage(callbacks, self._extract_usage(response), time.time() - start)
            return self._extract_chat_text(response)
        except Exception as e:
            return f"Error: {str(e)}"

    def generate(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
        return self.predict(prompt, stop=stop)

    async def agenerate(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
        return await self.apredict(prompt, stop=stop)

    def generate_prompt(self, prompts, stop=None, callbacks=None):
        if prompts and len(prompts) > 0:
            prompt = prompts[0]
            text = getattr(prompt, "text", None) or (prompt.get("text") if isinstance(prompt, dict) else None) or str(prompt)
            response = self.predict(text, callbacks=callbacks, stop=stop)
            return {"generations": [[{"text": response, "generation_info": {}}]]}
        return {"generations": []}

    async def agenerate_prompt(self, prompts, stop=None, callbacks=None):
        if prompts and len(prompts) > 0:
            prompt = prompts[0]
            text = getattr(prompt, "text", None) or (prompt.get("text") if isinstance(prompt, dict) else None) or str(prompt)
            response = await self.apredict(text, callbacks=callbacks, stop=stop)
            return {"generations": [[{"text": response, "generation_info": {}}]]}
        return {"generations": []}

    async def apredict_stream(self, text, callbacks=None, stop=None) -> AsyncGenerator[str, None]:
        errors: List[str] = []
        for mode in self._response_api_preference():
            try:
                start = time.time()
                usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
                if mode == "chat":
                    stream = await self.async_client.chat.completions.create(
                        **self._chat_request_kwargs(
                            [{"role": "user", "content": text}], stop=stop, stream=True
                        )
                    )
                    async for chunk in stream:
                        usage_obj = getattr(chunk, "usage", None)
                        if usage_obj is not None:
                            usage = self._extract_usage({"usage": usage_obj})
                        if chunk.choices and chunk.choices[0].delta.content:
                            yield chunk.choices[0].delta.content
                    self._record_usage(callbacks, usage, time.time() - start)
                    return
                stream = await self.async_client.responses.create(
                    **self._responses_request_kwargs(text, stop=stop, stream=True)
                )
                async for event in stream:
                    usage_obj = getattr(event, "usage", None)
                    if usage_obj is not None:
                        usage = self._extract_usage({"usage": usage_obj})
                    event_type = getattr(event, "type", "")
                    if event_type in {"response.output_text.delta", "response.output_text.annotation.added"}:
                        delta = getattr(event, "delta", None)
                        if isinstance(delta, str) and delta:
                            yield delta
                    elif event_type == "response.completed":
                        response = getattr(event, "response", None)
                        if response is not None:
                            usage = self._extract_usage(response)
                        break
                self._record_usage(callbacks, usage, time.time() - start)
                return
            except Exception as exc:
                errors.append(f"{mode}: {exc}")
                continue
        yield f"Error: {' | '.join(errors)}"

    async def agenerate_stream(self, prompts, stop=None, callbacks=None) -> AsyncGenerator[str, None]:
        if prompts and len(prompts) > 0:
            prompt = prompts[0]
            text = getattr(prompt, "text", None) or (prompt.get("text") if isinstance(prompt, dict) else None) or str(prompt)
            async for token in self.apredict_stream(text, callbacks=callbacks, stop=stop):
                yield token
        else:
            yield ""


class VLLMModelAdapter:
    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.config = kwargs

    def predict(self, text, callbacks=None, stop=None):
        return f"VLLM response to: {text[:100]}..."

    def generate(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
        return f"VLLM response to: {prompt[:100]}..."

    def generate_prompt(self, prompts, stop=None, callbacks=None):
        response = f"VLLM response to: {str(prompts)[:100]}..."
        return {"generations": [[{"text": response, "generation_info": {}}]]}

    async def apredict(self, text, callbacks=None, stop=None):
        return f"VLLM response to: {text[:100]}..."

    async def agenerate(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
        return f"VLLM response to: {prompt[:100]}..."

    async def agenerate_prompt(self, prompts, stop=None, callbacks=None):
        response = f"VLLM response to: {str(prompts)[:100]}..."
        return {"generations": [[{"text": response, "generation_info": {}}]]}

    async def _call_async(self, messages, callbacks=None, stop=None):
        return f"VLLM response to: {messages}"

    async def agenerate_stream(self, prompts, stop=None, callbacks=None):
        yield f"VLLM stream response to: {str(prompts)[:100]}..."


class DefaultModelAdapter:
    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.config = kwargs

    def predict(self, text, callbacks=None, stop=None):
        return f"Default response to: {text[:100]}..."

    def generate(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
        return f"Default response to: {prompt[:100]}..."

    def generate_prompt(self, prompts, stop=None, callbacks=None):
        response = f"Default response to: {str(prompts)[:100]}..."
        return {"generations": [[{"text": response, "generation_info": {}}]]}

    async def apredict(self, text, callbacks=None, stop=None):
        return f"Default response to: {text[:100]}..."

    async def agenerate(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
        return f"Default response to: {prompt[:100]}..."

    async def agenerate_prompt(self, prompts, stop=None, callbacks=None):
        response = f"Default response to: {str(prompts)[:100]}..."
        return {"generations": [[{"text": response, "generation_info": {}}]]}

    async def _call_async(self, messages, callbacks=None, stop=None):
        return f"Default response to: {messages}"

    async def agenerate_stream(self, prompts, stop=None, callbacks=None):
        yield f"Default stream response to: {str(prompts)[:100]}..."


def create_llm_adapter(model_type: str, model_name: str, **kwargs):
    if model_type in ["openai", "azure"]:
        # Prefer generic env names for runtime switching across providers.
        api_key = (
            os.environ.get("API_KEY")
            or os.environ.get("TOOL_LLM_API_KEY")
            or kwargs.get("api_key")
            or None
        )
        api_base = (
            os.environ.get("BASE_URL")
            or kwargs.get("api_base")
            or os.environ.get("OPENAI_API_BASE")
            or None
        )
        async_client = openai.AsyncOpenAI(base_url=api_base, api_key=api_key)
        sync_client = openai.OpenAI(base_url=api_base, api_key=api_key)
        return OpenAICompatibleAdapter(async_client, sync_client, model_name, **kwargs)
    if model_type == "vllm":
        return VLLMModelAdapter(model_name, **kwargs)
    return DefaultModelAdapter(model_name, **kwargs)
