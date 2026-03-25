from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Literal

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from hotactor import HostedState, launch_or_replace_actor, run_service_loop

ACTOR_NAME = "vllm-chat"
NAMESPACE = "hotactor"
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
HANDLER_MODULE = "examples.vllm_chat.client"
BackendName = Literal["transformers", "vllm"]


def _get_device() -> str:
    return "cpu"


class VllmChatState(HostedState):
    """Long-lived state for the chat example."""

    def __init__(
        self,
        model_name: str = MODEL_ID,
        backend: BackendName = "transformers",
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.backend = backend
        self.device = _get_device()
        self.tokenizer = None
        self.model = None
        self.llm = None
        self.chat_count = 0

    def build(self) -> None:
        if self.backend == "vllm":
            self._build_vllm()
        else:
            self._build_transformers()

        if self._host_view is not None:
            self.host.set_state("idle")

    def _build_vllm(self) -> None:
        try:
            from vllm import LLM
        except ImportError as exc:
            raise RuntimeError(
                "vLLM is not installed yet. Re-run in an environment with vLLM and set "
                "`--backend vllm`."
            ) from exc

        kwargs = {
            "model": self.model_name,
            "trust_remote_code": True,
            "dtype": "float32",
            "tensor_parallel_size": 1,
        }
        self.llm = LLM(**kwargs)

    def _build_transformers(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )
        if self.tokenizer.chat_template is None:
            self.tokenizer.chat_template = (
                "{% for message in messages %}"
                "{{ message['role'] }}: {{ message['content'] }}\\n"
                "{% endfor %}"
                "{% if add_generation_prompt %}assistant: {% endif %}"
            )

        model_kwargs = {
            "trust_remote_code": True,
            "dtype": torch.float32,
        }
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **model_kwargs,
            )
        except TypeError:
            model_kwargs["torch_dtype"] = model_kwargs.pop("dtype")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **model_kwargs,
            )
        self.model.to(self.device)
        self.model.eval()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
    ) -> str:
        if self.backend == "vllm":
            return self._chat_vllm(messages, max_tokens=max_tokens, temperature=temperature, top_p=top_p)
        return self._chat_transformers(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )

    def _chat_vllm(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> str:
        from vllm import SamplingParams

        if self.llm is None:
            raise RuntimeError("vLLM backend is not loaded.")

        params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        outputs = self.llm.chat(messages=messages, sampling_params=params)
        return outputs[0].outputs[0].text.strip()

    def _chat_transformers(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> str:
        import torch

        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Transformers backend is not loaded.")

        model_inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )

        if hasattr(model_inputs, "to"):
            model_inputs = model_inputs.to(self.device)

        if isinstance(model_inputs, torch.Tensor):
            input_ids = model_inputs
            attention_mask = torch.ones_like(input_ids, device=self.device)
        else:
            input_ids = model_inputs["input_ids"]
            attention_mask = model_inputs.get("attention_mask")
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids, device=self.device)

        generate_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": max_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if temperature > 0:
            generate_kwargs.update(
                {
                    "do_sample": True,
                    "temperature": temperature,
                    "top_p": top_p,
                }
            )
        else:
            generate_kwargs["do_sample"] = False

        with torch.no_grad():
            output_ids = self.model.generate(**generate_kwargs)

        new_tokens = output_ids[0, input_ids.shape[-1] :]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def teardown(self) -> None:
        self.llm = None
        self.model = None
        self.tokenizer = None

    def status(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "backend": self.backend,
            "device": self.device,
            "chat_count": self.chat_count,
            "model_loaded": self.llm is not None or self.model is not None,
            "tokenizer_loaded": self.tokenizer is not None,
        }


def run_server(model_name: str, backend: BackendName, heartbeat_interval: int) -> None:
    print(f"Launching {model_name!r} with backend {backend!r} as actor {ACTOR_NAME!r}...")
    client = launch_or_replace_actor(
        name="vllm-chat-service",
        state_cls=VllmChatState,
        state_kwargs={"model_name": model_name, "backend": backend},
        handler_modules=[HANDLER_MODULE],
        ray_options={"num_cpus": 1, "max_concurrency": 4},
        actor_name=ACTOR_NAME,
        namespace=NAMESPACE,
        get_if_exists=False,
        replace_existing=True,
    )

    def status_header(status: dict[str, Any]) -> list[str]:
        lines = [f"Actor ready. Lifecycle: {status['host']['lifecycle']}"]
        build_elapsed = status.get("build_elapsed_s")
        if build_elapsed is not None:
            lines.append(f"Build time: {build_elapsed:.1f}s")
        lines.append(f"Backend: {status['state']['backend']}")
        lines.append(f"Device: {status['state']['device']}")
        return lines

    run_service_loop(
        client,
        heartbeat_interval=heartbeat_interval,
        status_header=status_header,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="vLLM chat server")
    parser.add_argument("--model", default=MODEL_ID, help="Model ID to load")
    parser.add_argument(
        "--backend",
        choices=("transformers", "vllm"),
        default="transformers",
        help="Backend to use for generation",
    )
    parser.add_argument("--heartbeat-interval", type=int, default=30)
    args = parser.parse_args()
    run_server(args.model, args.backend, args.heartbeat_interval)


if __name__ == "__main__":
    main()
