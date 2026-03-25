from __future__ import annotations

import argparse
import os
import sys
from typing import Any

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from hotactor import HostedState, launch_or_replace_actor, run_service_loop

ACTOR_NAME = "qwen-chat"
NAMESPACE = "hotactor"
MODEL_ID = "Qwen/Qwen3.5-0.8B"
HANDLER_MODULE = "examples.qwen_chat.client"


def _get_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class QwenChatState(HostedState):
    """Long-lived state for the Qwen chat example."""

    def __init__(self, model_name: str = MODEL_ID) -> None:
        super().__init__()
        self.model_name = model_name
        self.device = _get_device()
        self.tokenizer = None
        self.model = None
        self.chat_count = 0

    def build(self) -> None:
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
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )
        self.model.to(self.device)
        self.model.eval()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.host.set_state("idle")

    def teardown(self) -> None:
        self.model = None
        self.tokenizer = None

    def status(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "device": self.device,
            "chat_count": self.chat_count,
            "model_loaded": self.model is not None,
            "tokenizer_loaded": self.tokenizer is not None,
        }


def run_server(model_name: str, heartbeat_interval: int) -> None:
    print(f"Launching {model_name!r} as actor {ACTOR_NAME!r}...")
    client = launch_or_replace_actor(
        name="qwen-chat-service",
        state_cls=QwenChatState,
        state_kwargs={"model_name": model_name},
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
        lines.append(f"Device: {status['state']['device']}")
        return lines

    run_service_loop(
        client,
        heartbeat_interval=heartbeat_interval,
        status_header=status_header,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen chat server")
    parser.add_argument("--model", default=MODEL_ID, help="Model ID to load")
    parser.add_argument("--heartbeat-interval", type=int, default=30)
    args = parser.parse_args()
    run_server(args.model, args.heartbeat_interval)


if __name__ == "__main__":
    main()
