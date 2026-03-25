from __future__ import annotations

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from hotactor import HotActorClient, actor_handler

ACTOR_NAME = "vllm-chat"
NAMESPACE = "hotactor"


class VllmChatClient(HotActorClient):
    """Service-specific client facade."""

    @classmethod
    def connect(cls) -> "VllmChatClient":
        try:
            return cls.connect_named(ACTOR_NAME, namespace=NAMESPACE)
        except ValueError as exc:
            raise RuntimeError(
                "Server not running. Start it first with "
                "`uv run --extra vllm-cpu examples/vllm_chat/server.py`."
            ) from exc

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> str:
        return self.call(
            "chat",
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )


@actor_handler
def chat(
    state,
    messages: list[dict[str, str]],
    max_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.95,
) -> str:
    state.chat_count += 1
    return state.chat(
        messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
    )


def run_client(
    system_prompt: str | None,
    max_tokens: int,
    temperature: float,
    top_p: float,
) -> None:
    client = VllmChatClient.connect()
    messages: list[dict[str, str]] = []

    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    print("Client connected to server")
    print("Enter prompts. Type 'quit' to exit.")

    while True:
        prompt = input("\nPrompt: ").strip()
        if prompt.lower() == "quit":
            break
        if not prompt:
            continue

        messages.append({"role": "user", "content": prompt})
        response = client.chat(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        messages.append({"role": "assistant", "content": response})
        print(f"\nResponse: {response}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="vLLM chat client")
    parser.add_argument("--system-prompt", default="You are a concise assistant.")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    args = parser.parse_args()

    run_client(
        args.system_prompt,
        args.max_tokens,
        args.temperature,
        args.top_p,
    )


if __name__ == "__main__":
    main()
