from __future__ import annotations

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from hotactor import HotActorClient, actor_handler

ACTOR_NAME = "qwen-chat"
NAMESPACE = "hotactor"


class QwenChatClient(HotActorClient):
    """Service-specific client facade."""

    @classmethod
    def connect(cls) -> "QwenChatClient":
        try:
            return cls.connect_named(ACTOR_NAME, namespace=NAMESPACE)
        except ValueError as exc:
            raise RuntimeError(
                "Server not running. Start it first with "
                "`uv run examples/qwen_chat/server.py`."
            ) from exc

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
    ) -> str:
        return self.call(
            "chat",
            messages,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )


@actor_handler
def chat(
    state,
    messages: list[dict[str, str]],
    max_new_tokens: int = 256,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = 20,
) -> str:
    import torch

    state.chat_count += 1

    model_inputs = state.tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )

    if hasattr(model_inputs, "to"):
        model_inputs = model_inputs.to(state.device)

    if isinstance(model_inputs, torch.Tensor):
        input_ids = model_inputs
        attention_mask = torch.ones_like(input_ids, device=state.device)
    else:
        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=state.device)

    generate_kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": max_new_tokens,
        "pad_token_id": state.tokenizer.pad_token_id,
        "eos_token_id": state.tokenizer.eos_token_id,
    }
    if temperature > 0:
        generate_kwargs.update(
            {
                "do_sample": True,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
            }
        )
    else:
        generate_kwargs["do_sample"] = False

    with torch.no_grad():
        output_ids = state.model.generate(**generate_kwargs)

    new_tokens = output_ids[0, input_ids.shape[-1]:]
    return state.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def run_client(
    system_prompt: str | None,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
) -> None:
    client = QwenChatClient.connect()
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
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
        messages.append({"role": "assistant", "content": response})
        print(f"\nResponse: {response}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen chat client")
    parser.add_argument("--system-prompt", default="You are a concise assistant.")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()

    run_client(
        args.system_prompt,
        args.max_new_tokens,
        args.temperature,
        args.top_p,
        args.top_k,
    )


if __name__ == "__main__":
    main()
