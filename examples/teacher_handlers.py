from __future__ import annotations

from typing import Any

from hotactor import actor_handler


@actor_handler
def forward(state, inputs: Any) -> dict[str, Any]:
    state.inference_count += 1
    return state.llm(inputs)


@actor_handler
def get_topk_logits(state, data: dict[str, Any], k: int, micro_batch_size: int | None = None) -> dict[str, Any]:
    state.inference_count += 1
    return {
        "backend": state.backend,
        "k": k,
        "micro_batch_size": micro_batch_size,
        "preview": data,
    }
