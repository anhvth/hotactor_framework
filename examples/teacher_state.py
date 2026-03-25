from __future__ import annotations

from typing import Any

from hotactor import HostedState


class FakeLLM:
    def __call__(self, inputs: Any) -> dict[str, Any]:
        return {"echo": inputs, "backend": "fake-llm"}


class TeacherState(HostedState):
    """Example state object.

    In production, replace FakeLLM with a vLLM LLM, a Policy, or any other
    heavy resource that is expensive to rebuild.
    """

    def __init__(self, model_name: str, backend: str = "vllm") -> None:
        super().__init__()
        self.model_name = model_name
        self.backend = backend
        self.llm = None
        self.inference_count = 0

    def build(self) -> None:
        self.llm = FakeLLM()
        self.host.set_state("idle")

    def status(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "backend": self.backend,
            "inference_count": self.inference_count,
            "llm_loaded": self.llm is not None,
        }

    def teardown(self) -> None:
        self.llm = None
