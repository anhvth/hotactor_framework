from __future__ import annotations

import ray

from hotactor import launch_actor
from teacher_state import TeacherState


if __name__ == "__main__":
    ray.init(ignore_reinit_error=True)

    teacher = launch_actor(
        name="teacher-service",
        state_cls=TeacherState,
        state_kwargs={"model_name": "demo-model", "backend": "vllm"},
        handler_modules=["teacher_handlers"],
        ray_options={"num_cpus": 0, "max_concurrency": 4},
    )

    print("status:", teacher.status())
    print("forward:", teacher.call("forward", {"text": "hello"}))
    print("topk:", teacher.call("get_topk_logits", {"input_ids": [1, 2, 3]}, k=8))
    print("reload:", teacher.reload_handlers())
    print("shutdown:", teacher.shutdown())
