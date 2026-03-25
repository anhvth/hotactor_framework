# hotactor

`hotactor` is a small framework for building long-lived Ray-hosted services with a clean separation between:

- **heavy state** that should stay up: models, policies, CUDA state, caches, distributed runtime objects
- **lightweight handlers** that should change often: `forward`, `get_topk_logits`, pre/post-processing, experiments

The framework wraps the Ray actor internally and gives user code a Pythonic model:

1. write a plain Python `HostedState` class
2. write handler functions in normal Python modules
3. launch the actor through `launch_actor(...)`
4. call handlers by name
5. reload handlers without rebuilding the heavy state

## Installation

```bash
git clone https://github.com/anhvth/hotactor_framework.git
cd hotactor_framework
pip install -e ./
```

## Why this exists

In a normal Ray actor, changing class methods often means recreating the actor. That is painful when the actor owns a large model. `hotactor` fixes that by keeping the actor process stable and moving the changeable logic into reloadable handler modules.

## Core concepts

### `HostedState`

A plain Python class that owns long-lived resources.

Put expensive state here:

- `self.llm`
- `self.policy`
- tokenizers
- placement-group-aware objects
- caches and metrics

### handler modules

Normal Python modules that export lightweight functions.

A handler function always looks like this:

```python
@actor_handler
def forward(state, inputs):
    return state.llm(inputs)
```

The first argument is the live `HostedState` instance.

### `launch_actor(...)`

Creates the underlying Ray actor and returns a `HotActorClient`.

### `HotActorClient`

Friendly wrapper that hides the raw Ray actor handle for common operations:

- `call(...)`
- `call_async(...)`
- `reload_handlers(...)`
- `status()`
- `shutdown()`

## Quick start

### 1. Define state

```python
from hotactor import HostedState

class TeacherState(HostedState):
    def __init__(self, model_name: str):
        super().__init__()
        self.model_name = model_name
        self.llm = None
        self.inference_count = 0

    def build(self):
        self.llm = build_llm(self.model_name)

    def status(self):
        return {
            "model_name": self.model_name,
            "inference_count": self.inference_count,
            "llm_loaded": self.llm is not None,
        }
```

### 2. Define handlers

```python
from hotactor import actor_handler

@actor_handler
def forward(state, inputs):
    state.inference_count += 1
    return state.llm(inputs)

@actor_handler
def get_topk_logits(state, data, k, micro_batch_size=None):
    state.inference_count += 1
    return state.llm.get_topk_logits(data, k=k, micro_batch_size=micro_batch_size)
```

### 3. Launch the service

```python
import ray
from hotactor import launch_actor
from myapp.teacher_state import TeacherState

ray.init(ignore_reinit_error=True)

teacher = launch_actor(
    name="teacher-service",
    state_cls=TeacherState,
    state_kwargs={"model_name": "big-model"},
    handler_modules=["myapp.teacher_handlers"],
    ray_options={"num_cpus": 0, "num_gpus": 1, "max_concurrency": 4},
)
```

### 4. Use it

```python
result = teacher.call("forward", {"text": "hello"})
status = teacher.status()
teacher.reload_handlers()
```

## Design rules

- Keep the Ray-facing runtime stable.
- Keep heavy resources on `HostedState`.
- Keep handlers thin and reloadable.
- Always define handlers in importable Python modules and reload with `reload_handlers()`.

## Named actors

`launch_actor(...)` supports Ray named actors:

```python
teacher = launch_actor(
    name="teacher-service",
    state_cls=TeacherState,
    handler_modules=["myapp.teacher_handlers"],
    actor_name="teacher-service",
    namespace="prod",
    get_if_exists=True,
)
```

## Files in this package

- `hotactor/state.py`: user-facing base state class and host lifecycle view
- `hotactor/registry.py`: `@actor_handler` decorator
- `hotactor/runtime.py`: internal framework runtime that owns the live state and handlers
- `hotactor/launcher.py`: one function to create the hidden Ray actor
- `hotactor/client.py`: friendly client wrapper
- `AGENTS.md`: canonical style guide for coding agents
- `examples/`: minimal working example
