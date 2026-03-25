# hotactor canonical guide for coding agents

This document defines the **canonical style** for building services with `hotactor`.

Follow it exactly unless the surrounding codebase has a stronger existing convention.

## The architectural rule

Always separate code into two layers:

### 1. long-lived state

This is the expensive part that should stay up.

Put it in a `HostedState` subclass.

Examples:
- `self.llm`
- `self.policy`
- distributed groups
- tokenizers
- caches
- persistent counters
- placement-group-aware objects

### 2. lightweight handlers

This is the changeable part that should be reloaded often.

Put it in handler modules.

Examples:
- `forward`
- `generate`
- `prepare_for_lp_inference`
- `get_topk_logits`
- input normalization
- output formatting
- experimental branching logic

Never bury changing business logic inside the state class.

## Canonical directory layout

Use this layout for every new service:

```text
my_service/
  __init__.py
  state.py
  handlers.py
  client.py              # optional, service-specific facade
  tests/
    test_handlers.py
    test_state.py
```

If the service has backend-specific logic, split handlers by backend:

```text
my_service/
  state.py
  handlers_vllm.py
  handlers_dtensor.py
```

## Canonical state style

A state class must:

- inherit from `HostedState`
- own only long-lived resources and service-level counters
- allocate heavy resources in `build()`
- release them in `teardown()`
- expose a small, serializable `status()` payload

### Example

```python
from hotactor import HostedState

class TeacherState(HostedState):
    def __init__(self, teacher_config, service_cfg, cluster_config=None):
        super().__init__()
        self.teacher_config = teacher_config
        self.service_cfg = service_cfg
        self.cluster_config = cluster_config
        self.backend = teacher_config.get("service", {}).get("backend", "vllm")

        self.llm = None
        self.policy = None
        self.inference_count = 0
        self.last_inference_time_s = None

    def build(self):
        if self.backend == "vllm":
            self.llm = build_vllm(self.teacher_config, self.service_cfg)

    def teardown(self):
        self.llm = None
        self.policy = None

    def status(self):
        return {
            "backend": self.backend,
            "inference_count": self.inference_count,
            "llm_loaded": self.llm is not None,
            "policy_loaded": self.policy is not None,
        }
```

## Canonical handler style

A handler must:

- be a plain function
- accept `state` as the first argument
- be decorated with `@actor_handler`
- avoid owning lifecycle
- avoid rebuilding heavy resources
- do one service operation cleanly

### Example

```python
from hotactor import actor_handler

@actor_handler
def prepare_for_lp_inference(state):
    state.host.set_state("ready")
    if state.policy is not None:
        state.policy.prepare_for_lp_inference()

@actor_handler
def get_topk_logits(state, data, k, micro_batch_size=None):
    state.inference_count += 1
    return state.policy.get_topk_logits(data, k=k, micro_batch_size=micro_batch_size)
```

## What belongs where

### Put this in `state.py`

- model construction
- policy construction
- tokenizer loading
- distributed setup
- cache initialization
- long-lived counters
- service-wide configuration
- teardown logic

### Put this in `handlers.py`

- `forward`
- `generate`
- `prepare_*`
- `offload_*`
- `get_topk_logits`
- pre/post-processing
- experimental feature flags
- orchestration over already-live state

## Naming conventions

Use these names consistently.

### State class

Use `<ServiceName>State`.

Examples:
- `TeacherState`
- `RewardModelState`
- `RolloutState`

### Handler module

Use `handlers.py` for one backend or `handlers_<backend>.py` for multiple backends.

Examples:
- `handlers.py`
- `handlers_vllm.py`
- `handlers_dtensor.py`

### Handler functions

Use verb-first names that describe the operation.

Good:
- `forward`
- `generate`
- `prepare_for_lp_inference`
- `get_topk_logits`
- `offload_after_refit`

Bad:
- `do_it`
- `main`
- `service_logic`
- `temp_handler`

## Launch pattern

This is the canonical launch pattern:

```python
teacher = launch_actor(
    name="teacher-service",
    state_cls=TeacherState,
    state_kwargs={
        "teacher_config": teacher_config,
        "service_cfg": service_cfg,
        "cluster_config": cluster_config,
    },
    handler_modules=["my_service.handlers_vllm"],
    ray_options={
        "num_cpus": 0,
        "num_gpus": 1,
        "max_concurrency": 4,
    },
    actor_name="teacher-service",
    namespace="prod",
    get_if_exists=True,
)
```

## Client-side use

Prefer named operations over reaching into raw Ray.

Good:

```python
result = teacher.call("get_topk_logits", data, k=32)
teacher.reload_handlers()
status = teacher.status()
```

Avoid raw framework internals unless necessary.

## Rule for reloading

Always reload by module:

```python
teacher.reload_handlers()
```

Never register handlers from strings. Always define handlers in importable Python modules.

## Service-specific facade pattern

When the service has a stable public API, add a small facade class.

### Example

```python
class TeacherClient:
    def __init__(self, actor_client):
        self._actor = actor_client

    def prepare_for_lp_inference(self):
        return self._actor.call("prepare_for_lp_inference")

    def get_topk_logits(self, data, k, micro_batch_size=None):
        return self._actor.call(
            "get_topk_logits",
            data,
            k=k,
            micro_batch_size=micro_batch_size,
        )
```

Use this when you want stronger typing and a clear service contract.

## Testing rules

### test handlers without Ray when possible

Handlers are plain functions, so test them against a fake state object first.

### test state separately

Test that `build()`, `status()`, and `teardown()` do the right thing.

### add one integration test with Ray

Have one end-to-end test that launches the actor and calls the main handler.

## Anti-patterns

Do not do these.

### 1. heavy logic inside handlers that reconstructs the model

Bad:

```python
@actor_handler
def forward(state, inputs):
    state.llm = build_llm(...)
    return state.llm(inputs)
```

### 2. business logic buried in `build()`

Bad:

```python
def build(self):
    self.llm = build_llm(...)
    self.cached_result = run_whole_pipeline(...)
```

`build()` should prepare resources, not run service operations.

### 3. framework-level status mixed with domain status in confusing ways

Keep host lifecycle on `state.host` and domain status in `status()`.

### 4. giant handler modules with unrelated logic

Split by backend or concern when the file grows large.

## Canonical summary

When writing code for `hotactor`, remember this sentence:

> `HostedState` owns what is expensive to build; handlers define what the live service does.

If a piece of code changes often, it belongs in a handler.
If a piece of code is expensive to rebuild, it belongs on the state object.
