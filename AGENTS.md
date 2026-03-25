# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`hotactor` is a framework for building long-lived Ray-hosted services with a clean separation between:
- **heavy state**: models, policies, CUDA state, caches, distributed runtime objects (expensive to build)
- **lightweight handlers**: forward, get_topk_logits, pre/post-processing (changeable and reloadable)

## Development Commands

### Installation
```bash
pip install -e ./
```

### Running the Example
The Qwen chat example demonstrates the server/client architecture:
```bash
# Start the server (loads the model)
uv run examples/qwen_chat/server.py

# Start the client (interactive chat interface)
uv run examples/qwen_chat/client.py
```

### Testing
No formal test suite exists. Testing should follow the patterns in AGENTS.md:
- Test handlers against fake state objects (no Ray)
- Test state class build/teardown/status separately
- Add one integration test with Ray for end-to-end validation

## Architecture Overview

### Core Components
1. **HostedState**: Base class for long-lived state
   - Override `build()` to allocate heavy resources (models, tokenizers, caches)
   - Override `teardown()` to release resources
   - Override `status()` to return service status
   - Access framework lifecycle via `self.host`

2. **@actor_handler decorator**: Marks functions as callable handlers
   - First argument must be the state object
   - Functions should be lightweight and avoid rebuilding resources

3. **launch_actor()**: Creates the Ray actor and returns HotActorClient
   - Named actors supported via `actor_name` and `namespace`
   - Handler modules are reloaded without recreating the actor

4. **HotActorClient**: Friendly wrapper for Ray actor operations
   - `call()`: Synchronous handler calls
   - `call_async()`: Asynchronous handler calls
   - `reload_handlers()`: Reload handler modules
   - `status()`: Get both framework and user status
   - `shutdown()`: Clean shutdown

### Directory Structure
```
hotactor/                    # Core framework
├── __init__.py             # Main exports (lazy imports)
├── launcher.py             # launch_actor, launch_or_replace_actor
├── state.py                # HostedState base class
├── registry.py             # @actor_handler decorator
├── runtime.py              # Internal _HotActorRuntime
├── client.py               # HotActorClient wrapper
└── serving.py              # run_service_loop

examples/                   # Working examples
└── qwen_chat/             # Chat service with Qwen model
    ├── server.py          # Model loading and actor launch
    └── client.py          # Interactive client and chat handler
```

## Key Patterns

### State Class Pattern
```python
class MyServiceState(HostedState):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = None
        self.tokenizer = None

    def build(self):
        # Load heavy resources here
        self.model = load_model(self.config.model_path)
        self.tokenizer = load_tokenizer()

    def teardown(self):
        # Clean up resources
        self.model = None
        self.tokenizer = None

    def status(self):
        return {
            "model_loaded": self.model is not None,
            "inference_count": self.inference_count
        }
```

### Handler Pattern
```python
from hotactor import actor_handler

@actor_handler
def forward(state, inputs):
    # Use state.model, state.tokenizer but don't recreate them
    return state.model.generate(inputs)

@actor_handler
def get_status(state):
    return state.status()
```

### Launch Pattern
```python
from hotactor import launch_actor

service = launch_actor(
    name="my-service",
    state_cls=MyServiceState,
    state_kwargs={"config": service_config},
    handler_modules=["my_service.handlers"],
    ray_options={
        "num_cpus": 0,
        "num_gpus": 1,
        "max_concurrency": 4,
    },
    actor_name="my-service",  # Optional named actor
    namespace="prod",         # Optional namespace
    get_if_exists=True,      # Reuse existing actor
)
```

### Client Pattern
```python
# Using HotActorClient directly
result = service.call("forward", {"text": "hello"})
status = service.status()
service.reload_handlers()

# Subclassing for service-specific API
class MyServiceClient(HotActorClient):
    @classmethod
    def connect(cls):
        return cls.connect_named("my-service", namespace="prod")

    def forward(self, text):
        return self.call("forward", {"text": text})
```

## Important Design Rules

1. **Separation of Concerns**: Heavy state in HostedState, lightweight logic in handlers
2. **No Handler Rebuilding**: Handlers should use existing state, not rebuild models
3. **Named Actors**: Always use named actors for services that need to persist
4. **Module Reloading**: Always reload handlers by module, not by string
5. **Status Separation**: Framework status in `state.host`, domain status in `state.status()`

## Framework Internals

- The `_HotActorRuntime` class wraps the Ray actor and manages the HostedState instance
- Handler modules are imported and reloaded using importlib
- The actor maintains a reference to the HostedState and host lifecycle view
- Ray actors are persistent while handlers can be reloaded without downtime