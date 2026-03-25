"""hotactor: a small framework for long-lived Ray-hosted Python state.

User code defines:
- a plain Python state class that owns heavy resources
- handler modules that define lightweight operations over that state

The framework hides the Ray actor and exposes a friendly client wrapper.
"""

from .registry import actor_handler
from .state import HostedState, HostLifecycleView

__all__ = [
    "HostLifecycleView",
    "HostedState",
    "actor_handler",
    "launch_actor",
    "launch_or_replace_actor",
    "run_service_loop",
    "HotActorClient",
]


def __getattr__(name: str):
    if name == "launch_actor":
        from .launcher import launch_actor
        return launch_actor
    if name == "launch_or_replace_actor":
        from .launcher import launch_or_replace_actor
        return launch_or_replace_actor
    if name == "run_service_loop":
        from .serving import run_service_loop
        return run_service_loop
    if name == "HotActorClient":
        from .client import HotActorClient
        return HotActorClient
    raise AttributeError(name)
