from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class HostLifecycleView:
    """Mutable runtime metadata that handlers may inspect and update.

    This object is intentionally small and generic. It tracks framework-level
    status only; user state lives on the HostedState subclass itself.
    """

    name: str
    lifecycle: str = "created"
    action: str | None = None
    last_error: str | None = None
    started_at_s: float = field(default_factory=time.time)
    last_started_handler: str | None = None
    last_finished_handler: str | None = None

    def set_state(self, lifecycle: str, action: str | None = None) -> None:
        self.lifecycle = lifecycle
        self.action = action

    def mark_error(self, message: str) -> None:
        self.lifecycle = "error"
        self.last_error = message

    def begin_handler(self, handler_name: str) -> None:
        self.last_started_handler = handler_name
        self.action = handler_name

    def finish_handler(self, handler_name: str, lifecycle: str = "idle") -> None:
        self.last_finished_handler = handler_name
        self.lifecycle = lifecycle
        self.action = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class HostedState:
    """Base class for user-owned, long-lived state.

    Subclasses should put heavy resources here: models, CUDA state,
    placement-group-aware objects, caches, tokenizers, distributed handles.

    Lifecycle hooks:
    - build(): allocate heavy resources once when the actor starts
    - teardown(): release resources on shutdown
    - status(): return user-defined status fields
    """

    def __init__(self) -> None:
        self._host_view: HostLifecycleView | None = None

    def _bind_host_view(self, host_view: HostLifecycleView) -> None:
        self._host_view = host_view

    @property
    def host(self) -> HostLifecycleView:
        if self._host_view is None:
            raise RuntimeError("HostedState has not been bound to a host view")
        return self._host_view

    def build(self) -> None:
        """Allocate heavy resources. Override in subclasses."""

    def teardown(self) -> None:
        """Release heavy resources. Override in subclasses."""

    def status(self) -> dict[str, Any]:
        """Return user-defined status fields. Override in subclasses."""
        return {}
