from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import ray


@dataclass
class HotActorClient:
    """Friendly wrapper around a hidden Ray actor handle.

    This class intentionally exposes both synchronous and asynchronous methods.
    Use sync methods for ergonomics and async methods when composing with other
    Ray tasks or actors.
    """

    _actor: Any
    name: str

    def call(self, handler_name: str, *args, **kwargs) -> Any:
        return ray.get(self._actor.call.remote(handler_name, *args, **kwargs))

    def call_async(self, handler_name: str, *args, **kwargs):
        return self._actor.call.remote(handler_name, *args, **kwargs)

    def register_handler_module(self, module_name: str, *, reload_module: bool = True) -> dict[str, Any]:
        return ray.get(
            self._actor.load_handlers_from_module.remote(
                module_name,
                reload_module=reload_module,
            )
        )

    def reload_handlers(self, module_names: list[str] | None = None) -> dict[str, Any]:
        return ray.get(self._actor.reload_handlers.remote(module_names))

    def list_handlers(self) -> dict[str, Any]:
        return ray.get(self._actor.list_handlers.remote())

    def status(self) -> dict[str, Any]:
        return ray.get(self._actor.status.remote())

    def shutdown(self) -> bool:
        return ray.get(self._actor.shutdown.remote())
