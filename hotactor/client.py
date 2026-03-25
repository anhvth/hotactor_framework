from __future__ import annotations

from dataclasses import dataclass
import time
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

    @classmethod
    def connect_named(
        cls,
        actor_name: str,
        *,
        namespace: str | None = None,
        address: str = "auto",
        ignore_reinit_error: bool = True,
        wait_timeout_s: float = 30.0,
        poll_interval_s: float = 0.5,
    ):
        """Connect to a named Ray actor.

        By default this waits briefly for the actor to appear, which makes
        service clients more resilient to server startup time or a slow rebuild.
        """
        ray.init(address=address, ignore_reinit_error=ignore_reinit_error)

        deadline = None
        if wait_timeout_s is not None and wait_timeout_s > 0:
            deadline = time.monotonic() + wait_timeout_s

        while True:
            try:
                handle = ray.get_actor(actor_name, namespace=namespace)
                return cls(_actor=handle, name=actor_name)
            except ValueError as exc:
                if deadline is None or time.monotonic() >= deadline:
                    namespace_note = f" in namespace {namespace!r}" if namespace is not None else ""
                    raise RuntimeError(
                        f"Timed out waiting for named actor {actor_name!r}{namespace_note}."
                    ) from exc
                time.sleep(poll_interval_s)

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
