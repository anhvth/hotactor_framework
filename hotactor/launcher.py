from __future__ import annotations

import time
from typing import Any

import ray

from .client import HotActorClient
from .runtime import _HotActorRuntime
from .state import HostedState


def launch_actor(
    *,
    name: str,
    state_cls: type[HostedState],
    state_kwargs: dict[str, Any] | None = None,
    handler_modules: list[str] | None = None,
    ray_options: dict[str, Any] | None = None,
    actor_name: str | None = None,
    namespace: str | None = None,
    get_if_exists: bool = False,
) -> HotActorClient:
    """Launch a managed actor and return a friendly client.

    Parameters
    ----------
    name:
        Human-readable service name used by framework status.
    state_cls:
        Plain Python class derived from HostedState. It owns long-lived,
        expensive resources.
    state_kwargs:
        Keyword arguments passed to state_cls(...).
    handler_modules:
        Python modules that define lightweight handlers.
    ray_options:
        Standard ray.remote actor options, for example:
        {"num_cpus": 0, "num_gpus": 1, "max_concurrency": 4}
    actor_name:
        Optional Ray named actor identifier.
    namespace:
        Optional Ray namespace for named actors.
    get_if_exists:
        If True and actor_name is set, reuse an existing named actor.
    """

    if not issubclass(state_cls, HostedState):
        raise TypeError("state_cls must inherit HostedState")

    ray_actor_options = dict(ray_options or {})
    if actor_name is not None:
        ray_actor_options["name"] = actor_name
        if get_if_exists:
            ray_actor_options["get_if_exists"] = True
    if namespace is not None:
        ray_actor_options["namespace"] = namespace

    RuntimeActor = ray.remote(**ray_actor_options)(_HotActorRuntime)
    actor_handle = RuntimeActor.remote(
        name=name,
        state_cls=state_cls,
        state_kwargs=state_kwargs,
        handler_modules=handler_modules,
        auto_build=True,
    )
    return HotActorClient(_actor=actor_handle, name=name)


def launch_or_replace_actor(
    *,
    name: str,
    state_cls: type[HostedState],
    state_kwargs: dict[str, Any] | None = None,
    handler_modules: list[str] | None = None,
    ray_options: dict[str, Any] | None = None,
    actor_name: str | None = None,
    namespace: str | None = None,
    get_if_exists: bool = False,
    replace_existing: bool = False,
    replace_wait_s: float = 0.5,
    auto_init_ray: bool = True,
    ray_init_kwargs: dict[str, Any] | None = None,
) -> HotActorClient:
    """Launch actor with optional Ray init and optional replacement of named actor.

    This helper is useful for service entrypoints that need consistent startup
    behavior without repeating ray.init + get_actor + shutdown boilerplate.
    """
    if auto_init_ray:
        init_kwargs = dict(ray_init_kwargs or {})
        init_kwargs.setdefault("ignore_reinit_error", True)
        ray.init(**init_kwargs)

    if replace_existing and actor_name is None:
        raise ValueError("replace_existing=True requires actor_name")

    if replace_existing:
        try:
            existing = ray.get_actor(actor_name, namespace=namespace)
        except ValueError:
            existing = None
        if existing is not None:
            HotActorClient(_actor=existing, name=actor_name).shutdown()
            if replace_wait_s > 0:
                time.sleep(replace_wait_s)

    return launch_actor(
        name=name,
        state_cls=state_cls,
        state_kwargs=state_kwargs,
        handler_modules=handler_modules,
        ray_options=ray_options,
        actor_name=actor_name,
        namespace=namespace,
        get_if_exists=get_if_exists,
    )
