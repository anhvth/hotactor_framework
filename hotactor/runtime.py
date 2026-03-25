from __future__ import annotations

import importlib
import sys
import time
import traceback
from collections.abc import Callable
from types import ModuleType
from typing import Any

from .state import HostLifecycleView, HostedState


class _HotActorRuntime:
    """Internal runtime that is intended to be wrapped by a Ray actor.

    User code should not instantiate this class directly.
    """

    def __init__(
        self,
        name: str,
        state_cls: type[HostedState],
        state_kwargs: dict[str, Any] | None = None,
        handler_modules: list[str] | None = None,
        auto_build: bool = True,
    ) -> None:
        self._host_view = HostLifecycleView(name=name)
        self._state = state_cls(**(state_kwargs or {}))
        if not isinstance(self._state, HostedState):
            raise TypeError(f"state_cls must construct a HostedState, got {type(self._state)!r}")
        self._state._bind_host_view(self._host_view)

        self._handlers: dict[str, Callable[..., Any]] = {}
        self._handler_sources: dict[str, str] = {}
        self._handler_versions: dict[str, int] = {}
        self._modules: dict[str, ModuleType] = {}
        self._known_handler_modules = list(handler_modules or [])
        self._build_elapsed_s: float | None = None

        try:
            if auto_build:
                t0 = time.monotonic()
                self._host_view.set_state("building", action="build")
                self._state.build()
                self._build_elapsed_s = time.monotonic() - t0
            if self._known_handler_modules:
                for module_name in self._known_handler_modules:
                    self.load_handlers_from_module(module_name, reload_module=False)
            self._host_view.set_state("idle")
        except Exception as exc:
            self._host_view.mark_error(str(exc))
            raise

    @property
    def state(self) -> HostedState:
        return self._state

    def _register_handler(self, name: str, fn: Callable[..., Any], *, source: str) -> None:
        self._handlers[name] = fn
        self._handler_sources[name] = source
        self._handler_versions[name] = self._handler_versions.get(name, 0) + 1

    def load_handlers_from_module(
        self,
        module_name: str,
        *,
        reload_module: bool = True,
        names: list[str] | None = None,
    ) -> dict[str, Any]:
        module = self._import_module(module_name, reload_module=reload_module)
        self._modules[module_name] = module
        if module_name not in self._known_handler_modules:
            self._known_handler_modules.append(module_name)

        loaded: list[str] = []
        for handler_name, fn in self._discover_module_handlers(module, names=names).items():
            self._register_handler(handler_name, fn, source=f"module:{module_name}")
            loaded.append(handler_name)

        return {
            "ok": True,
            "module": module_name,
            "handlers": loaded,
        }

    def reload_handlers(self, module_names: list[str] | None = None) -> dict[str, Any]:
        target_modules = module_names or list(self._known_handler_modules)
        reloaded = []
        for module_name in target_modules:
            self.load_handlers_from_module(module_name, reload_module=True)
            reloaded.append(module_name)
        return {
            "ok": True,
            "reloaded_modules": reloaded,
        }

    def call(self, name: str, *args, **kwargs) -> Any:
        fn = self._handlers.get(name)
        if fn is None:
            known = ", ".join(sorted(self._handlers)) or "<none>"
            raise KeyError(f"Unknown handler '{name}'. Known handlers: {known}")

        self._host_view.begin_handler(name)
        self._host_view.lifecycle = "running"
        try:
            result = fn(self._state, *args, **kwargs)
            self._host_view.finish_handler(name, lifecycle="idle")
            return result
        except Exception as exc:
            self._host_view.mark_error(f"{type(exc).__name__}: {exc}")
            raise RuntimeError(
                f"Handler '{name}' failed:\n{traceback.format_exc()}"
            ) from exc

    def list_handlers(self) -> dict[str, dict[str, Any]]:
        return {
            name: {
                "source": self._handler_sources.get(name),
                "version": self._handler_versions.get(name, 0),
            }
            for name in sorted(self._handlers)
        }

    def status(self) -> dict[str, Any]:
        return {
            "host": self._host_view.to_dict(),
            "build_elapsed_s": self._build_elapsed_s,
            "handlers": self.list_handlers(),
            "state": self._state.status(),
        }

    def shutdown(self) -> bool:
        try:
            self._host_view.set_state("stopping", action="teardown")
            self._state.teardown()
            self._host_view.set_state("shutdown")
            return True
        except Exception as exc:
            self._host_view.mark_error(str(exc))
            return False

    def _import_module(self, module_name: str, *, reload_module: bool) -> ModuleType:
        if module_name in self._modules and reload_module:
            return importlib.reload(self._modules[module_name])
        if module_name in sys.modules and reload_module:
            return importlib.reload(sys.modules[module_name])
        return importlib.import_module(module_name)

    def _discover_module_handlers(
        self,
        module: ModuleType,
        *,
        names: list[str] | None,
    ) -> dict[str, Callable[..., Any]]:
        explicit_names = names
        if explicit_names is None:
            exported = getattr(module, "__hotactor_handlers__", None)
            if exported is not None:
                explicit_names = list(exported)

        if explicit_names is not None:
            handlers = {}
            for name in explicit_names:
                fn = getattr(module, name)
                handler_name = getattr(fn, "__hotactor_name__", name)
                handlers[handler_name] = fn
            return handlers

        discovered = {}
        for attr_name, attr_value in vars(module).items():
            if not callable(attr_value):
                continue
            if not getattr(attr_value, "__hotactor_handler__", False):
                continue
            handler_name = getattr(attr_value, "__hotactor_name__", attr_name)
            discovered[handler_name] = attr_value
        return discovered
