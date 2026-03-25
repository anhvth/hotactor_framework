from __future__ import annotations

from collections.abc import Callable


def actor_handler(fn: Callable | None = None, *, name: str | None = None):
    """Mark a function as a framework handler.

    Usage:
        @actor_handler
        def forward(state, inputs):
            ...

        @actor_handler(name="predict")
        def my_forward(state, inputs):
            ...
    """

    def decorate(inner_fn: Callable) -> Callable:
        inner_fn.__hotactor_handler__ = True
        inner_fn.__hotactor_name__ = name or inner_fn.__name__
        return inner_fn

    if fn is None:
        return decorate
    return decorate(fn)
