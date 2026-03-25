from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import ray

from .client import HotActorClient


def default_status_header(status: dict[str, Any]) -> list[str]:
    lines = [f"Actor ready. Lifecycle: {status['host']['lifecycle']}"]
    build_elapsed = status.get("build_elapsed_s")
    if build_elapsed is not None:
        lines.append(f"Build time: {build_elapsed:.1f}s")
    return lines


def run_service_loop(
    client: HotActorClient,
    *,
    heartbeat_interval: int = 30,
    status_header: Callable[[dict[str, Any]], list[str]] | None = None,
    heartbeat_formatter: Callable[[dict[str, Any]], str] | None = None,
    on_shutdown: Callable[[HotActorClient], None] | None = None,
) -> None:
    """Print ready status, run heartbeat loop, and shutdown cleanly on Ctrl-C."""
    status = client.status()
    header_fn = status_header or default_status_header
    for line in header_fn(status):
        print(line)
    print("Waiting for requests. Ctrl-C to stop.")

    heartbeat_fn = heartbeat_formatter or (lambda s: f"[heartbeat] lifecycle={s['host']['lifecycle']}")

    try:
        while True:
            time.sleep(heartbeat_interval)
            print(heartbeat_fn(client.status()))
    except KeyboardInterrupt:
        print("\nShutting down actor...")
        if on_shutdown is not None:
            on_shutdown(client)
        else:
            client.shutdown()
        ray.shutdown()
        print("Done.")

