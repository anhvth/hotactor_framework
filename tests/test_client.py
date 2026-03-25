from __future__ import annotations

import unittest
from types import SimpleNamespace

import hotactor.client as client_mod


class _FakeRay:
    def __init__(self, outcomes: list[object]) -> None:
        self._outcomes = outcomes
        self.init_calls: list[dict[str, object]] = []
        self.get_actor_calls: list[tuple[str, str | None]] = []

    def init(self, **kwargs):
        self.init_calls.append(kwargs)

    def get_actor(self, actor_name: str, namespace: str | None = None):
        self.get_actor_calls.append((actor_name, namespace))
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class HotActorClientConnectNamedTests(unittest.TestCase):
    def test_waits_for_actor(self):
        fake_handle = SimpleNamespace()
        fake_ray = _FakeRay([ValueError("missing"), ValueError("missing"), fake_handle])
        clock = {"value": 0.0}

        def monotonic():
            return clock["value"]

        def sleep(_seconds):
            clock["value"] += 1.0

        original_ray = client_mod.ray
        original_monotonic = client_mod.time.monotonic
        original_sleep = client_mod.time.sleep
        try:
            client_mod.ray = fake_ray
            client_mod.time.monotonic = monotonic
            client_mod.time.sleep = sleep

            client = client_mod.HotActorClient.connect_named(
                "service",
                namespace="ns",
                wait_timeout_s=5.0,
                poll_interval_s=0.1,
            )
        finally:
            client_mod.ray = original_ray
            client_mod.time.monotonic = original_monotonic
            client_mod.time.sleep = original_sleep

        self.assertIs(client._actor, fake_handle)
        self.assertEqual(client.name, "service")
        self.assertEqual(
            fake_ray.init_calls,
            [{"address": "auto", "ignore_reinit_error": True}],
        )
        self.assertEqual(
            fake_ray.get_actor_calls,
            [
                ("service", "ns"),
                ("service", "ns"),
                ("service", "ns"),
            ],
        )

    def test_times_out(self):
        fake_ray = _FakeRay([ValueError("missing"), ValueError("missing")])
        clock = {"value": 0.0}

        def monotonic():
            return clock["value"]

        def sleep(_seconds):
            clock["value"] += 1.0

        original_ray = client_mod.ray
        original_monotonic = client_mod.time.monotonic
        original_sleep = client_mod.time.sleep
        try:
            client_mod.ray = fake_ray
            client_mod.time.monotonic = monotonic
            client_mod.time.sleep = sleep

            with self.assertRaises(RuntimeError) as ctx:
                client_mod.HotActorClient.connect_named(
                    "service",
                    namespace="ns",
                    wait_timeout_s=1.0,
                    poll_interval_s=0.1,
                )
        finally:
            client_mod.ray = original_ray
            client_mod.time.monotonic = original_monotonic
            client_mod.time.sleep = original_sleep

        self.assertIn(
            "Timed out waiting for named actor 'service' in namespace 'ns'.",
            str(ctx.exception),
        )


if __name__ == "__main__":
    unittest.main()
