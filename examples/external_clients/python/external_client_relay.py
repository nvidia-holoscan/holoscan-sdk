#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run a finite Python relay graph for the standalone client examples."""

from __future__ import annotations

import argparse
from pathlib import Path

import holoscan as hs

INPUT_ENDPOINT = "example.external_clients.input"
OUTPUT_ENDPOINT = "example.external_clients.output"


class Relay(hs.Operator):
    """Forward each imported scalar sample to the exported topic."""

    input = hs.Input("input", hs.payload.uint32)
    output = hs.Output("output", hs.payload.uint32)

    def setup(self, spec: hs.OperatorSpec) -> None:
        """Declare bounded scalar ports."""
        spec.input(self.input, queue_depth=8)
        spec.output(self.output)

    def contract(self) -> hs.Contract:
        """Run once for every imported sample."""
        return hs.Contract().trigger(hs.OnEach(self.input))

    def compute(self, context: hs.ExecutionContext) -> None:
        """Preserve each value and its accepted order."""
        try:
            value = self.input.receive_data()
        except hs.errors.NotReadyError:
            # A connection/control wake may precede cross-process sample materialization.
            return
        self.output.emit(value)


def parse_args() -> argparse.Namespace:
    """Parse the endpoint-catalog output path."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", required=True, type=Path)
    parser.add_argument("--instance", required=True)
    return parser.parse_args()


def build_plan(instance: str) -> hs.Plan:
    """Author and compile the finite import-relay-export graph."""
    input_endpoint = f"{INPUT_ENDPOINT}.{instance}"
    output_endpoint = f"{OUTPUT_ENDPOINT}.{instance}"
    graph = hs.Graph("python-external-client-relay")
    clock = graph.add_clock(hs.RealtimeClock, "runtime-clock", use_time_since_epoch=True)
    graph.set_default_clock(clock)
    required = hs.topics.TopicAttachPolicy(
        peer=hs.topics.PeerRequirement.REQUIRED,
        max_wait=hs.time.ms(10_000),
    )
    imported = graph.import_topic(
        f"holomq://{input_endpoint}",
        hs.payload.uint32,
        endpoint_name=input_endpoint,
        capture_time=clock,
        attach=required,
        bounds=hs.topics.EndpointBounds(
            publisher=hs.topics.PublisherEndpointBounds(max_loaned_samples=1)
        ),
    )
    exported = graph.export_topic(
        f"holomq://{output_endpoint}",
        hs.payload.uint32,
        endpoint_name=output_endpoint,
        attach=required,
        bounds=hs.topics.EndpointBounds(
            subscriber=hs.topics.SubscriberEndpointBounds(
                queue_depth=8,
                max_borrowed_samples=1,
            )
        ),
    )
    relay = graph.op(Relay, "relay")
    graph.add_flow(imported.output, relay.input, queue_depth=8)
    graph.add_flow(relay.output, exported.input, queue_depth=8)
    return hs.compile(graph)


def main() -> None:
    """Publish the catalog, then run until imported producer EOS drains."""
    args = parse_args()
    plan = build_plan(args.instance)
    if not plan:
        raise SystemExit(plan.summary())
    plan.endpoint_catalog.save_json(args.catalog)
    print(f"catalog ready: {args.catalog}", flush=True)

    session = hs.run(plan)
    if session.termination is not hs.RunTermination.COMPLETED or session.diagnostics:
        diagnostics = [(item.token, item.message) for item in session.diagnostics]
        raise RuntimeError(f"unexpected run result: {session!r}; diagnostics={diagnostics!r}")
    print("Python external client relay complete")


if __name__ == "__main__":
    main()
