#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared normalized scenario identity for the transport benchmark suite."""

from __future__ import annotations

import json
from typing import Any

SCENARIO_IDENTITY_SCHEMA = "holoscan.transport-benchmark-scenario/v3"
SCENARIO_IDENTITY_FIELDS = (
    "scope",
    "domain",
    "topology",
    "workload",
    "trigger",
    "payload",
    "storage",
    "bulk_work_profile",
    "delivery",
    "execution",
    "synchronization",
    "wake",
    "geometry",
)
SCENARIO_GEOMETRY_FIELDS = (
    "fanout",
    "relay_count",
    "queue_depth",
    "notification_capacity",
    "pool_slots",
    "release_capacity",
    "publication_backpressure_capacity",
    "max_fanout_release_streams",
    "process_count",
    "application_payload_bytes",
    "inline_transport_bytes",
    "external_buffer_bytes",
)


def scenario_identity_contract() -> dict[str, Any]:
    """Return the versioned identity vocabulary embedded in manifests and summaries."""
    return {
        "schema": SCENARIO_IDENTITY_SCHEMA,
        "fields": list(SCENARIO_IDENTITY_FIELDS),
        "geometry_fields": list(SCENARIO_GEOMETRY_FIELDS),
    }


def make_scenario_identity(row: dict[str, Any]) -> dict[str, Any]:
    """Build one exact normalized identity from a summary row."""
    identity = {
        "scope": row["scope"],
        "domain": row["scenario_domain"],
        "topology": row["topology"],
        "workload": row["workload_class"],
        "trigger": row["trigger_mechanism"],
        "payload": row["scenario_payload"],
        "storage": row["storage"],
        "bulk_work_profile": row["bulk_work_profile"],
        "delivery": row["delivery_path"],
        "execution": row["execution_model"],
        "synchronization": row["synchronization"],
        "wake": row["wake_mode"],
        "geometry": {field: row.get(field) for field in SCENARIO_GEOMETRY_FIELDS},
    }
    if tuple(identity) != SCENARIO_IDENTITY_FIELDS:
        raise RuntimeError("scenario identity fields do not match the published contract")
    return identity


def scenario_identity_key(identity: dict[str, Any]) -> str:
    """Return a stable machine-comparable spelling of a normalized identity."""
    return json.dumps(identity, separators=(",", ":"), sort_keys=True)
