#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the external subscriber close-boundary checker."""

from __future__ import annotations

import re
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import check_subscriber_close_boundaries as checker  # noqa: E402


def test_runtime_rejects_private_and_public_subscriber_close() -> None:
    direct_source = """
void retire(ChannelSubscriber<T>& subscriber) {
  close_local_subscriber(owner);
  subscriber.close();
}
"""
    alias_source = """
void retire(ChannelSubscriber<T>& subscriber) {
  close_local_subscriber(owner);
  auto& reader = subscriber;
  reader.close();
}
"""
    for source in (direct_source, alias_source):
        errors = checker.check_runtime_source(Path("runtime.cpp"), source)
        assert len(errors) == 2


def test_runtime_rejects_close_on_inferred_subscriber_factory_result() -> None:
    source = """
void retire(ChannelRuntime& runtime) {
  auto endpoint = runtime.create_subscriber<int>("x").value();
  endpoint.close();
  auto schema_endpoint = runtime.create_tensor_schema_subscriber<Payload>(
      catalog, endpoint_view).value();
  schema_endpoint.close();
}
"""
    errors = checker.check_runtime_source(Path("runtime.cpp"), source)
    assert len(errors) == 1


def test_runtime_rejects_reference_bound_subscriber_factory_results() -> None:
    for declaration in ("auto", "auto&", "auto&&", "const auto&"):
        source = f"""
void retire(ChannelRuntime& runtime) {{
  {declaration} endpoint = runtime.create_subscriber<int>("x").value();
  endpoint.close();
}}
"""
        errors = checker.check_runtime_source(Path("runtime.cpp"), source)
        assert len(errors) == 1


def test_runtime_allows_non_subscriber_close_with_alias_name() -> None:
    source = """
void retire(Resource& reader) {
  reader.close();
}
"""
    assert checker.check_runtime_source(Path("runtime.cpp"), source) == []


def test_runtime_allows_inferred_non_subscriber_factory_result() -> None:
    source = """
void retire(Runtime& runtime) {
  auto endpoint = runtime.create_publisher<int>("x").value();
  endpoint.close();
}
"""
    assert checker.check_runtime_source(Path("runtime.cpp"), source) == []


def test_runtime_ignores_close_spellings_in_comments_and_strings() -> None:
    source = """
// subscriber.close();
const char* text = "close_local_subscriber(owner)";
"""
    assert checker.check_runtime_source(Path("runtime.cpp"), source) == []


def test_close_body_rejects_message_error_factory() -> None:
    source = """
expected<void, Error> close() noexcept {
  return make_unexpected(make_channel_error(ErrorCode::kClosed, "closed"));
}
"""
    errors = checker.check_close_body(
        Path("subscriber.hpp"),
        source,
        re.compile(r"expected\s*<\s*void\s*,\s*Error\s*>\s+close\s*\(\s*\)\s*noexcept"),
        require_delegate=True,
    )
    assert any("make_channel_error" in error for error in errors)


def test_close_body_requires_common_retirement_delegate() -> None:
    signature = re.compile(r"expected\s*<\s*void\s*,\s*Error\s*>\s+close\s*\(\s*\)\s*noexcept")
    delegated_source = """
expected<void, Error> close() noexcept {
  return close_local_subscriber(owner);
}
"""
    assert (
        checker.check_close_body(
            Path("subscriber.hpp"),
            delegated_source,
            signature,
            require_delegate=True,
        )
        == []
    )

    missing_delegate_source = """
expected<void, Error> close() noexcept {
  return {};
}
"""
    errors = checker.check_close_body(
        Path("subscriber.hpp"),
        missing_delegate_source,
        signature,
        require_delegate=True,
    )
    assert any("must delegate" in error for error in errors)
