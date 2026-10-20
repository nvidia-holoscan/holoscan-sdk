// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Display selection for an ExclusiveDisplaySinkDesc (HolovizOp). Module-private: not exported,
// not installed.

#pragma once

#include <functional>
#include <string>
#include <vector>

#include <holoscan/core/errors.hpp>
#include <holoscan/core/expected.hpp>
#include <holoviz/adapter.hpp>
#include <holoviz/display.hpp>

namespace holoscan::visualization {

/// Enumeration seam for resolve_exclusive_display(). Defaults to the HoloViz entry points; tests
/// substitute fakes to drive per-adapter enumeration failures without a GPU.
struct DisplayEnumerator {
  std::function<holoviz::expected<std::vector<holoviz::Adapter>>()> enumerate_adapters = [] {
    return holoviz::enumerate_adapters();
  };
  std::function<holoviz::expected<std::vector<holoviz::Display>>(const holoviz::Adapter&)>
      enumerate_displays =
          [](const holoviz::Adapter& adapter) { return holoviz::enumerate_displays(adapter); };
};

/// Pick the display for an ExclusiveDisplaySinkDesc. A non-empty \p display_name is a
/// case-sensitive substring match against the driver/EDID display names, first match in adapter
/// order (the holoviz::get_display() rule); an empty name selects the first display classified as
/// free (DisplayOwnership::kAvailable), falling back to the first unclassified one (kUnknown,
/// logged as a warning); compositor-owned displays are never chosen.
///
/// Display enumeration is best-effort per adapter: an adapter whose enumeration fails is skipped so
/// a usable display on a later adapter still wins. When nothing usable matches, the returned
/// kNotFound error is prefixed with \p who and carries the connected displays with their ownership
/// plus, if any adapter was skipped, its name and the enumeration error, so a driver failure is not
/// mistaken for a missing panel. An enumerate_adapters() failure is returned through
/// translate_error() with its own code. Never throws: HolovizOp::start(), which has no error return
/// channel, is the one place that turns the error into a RuntimeError.
[[nodiscard]] holoscan::expected<holoviz::Display, holoscan::Error> resolve_exclusive_display(
    const std::string& display_name, const char* who,
    const DisplayEnumerator& enumerator = DisplayEnumerator{});

}  // namespace holoscan::visualization
