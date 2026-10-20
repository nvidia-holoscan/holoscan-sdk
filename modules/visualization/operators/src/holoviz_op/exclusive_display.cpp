// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "holoviz_op/exclusive_display.hpp"

#include <string>
#include <vector>

#include <fmt/format.h>

#include <holoscan/core/errors.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/logger/logger.hpp>
#include <holoviz/adapter.hpp>

#include "holoviz_op/error_translation.hpp"

namespace holoscan::visualization {

// Keep the free-before-unknown ownership policy and complete enumeration diagnostics together.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
holoscan::expected<holoviz::Display, holoscan::Error> resolve_exclusive_display(
    const std::string& display_name, const char* who, const DisplayEnumerator& enumerator) {
  auto adapters = enumerator.enumerate_adapters();
  if (!adapters) {
    return holoscan::make_unexpected(translate_error(adapters.error(), "enumerate_adapters"));
  }

  // One enumeration pass feeds the match, the listing, and the diagnostics. Enumeration is
  // best-effort per adapter: an adapter without a display path (or with a transient failure) must
  // not hide a usable display on a later adapter, so it is skipped here and named only in the
  // terminal error, where it keeps a driver failure from reading as a missing panel.
  std::vector<holoviz::Display> candidates;
  std::string skipped;
  for (const auto& adapter : *adapters) {
    auto displays = enumerator.enumerate_displays(adapter);
    if (!displays) {
      skipped += fmt::format(" \"{}\": {};", adapter.name.data(), displays.error().message);
      continue;
    }
    candidates.insert(candidates.end(), displays->begin(), displays->end());
  }

  const auto diagnostics = [&] {
    std::string listing;
    for (const auto& display : candidates) {
      const char* ownership =
          display.ownership == holoviz::DisplayOwnership::kAvailable    ? "available"
          : display.ownership == holoviz::DisplayOwnership::kCompositor ? "compositor-owned"
                                                                        : "unknown";
      listing += fmt::format(" \"{}\" ({}x{}@{}mHz, {});",
                             display.name.data(),
                             display.current_mode.resolution.width,
                             display.current_mode.resolution.height,
                             display.current_mode.refresh_rate_mhz,
                             ownership);
    }
    std::string text = "; connected displays:" + (listing.empty() ? std::string(" none") : listing);
    if (!skipped.empty()) {
      text += "; adapters skipped after enumeration errors:" + skipped;
    }
    return text;
  };

  if (!display_name.empty()) {
    for (const auto& display : candidates) {
      if (std::string(display.name.data()).find(display_name) != std::string::npos) {
        return display;
      }
    }
    return holoscan::make_unexpected(holoscan::Error{
        holoscan::ErrorCode::kNotFound,
        fmt::format("{}: no display matches \"{}\"{}", who, display_name, diagnostics())});
  }

  // A display classified as free wins over one whose ownership could not be classified (no
  // DRM view, e.g. /dev/dri not mapped into the container): kUnknown is still attempted, as
  // Display::create_surface() documents, but only when nothing better is connected. A
  // compositor-owned display is never selected; acquiring it cannot succeed.
  const holoviz::Display* unclassified = nullptr;
  for (const auto& display : candidates) {
    if (display.ownership == holoviz::DisplayOwnership::kAvailable) {
      return display;
    }
    if (display.ownership == holoviz::DisplayOwnership::kUnknown && unclassified == nullptr) {
      unclassified = &display;
    }
  }
  if (unclassified != nullptr) {
    HOLOSCAN_LOG_WARN(
        "{}: ownership of display \"{}\" could not be classified; attempting exclusive "
        "acquisition anyway (name a display explicitly to override this choice)",
        who,
        unclassified->name.data());
    return *unclassified;
  }
  return holoscan::make_unexpected(holoscan::Error{
      holoscan::ErrorCode::kNotFound,
      fmt::format("{}: no display is free for exclusive scanout (detach one from the desktop "
                  "session or name it explicitly){}",
                  who,
                  diagnostics())});
}

}  // namespace holoscan::visualization
