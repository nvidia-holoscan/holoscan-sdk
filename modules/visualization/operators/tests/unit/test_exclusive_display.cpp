// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Pins the display-selection contract of resolve_exclusive_display() (HolovizOp's
// ExclusiveDisplaySinkDesc) through its enumeration seam: name matching, the ownership
// preference order, and, above all, that a per-adapter enumeration failure is skipped (a later
// adapter still wins) yet named in the terminal error so a driver failure is not read as a missing
// panel. Everything here runs headless: no instance, no GPU, no display.

#include <gtest/gtest.h>

#include <cstring>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/core/errors.hpp>
#include <holoviz/adapter.hpp>
#include <holoviz/display.hpp>

#include "holoviz_op/exclusive_display.hpp"

namespace holoscan::visualization {
namespace {

using holoviz::DisplayOwnership;

constexpr const char* kWho = "probe";
constexpr const char* kVulkanFailure =
    "VK_ERROR_OUT_OF_HOST_MEMORY from "
    "vkGetPhysicalDeviceDisplayPropertiesKHR";

holoviz::Adapter make_adapter(const char* name) {
  holoviz::Adapter adapter{};
  std::strncpy(adapter.name.data(), name, adapter.name.size() - 1);
  return adapter;
}

holoviz::Display make_display(const char* name, DisplayOwnership ownership) {
  holoviz::Display display{};
  std::strncpy(display.name.data(), name, display.name.size() - 1);
  display.ownership = ownership;
  display.current_mode = {{1920, 1080}, 60000};
  return display;
}

holoviz::Error vulkan_error() {
  return holoviz::Error{holoviz::ErrorCode::kVulkanError, kVulkanFailure};
}

/// A fake enumeration: adapters in declaration order, each mapped to either its displays or an
/// enumeration error. Records the adapters queried so tests can pin that no adapter is skipped
/// by an early return.
class FakeEnumerator {
 public:
  FakeEnumerator& adapter(const char* name, std::vector<holoviz::Display> displays) {
    adapters_.push_back(make_adapter(name));
    results_.emplace(name, std::move(displays));
    return *this;
  }

  FakeEnumerator& failing_adapter(const char* name, holoviz::Error error = vulkan_error()) {
    adapters_.push_back(make_adapter(name));
    results_.emplace(name, holoviz::make_unexpected(std::move(error)));
    return *this;
  }

  DisplayEnumerator seam() {
    return DisplayEnumerator{
        .enumerate_adapters =
            [this] { return holoviz::expected<std::vector<holoviz::Adapter>>(adapters_); },
        .enumerate_displays =
            [this](const holoviz::Adapter& adapter) {
              queried_.emplace_back(adapter.name.data());
              return results_.at(adapter.name.data());
            },
    };
  }

  const std::vector<std::string>& queried() const { return queried_; }

 private:
  std::vector<holoviz::Adapter> adapters_;
  std::map<std::string, holoviz::expected<std::vector<holoviz::Display>>> results_;
  std::vector<std::string> queried_;
};

bool contains(const std::string& haystack, const std::string& needle) {
  return haystack.find(needle) != std::string::npos;
}

}  // namespace

TEST(ExclusiveDisplay, NamedMatchOnLaterAdapterSurvivesEarlierEnumerationFailure) {
  FakeEnumerator fake;
  fake.failing_adapter("gpu-a").adapter("gpu-b",
                                        {make_display("DP-2", DisplayOwnership::kUnknown)});
  const auto display = resolve_exclusive_display("DP-2", kWho, fake.seam());
  ASSERT_TRUE(display.has_value()) << runtime_error_message(display.error());
  EXPECT_STREQ(display->name.data(), "DP-2");
  EXPECT_EQ(fake.queried(), (std::vector<std::string>{"gpu-a", "gpu-b"}));
}

TEST(ExclusiveDisplay, NamedFailureReportsSkippedAdapter) {
  FakeEnumerator fake;
  fake.failing_adapter("gpu-a").adapter("gpu-b",
                                        {make_display("HDMI-A-1", DisplayOwnership::kCompositor)});
  const auto display = resolve_exclusive_display("DP-2", kWho, fake.seam());
  ASSERT_FALSE(display.has_value());
  EXPECT_EQ(display.error().code, holoscan::ErrorCode::kNotFound);
  const std::string& message = display.error().message;
  EXPECT_TRUE(contains(message, "probe: no display matches \"DP-2\"")) << message;
  EXPECT_TRUE(contains(message, "\"HDMI-A-1\" (1920x1080@60000mHz, compositor-owned)")) << message;
  EXPECT_TRUE(contains(
      message,
      "adapters skipped after enumeration errors: \"gpu-a\": " + std::string(kVulkanFailure)))
      << message;
}

TEST(ExclusiveDisplay, NamedFailureWithEveryAdapterSkippedStillNamesThem) {
  // Every adapter failing is still kNotFound (the holoviz::get_display() contract), but the
  // message must carry the failure instead of an empty "none" listing.
  FakeEnumerator fake;
  fake.failing_adapter("gpu-a").failing_adapter("gpu-b");
  const auto display = resolve_exclusive_display("DP-2", kWho, fake.seam());
  ASSERT_FALSE(display.has_value());
  EXPECT_EQ(display.error().code, holoscan::ErrorCode::kNotFound);
  const std::string& message = display.error().message;
  EXPECT_TRUE(contains(message, "connected displays: none")) << message;
  EXPECT_TRUE(contains(message, "\"gpu-a\": ")) << message;
  EXPECT_TRUE(contains(message, "\"gpu-b\": ")) << message;
  EXPECT_EQ(fake.queried().size(), 2u);
}

TEST(ExclusiveDisplay, AutomaticSelectionOnLaterAdapterSurvivesEarlierEnumerationFailure) {
  FakeEnumerator fake;
  fake.failing_adapter("gpu-a").adapter("gpu-b",
                                        {make_display("DP-3", DisplayOwnership::kAvailable)});
  const auto display = resolve_exclusive_display("", kWho, fake.seam());
  ASSERT_TRUE(display.has_value()) << runtime_error_message(display.error());
  EXPECT_STREQ(display->name.data(), "DP-3");
  EXPECT_EQ(fake.queried(), (std::vector<std::string>{"gpu-a", "gpu-b"}));
}

TEST(ExclusiveDisplay, AutomaticFailureReportsSkippedAdapter) {
  FakeEnumerator fake;
  fake.failing_adapter("gpu-a").adapter("gpu-b",
                                        {make_display("DP-1", DisplayOwnership::kCompositor)});
  const auto display = resolve_exclusive_display("", kWho, fake.seam());
  ASSERT_FALSE(display.has_value());
  EXPECT_EQ(display.error().code, holoscan::ErrorCode::kNotFound);
  const std::string& message = display.error().message;
  EXPECT_TRUE(contains(message, "probe: no display is free for exclusive scanout")) << message;
  EXPECT_TRUE(contains(message, "\"DP-1\" (1920x1080@60000mHz, compositor-owned)")) << message;
  EXPECT_TRUE(contains(
      message,
      "adapters skipped after enumeration errors: \"gpu-a\": " + std::string(kVulkanFailure)))
      << message;
}

TEST(ExclusiveDisplay, NoSkippedAdaptersLeavesMessageWithoutTheSkippedFragment) {
  FakeEnumerator fake;
  fake.adapter("gpu-a", {make_display("DP-1", DisplayOwnership::kCompositor)});
  const auto named = resolve_exclusive_display("DP-2", kWho, fake.seam());
  ASSERT_FALSE(named.has_value());
  EXPECT_FALSE(contains(named.error().message, "adapters skipped")) << named.error().message;
  const auto automatic = resolve_exclusive_display("", kWho, fake.seam());
  ASSERT_FALSE(automatic.has_value());
  EXPECT_FALSE(contains(automatic.error().message, "adapters skipped"))
      << automatic.error().message;
}

TEST(ExclusiveDisplay, AdapterEnumerationFailureIsForwardedWithItsOwnCode) {
  DisplayEnumerator seam;
  seam.enumerate_adapters = [] {
    return holoviz::expected<std::vector<holoviz::Adapter>>(
        holoviz::make_unexpected(vulkan_error()));
  };
  seam.enumerate_displays = [](const holoviz::Adapter&) {
    ADD_FAILURE() << "enumerate_displays must not run when adapter enumeration failed";
    return holoviz::expected<std::vector<holoviz::Display>>(std::vector<holoviz::Display>{});
  };
  const auto display = resolve_exclusive_display("DP-2", kWho, seam);
  ASSERT_FALSE(display.has_value());
  EXPECT_NE(display.error().code, holoscan::ErrorCode::kNotFound);
  EXPECT_TRUE(contains(display.error().message, "enumerate_adapters")) << display.error().message;
  EXPECT_TRUE(contains(display.error().message, kVulkanFailure)) << display.error().message;
}

TEST(ExclusiveDisplay, NamedMatchIsASubstringInAdapterOrder) {
  FakeEnumerator fake;
  fake.adapter("gpu-a", {make_display("HDMI-A-1", DisplayOwnership::kCompositor)})
      .adapter("gpu-b",
               {make_display("DP-1", DisplayOwnership::kCompositor),
                make_display("DP-10", DisplayOwnership::kAvailable)});
  const auto display = resolve_exclusive_display("DP-1", kWho, fake.seam());
  ASSERT_TRUE(display.has_value()) << runtime_error_message(display.error());
  EXPECT_STREQ(display->name.data(), "DP-1");  // first substring match, ownership not consulted
}

TEST(ExclusiveDisplay, AutomaticSelectionPrefersAvailableOverEarlierUnclassified) {
  FakeEnumerator fake;
  fake.adapter("gpu-a", {make_display("DP-1", DisplayOwnership::kUnknown)})
      .adapter("gpu-b",
               {make_display("DP-2", DisplayOwnership::kCompositor),
                make_display("DP-3", DisplayOwnership::kAvailable)});
  const auto display = resolve_exclusive_display("", kWho, fake.seam());
  ASSERT_TRUE(display.has_value()) << runtime_error_message(display.error());
  EXPECT_STREQ(display->name.data(), "DP-3");
}

TEST(ExclusiveDisplay, AutomaticSelectionFallsBackToTheFirstUnclassified) {
  FakeEnumerator fake;
  fake.adapter("gpu-a",
               {make_display("DP-1", DisplayOwnership::kCompositor),
                make_display("DP-2", DisplayOwnership::kUnknown),
                make_display("DP-3", DisplayOwnership::kUnknown)});
  const auto display = resolve_exclusive_display("", kWho, fake.seam());
  ASSERT_TRUE(display.has_value()) << runtime_error_message(display.error());
  EXPECT_STREQ(display->name.data(), "DP-2");
}

TEST(ExclusiveDisplay, AutomaticSelectionNeverPicksACompositorOwnedDisplay) {
  FakeEnumerator fake;
  fake.adapter("gpu-a", {make_display("DP-1", DisplayOwnership::kCompositor)});
  const auto display = resolve_exclusive_display("", kWho, fake.seam());
  ASSERT_FALSE(display.has_value());
  EXPECT_EQ(display.error().code, holoscan::ErrorCode::kNotFound);
}

}  // namespace holoscan::visualization
