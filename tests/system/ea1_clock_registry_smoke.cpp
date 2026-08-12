/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Smoke: ClockRegistryBuilder, TagAllocator, and SourceAcceptanceAllocator
// sequence allocations correctly; a ManualClock advanced after registration
// is visible through the resolved default-clock handle.

#include <cstdint>
#include <memory>
#include <utility>

#include <holoscan/core/source_id.hpp>
#include <holoscan/core/tag.hpp>
#include <holoscan/time/clock_registry.hpp>
#include <holoscan/time/manual_clock.hpp>

#include <core/source_id_internal.hpp>
#include <core/tag_internal.hpp>
#include <time/clock_registry_internal.hpp>

int main() {
  using holoscan::ClockKey;
  using holoscan::ManualClock;

  auto manual_clock = std::make_shared<ManualClock>(100);
  holoscan::detail::ClockRegistryBuilder builder;
  const auto clock_key = ClockKey::named("manual");
  if (!builder.add_clock(clock_key,
                         manual_clock,
                         holoscan::detail::ClockCapabilityOptions{
                             .can_read_now = true,
                             .can_wait_until = true,
                             .steady = true,
                         }) ||
      !builder.set_default(clock_key)) {
    return 1;  // exit 1: clock registration or default-clock assignment failed
  }

  auto registry_result = builder.build();
  if (!registry_result) {
    return 2;  // exit 2: ClockRegistryBuilder::build() returned an error
  }
  auto registry = std::move(registry_result).value();
  auto default_clock = registry.resolve(ClockKey::default_clock());
  if (!default_clock || !default_clock->now() || default_clock->now()->timestamp_ns != 100) {
    return 3;  // exit 3: default clock not resolvable or initial timestamp (100 ns) wrong
  }

  // TagAllocator issues monotonically increasing sequence numbers per generation;
  // publish_restart() advances the generation and resets the sequence to 1.
  holoscan::detail::TagAllocator tags;
  auto first = tags.reserve_candidate();
  auto no_compute = tags.reserve_candidate();
  if (!first || !no_compute || first->sequence != 1 || no_compute->sequence != 2 ||
      first->generation != 1 || !tags.publish_restart()) {
    return 4;  // exit 4: tag sequences not 1/2 in generation 1, or restart failed
  }
  auto after_restart = tags.reserve_candidate();
  if (!after_restart || after_restart->sequence != 1 || after_restart->generation != 2) {
    return 5;  // exit 5: post-restart tag does not have sequence 1 / generation 2
  }

  holoscan::detail::SourceAcceptanceAllocator source_acceptances;
  auto source_acceptance = source_acceptances.reserve_acceptance();
  if (!source_acceptance || source_acceptance->generation != 1 ||
      source_acceptance->sequence != 1) {
    return 6;  // exit 6: first source acceptance is not generation 1 / sequence 1
  }

  // Advancing the ManualClock after build must be visible through the resolved handle.
  manual_clock->advance_to(250);
  auto now = default_clock->now();
  return now && now->timestamp_ns == 250 && now->clock_id == default_clock->id() ? 0 : 7;
  // Exit 7: advanced timestamp (250 ns) or clock ID not reflected through resolved handle
}
