/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <chrono>
#include <cstdint>
#include <type_traits>
#include <vector>

#include <holoscan/core/predicate.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/core/temporal_contract_records.hpp>
#include <holoscan/holoscan.hpp>

namespace {

using namespace std::chrono_literals;

class InstalledTemporalOperator final : public holoscan::Operator<> {
 public:
  holoscan::Input<std::uint32_t> trigger_{"trigger"};
  holoscan::Input<std::uint32_t> latest_{"latest"};

  holoscan::Contract contract() const override {
    holoscan::Contract value;
    value.trigger(holoscan::OnEach{trigger_})
        .input(latest_, holoscan::LatestWithin{.max_age = 5ms})
        .on_missing(latest_, holoscan::ReuseLast{.max_age = 20ms})
        .deadline(2ms)
        .on_violation(holoscan::OnViolation::kFailRun);
    return value;
  }

  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext&) override {
    return {};
  }
};

static_assert(std::is_copy_constructible_v<holoscan::Contract>);
static_assert(std::is_copy_constructible_v<holoscan::PredicateBinding>);
template <typename T>
concept HasMaxSamples = requires(T value) { value.max_samples; };

static_assert(!HasMaxSamples<holoscan::DrainAll>);

}  // namespace

// REQ: CONTRACT-013
int main() {
  InstalledTemporalOperator op;
  auto contract = op.contract();
  auto copied = contract;
  auto binding =
      holoscan::predicate(holoscan::PredicateKey{"nvidia.holoscan.installed.always", {1, 0, 0}},
                          [](const holoscan::PredicateEvaluationContext&) -> bool { return true; });
  if (!binding || binding.name() != "nvidia.holoscan.installed.always") {
    return 1;
  }
  const holoscan::ContractRecord record{
      .operator_name = "installed",
      .trigger = holoscan::TemporalRecordTriggerKind::kAnyInput,
      .trigger_pins = {{.input_name = "trigger",
                        .retained_pin_capacity = 1,
                        .capacity_origin = holoscan::TemporalRecordBoundOrigin::kResolvedRetention,
                        .provenance = {.source = "compiler.trigger-pin", .declaration_order = 1}}},
      .provenance = {.source = "operator.contract", .declaration_order = 0},
  };
  const std::vector<holoscan::EdgeSelectionRecord> edges;
  const auto serialized = holoscan::serialize_temporal_records(record, edges);
  if (!serialized || serialized->empty()) {
    return 2;
  }
  static_cast<void>(copied);
  return 0;
}
