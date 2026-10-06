/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <array>
#include <cstring>
#include <holoscan/testing/operator_test_harness.hpp>
#include <iostream>
#include <utility>

namespace {
class HostTensorProducer final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    spec.output(out, "out")
        .produces_tensor(holoscan::TensorOutputSpec{
            .representation = {.memory_kind = holoscan::MemoryKind::kHost,
                               .dtype = DLDataType{kDLInt, 32, 1},
                               .rank = 1U},
            .bounds = holoscan::tensor_bounds(16U),
            .storage = holoscan::TensorOutputStorage::kRuntimePool});
  }
  holoscan::Contract contract() const override {
    holoscan::Contract c;
    c.trigger(holoscan::OnClock{.period = std::chrono::milliseconds{1}})
        .complete_after(holoscan::ComputeCalls{.limit = 2U});
    return c;
  }
  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext&) override {
    const std::array<std::int64_t, 1> shape{4};
    auto loan = out.allocate_tensor(shape);
    if (!loan)
      return holoscan::make_unexpected(std::move(loan).error());
    auto filled = loan->write_host_and_commit([&](const auto& write) {
      for (int i = 0; i < 4; ++i)
        write.template data_as<std::int32_t>()[i] = next_ + i;
    });
    if (!filled)
      return filled;
    next_ += 10;
    return out.emit(std::move(*loan));
  }
  holoscan::Output<holoscan::Tensor> out;

 private:
  std::int32_t next_{10};
};
}  // namespace
int main() {
  auto h = holoscan::testing::create_operator_test<HostTensorProducer>();
  h.capture_snapshot<holoscan::Tensor>("out", 4U, 64U);
  const auto result = h.run();
  if (!result.ok()) {
    std::cerr << result.describe_failure() << '\n';
    return 1;
  }
  const auto snapshots = result.snapshots<holoscan::Tensor>("out");
  if (snapshots.size() != 2U)
    return 1;
  for (std::size_t i = 0; i < snapshots.size(); ++i) {
    if (snapshots[i].bytes().size() != 16U)
      return 1;
    std::array<std::int32_t, 4> values{};
    std::memcpy(values.data(), snapshots[i].bytes().data(), snapshots[i].bytes().size());
    if (values[0] != static_cast<std::int32_t>((i + 1U) * 10U))
      return 1;
  }
  h.capture<holoscan::Tensor>("out", holoscan::testing::Retain{.max_records = 8U});
  const auto retained = h.run();
  if (!retained.ok() || retained.termination() != holoscan::RunTermination::kStopped) {
    std::cerr << retained.describe_failure() << '\n';
    return 1;
  }
  const auto values = retained.values<holoscan::Tensor>("out");
  if (values.size() != 2U || values[0].data_as<std::int32_t>()[0] != 10 ||
      values[1].data_as<std::int32_t>()[0] != 20)
    return 1;
  std::cout << "tensor snapshot example complete: independent pool content verified\n";
}
