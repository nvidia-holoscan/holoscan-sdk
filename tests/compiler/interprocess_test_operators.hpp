/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <cstdint>

#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/port.hpp>

namespace holoscan::test {

// Copyable hooks keep executable-specific probe state outside the shared Operator implementation.
struct ScalarProcessBoundaryHooks {
  std::uint32_t payload{};
  void (*source_started)() = nullptr;
  void (*sink_started)() = nullptr;
  void (*sample_received)(std::uint32_t) = nullptr;
};

struct FanoutProcessBoundaryHooks {
  std::uint32_t sample_count{};
  void (*source_started)() = nullptr;
  void (*sink_started)(std::size_t) = nullptr;
  void (*sample_emitted)() = nullptr;
  void (*sample_received)(std::size_t, std::uint32_t) = nullptr;
};

// Clock-driven scalar source shared by in-process and exec-isolated boundary tests.
class ScalarProcessBoundarySource final : public Operator<> {
 public:
  explicit ScalarProcessBoundarySource(ScalarProcessBoundaryHooks hooks);

  void setup(OperatorSpec& spec) override;
  [[nodiscard]] Contract contract() const override;
  void start() override;
  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override;

  Output<std::uint32_t> output;

 private:
  ScalarProcessBoundaryHooks hooks_;
};

// Latest-within scalar sink shared by in-process and exec-isolated boundary tests.
class ScalarProcessBoundarySink final : public Operator<> {
 public:
  explicit ScalarProcessBoundarySink(ScalarProcessBoundaryHooks hooks);

  void setup(OperatorSpec& spec) override;
  [[nodiscard]] Contract contract() const override;
  void start() override;
  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override;

  Input<std::uint32_t> input;

 private:
  ScalarProcessBoundaryHooks hooks_;
};

// Clock-driven ordered sequence source shared by process-local and cross-process fanout tests.
class FanoutProcessBoundarySource final : public Operator<> {
 public:
  explicit FanoutProcessBoundarySource(FanoutProcessBoundaryHooks hooks);

  void setup(OperatorSpec& spec) override;
  [[nodiscard]] Contract contract() const override;
  void start() override;
  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override;

  Output<std::uint32_t> output;

 private:
  FanoutProcessBoundaryHooks hooks_;
  std::uint32_t next_value_{1U};
};

// One indexed fanout consumer; the executable-specific hook decides how to retain observations.
class FanoutProcessBoundarySink final : public Operator<> {
 public:
  FanoutProcessBoundarySink(FanoutProcessBoundaryHooks hooks, std::size_t sink_index);

  void setup(OperatorSpec& spec) override;
  [[nodiscard]] Contract contract() const override;
  void start() override;
  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override;

  Input<std::uint32_t> input;

 private:
  FanoutProcessBoundaryHooks hooks_;
  std::size_t sink_index_{};
};

}  // namespace holoscan::test
