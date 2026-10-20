// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <string_view>

#include <nanobind/nanobind.h>

#include <holoscan/testing/operator_test_harness.hpp>

#include "../core/binding_services.hpp"
#include "../core/native_operator.hpp"

namespace holoscan::testing::python {

namespace nb = nanobind;

struct BindingOwner {
  GilSafeObject core_module;
  GilSafeObject testing_module;
  const holoscan::python::BindingServices* services{};
};

enum class CapturePolicy { kValue, kSnapshot, kNative };

// Conversion remains in Core. These adapters own only finite testing staging/capture policy.
class PayloadAdapter {
 public:
  virtual ~PayloadAdapter() = default;
  virtual StagedInput stage(const nb::sequence& values, const nb::sequence& times) const = 0;
  virtual CaptureRequest capture(CapturePolicy policy, std::size_t capacity,
                                 std::size_t maximum_bytes) const = 0;
  virtual nb::list samples(const TestResult& result, std::string_view name) const = 0;
  virtual nb::list snapshots(const TestResult& result, std::string_view name) const = 0;
  virtual CapturePolicy default_policy() const noexcept = 0;
};

std::shared_ptr<const PayloadAdapter> payload_adapter(const PortRecord& port,
                                                      std::shared_ptr<const BindingOwner> owner);

}  // namespace holoscan::testing::python
