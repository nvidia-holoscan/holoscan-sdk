// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "adapters.hpp"
#include <memory>
#include "registered_adapters.hpp"

#include <chrono>
#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include <nanobind/stl/shared_ptr.h>

namespace holoscan::testing::python {
namespace {

template <typename T>
class TypedAdapter final : public PayloadAdapter {
 public:
  TypedAdapter(ScalarIdentityKind kind, std::shared_ptr<const BindingOwner> owner)
      : kind_(kind), owner_(std::move(owner)) {}

  StagedInput stage(const nb::sequence& values, const nb::sequence& times) const override {
    if (nb::len(values) != nb::len(times)) {
      throw nb::value_error("sample values and timestamps must have the same length");
    }
    std::vector<TestSample<T>> samples;
    samples.reserve(nb::len(values));
    for (std::size_t index = 0; index < nb::len(values); ++index) {
      T value{};
      if constexpr (std::same_as<T, Tensor>) {
        value = owner_->services->tensor_from_python(values[index]);
      } else {
        const auto scalar = owner_->services->scalar_from_python(values[index], kind_);
        if (!scalar.try_as(value)) {
          throw nb::type_error("canonical scalar conversion did not preserve the declared width");
        }
      }
      TestSample<T> sample{.data = std::move(value)};
      if (!times[index].is_none()) {
        sample.capture_coordinate = std::chrono::nanoseconds{nb::cast<std::int64_t>(times[index])};
      }
      samples.push_back(std::move(sample));
    }
    return staged_samples<T>(std::move(samples));
  }

  CaptureRequest capture(CapturePolicy policy, std::size_t capacity,
                         std::size_t maximum_bytes) const override {
    if constexpr (std::same_as<T, Tensor>) {
      if (policy == CapturePolicy::kSnapshot) {
        return snapshot_capture<T>(capacity, maximum_bytes);
      }
      if (policy == CapturePolicy::kNative) {
        return retain_capture<T>(capacity);
      }
    } else {
      if (policy == CapturePolicy::kValue) {
        return value_capture<T>(capacity);
      }
    }
    throw nb::value_error("capture policy does not match the declared output payload");
  }

  nb::list samples(const TestResult& result, std::string_view name) const override {
    nb::list output;
    for (const auto& sample : result.samples<T>(name)) {
      if constexpr (std::same_as<T, Tensor>) {
        output.append(owner_->services->tensor_sample(authoring::ForeignTensorSample{
            .has_data = true, .data = sample.data, .metadata = sample.metadata}));
      } else {
        output.append(owner_->services->scalar_sample(
            authoring::ForeignScalarSample{.has_data = true,
                                           .data = ParamScalarValue::from(sample.data),
                                           .metadata = sample.metadata}));
      }
    }
    return output;
  }

  nb::list snapshots(const TestResult& result, std::string_view name) const override {
    if constexpr (std::same_as<T, Tensor>) {
      nb::list output;
      for (auto& snapshot : result.snapshots<T>(name)) {
        output.append(nb::cast(std::move(snapshot)));
      }
      return output;
    }
    throw nb::value_error("output was captured as values, not snapshots");
  }

  CapturePolicy default_policy() const noexcept override {
    return std::same_as<T, Tensor> ? CapturePolicy::kSnapshot : CapturePolicy::kValue;
  }

 private:
  ScalarIdentityKind kind_;
  std::shared_ptr<const BindingOwner> owner_;
};

}  // namespace

std::shared_ptr<const PayloadAdapter> payload_adapter(const PortRecord& port,
                                                      std::shared_ptr<const BindingOwner> owner) {
  if (port.payload_identity.kind == PayloadIdentityKind::kTensor) {
    return std::make_shared<TypedAdapter<Tensor>>(ScalarIdentityKind::kBoolean, std::move(owner));
  }
  if (port.payload_identity.kind == PayloadIdentityKind::kScalar && port.payload_identity.scalar) {
    const auto kind = port.payload_identity.scalar->kind;
    switch (kind) {
      case ScalarIdentityKind::kBoolean:
        return std::make_shared<TypedAdapter<bool>>(kind, std::move(owner));
      case ScalarIdentityKind::kUInt8:
        return std::make_shared<TypedAdapter<std::uint8_t>>(kind, std::move(owner));
      case ScalarIdentityKind::kInt8:
        return std::make_shared<TypedAdapter<std::int8_t>>(kind, std::move(owner));
      case ScalarIdentityKind::kUInt16:
        return std::make_shared<TypedAdapter<std::uint16_t>>(kind, std::move(owner));
      case ScalarIdentityKind::kInt16:
        return std::make_shared<TypedAdapter<std::int16_t>>(kind, std::move(owner));
      case ScalarIdentityKind::kUInt32:
        return std::make_shared<TypedAdapter<std::uint32_t>>(kind, std::move(owner));
      case ScalarIdentityKind::kInt32:
        return std::make_shared<TypedAdapter<std::int32_t>>(kind, std::move(owner));
      case ScalarIdentityKind::kUInt64:
        return std::make_shared<TypedAdapter<std::uint64_t>>(kind, std::move(owner));
      case ScalarIdentityKind::kInt64:
        return std::make_shared<TypedAdapter<std::int64_t>>(kind, std::move(owner));
      case ScalarIdentityKind::kFloat32:
        return std::make_shared<TypedAdapter<float>>(kind, std::move(owner));
      case ScalarIdentityKind::kFloat64:
        return std::make_shared<TypedAdapter<double>>(kind, std::move(owner));
      case ScalarIdentityKind::kLocalEnum:
        break;
    }
  }
  return registered_payload_adapter(port, std::move(owner));
}

}  // namespace holoscan::testing::python
