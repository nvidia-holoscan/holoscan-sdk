/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <utility>

#include <gtest/gtest.h>

#include <holoscan/core/port.hpp>
#include <holoscan/schema/sdk_schemas.hpp>
#include <holoscan/sensor_io/provenance.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>

#include "common/allocation_counter.hpp"

namespace holoscan {
namespace {

template <typename T>
concept HasReferenceEmission = requires(Output<T>& output, const T& value) {
  output.emit_ref(value);
  output.emit_ref(value, EmitOptions{});
};

// REQ: SCHEMA-010
static_assert(HasReferenceEmission<schema::ImuT>);
static_assert(HasReferenceEmission<schema::TimeT>);
static_assert(HasReferenceEmission<schema::StringT>);  // Native-bounds opt-in is not required.
static_assert(!HasReferenceEmission<schema::ImageT>);
static_assert(!HasReferenceEmission<Tensor>);
static_assert(!HasReferenceEmission<std::uint32_t>);

struct RefEncodeBinding {
  alignas(std::max_align_t)
      std::array<std::byte, GeneratedSchemaNativeTraits<schema::ImuT>::max_serialized_size> bytes{};
  std::size_t size{};
  const void* expected_address{};
  bool received_original_address{};
  std::size_t calls{};
  EmitOptions options{};
  ErrorCode status{ErrorCode::kSuccess};

  static expected<void, Error> emit(void* opaque, const void* value, EmitOptions options) noexcept {
    auto& state = *static_cast<RefEncodeBinding*>(opaque);
    ++state.calls;
    state.received_original_address = value == state.expected_address;
    state.options = options;
    if (state.status != ErrorCode::kSuccess)
      return make_unexpected(Error{state.status});
    auto encoded =
        detail::runtime_port_type_ops<schema::ImuT>()->encode_payload(value, state.bytes);
    if (!encoded)
      return make_unexpected(std::move(encoded).error());
    state.size = *encoded;
    return {};
  }
  inline static const detail::RuntimeOutputBindingOps ops{.emit_ref = &emit};
};

// REQ: SCHEMA-010
TEST(SchemaEmitRef, BorrowedPassThroughPreservesProvenanceWithoutScratchOrAllocation) {
  struct Selected {
    Sample<schema::ImuT> sample{schema::ImuT{}, SampleMetadata{}};
    std::uint64_t epoch{1};
  } selected;
  selected.sample.data.header = std::make_shared<schema::HeaderT>();
  selected.sample.data.header->frame_id.assign(128, 'b');
  selected.sample.metadata = {.capture_timestamp_ns = 17,
                              .source_clock_id = ClockId{3},
                              .trace_id = 41,
                              .frame_id = 42,
                              .flags = SampleFlags::kDegraded};
  const detail::RuntimeInputBindingOps input_ops{
      .scalar = [](const void* opaque,
                   std::uint64_t,
                   bool claim) noexcept -> expected<detail::RuntimeScalarReadBinding, Error> {
        if (claim)
          return make_unexpected(Error{ErrorCode::kNotSupported});
        const auto& current = *static_cast<const Selected*>(opaque);
        return detail::RuntimeScalarReadBinding{.sample = &current.sample,
                                                .view_generation = &current.epoch,
                                                .expected_view_generation = current.epoch};
      }};
  detail::RuntimeInputBinding input_binding{.state = &selected, .ops = &input_ops, .generation = 1};
  RefEncodeBinding encoder;
  encoder.expected_address = &selected.sample.data;
  detail::RuntimeOutputBinding output_binding{.state = &encoder, .ops = &RefEncodeBinding::ops};
  Input<schema::ImuT> input;
  Output<schema::ImuT> output;
  const auto* ops = detail::runtime_port_type_ops<schema::ImuT>();
  ops->bind_input(&input, &input_binding);
  ops->bind_output(&output, &output_binding);
  bool accepted = false;
  std::size_t allocations = 0;
  {
    const test::ScopedAllocationCounter counter;
    auto borrowed = input.selected_sample();
    if (borrowed)
      accepted = static_cast<bool>(
          output.emit_ref(borrowed->data(), sensor_io::derive_from(borrowed->metadata())));
    allocations = counter.count();
  }
  ASSERT_TRUE(accepted);
  EXPECT_EQ(allocations, 0U);
  EXPECT_TRUE(encoder.received_original_address);
  EXPECT_EQ(selected.sample.data.header.use_count(), 1);
  ASSERT_TRUE(encoder.options.capture_time);
  EXPECT_EQ(encoder.options.capture_time->timestamp_ns, 17);
  EXPECT_EQ(encoder.options.capture_time->clock_id, ClockId{3});
  EXPECT_EQ(encoder.options.frame_id, 42U);
  EXPECT_EQ(encoder.options.trace->trace_id, 41U);
  EXPECT_EQ(encoder.options.flags, SampleFlags::kDegraded);
  ops->unbind_input(&input, &input_binding);
  ops->unbind_output(&output, &output_binding);
}

// REQ: SCHEMA-010
TEST(SchemaEmitRef, FirstAndRepeatedScratchPublicationEncodeWithoutNativeCopyOrAllocation) {
  schema::ImuT source;
  source.header = std::make_shared<schema::HeaderT>();
  source.header->frame_id.assign(128, 'i');
  source.header->stamp = std::make_shared<schema::TimeT>();
  source.orientation_covariance = std::make_shared<schema::Matrix3f>();
  auto scratch = schema::SchemaScratch<schema::ImuT>::create();
  ASSERT_TRUE(scratch);
  ASSERT_TRUE(scratch->copy_from(source));
  RefEncodeBinding encoder;
  encoder.expected_address = &scratch->value();
  detail::RuntimeOutputBinding binding{.state = &encoder, .ops = &RefEncodeBinding::ops};
  Output<schema::ImuT> output;
  const auto* ops = detail::runtime_port_type_ops<schema::ImuT>();
  ops->bind_output(&output, &binding);
  const auto* header = scratch->value().header.get();
  for (unsigned iteration = 0; iteration < 64; ++iteration) {
    scratch->value().header->device_sequence = iteration;
    bool accepted = false;
    std::size_t allocations = 0;
    {
      const test::ScopedAllocationCounter counter;
      accepted = static_cast<bool>(output.emit_ref(
          scratch->value(),
          {.frame_id = 42, .trace = TraceContext::root(43), .flags = SampleFlags::kDegraded}));
      allocations = counter.count();
    }
    ASSERT_TRUE(accepted);
    EXPECT_EQ(allocations, 0U);
    EXPECT_TRUE(encoder.received_original_address);
    EXPECT_EQ(scratch->value().header.get(), header);
    EXPECT_EQ(encoder.options.frame_id, 42U);
    EXPECT_EQ(encoder.options.trace->trace_id, 43U);
    EXPECT_EQ(encoder.options.flags, SampleFlags::kDegraded);
    // A delayed consumer must observe the published value, not the reused caller scratch.
    ASSERT_TRUE(scratch->reset());
    auto decoded = schema_package<schema::ImuT>().materialize(
        std::span<const std::byte>{encoder.bytes.data(), encoder.size});
    ASSERT_TRUE(decoded);
    ASSERT_NE(decoded->header, nullptr);
    EXPECT_EQ(decoded->header->device_sequence, iteration);
    EXPECT_EQ(decoded->header->frame_id, std::string(128, 'i'));
    ASSERT_TRUE(scratch->copy_from(source));
  }
  EXPECT_EQ(encoder.calls, 64U);
  ops->unbind_output(&output, &binding);
}

// REQ: SCHEMA-010
TEST(SchemaEmitRef, MissingCapabilityFailsClosedAndEveryErrorLeavesCallerOwnedValueIntact) {
  schema::ImuT value;
  value.header = std::make_shared<schema::HeaderT>();
  value.header->frame_id.assign(128, 'e');
  const auto* header = value.header.get();
  Output<schema::ImuT> output;
  auto unbound = output.emit_ref(value);
  ASSERT_FALSE(unbound);
  EXPECT_EQ(unbound.error().code, ErrorCode::kNotReady);
  RefEncodeBinding encoder;
  encoder.expected_address = &value;
  const detail::RuntimeOutputBindingOps unsupported{};
  detail::RuntimeOutputBinding binding{.state = &encoder, .ops = &unsupported};
  const auto* ops = detail::runtime_port_type_ops<schema::ImuT>();
  ops->bind_output(&output, &binding);
  auto missing = output.emit_ref(value);
  ASSERT_FALSE(missing);
  EXPECT_EQ(missing.error().code, ErrorCode::kNotSupported);
  EXPECT_EQ(encoder.calls, 0U);
  binding.ops = &RefEncodeBinding::ops;
  for (const auto code : {ErrorCode::kInvalidArgument,
                          ErrorCode::kResourceExhausted,
                          ErrorCode::kBackpressured,
                          ErrorCode::kStaleEndpoint,
                          ErrorCode::kNotReady,
                          ErrorCode::kClosed,
                          ErrorCode::kInterrupted,
                          ErrorCode::kFailure,
                          ErrorCode::kPublicationPartial,
                          ErrorCode::kPublicationIndeterminate}) {
    encoder.status = code;
    std::size_t allocations = 0;
    expected<void, Error> result;
    {
      const test::ScopedAllocationCounter counter;
      result = output.emit_ref(value);
      allocations = counter.count();
    }
    ASSERT_FALSE(result);
    EXPECT_EQ(result.error().code, code);
    EXPECT_EQ(allocations, 0U);
    EXPECT_TRUE(encoder.received_original_address);
    EXPECT_EQ(value.header.get(), header);
    EXPECT_EQ(value.header.use_count(), 1);
    EXPECT_EQ(value.header->frame_id, std::string(128, 'e'));
  }
  ops->unbind_output(&output, &binding);
}

}  // namespace
}  // namespace holoscan
