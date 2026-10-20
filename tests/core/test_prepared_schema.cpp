/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <holoscan/schema/prepared_schema_storage.hpp>
#include <holoscan/schema/sdk_schemas.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>

#include "common/allocation_counter.hpp"
#include "prepared_descriptor_schema_traits.hpp"
#include "prepared_keyword_descriptor_schema_traits.hpp"
#include "prepared_keyword_scalar_schema_traits.hpp"
#include "prepared_ordinary_schema_traits.hpp"

// REQ: SCHEMA-008

namespace holoscan {
namespace {

using Packet = prepared_schema_test::PacketT;
using PacketScratch = schema::DescriptorMetadataScratch<Packet>;
using ImuScratch = schema::SchemaScratch<schema::ImuT>;

struct CountedResult {
  bool success{};
  ErrorCode code{};
  std::size_t allocations{};
};
template <typename Function>
CountedResult count_operation(Function&& operation) {
  CountedResult result;
  {
    const test::ScopedAllocationCounter counter;
    auto status = operation();
    result.success = static_cast<bool>(status);
    result.code = status ? ErrorCode::kSuccess : status.error().code;
    result.allocations = counter.count();
  }
  return result;
}

schema::ImuT imu(unsigned mask, std::size_t length = 128) {
  schema::ImuT value;
  value.orientation = schema::Quaternionf{1.F, 2.F, 3.F, 4.F};
  value.angular_velocity = schema::Vector3f{5.F, 6.F, 7.F};
  value.linear_acceleration = schema::Vector3f{8.F, 9.F, 10.F};
  value.flags = schema::ImuFlags_UNCALIBRATED;
  if (mask & 1U) {
    value.header = std::make_shared<schema::HeaderT>();
    value.header->frame_id.assign(length, 'i');
    value.header->device_sequence = 72;
    value.header->capture_timestamp_ns = 73;
    if (mask & 2U) {
      value.header->stamp = std::make_shared<schema::TimeT>();
      value.header->stamp->time_ns = 74;
      value.header->stamp->acqtime_ns = 75;
    }
  }
  const float matrix_values[9]{1.F, 2.F, 3.F, 4.F, 5.F, 6.F, 7.F, 8.F, 9.F};
  const schema::Matrix3f matrix{flatbuffers::span<const float, 9>{matrix_values}};
  if (mask & 4U)
    value.orientation_covariance = std::make_shared<schema::Matrix3f>(matrix);
  if (mask & 8U)
    value.angular_velocity_covariance = std::make_shared<schema::Matrix3f>(matrix);
  if (mask & 16U)
    value.linear_acceleration_covariance = std::make_shared<schema::Matrix3f>(matrix);
  return value;
}

Packet packet(unsigned mask, std::size_t length = 128, std::size_t elements = 4) {
  Packet value;
  value.sequence = 42;
  value.label.assign(length, 'p');
  value.readings.assign(elements * 2, 4.F);
  value.tags.assign(elements, std::string(length, 't'));
  if (mask & 1U) {
    value.metadata = std::make_shared<prepared_schema_test::MetadataT>();
    value.metadata->label.assign(length, 'm');
    value.metadata->tags.assign(elements == 4 ? 3 : elements, std::string(length, 'n'));
    if (mask & 2U)
      value.metadata->calibration = std::make_shared<prepared_schema_test::Calibration>(2.F, 3);
    if (mask & 4U) {
      value.metadata->child = std::make_shared<prepared_schema_test::ChildT>();
      value.metadata->child->label.assign(length, 'c');
    }
  }
  if (mask & 8U)
    value.calibration = std::make_shared<prepared_schema_test::Calibration>(4.F, 5);
  return value;
}

TEST(PreparedSchema, ScalarCopySelfCopyResetAndMoves) {
  using Scratch = schema::SchemaScratch<schema::TimeT>;
  static_assert(!std::is_copy_constructible_v<Scratch>);
  static_assert(std::is_nothrow_move_constructible_v<Scratch>);
  auto created = Scratch::create();
  ASSERT_TRUE(created);
  auto scratch = std::move(*created);
  const schema::TimeT input{.time_ns = 17, .acqtime_ns = 18};
  auto result = count_operation([&] { return scratch.copy_from(input); });
  EXPECT_TRUE(result.success);
  EXPECT_EQ(result.allocations, 0U);
  EXPECT_EQ(scratch.value().time_ns, 17);
  result = count_operation([&] { return scratch.copy_from(scratch.value()); });
  EXPECT_TRUE(result.success);
  EXPECT_EQ(result.allocations, 0U);
  auto moved = std::move(scratch);
  result = count_operation([&] { return moved.reset(); });
  EXPECT_TRUE(result.success);
  EXPECT_EQ(result.allocations, 0U);
  EXPECT_EQ(moved.value().time_ns, 0);
  EXPECT_EQ(moved.memory_usage().native_bytes, sizeof(schema::TimeT));
  EXPECT_EQ(moved.memory_usage().optional_bytes, 0U);
  EXPECT_LE(moved.memory_usage().total_bytes(), 256U);
}

TEST(PreparedSchema, ImuAllPresencePatternsFirstCopyAndResetAreAllocationFree) {
  std::vector<schema::ImuT> inputs;
  for (unsigned mask = 0; mask < 32; ++mask)
    inputs.push_back(imu(mask, mask % 2 ? 512 : 0));
  for (unsigned initial = 0; initial < 32; ++initial) {
    auto scratch = ImuScratch::create();
    ASSERT_TRUE(scratch);
    for (unsigned i = 0; i < 64; ++i) {
      const auto& source = inputs[(i + initial) % 32];
      const auto result = count_operation([&] { return scratch->copy_from(source); });
      ASSERT_TRUE(result.success);
      EXPECT_EQ(result.allocations, 0U);
      const auto& value = scratch->value();
      EXPECT_EQ(static_cast<bool>(value.header), static_cast<bool>(source.header));
      EXPECT_EQ(static_cast<bool>(value.orientation_covariance),
                static_cast<bool>(source.orientation_covariance));
      EXPECT_EQ(static_cast<bool>(value.angular_velocity_covariance),
                static_cast<bool>(source.angular_velocity_covariance));
      EXPECT_EQ(static_cast<bool>(value.linear_acceleration_covariance),
                static_cast<bool>(source.linear_acceleration_covariance));
      if (value.header) {
        EXPECT_NE(value.header.get(), source.header.get());
        EXPECT_EQ(value.header->frame_id, source.header->frame_id);
        EXPECT_EQ(static_cast<bool>(value.header->stamp), static_cast<bool>(source.header->stamp));
        if (value.header->stamp)
          EXPECT_EQ(value.header->stamp->acqtime_ns, 75);
      }
      if (value.orientation_covariance)
        EXPECT_EQ(value.orientation_covariance->m()->Get(8), 9.F);
      EXPECT_EQ(value.orientation.w(), 4.F);
      EXPECT_EQ(value.linear_acceleration.z(), 10.F);
    }
    const auto reset = count_operation([&] { return scratch->reset(); });
    EXPECT_TRUE(reset.success);
    EXPECT_EQ(reset.allocations, 0U);
    EXPECT_FALSE(scratch->value().header);
    EXPECT_FALSE(scratch->value().orientation_covariance);
  }
}

TEST(PreparedSchema, StringVectorsRetainNestedCapacityAcrossAllTransitions) {
  std::vector<Packet> inputs;
  for (unsigned mask = 0; mask < 16; ++mask)
    for (const auto size : {0U, 1U, 4U})
      inputs.push_back(packet(mask, size == 1 ? 6 : 128, size));
  auto scratch = PacketScratch::create();
  ASSERT_TRUE(scratch);
  for (unsigned pass = 0; pass < 3; ++pass) {
    for (const auto& source : inputs) {
      auto result = count_operation([&] { return scratch->copy_metadata_from(source); });
      ASSERT_TRUE(result.success);
      EXPECT_EQ(result.allocations, 0U);
      const auto& value = scratch->descriptor();
      EXPECT_EQ(value.tags, source.tags);
      EXPECT_EQ(value.readings, source.readings);
      EXPECT_EQ(value.label, source.label);
      EXPECT_EQ(static_cast<bool>(value.metadata), static_cast<bool>(source.metadata));
      if (value.metadata) {
        EXPECT_NE(value.metadata.get(), source.metadata.get());
        EXPECT_EQ(value.metadata->tags, source.metadata->tags);
        EXPECT_EQ(static_cast<bool>(value.metadata->child),
                  static_cast<bool>(source.metadata->child));
        if (value.metadata->child)
          EXPECT_EQ(value.metadata->child->label, source.metadata->child->label);
      }
      EXPECT_FALSE(value.first);
      EXPECT_FALSE(value.second);
      if (pass == 1) {
        result = count_operation([&] { return scratch->reset(); });
        EXPECT_TRUE(result.success);
        EXPECT_EQ(result.allocations, 0U);
      }
    }
  }
}

TEST(PreparedSchema, OversizeNestedInputRejectsAtomicallyWithoutAllocating) {
  auto scratch = PacketScratch::create();
  ASSERT_TRUE(scratch);
  auto source = packet(15);
  ASSERT_TRUE(scratch->copy_metadata_from(source));
  source.sequence = 99;
  source.metadata->tags[1].assign(129, 'x');
  auto result = count_operation([&] { return scratch->copy_metadata_from(source); });
  EXPECT_EQ(result.code, ErrorCode::kResourceExhausted);
  EXPECT_EQ(result.allocations, 0U);
  EXPECT_EQ(scratch->descriptor().sequence, 42U);
  EXPECT_EQ(scratch->descriptor().metadata->tags[1], std::string(128, 'n'));
  source.metadata->tags[1].assign(128, 'n');
  source.tags.push_back("overflow");
  result = count_operation([&] { return scratch->copy_metadata_from(source); });
  EXPECT_EQ(result.code, ErrorCode::kResourceExhausted);
  EXPECT_EQ(result.allocations, 0U);
  EXPECT_EQ(scratch->descriptor().sequence, 42U);
}

TEST(PreparedSchema, AlteredOwnersAliasesAndLostStringBuffersReject) {
  auto scratch = PacketScratch::create();
  ASSERT_TRUE(scratch);
  const auto source = packet(15);
  ASSERT_TRUE(scratch->copy_metadata_from(source));
  auto alias = scratch->descriptor().metadata;
  auto result = count_operation([&] { return scratch->reset(); });
  EXPECT_EQ(result.code, ErrorCode::kPreparedStorageInvalid);
  EXPECT_EQ(result.allocations, 0U);
  alias.reset();
  scratch->descriptor().tags.clear();  // Deliberately destroys active prepared element buffers.
  result = count_operation([&] { return scratch->copy_metadata_from(source); });
  EXPECT_EQ(result.code, ErrorCode::kPreparedStorageInvalid);
  EXPECT_EQ(result.allocations, 0U);
  EXPECT_TRUE(scratch->descriptor().metadata);
}

TEST(PreparedSchema, MetadataScratchDoesNotRetainEitherTensorBacking) {
  std::array<std::uint8_t, 16> pixels{};
  std::int64_t shape = 16;
  int released = 0;
  DLManagedTensor managed{};
  managed.dl_tensor = {.data = pixels.data(),
                       .device = {kDLCPU, 0},
                       .ndim = 1,
                       .dtype = {kDLUInt, 8, 1},
                       .shape = &shape};
  managed.manager_ctx = &released;
  managed.deleter = [](DLManagedTensor* tensor) { ++*static_cast<int*>(tensor->manager_ctx); };
  auto second_managed = managed;
  auto source = packet(15);
  source.first = std::make_shared<Tensor>(&managed);
  source.second = std::make_shared<Tensor>(&second_managed);
  auto scratch = PacketScratch::create();
  ASSERT_TRUE(scratch);
  const auto result = count_operation([&] { return scratch->copy_metadata_from(source); });
  EXPECT_TRUE(result.success);
  EXPECT_EQ(result.allocations, 0U);
  EXPECT_FALSE(scratch->descriptor().first);
  EXPECT_FALSE(scratch->descriptor().second);
  source.first.reset();
  EXPECT_EQ(released, 1);
  source.second.reset();
  EXPECT_EQ(released, 2);
}

TEST(PreparedSchema, ColdAllocationFailuresCleanUpAndTighterBudgetsFailClosed) {
  std::size_t allocations = 0;
  bool success = false;
  {
    const test::ScopedAllocationCounter counter;
    auto prepared = PacketScratch::create();
    allocations = counter.count();
    success = static_cast<bool>(prepared);
  }
  ASSERT_TRUE(success);
  ASSERT_GT(allocations, 0U);
  for (std::size_t fail_at = 0; fail_at < allocations; ++fail_at) {
    auto prepared = [&] {
      const test::ScopedAllocationExhaustion failure(fail_at);
      return PacketScratch::create();
    }();
    ASSERT_FALSE(prepared) << fail_at;
    EXPECT_EQ(prepared.error().code, ErrorCode::kResourceExhausted);
  }
  auto prepared = PacketScratch::create();
  ASSERT_TRUE(prepared);
  const auto bytes = prepared->memory_usage().total_bytes();
  EXPECT_TRUE(PacketScratch::create(bytes));
  EXPECT_FALSE(PacketScratch::create(bytes - 1));
  EXPECT_FALSE(PacketScratch::create(0));
  EXPECT_GT(prepared->memory_usage().optional_bytes, 0U);
  EXPECT_GT(prepared->memory_usage().container_bytes, 0U);
}

TEST(PreparedSchema, GeneratedWireAssignmentReusesPreparedOptionalOwners) {
  schema::detail::PreparedMemoryBudget budget{4096};
  schema::PreparedSchemaTraits<schema::ImuT>::Storage storage;
  schema::ImuT destination;
  storage.prepare(destination, budget);
  for (unsigned mask = 0; mask < 32; ++mask) {
    const auto source = imu(mask);
    flatbuffers::FlatBufferBuilder builder;
    builder.Finish(schema::Imu::Pack(builder, &source));
    const auto* root = flatbuffers::GetRoot<schema::Imu>(builder.GetBufferPointer());
    const auto result = count_operation([&]() -> expected<void, Error> {
      if (!storage.valid(destination) || !storage.wire_fits(*root))
        return make_unexpected(Error{ErrorCode::kFailure});
      storage.assign_root_unchecked(destination, *root);
      return {};
    });
    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.allocations, 0U);
    EXPECT_EQ(static_cast<bool>(destination.header), static_cast<bool>(source.header));
    EXPECT_EQ(static_cast<bool>(destination.orientation_covariance),
              static_cast<bool>(source.orientation_covariance));
    EXPECT_EQ(destination.orientation.w(), source.orientation.w());
  }
}

// REQ: SCHEMA-008
TEST(PreparedSchema, KeywordFieldsCopyResetAndAssignWireWithoutAllocation) {
  using Native = prepared_schema_test::KeywordScalarT;
  using Root = prepared_schema_test::KeywordScalar;
  auto scratch = schema::SchemaScratch<Native>::create();
  ASSERT_TRUE(scratch);
  schema::detail::PreparedMemoryBudget budget{4096};
  schema::PreparedSchemaTraits<Native>::Storage storage;
  Native destination;
  storage.prepare(destination, budget);

  // Raw schema names in the bounds file must still select the escaped C++ member paths.
  for (const bool present : {true, false, true}) {
    Native source;
    source.class_ = 42;
    if (present) {
      source.namespace_ = std::make_shared<prepared_schema_test::KeywordChildT>();
      source.namespace_->template_ = {1, 2, 3, 4};
      source.namespace_->switch_.assign(2, std::string(64, 's'));
      source.namespace_->return_.assign(64, 'r');
    }
    const auto copied = count_operation([&] { return scratch->copy_from(source); });
    ASSERT_TRUE(copied.success);
    EXPECT_EQ(copied.allocations, 0U);
    EXPECT_EQ(scratch->value().class_, source.class_);
    ASSERT_EQ(static_cast<bool>(scratch->value().namespace_), present);
    if (present) {
      EXPECT_NE(scratch->value().namespace_.get(), source.namespace_.get());
      EXPECT_EQ(scratch->value().namespace_->template_, source.namespace_->template_);
      EXPECT_EQ(scratch->value().namespace_->switch_, source.namespace_->switch_);
      EXPECT_EQ(scratch->value().namespace_->return_, source.namespace_->return_);
    }
    flatbuffers::FlatBufferBuilder builder;
    builder.Finish(Root::Pack(builder, &source));
    const auto* root = flatbuffers::GetRoot<Root>(builder.GetBufferPointer());
    const auto assigned = count_operation([&]() -> expected<void, Error> {
      if (!storage.valid(destination) || !storage.wire_fits(*root))
        return make_unexpected(Error{ErrorCode::kFailure});
      storage.assign_root_unchecked(destination, *root);
      return {};
    });
    ASSERT_TRUE(assigned.success);
    EXPECT_EQ(assigned.allocations, 0U);
    EXPECT_EQ(destination.class_, source.class_);
    ASSERT_EQ(static_cast<bool>(destination.namespace_), present);
    if (present) {
      EXPECT_EQ(destination.namespace_->template_, source.namespace_->template_);
      EXPECT_EQ(destination.namespace_->switch_, source.namespace_->switch_);
      EXPECT_EQ(destination.namespace_->return_, source.namespace_->return_);
    }
  }
  const auto reset = count_operation([&] { return scratch->reset(); });
  ASSERT_TRUE(reset.success);
  EXPECT_EQ(reset.allocations, 0U);
  EXPECT_EQ(scratch->value().class_, 7);
  EXPECT_FALSE(scratch->value().namespace_);
}

TEST(PreparedSchema, OptionalScalarsEnumsInlineArraysAndDefaultsRoundTrip) {
  using Native = prepared_schema_test::ScalarsT;
  using Root = prepared_schema_test::Scalars;
  auto scratch = schema::SchemaScratch<Native>::create();
  ASSERT_TRUE(scratch);
  schema::detail::PreparedMemoryBudget budget{1024};
  schema::PreparedSchemaTraits<Native>::Storage storage;
  Native destination;
  storage.prepare(destination, budget);
  for (unsigned pass = 0; pass < 4; ++pass) {
    Native source;
    if (pass % 2) {
      source.counter = 64;
      source.child = std::make_shared<prepared_schema_test::ScalarsChildT>();
      source.child->counter = 32;
      const int values[3]{1, 2, 3};
      source.fixed = prepared_schema_test::Fixed{flatbuffers::span<const int, 3>{values}};
      source.modes.assign(4, prepared_schema_test::Mode_Active);
    }
    auto result = count_operation([&] { return scratch->copy_from(source); });
    ASSERT_TRUE(result.success);
    EXPECT_EQ(result.allocations, 0U);
    EXPECT_EQ(scratch->value().counter, source.counter);
    flatbuffers::FlatBufferBuilder builder;
    builder.Finish(Root::Pack(builder, &source));
    const auto* root = flatbuffers::GetRoot<Root>(builder.GetBufferPointer());
    result = count_operation([&]() -> expected<void, Error> {
      if (!storage.valid(destination) || !storage.wire_fits(*root))
        return make_unexpected(Error{ErrorCode::kFailure});
      storage.assign_root_unchecked(destination, *root);
      return {};
    });
    ASSERT_TRUE(result.success);
    EXPECT_EQ(result.allocations, 0U);
    EXPECT_EQ(destination.counter, source.counter);
    EXPECT_EQ(destination.mode, prepared_schema_test::Mode_Active);
    EXPECT_EQ(destination.minimum, std::numeric_limits<std::int64_t>::min());
    EXPECT_EQ(destination.maximum, std::numeric_limits<std::uint64_t>::max());
    EXPECT_EQ(destination.weight, 1.25F);
    EXPECT_EQ(destination.fixed.values()->Get(2), source.fixed.values()->Get(2));
    EXPECT_EQ(destination.modes, source.modes);
    EXPECT_EQ(static_cast<bool>(destination.child), static_cast<bool>(source.child));
    if (destination.child)
      EXPECT_EQ(destination.child->counter, source.child->counter);
  }
  auto result = count_operation([&] { return scratch->reset(); });
  EXPECT_TRUE(result.success);
  EXPECT_EQ(result.allocations, 0U);
  EXPECT_FALSE(scratch->value().counter.has_value());
  EXPECT_EQ(scratch->value().mode, prepared_schema_test::Mode_Active);
  EXPECT_FALSE(scratch->value().child);
}

TEST(PreparedSchema, ParkedStorageSurvivesMovesAndRejectsCheckedBudgetOverflow) {
  auto scratch = PacketScratch::create();
  ASSERT_TRUE(scratch);
  const auto source = packet(15);
  ASSERT_TRUE(scratch->copy_metadata_from(source));
  ASSERT_TRUE(scratch->reset());
  auto moved = std::move(*scratch);
  auto result = count_operation([&] { return moved.copy_metadata_from(source); });
  EXPECT_TRUE(result.success);
  EXPECT_EQ(result.allocations, 0U);
  auto other = PacketScratch::create();
  ASSERT_TRUE(other);
  *other = std::move(moved);
  result = count_operation([&] { return other->copy_metadata_from(other->descriptor()); });
  EXPECT_TRUE(result.success);
  EXPECT_EQ(result.allocations, 0U);
  EXPECT_EQ(other->descriptor().metadata->tags, source.metadata->tags);
  schema::detail::PreparedMemoryBudget budget{std::numeric_limits<std::size_t>::max()};
  budget.charge(std::numeric_limits<std::size_t>::max(),
                &schema::PreparedSchemaMemory::native_bytes);
  EXPECT_THROW(budget.charge(1, &schema::PreparedSchemaMemory::container_bytes), std::bad_alloc);
  EXPECT_THROW(
      schema::detail::PreparedMemoryBudget::product(std::numeric_limits<std::size_t>::max(), 2),
      std::bad_alloc);
}

template <typename T>
class PreparedRuntimeSlot {
 public:
  const detail::RuntimePortTypeOps* ops = detail::runtime_port_type_ops<T>();
  void* storage = ::operator new(ops->sample_size, std::align_val_t{ops->sample_alignment});
  bool ready = false;
  PreparedRuntimeSlot() = default;
  explicit PreparedRuntimeSlot(const detail::RuntimePortTypeOps* selected_ops)
      : ops(selected_ops) {}
  PreparedRuntimeSlot(const PreparedRuntimeSlot&) = delete;
  PreparedRuntimeSlot& operator=(const PreparedRuntimeSlot&) = delete;
  ~PreparedRuntimeSlot() {
    if (ready)
      ops->destroy_sample_storage(storage);
    ::operator delete(storage, std::align_val_t{ops->sample_alignment});
  }
  expected<void, Error> prepare() {
    auto result = ops->prepare_sample_storage(storage);
    ready = static_cast<bool>(result);
    return result;
  }
  const Sample<T>& sample() const { return *static_cast<const Sample<T>*>(storage); }
};

using DefaultPacketCodec = detail::RuntimePreparedSchemaTraits<Packet>::DefaultDescriptorCodec;

struct CustomPacketCodecProbe {
  static inline std::size_t prepares{};
  static inline std::size_t resets{};
  static inline std::size_t assignments{};
  static inline bool fail_prepare{};
  static inline bool reject_assignment{};
  static inline std::weak_ptr<Tensor> failed_holder;
};

// Inheriting the default encoder must not opt overridden lifecycle/validation hooks into bypass.
struct CustomPacketCodec : DefaultPacketCodec {
  [[nodiscard]] expected<void, Error> prepare(Packet& value) const {
    ++CustomPacketCodecProbe::prepares;
    if (CustomPacketCodecProbe::fail_prepare) {
      CustomPacketCodecProbe::failed_holder = value.first;
      return make_unexpected(Error{ErrorCode::kFailure});
    }
    auto prepared = DefaultPacketCodec::prepare(value);
    if (prepared)
      value.sequence = 71U;
    return prepared;
  }

  void reset(Packet& value) const noexcept {
    ++CustomPacketCodecProbe::resets;
    DefaultPacketCodec::reset(value);
  }

  [[nodiscard]] expected<void, Error> assign_root(Packet& value,
                                                  const prepared_schema_test::Packet& root) const {
    ++CustomPacketCodecProbe::assignments;
    if (CustomPacketCodecProbe::reject_assignment)
      return make_unexpected(Error{ErrorCode::kCodecError});
    return DefaultPacketCodec::assign_root(value, root);
  }
};

// REQ: SCHEMA-009
TEST(PreparedSchemaRuntime, CustomDescriptorCodecPreservesHooksAndValidation) {
  using Probe = CustomPacketCodecProbe;
  static_assert(std::is_empty_v<CustomPacketCodec>);
  Probe::prepares = Probe::resets = Probe::assignments = 0U;
  Probe::fail_prepare = true;
  Probe::reject_assignment = true;
  Probe::failed_holder.reset();
  const auto ops = schema::make_standard_descriptor_tensor_port_type_ops<
      Packet,
      prepared_schema_test::Packet,
      schema::flatbuffer_descriptor_metadata_capacity_v<prepared_schema_test::Packet>,
      &Packet::first,
      &Packet::second>(CustomPacketCodec{});
  EXPECT_EQ(ops.sample_size, sizeof(Sample<Packet>));
  // The generated default keeps its prepared inventory; a different codec uses ordinary storage.
  EXPECT_GT(detail::runtime_port_type_ops<Packet>()->sample_size, ops.sample_size);
  {
    PreparedRuntimeSlot<Packet> failed{&ops};
    const auto prepared = failed.prepare();
    ASSERT_FALSE(prepared);
    EXPECT_EQ(prepared.error().code, ErrorCode::kFailure);
    EXPECT_EQ(Probe::prepares, 1U);
    EXPECT_TRUE(Probe::failed_holder.expired());
  }
  Probe::fail_prepare = false;
  PreparedRuntimeSlot<Packet> slot{&ops};
  ASSERT_TRUE(slot.prepare());
  EXPECT_EQ(Probe::prepares, 2U);
  EXPECT_EQ(slot.sample().data.sequence, 71U);
  const auto* first_holder = slot.sample().data.first.get();
  const auto* second_holder = slot.sample().data.second.get();
  ASSERT_NE(first_holder, nullptr);
  ASSERT_NE(second_holder, nullptr);

  std::array<std::uint8_t, 2> pixels{};
  DLManagedTensor first_managed{};
  first_managed.dl_tensor = {
      .data = pixels.data(), .device = {kDLCPU, 0}, .ndim = 0, .dtype = {kDLUInt, 8, 1}};
  first_managed.deleter = [](DLManagedTensor*) {};
  auto second_managed = first_managed;
  second_managed.dl_tensor.data = pixels.data() + 1;
  const std::array<Tensor, 2> originals{Tensor{&first_managed}, Tensor{&second_managed}};
  Packet source;
  source.sequence = 42U;
  source.label = "wire";
  const SampleMetadata metadata{
      .sequence = 17U, .byte_size = 2U, .payload_kind = PayloadKind::kDescriptorBacked};
  std::vector<std::max_align_t> aligned_wire((ops.wire_capacity + sizeof(std::max_align_t) - 1U) /
                                             sizeof(std::max_align_t));
  auto destination = std::as_writable_bytes(std::span{aligned_wire}).first(ops.wire_capacity);
  auto encoded = ops.encode_descriptor_tensor_loans(&source, originals, metadata, 1U, destination);
  ASSERT_TRUE(encoded);
  const auto wire = destination.first(*encoded);
  auto verified = ops.descriptor_tensor_sample_metadata(wire, 0U, 1U);
  ASSERT_TRUE(verified);
  auto tensors = originals;
  auto assigned =
      ops.assign_descriptor_tensors_sample(slot.storage, wire, tensors, {*verified}, {*verified});
  ASSERT_FALSE(assigned);
  EXPECT_EQ(assigned.error().code, ErrorCode::kCodecError);
  EXPECT_EQ(Probe::assignments, 1U);
  EXPECT_EQ(slot.sample().data.sequence, 71U);
  EXPECT_EQ(slot.sample().metadata.sequence, 0U);
  EXPECT_EQ(slot.sample().data.first->data(), nullptr);
  EXPECT_EQ(slot.sample().data.second->data(), nullptr);

  ops.reset_sample_storage(slot.storage);
  EXPECT_EQ(Probe::resets, 1U);
  EXPECT_EQ(slot.sample().data.sequence, 0U);
  Probe::reject_assignment = false;
  tensors = originals;
  assigned =
      ops.assign_descriptor_tensors_sample(slot.storage, wire, tensors, {*verified}, {*verified});
  ASSERT_TRUE(assigned);
  EXPECT_EQ(Probe::assignments, 2U);
  EXPECT_EQ(slot.sample().data.sequence, source.sequence);
  EXPECT_EQ(slot.sample().data.label, source.label);
  EXPECT_EQ(slot.sample().metadata.sequence, metadata.sequence);
  EXPECT_EQ(slot.sample().data.first->data(), pixels.data());
  EXPECT_EQ(slot.sample().data.second->data(), pixels.data() + 1);
  ops.reset_sample_storage(slot.storage);
  EXPECT_EQ(Probe::resets, 2U);
  EXPECT_EQ(slot.sample().data.first.get(), first_holder);
  EXPECT_EQ(slot.sample().data.second.get(), second_holder);
  EXPECT_EQ(slot.sample().data.first->data(), nullptr);
  EXPECT_EQ(slot.sample().data.second->data(), nullptr);
  EXPECT_EQ(slot.sample().metadata.sequence, 0U);
}

// Each rejected encode names its cause. Storage alignment in particular depends on the caller's
// platform and compiler, so a bare error code would not identify it.
// REQ: SCHEMA-009
TEST(PreparedSchemaRuntime, DescriptorEnvelopeEncoderNamesEachRejection) {
  const detail::RuntimePortTypeOps* ops = detail::runtime_port_type_ops<Packet>();
  ASSERT_NE(ops, nullptr);
  std::array<std::uint8_t, 2> pixels{};
  DLManagedTensor first_managed{};
  first_managed.dl_tensor = {
      .data = pixels.data(), .device = {kDLCPU, 0}, .ndim = 0, .dtype = {kDLUInt, 8, 1}};
  first_managed.deleter = [](DLManagedTensor*) {};
  auto second_managed = first_managed;
  second_managed.dl_tensor.data = pixels.data() + 1;
  const std::array<Tensor, 2> tensors{Tensor{&first_managed}, Tensor{&second_managed}};
  Packet source;
  const SampleMetadata metadata{
      .sequence = 1U, .byte_size = 2U, .payload_kind = PayloadKind::kDescriptorBacked};
  std::vector<std::max_align_t> aligned_wire(ops->wire_capacity / sizeof(std::max_align_t) + 2U);
  const auto storage = std::as_writable_bytes(std::span{aligned_wire});
  const auto exact = storage.first(ops->wire_capacity);
  const auto expect_rejected = [&](const void* payload,
                                   std::span<const Tensor>
                                       fields,
                                   std::uint64_t generation,
                                   std::span<std::byte>
                                       destination,
                                   std::string_view message) {
    const auto encoded =
        ops->encode_descriptor_tensor_loans(payload, fields, metadata, generation, destination);
    ASSERT_FALSE(encoded) << message;
    EXPECT_EQ(encoded.error().code, ErrorCode::kInvalidArgument);
    EXPECT_EQ(encoded.error().message, message);
  };
  expect_rejected(nullptr, tensors, 1U, exact, "descriptor Tensor envelope payload is null");
  expect_rejected(&source,
                  std::span{tensors}.first(1U),
                  1U,
                  exact,
                  "descriptor Tensor envelope Tensor count does not match");
  expect_rejected(&source,
                  tensors,
                  1U,
                  storage.first(ops->wire_capacity - 1U),
                  "descriptor Tensor envelope storage size does not match");
  expect_rejected(&source,
                  tensors,
                  1U,
                  storage.subspan(alignof(std::uint64_t), ops->wire_capacity),
                  "descriptor Tensor envelope storage is not max_align_t aligned");
  expect_rejected(
      &source, tensors, 0U, exact, "descriptor Tensor envelope endpoint generation is zero");
  EXPECT_TRUE(ops->encode_descriptor_tensor_loans(&source, tensors, metadata, 1U, exact));
}

// REQ: SCHEMA-008
// REQ: SCHEMA-009
TEST(PreparedSchemaRuntime, KeywordTensorFieldSeparatesScratchFromRuntimeHolder) {
  using Native = prepared_keyword_test::KeywordDescriptorT;
  auto scratch = schema::DescriptorMetadataScratch<Native>::create();
  ASSERT_TRUE(scratch);
  Native source;
  source.class_ = std::make_shared<Tensor>();
  source.delete_ = 42;
  const auto copied = count_operation([&] { return scratch->copy_metadata_from(source); });
  ASSERT_TRUE(copied.success);
  EXPECT_EQ(copied.allocations, 0U);
  EXPECT_FALSE(scratch->descriptor().class_);
  EXPECT_EQ(source.class_.use_count(), 1);
  EXPECT_EQ(scratch->descriptor().delete_, 42U);

  // Runtime preparation must use the escaped Tensor member pointer, not retain the source owner.
  PreparedRuntimeSlot<Native> slot;
  ASSERT_TRUE(slot.prepare());
  ASSERT_TRUE(slot.sample().data.class_);
  EXPECT_NE(slot.sample().data.class_.get(), source.class_.get());
  const auto reset = count_operation([&] { return scratch->reset(); });
  ASSERT_TRUE(reset.success);
  EXPECT_EQ(reset.allocations, 0U);
  EXPECT_FALSE(scratch->descriptor().class_);
  EXPECT_EQ(scratch->descriptor().delete_, 0U);
}

// REQ: SCHEMA-009
TEST(PreparedSchemaRuntime, EveryOrdinarySlotAssignsFirstMessageAndTransitionsWithoutAllocation) {
  std::array<PreparedRuntimeSlot<schema::ImuT>, 8> slots;
  for (auto& slot : slots) {
    ASSERT_TRUE(slot.ops->persistent_sample_storage);
    ASSERT_NE(slot.ops->assign_sample_storage, nullptr);
    ASSERT_TRUE(slot.prepare());
  }
  for (unsigned iteration = 0; iteration < 64; ++iteration) {
    const auto source = imu(iteration, iteration % 2 ? 128 : 0);
    flatbuffers::FlatBufferBuilder builder;
    builder.Finish(schema::Imu::Pack(builder, &source), schema::ImuIdentifier());
    const auto wire = std::as_bytes(std::span{builder.GetBufferPointer(), builder.GetSize()});
    for (auto& slot : slots) {
      const SampleMetadata metadata{.sequence = iteration, .frame_id = 73};
      const auto result = count_operation(
          [&] { return slot.ops->assign_sample_storage(slot.storage, wire, metadata); });
      ASSERT_TRUE(result.success) << static_cast<int>(result.code);
      EXPECT_EQ(result.allocations, 0U);
      EXPECT_EQ(slot.sample().metadata.sequence, iteration);
      EXPECT_EQ(slot.sample().metadata.frame_id, 73U);
      const auto& received = slot.sample().data;
      EXPECT_EQ(static_cast<bool>(received.header), static_cast<bool>(source.header));
      if (received.header) {
        EXPECT_EQ(received.header->frame_id, source.header->frame_id);
        EXPECT_EQ(static_cast<bool>(received.header->stamp),
                  static_cast<bool>(source.header->stamp));
      }
      EXPECT_EQ(static_cast<bool>(received.orientation_covariance),
                static_cast<bool>(source.orientation_covariance));
      EXPECT_EQ(static_cast<bool>(received.angular_velocity_covariance),
                static_cast<bool>(source.angular_velocity_covariance));
      EXPECT_EQ(static_cast<bool>(received.linear_acceleration_covariance),
                static_cast<bool>(source.linear_acceleration_covariance));
      if (iteration % 3 == 0) {
        const auto reset = count_operation([&]() -> expected<void, Error> {
          slot.ops->reset_sample_storage(slot.storage);
          return {};
        });
        EXPECT_EQ(reset.allocations, 0U);
      }
    }
  }
}

// REQ: SCHEMA-009
TEST(PreparedSchemaRuntime, RetainedNestedAliasRejectsReuseWithoutAllocationOrMutation) {
  PreparedRuntimeSlot<schema::ImuT> slot;
  ASSERT_TRUE(slot.prepare());
  auto source = imu(3);
  flatbuffers::FlatBufferBuilder builder;
  builder.Finish(schema::Imu::Pack(builder, &source), schema::ImuIdentifier());
  auto wire = std::as_bytes(std::span{builder.GetBufferPointer(), builder.GetSize()});
  ASSERT_TRUE(slot.ops->assign_sample_storage(slot.storage, wire, {}));
  auto alias = slot.sample().data.header->stamp;
  slot.ops->reset_sample_storage(slot.storage);
  source.header->stamp->time_ns = 99;
  builder.Clear();
  builder.Finish(schema::Imu::Pack(builder, &source), schema::ImuIdentifier());
  wire = std::as_bytes(std::span{builder.GetBufferPointer(), builder.GetSize()});
  auto result =
      count_operation([&] { return slot.ops->assign_sample_storage(slot.storage, wire, {}); });
  EXPECT_EQ(result.code, ErrorCode::kPreparedStorageInvalid);
  EXPECT_EQ(result.allocations, 0U);
  EXPECT_EQ(alias->time_ns, 74);
  alias.reset();
  result = count_operation([&] { return slot.ops->assign_sample_storage(slot.storage, wire, {}); });
  EXPECT_TRUE(result.success);
  EXPECT_EQ(result.allocations, 0U);
  EXPECT_EQ(slot.sample().data.header->stamp->time_ns, 99);
}

TEST(PreparedSchemaRuntime, MalformedAndOversizedAssignmentPreservePriorValueWithoutAllocation) {
  using Native = prepared_schema_test::ScalarsT;
  PreparedRuntimeSlot<Native> slot;
  ASSERT_TRUE(slot.prepare());
  Native source;
  source.counter = 17;
  flatbuffers::FlatBufferBuilder builder;
  builder.Finish(prepared_schema_test::Scalars::Pack(builder, &source), "PRSC");
  auto wire = std::as_bytes(std::span{builder.GetBufferPointer(), builder.GetSize()});
  ASSERT_TRUE(slot.ops->assign_sample_storage(slot.storage, wire, {.sequence = 42}));
  auto result = count_operation(
      [&] { return slot.ops->assign_sample_storage(slot.storage, wire.first(4), {}); });
  EXPECT_EQ(result.code, ErrorCode::kCodecError);
  EXPECT_EQ(result.allocations, 0U);
  source.counter = 19;
  source.modes.assign(5, prepared_schema_test::Mode_Active);
  builder.Clear();
  builder.Finish(prepared_schema_test::Scalars::Pack(builder, &source), "PRSC");
  wire = std::as_bytes(std::span{builder.GetBufferPointer(), builder.GetSize()});
  result = count_operation([&] { return slot.ops->assign_sample_storage(slot.storage, wire, {}); });
  EXPECT_EQ(result.code, ErrorCode::kResourceExhausted);
  EXPECT_EQ(result.allocations, 0U);
  EXPECT_EQ(slot.sample().data.counter, 17);
  EXPECT_EQ(slot.sample().metadata.sequence, 42U);
}

TEST(PreparedSchemaRuntime, EveryDescriptorSlotReusesMetadataAndAuthoritativeTensorHolders) {
  std::array<std::uint8_t, 16> first_pixels{};
  std::array<std::uint8_t, 16> second_pixels{};
  std::int64_t shape = 16;
  DLManagedTensor first_managed{};
  first_managed.dl_tensor = {.data = first_pixels.data(),
                             .device = {kDLCPU, 0},
                             .ndim = 1,
                             .dtype = {kDLUInt, 8, 1},
                             .shape = &shape};
  first_managed.deleter = [](DLManagedTensor*) {};
  auto second_managed = first_managed;
  second_managed.dl_tensor.data = second_pixels.data();
  const std::array<Tensor, 2> originals{Tensor{&first_managed}, Tensor{&second_managed}};
  std::array<PreparedRuntimeSlot<Packet>, 8> slots;
  for (auto& slot : slots)
    ASSERT_TRUE(slot.prepare());
  const auto* ops = slots.front().ops;
  std::vector<std::max_align_t> aligned_wire((ops->wire_capacity + sizeof(std::max_align_t) - 1) /
                                             sizeof(std::max_align_t));
  auto destination = std::as_writable_bytes(std::span{aligned_wire}).first(ops->wire_capacity);
  for (unsigned iteration = 0; iteration < 64; ++iteration) {
    const auto source = packet(iteration, iteration % 2 ? 128 : 0, iteration % 5);
    const SampleMetadata metadata{
        .sequence = iteration + 1, .byte_size = 32, .payload_kind = PayloadKind::kDescriptorBacked};
    auto encoded =
        ops->encode_descriptor_tensor_loans(&source, originals, metadata, 1, destination);
    ASSERT_TRUE(encoded);
    const auto wire = destination.first(*encoded);
    auto verified = ops->descriptor_tensor_sample_metadata(wire, 0, 1);
    ASSERT_TRUE(verified);
    for (auto& slot : slots) {
      auto tensors = originals;
      const auto result = count_operation([&] {
        return ops->assign_descriptor_tensors_sample(
            slot.storage, wire, tensors, {*verified}, {*verified});
      });
      ASSERT_TRUE(result.success) << static_cast<int>(result.code);
      EXPECT_EQ(result.allocations, 0U);
      EXPECT_EQ(slot.sample().data.tags, source.tags);
      EXPECT_EQ(slot.sample().data.readings, source.readings);
      EXPECT_EQ(slot.sample().data.first->data(), first_pixels.data());
      EXPECT_EQ(slot.sample().data.second->data(), second_pixels.data());
      EXPECT_EQ(slot.sample().metadata.sequence, iteration + 1);
      if (iteration % 3 == 0) {
        const auto reset = count_operation([&]() -> expected<void, Error> {
          ops->reset_sample_storage(slot.storage);
          return {};
        });
        EXPECT_EQ(reset.allocations, 0U);
        EXPECT_TRUE(slot.sample().data.first);
        EXPECT_EQ(slot.sample().data.first->data(), nullptr);
      }
    }
  }
}

TEST(PreparedSchemaRuntime, EveryColdFailureCanRetryTheSameSlot) {
  PreparedRuntimeSlot<Packet> measured;
  const auto result = count_operation([&] { return measured.prepare(); });
  ASSERT_TRUE(result.success);
  ASSERT_GT(result.allocations, 0U);
  for (std::size_t fail_at = 0; fail_at < result.allocations; ++fail_at) {
    PreparedRuntimeSlot<Packet> slot;
    auto failed = [&] {
      const test::ScopedAllocationExhaustion exhaustion{fail_at};
      return slot.prepare();
    }();
    ASSERT_FALSE(failed) << fail_at;
    EXPECT_EQ(failed.error().code, ErrorCode::kResourceExhausted);
    EXPECT_TRUE(slot.prepare());
  }
}

}  // namespace
}  // namespace holoscan
