/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <new>
#include <optional>
#include <span>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <benchmark/benchmark.h>

#include <holoscan/core/port.hpp>
#include <holoscan/schema/sdk_schemas.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>

#include "common/allocation_counter.hpp"
#include "runtime_native_schema_benchmark_traits.hpp"

namespace holoscan {
namespace {

using Packet = native_schema_benchmark::PacketT;
enum class Operation {
  kMaterialize,
  kReceive,
  kBorrow,
  kCopyEdit,
  kEncode,
  kEmitEncodeBinding,
  kPreparedMaterialize,
  kPreparedCopyEdit,
  kEmitRefEncodeBinding,
  kColdPrepareSlot,
  kColdPrepareScratch
};
constexpr std::size_t kSourceCount = 35;
constexpr std::size_t kCountedOperations = 128;

// Sources, wire buffers, and real host Tensor backing are all prepared outside measurement.
// Indices 0/1/2 are empty/short/full; 3..34 enumerate five independent optional-presence bits.
struct HostPixels {
  std::array<std::uint8_t, 16> bytes{};
  std::array<std::int64_t, 1> shape{16};
  DLManagedTensor managed{};
  HostPixels() {
    managed.dl_tensor = {.data = bytes.data(),
                         .device = {kDLCPU, 0},
                         .ndim = 1,
                         .dtype = {kDLUInt, 8, 1},
                         .shape = shape.data()};
    managed.deleter = [](DLManagedTensor*) {};
  }
};

template <typename T>
T make_source(std::size_t index, const Tensor& first, const Tensor& second) {
  T value;
  const auto mask = index == 0 ? 0U : (index < 3 ? 31U : static_cast<unsigned>(index - 3));
  const std::size_t length = index == 1 ? 6 : 128;
  if constexpr (std::is_same_v<T, schema::TimeT>) {
    value.time_ns = 123;
    value.acqtime_ns = 456;
  } else if constexpr (std::is_same_v<T, schema::ImuT>) {
    value.orientation = schema::Quaternionf{0.1F, 0.2F, 0.3F, 1.0F};
    value.angular_velocity = schema::Vector3f{1.0F, 2.0F, 3.0F};
    value.linear_acceleration = schema::Vector3f{4.0F, 5.0F, 6.0F};
    if (mask & 1U) {
      value.header = std::make_shared<schema::HeaderT>();
      value.header->frame_id.assign(length, 'h');
      value.header->device_sequence = 17;
      value.header->capture_timestamp_ns = 123;
      if (mask & 2U) {
        value.header->stamp = std::make_shared<schema::TimeT>();
        value.header->stamp->time_ns = 456;
      }
    }
    if (mask & 4U)
      value.orientation_covariance = std::make_shared<schema::Matrix3f>();
    if (mask & 8U)
      value.angular_velocity_covariance = std::make_shared<schema::Matrix3f>();
    if (mask & 16U)
      value.linear_acceleration_covariance = std::make_shared<schema::Matrix3f>();
  } else {
    value.sequence = 17;
    value.first = std::make_shared<Tensor>(first);
    value.second = std::make_shared<Tensor>(second);
    if (index != 0) {
      value.label.assign(length, 'l');
      value.readings.assign(index == 1 ? 1 : 8, 3.5F);
      value.tags.assign(index == 1 ? 1 : 4, std::string(length, 't'));
    }
    if (mask & 1U) {
      value.metadata = std::make_shared<native_schema_benchmark::MetadataT>();
      value.metadata->label.assign(length, 'm');
      value.metadata->tags.assign(index == 1 ? 1 : 3, std::string(length, 'n'));
      if (mask & 2U)
        value.metadata->calibration =
            std::make_shared<native_schema_benchmark::Calibration>(2.F, 3);
      if (mask & 4U) {
        value.metadata->child = std::make_shared<native_schema_benchmark::ChildT>();
        value.metadata->child->label.assign(length, 'c');
      }
    }
    if (mask & 8U)
      value.calibration = std::make_shared<native_schema_benchmark::Calibration>(4.F, 5);
    if (!(mask & 16U))
      value.tags.clear();
  }
  return value;
}

template <typename T, typename Equal>
bool optional_equal(const std::shared_ptr<T>& a, const std::shared_ptr<T>& b, Equal equal) {
  return static_cast<bool>(a) == static_cast<bool>(b) && (!a || equal(*a, *b));
}

bool equal_packet_metadata(const Packet& a, const Packet& b) {
  const auto calibration = [](const auto& x, const auto& y) {
    return x.scale() == y.scale() && x.revision() == y.revision();
  };
  return a.sequence == b.sequence && a.label == b.label && a.readings == b.readings &&
         a.tags == b.tags && optional_equal(a.calibration, b.calibration, calibration) &&
         optional_equal(a.metadata, b.metadata, [&](const auto& x, const auto& y) {
           return x.label == y.label && x.tags == y.tags &&
                  optional_equal(x.calibration, y.calibration, calibration) &&
                  optional_equal(x.child, y.child, [](const auto& u, const auto& v) {
                    return u.label == v.label;
                  });
         });
}

template <typename T>
bool equal_value(const T& a, const T& b) {
  if constexpr (std::is_same_v<T, schema::TimeT>) {
    return a.time_ns == b.time_ns && a.acqtime_ns == b.acqtime_ns;
  } else if constexpr (std::is_same_v<T, schema::ImuT>) {
    const auto matrix = [](const auto& x, const auto& y) {
      return std::equal(x.m()->begin(), x.m()->end(), y.m()->begin());
    };
    return a.flags == b.flags && a.orientation.x() == b.orientation.x() &&
           a.orientation.y() == b.orientation.y() && a.orientation.z() == b.orientation.z() &&
           a.orientation.w() == b.orientation.w() &&
           a.angular_velocity.x() == b.angular_velocity.x() &&
           a.angular_velocity.y() == b.angular_velocity.y() &&
           a.angular_velocity.z() == b.angular_velocity.z() &&
           a.linear_acceleration.x() == b.linear_acceleration.x() &&
           a.linear_acceleration.y() == b.linear_acceleration.y() &&
           a.linear_acceleration.z() == b.linear_acceleration.z() &&
           optional_equal(a.orientation_covariance, b.orientation_covariance, matrix) &&
           optional_equal(a.angular_velocity_covariance, b.angular_velocity_covariance, matrix) &&
           optional_equal(
               a.linear_acceleration_covariance, b.linear_acceleration_covariance, matrix) &&
           optional_equal(a.header, b.header, [](const auto& x, const auto& y) {
             return x.frame_id == y.frame_id && x.device_sequence == y.device_sequence &&
                    x.capture_timestamp_ns == y.capture_timestamp_ns &&
                    optional_equal(x.stamp, y.stamp, equal_value<schema::TimeT>);
           });
  } else {
    const auto tensor = [](const Tensor& x, const Tensor& y) {
      return x.data() == y.data() && x.nbytes() == y.nbytes();
    };
    return equal_packet_metadata(a, b) && optional_equal(a.first, b.first, tensor) &&
           optional_equal(a.second, b.second, tensor);
  }
}

std::size_t source_index(std::int64_t pattern, std::size_t operation) {
  switch (pattern) {
    case 0:
      return 0;
    case 1:
      return 1;
    case 2:
      return 2;
    case 3:
      return operation % 2 == 0 ? 2 : 0;
    case 4:
      return std::array<std::size_t, 4>{1, 2, 0, 2}[operation % 4];
    default:
      return 3 + operation % 32;
  }
}

// This binding isolates the public accessor; it does not model Runtime admission or retirement.
struct Selection {
  const void* sample{};
  std::size_t claims{};
  std::uint64_t epoch{1};
  static void retain(void* opaque) noexcept { ++static_cast<Selection*>(opaque)->claims; }
  static void release(void* opaque) noexcept { --static_cast<Selection*>(opaque)->claims; }
  static expected<detail::RuntimeScalarReadBinding, Error> read(const void* opaque, std::uint64_t,
                                                                bool claim) noexcept {
    auto& self = *const_cast<Selection*>(static_cast<const Selection*>(opaque));
    if (claim && self.claims != 0)
      return make_unexpected(Error{ErrorCode::kResourceExhausted});
    detail::PublishedReadClaimBinding credit;
    if (claim) {
      ++self.claims;
      credit = {.state = &self, .retain = retain, .release = release};
    }
    return detail::RuntimeScalarReadBinding{.sample = self.sample,
                                            .claim = credit,
                                            .view_generation = &self.epoch,
                                            .expected_view_generation = self.epoch};
  }
  static inline const detail::RuntimeInputBindingOps kOps{.scalar = read};
};

struct StorageDeleter {
  std::align_val_t alignment;
  void operator()(std::byte* value) const noexcept { ::operator delete(value, alignment); }
};

template <typename T>
struct ScratchFor {
  using Type = schema::SchemaScratch<T>;
};
template <>
struct ScratchFor<Packet> {
  using Type = schema::DescriptorMetadataScratch<Packet>;
};

template <typename T>
const detail::RuntimePortTypeOps* operation_type_ops(Operation operation) {
  if constexpr (std::is_same_v<T, Packet>) {
    // Preserve the original owning descriptor baseline even though the generated package now
    // opts into prepared storage. Do not silently relabel its old Materialize measurements.
    using PackageCodec = native_schema_benchmark::holoscan_generated_Packet_package::detail::
        GeneratedDescriptorCodec;
    using Callbacks =
        schema::detail::StandardDescriptorTensorCallbacks<Packet,
                                                          native_schema_benchmark::Packet,
                                                          PackageCodec,
                                                          &Packet::first,
                                                          &Packet::second>;
    static const auto legacy = schema::make_descriptor_tensor_port_type_ops<
        Packet,
        native_schema_benchmark::Packet,
        DescriptorTensorSchemaNativeTraits<Packet>::metadata_capacity>(
        typename Callbacks::Prepare{},
        typename Callbacks::Reset{},
        typename Callbacks::Encode{},
        typename Callbacks::Assign{});
    if (operation != Operation::kPreparedMaterialize && operation != Operation::kColdPrepareSlot)
      return &legacy;
  }
  return detail::runtime_port_type_ops<T>();
}

template <typename T>
class Fixture {
 public:
  explicit Fixture(Operation operation)
      : ops_(operation_type_ops<T>(operation)),
        storage_(static_cast<std::byte*>(
                     ::operator new(ops_->sample_size, std::align_val_t{ops_->sample_alignment})),
                 StorageDeleter{std::align_val_t{ops_->sample_alignment}}),
        output_wire_(ops_->wire_capacity) {
    sources_.reserve(kSourceCount);
    for (std::size_t i = 0; i < kSourceCount; ++i) {
      sources_.emplace_back(make_source<T>(i, first_, second_), SampleMetadata{});
      sources_[i].metadata.frame_id = 23;
      if constexpr (std::is_same_v<T, Packet>) {
        sources_[i].metadata.byte_size = 32;
        sources_[i].metadata.payload_kind = PayloadKind::kDescriptorBacked;
      }
      wires_[i].resize(ops_->wire_capacity);
      auto encoded = encode(sources_[i].data, sources_[i].metadata, wires_[i]);
      if (!encoded)
        throw RuntimeError(encoded.error());
      sizes_[i] = *encoded;
    }
    // Perform fallible scratch creation before constructing a slot with explicit destruction.
    if (operation == Operation::kPreparedCopyEdit || operation == Operation::kColdPrepareScratch) {
      const test::ScopedAllocationCounter counter;
      auto prepared = ScratchFor<T>::Type::create();
      cold_scratch_allocations = counter.count();
      if (!prepared)
        throw RuntimeError(prepared.error());
      scratch_.emplace(std::move(*prepared));
      cold_scratch_bytes = scratch_->memory_usage().total_bytes();
    }
    // This existing materialize case remains the owning-codec baseline for ordinary schemas.
    // Never destroy a prepared hidden inventory through the owning Sample-only destructor.
    if (operation == Operation::kPreparedMaterialize || operation == Operation::kColdPrepareSlot ||
        (std::is_same_v<T, Packet> && operation == Operation::kMaterialize)) {
      const test::ScopedAllocationCounter counter;
      auto prepared = ops_->prepare_sample_storage(storage_.get());
      cold_allocations = counter.count();
      if (!prepared)
        throw RuntimeError(prepared.error());
      constructed_ = true;
      prepared_ = true;
      if (operation == Operation::kPreparedMaterialize ||
          operation == Operation::kColdPrepareSlot) {
        using Prepared = detail::RuntimePreparedSchemaTraits<T>;
        cold_slot_bytes = Prepared::inventory(storage_.get()).budget.memory().total_bytes();
      }
    }
    detail::RuntimePortBindingAccess::bind_input(input_, &input_binding_);
    detail::RuntimePortBindingAccess::bind_output(output_, &output_binding_);
  }
  Fixture(const Fixture&) = delete;
  Fixture& operator=(const Fixture&) = delete;
  ~Fixture() {
    detail::RuntimePortBindingAccess::unbind_input(input_, &input_binding_);
    detail::RuntimePortBindingAccess::unbind_output(output_, &output_binding_);
    if (prepared_)
      ops_->destroy_sample_storage(storage_.get());
    else if (constructed_)
      ops_->destroy_sample(storage_.get());
  }

  bool run(Operation operation, std::size_t index) {
    const auto& source = sources_[index];
    selection_.sample = &source;
    switch (operation) {
      case Operation::kColdPrepareSlot: {
        if (prepared_) {
          ops_->destroy_sample_storage(storage_.get());
          prepared_ = false;
          constructed_ = false;
        }
        auto prepared = ops_->prepare_sample_storage(storage_.get());
        prepared_ = static_cast<bool>(prepared);
        return prepared_;
      }
      case Operation::kColdPrepareScratch: {
        scratch_.reset();
        auto prepared = ScratchFor<T>::Type::create();
        if (!prepared)
          return false;
        scratch_.emplace(std::move(*prepared));
        return scratch_->memory_usage().total_bytes() == cold_scratch_bytes;
      }
      case Operation::kMaterialize:
      case Operation::kPreparedMaterialize: {
        if constexpr (std::is_same_v<T, Packet>) {
          ops_->reset_sample_storage(storage_.get());
          const auto bytes = std::span<const std::byte>{wires_[index]}.first(sizes_[index]);
          auto metadata = ops_->descriptor_tensor_sample_metadata(bytes, 0, 1);
          if (!metadata)
            return false;
          std::array<Tensor, 2> tensors{first_, second_};
          auto assigned = ops_->assign_descriptor_tensors_sample(
              storage_.get(), bytes, tensors, {*metadata}, {*metadata});
          return assigned && equal_value(sample().data, source.data) &&
                 sample().metadata.frame_id == source.metadata.frame_id;
        } else {
          if (prepared_) {
            ops_->reset_sample_storage(storage_.get());
            auto assigned = ops_->assign_sample_storage(
                storage_.get(),
                std::span<const std::byte>{wires_[index]}.first(sizes_[index]),
                source.metadata);
            return assigned && equal_value(sample().data, source.data) &&
                   sample().metadata.frame_id == source.metadata.frame_id;
          }
          if (constructed_) {
            ops_->destroy_sample(storage_.get());
            constructed_ = false;
          }
          auto decoded =
              ops_->construct_sample(storage_.get(),
                                     std::span<const std::byte>{wires_[index]}.first(sizes_[index]),
                                     source.metadata);
          constructed_ = static_cast<bool>(decoded);
          return decoded && equal_value(sample().data, source.data) &&
                 sample().metadata.frame_id == source.metadata.frame_id;
        }
      }
      case Operation::kReceive: {
        auto received = input_.receive();
        return received && equal_value(received->data, source.data) &&
               received->metadata.frame_id == source.metadata.frame_id;
      }
      case Operation::kBorrow: {
        auto borrowed = input_.selected_sample();
        return borrowed && &borrowed->sample() == &source && selection_.claims == 0;
      }
      case Operation::kCopyEdit: {
        edited_ = source.data;
        const bool valid = equal_value(edited_, source.data);
        if constexpr (std::is_same_v<T, schema::TimeT>) {
          ++edited_.time_ns;
          return valid && edited_.time_ns != source.data.time_ns;
        } else if constexpr (std::is_same_v<T, schema::ImuT>) {
          edited_.flags = schema::ImuFlags_UNCALIBRATED;
          return valid && edited_.flags != source.data.flags;
        } else {
          ++edited_.sequence;
          return valid && edited_.sequence != source.data.sequence;
        }
      }
      case Operation::kPreparedCopyEdit: {
        if constexpr (std::is_same_v<T, Packet>) {
          auto copied = scratch_->copy_metadata_from(source.data);
          auto& value = scratch_->descriptor();
          const bool valid = copied && equal_packet_metadata(value, source.data) && !value.first &&
                             value.first.use_count() == 0 && !value.second &&
                             value.second.use_count() == 0;
          ++value.sequence;
          return valid && value.sequence != source.data.sequence;
        } else {
          auto copied = scratch_->copy_from(source.data);
          auto& value = scratch_->value();
          const bool valid = copied && equal_value(value, source.data);
          if constexpr (std::is_same_v<T, schema::TimeT>) {
            ++value.time_ns;
            return valid && value.time_ns != source.data.time_ns;
          } else {
            value.flags = schema::ImuFlags_UNCALIBRATED;
            return valid && value.flags != source.data.flags;
          }
        }
      }
      case Operation::kEncode: {
        auto encoded = encode(source.data, source.metadata, output_wire_);
        return encoded && *encoded == sizes_[index] &&
               std::equal(
                   output_wire_.begin(), output_wire_.begin() + *encoded, wires_[index].begin());
      }
      case Operation::kEmitEncodeBinding:
        // Isolate by-value argument copying plus synchronous encoding; not Runtime publication.
        expected_source_ = &source;
        return static_cast<bool>(output_.emit(source.data, EmitOptions{.frame_id = 23}));
      case Operation::kEmitRefEncodeBinding:
        if constexpr (!std::is_same_v<T, Packet>) {
          expected_source_ = &source;
          return static_cast<bool>(output_.emit_ref(source.data, EmitOptions{.frame_id = 23}));
        }
        return false;
    }
    return false;
  }

  bool claims_released() const { return selection_.claims == 0; }
  std::size_t cold_allocations{};
  std::size_t cold_slot_bytes{};
  std::size_t cold_scratch_allocations{};
  std::size_t cold_scratch_bytes{};

 private:
  expected<std::size_t, Error> encode(const T& value, SampleMetadata metadata,
                                      std::span<std::byte> wire) {
    if constexpr (std::is_same_v<T, Packet>)
      return ops_->encode_descriptor_tensor_sample(&value, metadata, 1, wire);
    else
      return ops_->encode_payload(&value, wire);
  }
  static expected<void, Error> emit(void* opaque, const void* raw) noexcept {
    auto& self = *static_cast<Fixture*>(opaque);
    const auto& sample = *static_cast<const Sample<T>*>(raw);
    if (!equal_value(sample.data, self.expected_source_->data) || sample.metadata.frame_id != 23)
      return make_unexpected(Error{ErrorCode::kFailure});
    auto metadata = sample.metadata;
    if constexpr (std::is_same_v<T, Packet>) {
      metadata.byte_size = 32;
      metadata.payload_kind = PayloadKind::kDescriptorBacked;
    }
    auto encoded = self.encode(sample.data, metadata, self.output_wire_);
    if (!encoded)
      return make_unexpected(std::move(encoded).error());
    return {};
  }
  static expected<void, Error> emit_ref(void* opaque, const void* raw,
                                        EmitOptions options) noexcept {
    auto& self = *static_cast<Fixture*>(opaque);
    if (raw != &self.expected_source_->data || options.frame_id != 23)
      return make_unexpected(Error{ErrorCode::kFailure});
    auto encoded = self.encode(
        *static_cast<const T*>(raw), self.expected_source_->metadata, self.output_wire_);
    if (!encoded)
      return make_unexpected(std::move(encoded).error());
    return {};
  }
  Sample<T>& sample() { return *reinterpret_cast<Sample<T>*>(storage_.get()); }

  const detail::RuntimePortTypeOps* ops_;
  HostPixels first_pixels_;
  HostPixels second_pixels_;
  Tensor first_{&first_pixels_.managed};
  Tensor second_{&second_pixels_.managed};
  std::vector<Sample<T>> sources_;
  std::array<std::vector<std::byte>, kSourceCount> wires_{};
  std::array<std::size_t, kSourceCount> sizes_{};
  std::unique_ptr<std::byte, StorageDeleter> storage_;
  std::vector<std::byte> output_wire_;
  T edited_{};
  std::optional<typename ScratchFor<T>::Type> scratch_;
  Selection selection_;
  Input<T> input_{"input"};
  Output<T> output_{"output"};
  detail::RuntimeInputBinding input_binding_{
      .state = &selection_, .ops = &Selection::kOps, .generation = 1};
  static inline const detail::RuntimeOutputBindingOps kOutputOps{.emit_sample = emit,
                                                                 .emit_ref = emit_ref};
  detail::RuntimeOutputBinding output_binding_{.state = this, .ops = &kOutputOps};
  const Sample<T>* expected_source_{};
  bool constructed_{};
  bool prepared_{};
};

template <typename T>
void measure(benchmark::State& state, Operation operation) {
  const auto pattern = state.range(0);
  std::size_t first_allocations{}, repeated_allocations{}, maximum_allocations{};
  bool valid = true;
  {
    Fixture<T> counted{operation};
    state.counters["cold_slot_allocations"] = static_cast<double>(counted.cold_allocations);
    state.counters["cold_prepared_slot_bytes"] = static_cast<double>(counted.cold_slot_bytes);
    state.counters["cold_scratch_allocations"] =
        static_cast<double>(counted.cold_scratch_allocations);
    state.counters["cold_scratch_bytes"] = static_cast<double>(counted.cold_scratch_bytes);
    for (std::size_t i = 0; i <= kCountedOperations; ++i) {
      std::size_t allocations{};
      {
        const test::ScopedAllocationCounter counter;
        valid = counted.run(operation, source_index(pattern, i)) && valid;
        allocations = counter.count();
      }
      if (i == 0)
        first_allocations = allocations;
      else
        repeated_allocations += allocations;
      maximum_allocations = std::max(maximum_allocations, allocations);
      valid = counted.claims_released() && valid;
    }
  }
  const bool allocation_free =
      operation == Operation::kBorrow || operation == Operation::kPreparedMaterialize ||
      operation == Operation::kPreparedCopyEdit || operation == Operation::kEmitRefEncodeBinding;
  if (!valid || (allocation_free && maximum_allocations != 0)) {
    state.SkipWithError(
        "native schema validation, read-credit, or zero-allocation contract failed");
    return;
  }
  // A separate fixture prevents the counted pass from pre-warming the timed operation's state.
  Fixture<T> timed{operation};
  std::size_t iteration{};
  for (auto _ : state) {
    if (!timed.run(operation, source_index(pattern, iteration++))) {
      state.SkipWithError("native schema timed operation validation failed");
      break;
    }
    benchmark::ClobberMemory();
  }
  state.counters["first_operation_allocations"] = static_cast<double>(first_allocations);
  state.counters["allocations_per_operation"] =
      static_cast<double>(repeated_allocations) / static_cast<double>(kCountedOperations);
  state.counters["maximum_operation_allocations"] = static_cast<double>(maximum_allocations);
  state.counters["counted_operations"] = static_cast<double>(kCountedOperations + 1);
  state.counters["native_bytes"] = static_cast<double>(sizeof(T));
  state.SetItemsProcessed(state.iterations());
}

template <typename T>
void register_type(const char* name, bool scalar_only) {
  for (const auto& [operation, suffix] :
       std::array{std::pair{Operation::kMaterialize, "Materialize"},
                  std::pair{Operation::kReceive, "Receive"},
                  std::pair{Operation::kBorrow, "Borrow"},
                  std::pair{Operation::kCopyEdit, "CopyEdit"},
                  std::pair{Operation::kEncode, "Encode"},
                  std::pair{Operation::kEmitEncodeBinding, "EmitEncodeBinding"},
                  std::pair{Operation::kPreparedMaterialize, "PreparedMaterialize"},
                  std::pair{Operation::kPreparedCopyEdit, "PreparedCopyEdit"},
                  std::pair{Operation::kEmitRefEncodeBinding, "EmitRefEncodeBinding"},
                  std::pair{Operation::kColdPrepareSlot, "ColdPrepareSlot"},
                  std::pair{Operation::kColdPrepareScratch, "ColdPrepareScratch"}}) {
    if (std::is_same_v<T, Packet> && operation == Operation::kEmitRefEncodeBinding)
      continue;
    const std::string label = std::string("NativeSchema/") + name + "/" + suffix;
    auto* benchmark = benchmark::RegisterBenchmark(
        label.c_str(), [operation](benchmark::State& state) { measure<T>(state, operation); });
    benchmark->ArgName("pattern");
    const bool cold =
        operation == Operation::kColdPrepareSlot || operation == Operation::kColdPrepareScratch;
    for (std::int64_t pattern = 0; pattern < (scalar_only || cold ? 1 : 6); ++pattern)
      benchmark->Arg(pattern);
  }
}

const bool registered = [] {
  register_type<schema::TimeT>("Time", true);
  register_type<schema::ImuT>("Imu", false);
  register_type<Packet>("Packet", false);
  return true;
}();

}  // namespace
}  // namespace holoscan
