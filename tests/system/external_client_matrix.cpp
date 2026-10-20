/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// This system fixture exercises the complete external-client matrix with the
// same public source in both localities. Each payload enters through a compiled
// Graph import, is observed and forwarded by an ordinary Operator, and leaves
// through a compiled Graph export. The real-child path receives only the
// canonical catalog file and endpoint names; it inherits no Graph, plan, or
// transport object from the parent.
//
// Data-flow topology (three parallel echo paths inside a single compiled Graph):
//
//   External publisher
//        |
//        v  holomq://[import endpoint]
//   ┌─────────────────────────────────────────┐
//   │  MatrixEcho<MatrixPod>                  │
//   │  MatrixEcho<schema::StringT>   (Graph)  │ ← MatrixProbe<T> records each sample
//   │  MatrixEcho<Tensor>                     │
//   └─────────────────────────────────────────┘
//        |
//        v  holomq://[export endpoint]
//   External subscriber  →  validate_sample()
//
// The 12-cell matrix covers: 3 payload types × 2 locality modes × 2 direction
// checks (import-side probe vs. export-side probe). Every cell exercises and
// closes all three producer generations so the complete graph can drain
// naturally; the selected cell controls which observation is reported.

#include <poll.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <limits>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <type_traits>
#include <utility>

#include <holoscan/channel/channel.hpp>
#include <holoscan/channel/endpoint_catalog.hpp>
#include <holoscan/channel/pod_type.hpp>
#include <holoscan/channel/runtime.hpp>
#include <holoscan/core/compile.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/run.hpp>
// Provides generated owning types and schema-trait specializations.
#include <holoscan/schema/sdk_schemas.hpp>  // NOLINT(misc-include-cleaner)
#include <holoscan/time/realtime_clock.hpp>

namespace {

using namespace std::chrono_literals;

constexpr DLDataType kByteDtype{.code = kDLUInt, .bits = 8U, .lanes = 1U};
constexpr std::array<std::int64_t, 1U> kTensorShape{8};
constexpr std::array kTensorAxes{holoscan::TensorAxisBounds{
    .max_extent = 8U,
    .max_abs_stride_bytes = 1U,
}};
constexpr holoscan::TensorRepresentation kHostByteVector{
    .memory_kind = holoscan::MemoryKind::kHost,
    .dtype = kByteDtype,
    .rank = 1U,
};

// Minimal two-field POD used as the kPod payload. Small enough to travel
// inline through HoloMQ without FlatBuffers serialization.
struct MatrixPod {
  std::uint32_t frame{};
  std::int32_t value{};

  friend bool operator==(const MatrixPod&, const MatrixPod&) noexcept = default;
};

}  // namespace

namespace holoscan {

template <>
struct ChannelTypeIdentity<::MatrixPod> {
  static constexpr std::string_view name = "holoscan.test.MatrixPod";
  static constexpr std::uint32_t version = 1U;
};

template <>
struct ChannelPodLayout<::MatrixPod> {
  static constexpr auto fields =
      pod_fields(pod_field("frame", &::MatrixPod::frame, offsetof(::MatrixPod, frame)),
                 pod_field("value", &::MatrixPod::value, offsetof(::MatrixPod, value)));
};

}  // namespace holoscan

namespace {

enum class MatrixPayload : std::uint8_t { kPod, kSchema, kTensor };
enum class MatrixDirection : std::uint8_t { kImport, kExport };
enum class MatrixLocality : std::uint8_t { kSameProcess, kRealChild };

// One cell in the 12-entry test matrix, identifying which direction is under
// scrutiny, which locality mode to use, and which payload type to exercise.
struct MatrixCell {
  MatrixDirection direction;
  MatrixLocality locality;
  MatrixPayload payload;
};

constexpr std::array kMatrixCells{
    MatrixCell{MatrixDirection::kImport, MatrixLocality::kSameProcess, MatrixPayload::kPod},
    MatrixCell{MatrixDirection::kExport, MatrixLocality::kSameProcess, MatrixPayload::kPod},
    MatrixCell{MatrixDirection::kImport, MatrixLocality::kSameProcess, MatrixPayload::kSchema},
    MatrixCell{MatrixDirection::kExport, MatrixLocality::kSameProcess, MatrixPayload::kSchema},
    MatrixCell{MatrixDirection::kImport, MatrixLocality::kSameProcess, MatrixPayload::kTensor},
    MatrixCell{MatrixDirection::kExport, MatrixLocality::kSameProcess, MatrixPayload::kTensor},
    MatrixCell{MatrixDirection::kImport, MatrixLocality::kRealChild, MatrixPayload::kPod},
    MatrixCell{MatrixDirection::kExport, MatrixLocality::kRealChild, MatrixPayload::kPod},
    MatrixCell{MatrixDirection::kImport, MatrixLocality::kRealChild, MatrixPayload::kSchema},
    MatrixCell{MatrixDirection::kExport, MatrixLocality::kRealChild, MatrixPayload::kSchema},
    MatrixCell{MatrixDirection::kImport, MatrixLocality::kRealChild, MatrixPayload::kTensor},
    MatrixCell{MatrixDirection::kExport, MatrixLocality::kRealChild, MatrixPayload::kTensor},
};
static_assert(kMatrixCells.size() == 12U);

// Payload-type-agnostic snapshot of a received Sample's observable fields.
// Collapsing the payload into a hash lets import-side and export-side
// observations be compared without templating the comparison logic on T.
struct NormalizedObservation {
  std::uint64_t payload_hash{};
  std::uint64_t trace_id{};
  std::uint64_t frame_id{};
  holoscan::SampleFlags flags{};
  holoscan::PayloadKind payload_kind{};
  std::size_t byte_size{};
  holoscan::ClockId source_clock_id{};
  bool capture_present{};
  bool publish_present{};

  friend bool operator==(const NormalizedObservation&,
                         const NormalizedObservation&) noexcept = default;
};

[[nodiscard]] constexpr bool capture_present(const holoscan::SampleMetadata& metadata) noexcept {
  return metadata.capture_time().valid();
}

// An exported sample must carry both halves of the publication coordinate. This mirrors the
// external-client assertion the reproducer used: a client that receives one half cannot place the
// sample on the runtime timeline, so a half-populated coordinate is a failure and not a partial
// success.
[[nodiscard]] constexpr bool publish_present(const holoscan::SampleMetadata& metadata) noexcept {
  return metadata.publish_clock_id.valid() && metadata.publish_timestamp_ns != 0 &&
         metadata.publish_time().valid();
}

// Canonical test value and deterministic content hash for each payload type.
// The hash collapses the payload to a uint64 so NormalizedObservation can
// carry it without being templated on T.
template <typename T>
struct MatrixTraits;

// MatrixPod specialization: bit-packs frame and value into a single uint64.
template <>
struct MatrixTraits<MatrixPod> {
  static MatrixPod value() noexcept { return MatrixPod{.frame = 17U, .value = -42}; }

  static std::uint64_t hash(const MatrixPod& value) noexcept {
    return (static_cast<std::uint64_t>(value.frame) << 32U) |
           static_cast<std::uint32_t>(value.value);
  }
};

// StringT specialization: FNV-1a hash over the string bytes.
template <>
struct MatrixTraits<holoscan::schema::StringT> {
  static holoscan::schema::StringT value() {
    holoscan::schema::StringT result;
    result.data = "holoscan-system-schema";
    return result;
  }

  static std::uint64_t hash(const holoscan::schema::StringT& value) noexcept {
    std::uint64_t result = 1469598103934665603ULL;
    for (const unsigned char byte : value.data) {
      result = (result ^ byte) * 1099511628211ULL;
    }
    return result;
  }
};

// Returns the wire byte size for the canonical StringT value so import- and
// export-side validators can agree on the expected byte_size metadata field
// without hard-coding a schema-version-specific constant.
[[nodiscard]] std::optional<std::size_t> canonical_schema_byte_size() {
  constexpr std::size_t kCapacity =
      holoscan::generated_schema_max_serialized_size_v<holoscan::schema::StringT>;
  // The generated schema encoder may materialize naturally aligned wire scalars in the caller's
  // scratch buffer. x86 tolerates an accidentally unaligned std::array, but portable callers must
  // provide the same maximum alignment required by the public schema-package contract.
  alignas(std::max_align_t) std::array<std::byte, kCapacity> encoded{};
  auto encoded_size = holoscan::schema_package<holoscan::schema::StringT>().encode(
      MatrixTraits<holoscan::schema::StringT>::value(), encoded);
  if (!encoded_size) {
    return std::nullopt;
  }
  return *encoded_size;
}

template <typename T>
[[nodiscard]] NormalizedObservation normalize_sample(const holoscan::Sample<T>& sample) {
  return NormalizedObservation{
      .payload_hash = MatrixTraits<T>::hash(sample.data),
      .trace_id = sample.metadata.trace_id,
      .frame_id = sample.metadata.frame_id,
      .flags = sample.metadata.flags,
      .payload_kind = sample.metadata.payload_kind,
      .byte_size = sample.metadata.byte_size,
      .source_clock_id = sample.metadata.source_clock_id,
      .capture_present = capture_present(sample.metadata),
      .publish_present = publish_present(sample.metadata),
  };
}

template <>
[[nodiscard]] NormalizedObservation normalize_sample(
    const holoscan::Sample<holoscan::Tensor>& sample) {
  std::uint64_t hash = 1469598103934665603ULL;
  const auto* bytes = static_cast<const std::byte*>(sample.data.data());
  if (bytes == nullptr || sample.data.nbytes() != static_cast<std::size_t>(kTensorShape.front())) {
    hash = 0U;
  } else {
    for (std::size_t index = 0U; index < sample.data.nbytes(); ++index) {
      hash = (hash ^ std::to_integer<std::uint8_t>(bytes[index])) * 1099511628211ULL;
    }
  }
  return NormalizedObservation{
      .payload_hash = hash,
      .trace_id = sample.metadata.trace_id,
      .frame_id = sample.metadata.frame_id,
      .flags = sample.metadata.flags,
      .payload_kind = sample.metadata.payload_kind,
      .byte_size = sample.metadata.byte_size,
      .source_clock_id = sample.metadata.source_clock_id,
      .capture_present = capture_present(sample.metadata),
      .publish_present = publish_present(sample.metadata),
  };
}

// Thread-safe static store for the last sample observed by MatrixEcho<T>.
// MatrixEcho::compute() calls record(); the test's validate_graph_import()
// calls snapshot() after the runtime has processed the sample.
template <typename T>
struct MatrixProbe {
  static void reset() {
    const std::lock_guard lock{mutex};
    observation.reset();
  }

  static void record(const holoscan::Sample<T>& sample) {
    const std::lock_guard lock{mutex};
    observation = normalize_sample(sample);
  }

  [[nodiscard]] static std::optional<NormalizedObservation> snapshot() {
    const std::lock_guard lock{mutex};
    return observation;
  }

  static inline std::mutex mutex;
  static inline std::optional<NormalizedObservation> observation;
};

// Barrier that counts MatrixEcho::start() calls and blocks until all three
// echo operators (Pod, Schema, Tensor) have entered the runtime. This prevents
// the external-client publisher from emitting before the Graph is ready to
// receive at the import boundary.
struct MatrixRuntimeStartupProbe {
  static void reset() noexcept { started.store(0U, std::memory_order_release); }

  static void record_start() noexcept { started.fetch_add(1U, std::memory_order_acq_rel); }

  [[nodiscard]] static bool wait_for_all() noexcept {
    constexpr std::uint32_t kExpectedOperators = 3U;
    const auto deadline = std::chrono::steady_clock::now() + 10s;
    while (std::chrono::steady_clock::now() < deadline) {
      if (started.load(std::memory_order_acquire) >= kExpectedOperators) {
        return true;
      }
      std::this_thread::sleep_for(2ms);
    }
    return false;
  }

  static inline std::atomic<std::uint32_t> started{};
};

// Bounds failure cleanup without making request_stop() part of the success path.
[[nodiscard]] bool wait_for_session_terminal(holoscan::RunSession& session) noexcept {
  const auto deadline = std::chrono::steady_clock::now() + 10s;
  while (session.is_running() && std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(2ms);
  }
  return !session.is_running();
}

// Tensor specialization of MatrixProbe: same interface, separate static
// state because Tensor is not a regular value type and needs explicit
// specialization rather than the generic template.
template <>
struct MatrixProbe<holoscan::Tensor> {
  static void reset() {
    const std::lock_guard lock{mutex};
    observation.reset();
  }

  static void record(const holoscan::Sample<holoscan::Tensor>& sample) {
    const std::lock_guard lock{mutex};
    observation = normalize_sample(sample);
  }

  [[nodiscard]] static std::optional<NormalizedObservation> snapshot() {
    const std::lock_guard lock{mutex};
    return observation;
  }

  static inline std::mutex mutex;
  static inline std::optional<NormalizedObservation> observation;
};

// Pass-through operator that records each received sample (via MatrixProbe<T>)
// and re-emits it unchanged. Verifies that Graph-internal routing preserves
// all metadata fields across the import → operator → export path.
template <typename T>
class MatrixEcho final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    if constexpr (std::same_as<T, holoscan::Tensor>) {
      spec.input(input, "input")
          .queue_depth(1U)
          .expects_tensor(holoscan::TensorInputSpec{.representation = kHostByteVector});
      spec.output(output, "output")
          .max_emits_per_compute(1U)
          .produces_tensor(holoscan::TensorOutputSpec{.representation = kHostByteVector});
    } else {
      spec.input(input, "input").queue_depth(1U);
      spec.output(output, "output").max_emits_per_compute(1U);
    }
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnEach{input});
    return result;
  }

  void start() override { MatrixRuntimeStartupProbe::record_start(); }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext&) override {
    auto sample = input.receive();
    if (!sample) {
      return holoscan::make_unexpected(std::move(sample.error()));
    }
    MatrixProbe<T>::record(*sample);
    return output.emit(std::move(*sample));
  }

  holoscan::Input<T> input;
  holoscan::Output<T> output;
};

// Endpoint names assigned during graph construction; passed to external clients
// so they can look up the corresponding catalog entries by name.
struct MatrixNames {
  std::string pod_import;
  std::string pod_export;
  std::string schema_import;
  std::string schema_export;
  std::string tensor_import;
  std::string tensor_export;
};

// Compiled execution plan for the 3-echo graph, bundled with the endpoint
// names needed to locate publisher/subscriber catalog entries at runtime.
struct MatrixGraph {
  holoscan::ExecutionPlan plan;
  MatrixNames names;
};

// Returns a process-unique endpoint name by embedding the PID and a monotonic
// ordinal. Prevents collisions when the test binary re-runs in the same HoloMQ
// namespace without restarting the service.
[[nodiscard]] std::string unique_name(std::string_view stem) {
  static std::atomic<std::uint64_t> ordinal{};
  return "system_" + std::string(stem) + "_" + std::to_string(::getpid()) + "_" +
         std::to_string(ordinal.fetch_add(1U, std::memory_order_relaxed));
}

// TopicOptions for an import endpoint: publisher side with one loan slot.
[[nodiscard]] holoscan::TopicOptions ordinary_import_options(std::string_view name) {
  holoscan::TopicOptions options;
  options.endpoint_name = name;
  options.bounds.publisher = holoscan::PublisherEndpointBounds{.max_loaned_samples = 1U};
  return options;
}

// TopicOptions for an export endpoint: subscriber side with depth-1 queue.
[[nodiscard]] holoscan::TopicOptions ordinary_export_options(std::string_view name) {
  holoscan::TopicOptions options;
  options.endpoint_name = name;
  options.bounds.subscriber = holoscan::SubscriberEndpointBounds{
      .queue_depth = 1U,
      .max_borrowed_samples = 1U,
  };
  return options;
}

// Appends Tensor layout requirements to TopicOptions so the channel can
// allocate and validate the shared-memory descriptor for a Tensor payload.
void add_tensor_contract(holoscan::TopicOptions& options) {
  options.tensor = holoscan::TensorBoundaryContract{
      .representation =
          holoscan::TensorRepresentation{
              .memory_kind = holoscan::MemoryKind::kHost,
              .dtype = kByteDtype,
              .rank = 1U,
              .strides = holoscan::TensorStridePolicy::kContiguousOnly,
          },
      .bounds =
          holoscan::TensorBounds{
              .max_rank = 1U,
              .max_byte_span = 8U,
              .axes = kTensorAxes,
          },
      .transfer =
          holoscan::TensorTransferPolicy{.foreign_emit_policy =
                                             holoscan::InterfaceTensorEmitPolicy::kCopyIfNeeded},
  };
}

// Wires one import → MatrixEcho<T> → export path into the graph and records
// the generated endpoint names in import_name / export_name for later use by
// external clients.
template <typename T>
void add_echo_path(holoscan::Graph& graph, std::string_view stem, std::string& import_name,
                   std::string& export_name, const holoscan::ClockKey& capture_clock, bool tensor) {
  import_name = unique_name(std::string(stem) + "_import");
  export_name = unique_name(std::string(stem) + "_export");
  auto import_options = ordinary_import_options(import_name);
  auto export_options = ordinary_export_options(export_name);
  if (tensor) {
    add_tensor_contract(import_options);
    add_tensor_contract(export_options);
  } else {
    import_options.capture_time = holoscan::RequireCaptureTime{.clock = capture_clock};
    export_options.capture_time = holoscan::RequireCaptureTime{.clock = capture_clock};
  }
  const auto imported = graph.import_topic<T>("holomq://" + import_name, std::move(import_options));
  const auto exported = graph.export_topic<T>("holomq://" + export_name, std::move(export_options));
  const auto echo = graph.op<MatrixEcho<T>>(std::string(stem) + "-echo");
  graph.add_flow(imported.output(), echo->input, holoscan::ConnectionOptions{.queue_depth = 1U});
  graph.add_flow(echo->output, exported.input(), holoscan::ConnectionOptions{.queue_depth = 1U});
  graph.partition(std::string(stem) + "-path").add(echo);
}

// Builds and compiles the full 3-echo graph. An extra clock is added first
// so the default-clock ID is not 1, making catalog-to-plan clock translation
// observable in validate_graph_import.
[[nodiscard]] MatrixGraph make_matrix_graph() {
  holoscan::Graph graph{"holoscan-external-client-matrix"};
  // Keep the plan clock namespace intentionally different from the external catalog's dense
  // namespace so the system path proves catalog-to-plan translation instead of passing through a
  // coincidentally equal numeric ClockId.
  static_cast<void>(graph.add_clock<holoscan::RealtimeClock>("aaa-unused-clock"));
  const auto clock = graph.add_clock<holoscan::RealtimeClock>("matrix-clock");
  graph.set_default_clock(clock);
  MatrixNames names;
  add_echo_path<MatrixPod>(graph, "pod", names.pod_import, names.pod_export, clock, false);
  add_echo_path<holoscan::schema::StringT>(
      graph, "schema", names.schema_import, names.schema_export, clock, false);
  add_echo_path<holoscan::Tensor>(
      graph, "tensor", names.tensor_import, names.tensor_export, clock, true);
  return MatrixGraph{.plan = holoscan::compile(graph), .names = std::move(names)};
}

// Polls both sides until each sees the other's presence, then waits for that
// state to be stable over kStableConnectionObservations consecutive checks.
// The stability window guards against zero-copy receiver attachment races.
template <typename Publisher, typename Subscriber>
[[nodiscard]] bool wait_for_connection(Publisher& publisher, Subscriber& subscriber,
                                       std::string_view label) {
  const auto deadline = std::chrono::steady_clock::now() + 10s;
  // HoloMQ can publish registry-visible peer state before the zero-copy receiver attachment is
  // ready to accept a sample. Tensor sends cross the submit boundary even when that first delivery
  // count is zero, so require a short bounded stability interval instead of treating one advisory
  // registry observation as a delivery guarantee.
  constexpr std::uint32_t kStableConnectionObservations = 16U;
  std::uint32_t consecutive{};
  bool publisher_ready = false;
  bool subscriber_ready = false;
  while (std::chrono::steady_clock::now() < deadline) {
    auto updated = publisher.update_connections();
    auto ready = subscriber.has_samples();
    if (!updated || !ready) {
      std::cerr << "matrix " << label << " connection refresh failed: publisher="
                << (updated ? "ok" : holoscan::runtime_error_message(updated.error()))
                << ", subscriber="
                << (ready ? "ok" : holoscan::runtime_error_message(ready.error())) << '\n';
      return false;
    }
    publisher_ready = publisher.has_subscribers();
    subscriber_ready = subscriber.has_publishers();
    if (publisher_ready && subscriber_ready) {
      if (++consecutive == kStableConnectionObservations) {
        return true;
      }
    } else {
      consecutive = 0U;
    }
    std::this_thread::sleep_for(2ms);
  }
  std::cerr << "matrix " << label
            << " connection timed out: publisher_has_subscribers=" << publisher_ready
            << ", subscriber_has_publishers=" << subscriber_ready << '\n';
  return false;
}

// Retries receive_sample() until a non-kNotReady result arrives or the
// deadline expires, returning the last result (success or non-transient error).
template <typename Subscriber>
[[nodiscard]] auto receive_until(Subscriber& subscriber) {
  using Result = decltype(subscriber.receive_sample());
  const auto deadline = std::chrono::steady_clock::now() + 10s;
  while (std::chrono::steady_clock::now() < deadline) {
    Result sample = subscriber.receive_sample();
    if (sample || sample.error().code != holoscan::ErrorCode::kNotReady) {
      return sample;
    }
    std::this_thread::sleep_for(2ms);
  }
  return Result{holoscan::make_unexpected(holoscan::Error{holoscan::ErrorCode::kTimeout})};
}

// Builds EmitOptions with per-ordinal frame_id/trace_id and kDegraded flag so
// validate_sample() can confirm all metadata fields survive the round trip
// through the channel and echo operator.
[[nodiscard]] holoscan::EmitOptions emit_options(std::uint64_t ordinal) {
  return holoscan::EmitOptions{
      .frame_id = 100U + ordinal,
      .trace =
          holoscan::TraceContext{
              .trace_id = 1000U + ordinal,
              .span_id = 2000U + ordinal,
              .parent_span_id = 3000U + ordinal,
          },
      .flags = holoscan::SampleFlags::kDegraded,
  };
}

[[nodiscard]] bool validate_sample(const holoscan::Sample<MatrixPod>& sample) {
  return sample.data == MatrixTraits<MatrixPod>::value() && sample.metadata.frame_id == 101U &&
         sample.metadata.trace_id == 1001U &&
         sample.metadata.flags == holoscan::SampleFlags::kDegraded &&
         sample.metadata.sequence != 0U && sample.metadata.byte_size == sizeof(MatrixPod) &&
         capture_present(sample.metadata) && publish_present(sample.metadata);
}

[[nodiscard]] bool validate_sample(const holoscan::Sample<holoscan::schema::StringT>& sample) {
  const auto expected_byte_size = canonical_schema_byte_size();
  return expected_byte_size &&
         sample.data.data == MatrixTraits<holoscan::schema::StringT>::value().data &&
         sample.metadata.frame_id == 102U && sample.metadata.trace_id == 1002U &&
         sample.metadata.flags == holoscan::SampleFlags::kDegraded &&
         sample.metadata.sequence != 0U && sample.metadata.byte_size == *expected_byte_size &&
         capture_present(sample.metadata) && publish_present(sample.metadata);
}

// Tensor payloads travel via descriptor (zero-copy shared memory), so the
// channel does not stamp a capture time — !capture_present is intentional and
// distinguishes the Tensor path from the value-typed paths above. Publication time is
// unconditional by contrast: Runtime owns it and stamps it on every exported path, so requiring it
// here is what keeps the descriptor path from silently diverging from the value-typed ones.
[[nodiscard]] bool validate_sample(const holoscan::Sample<holoscan::Tensor>& sample) {
  if (!sample.data.is_read_only() || sample.data.nbytes() != 8U || sample.data.data() == nullptr ||
      sample.metadata.frame_id != 103U || sample.metadata.trace_id != 1003U ||
      sample.metadata.flags != holoscan::SampleFlags::kDegraded || sample.metadata.sequence == 0U ||
      sample.metadata.payload_kind != holoscan::PayloadKind::kDescriptorBacked ||
      capture_present(sample.metadata) || !publish_present(sample.metadata)) {
    return false;
  }
  const auto* bytes = static_cast<const std::byte*>(sample.data.data());
  for (std::size_t index = 0U; index < 8U; ++index) {
    if (bytes[index] != static_cast<std::byte>(index + 1U)) {
      return false;
    }
  }
  return true;
}

// Holds export-side observations collected by external subscriber clients,
// indexed by MatrixPayload ordinal (kPod=0, kSchema=1, kTensor=2).
struct ClientResult {
  holoscan::ChannelRuntime runtime;
  std::array<NormalizedObservation, 3U> exports;
  std::optional<holoscan::ChannelSubscriber<MatrixPod>> pod_subscriber;
  std::optional<holoscan::ChannelSubscriber<holoscan::schema::StringT>> schema_subscriber;
  std::optional<holoscan::ChannelSubscriber<holoscan::Tensor>> tensor_subscriber;
};

// Creates external publisher and subscriber for a value-typed (Pod or Schema)
// payload, sends one sample with emit_captured, closes the producer, validates
// final-data-before-EOS at the export, and retains the remotely closed
// subscriber until the parent observes session completion.
template <typename T>
[[nodiscard]] bool run_value_clients(
    holoscan::ChannelRuntime& runtime, const holoscan::ChannelEndpointCatalog& catalog,
    std::string_view import_name, std::string_view export_name, std::uint64_t ordinal,
    NormalizedObservation& observation,
    std::optional<holoscan::ChannelSubscriber<T>>& retained_subscriber) {
  auto publisher = runtime.create_publisher<T>(catalog, import_name);
  auto subscriber = runtime.create_subscriber<T>(catalog, export_name);
  if (!publisher || !subscriber) {
    std::cerr << "matrix client creation failed for " << import_name << ": publisher="
              << (publisher ? "ok" : holoscan::runtime_error_message(publisher.error()))
              << ", subscriber="
              << (subscriber ? "ok" : holoscan::runtime_error_message(subscriber.error())) << '\n';
    return false;
  }
  if (!wait_for_connection(*publisher, *subscriber, import_name)) {
    return false;
  }
  auto emitted =
      publisher->emit_captured(MatrixTraits<T>::value(), 10'000 + ordinal, emit_options(ordinal));
  if (!emitted) {
    std::cerr << "matrix emit failed for " << import_name << ": "
              << holoscan::runtime_error_message(emitted.error()) << '\n';
    return false;
  }
  const auto closed = publisher->close();
  const auto closed_twice = publisher->close();
  if (!closed || closed_twice || closed_twice.error().code != holoscan::ErrorCode::kClosed) {
    std::cerr << "matrix producer close validation failed for " << import_name << '\n';
    return false;
  }
  auto sample = receive_until(*subscriber);
  if (!sample) {
    std::cerr << "matrix receive failed for " << export_name << ": "
              << holoscan::runtime_error_message(sample.error()) << '\n';
    return false;
  }
  if (!validate_sample(*sample)) {
    std::cerr << "matrix sample validation failed for " << export_name << '\n';
    return false;
  }
  observation = normalize_sample(*sample);
  auto remote_end_of_stream = receive_until(*subscriber);
  if (remote_end_of_stream || remote_end_of_stream.error().code != holoscan::ErrorCode::kClosed) {
    std::cerr << "matrix remote EOS validation failed for " << export_name << '\n';
    return false;
  }
  retained_subscriber.emplace(std::move(*subscriber));
  return true;
}

// Creates external publisher and subscriber for a Tensor payload, loans a
// write buffer, fills it with bytes 1..8, commits and emits it, then closes the
// producer and validates descriptor data before remote EOS.
[[nodiscard]] bool run_tensor_clients(
    holoscan::ChannelRuntime& runtime, const holoscan::ChannelEndpointCatalog& catalog,
    std::string_view import_name, std::string_view export_name, NormalizedObservation& observation,
    std::optional<holoscan::ChannelSubscriber<holoscan::Tensor>>& retained_subscriber) {
  auto publisher = runtime.create_publisher<holoscan::Tensor>(catalog, import_name);
  auto subscriber = runtime.create_subscriber<holoscan::Tensor>(catalog, export_name);
  if (!publisher || !subscriber) {
    std::cerr << "matrix Tensor client creation failed for " << import_name << ": publisher="
              << (publisher ? "ok" : holoscan::runtime_error_message(publisher.error()))
              << ", subscriber="
              << (subscriber ? "ok" : holoscan::runtime_error_message(subscriber.error())) << '\n';
    return false;
  }
  if (!wait_for_connection(*publisher, *subscriber, import_name)) {
    return false;
  }
  auto loan = publisher->loan_tensor(
      holoscan::TensorLoanRequest{.shape = kTensorShape, .dtype = kByteDtype});
  if (!loan) {
    std::cerr << "matrix Tensor loan failed for " << import_name << ": "
              << holoscan::runtime_error_message(loan.error()) << '\n';
    return false;
  }
  auto writer = loan->write_host();
  if (!writer || writer->byte_size() != 8U) {
    std::cerr << "matrix Tensor writer failed for " << import_name << ": "
              << (writer ? "unexpected byte size" : holoscan::runtime_error_message(writer.error()))
              << '\n';
    return false;
  }
  auto* bytes = static_cast<std::byte*>(writer->data());
  for (std::size_t index = 0U; index < 8U; ++index) {
    bytes[index] = static_cast<std::byte>(index + 1U);
  }
  auto committed = std::move(*writer).commit();
  if (!committed) {
    std::cerr << "matrix Tensor commit failed for " << import_name << ": "
              << holoscan::runtime_error_message(committed.error()) << '\n';
    return false;
  }
  auto emitted = publisher->emit(std::move(*loan), emit_options(3U));
  if (!emitted) {
    std::cerr << "matrix Tensor emit failed for " << import_name << ": "
              << holoscan::runtime_error_message(emitted.error()) << '\n';
    return false;
  }
  const auto producer_closed = publisher->close();
  if (!producer_closed) {
    std::cerr << "matrix Tensor producer close failed for " << import_name << ": "
              << holoscan::runtime_error_message(producer_closed.error()) << '\n';
    return false;
  }
  {
    auto sample = receive_until(*subscriber);
    if (!sample) {
      std::cerr << "matrix Tensor receive failed for " << export_name << ": "
                << holoscan::runtime_error_message(sample.error()) << '\n';
      return false;
    }
    if (!validate_sample(*sample)) {
      std::cerr << "matrix Tensor sample validation failed for " << export_name << '\n';
      return false;
    }
    observation = normalize_sample(*sample);
  }
  auto remote_end_of_stream = receive_until(*subscriber);
  if (remote_end_of_stream || remote_end_of_stream.error().code != holoscan::ErrorCode::kClosed) {
    std::cerr << "matrix Tensor remote EOS validation failed for " << export_name << '\n';
    return false;
  }
  retained_subscriber.emplace(std::move(*subscriber));
  return true;
}

[[nodiscard]] bool close_client_subscribers(ClientResult& result) {
  auto close = [](auto& subscriber) {
    if (!subscriber) {
      return false;
    }
    auto closed = subscriber->close();
    subscriber.reset();
    return static_cast<bool>(closed);
  };
  const bool pod_closed = close(result.pod_subscriber);
  const bool schema_closed = close(result.schema_subscriber);
  const bool tensor_closed = close(result.tensor_subscriber);
  return pod_closed && schema_closed && tensor_closed;
}

// Exercises all three finite paths so every graph input reaches EOS. The caller
// still selects one import/export observation for each matrix cell.
[[nodiscard]] std::optional<ClientResult> run_clients(
    const holoscan::ChannelEndpointCatalog& catalog, const MatrixNames& names) {
  auto runtime = holoscan::ChannelRuntime::create();
  if (!runtime) {
    return std::nullopt;
  }
  ClientResult result{.runtime = std::move(*runtime)};
  const bool pod = run_value_clients<MatrixPod>(result.runtime,
                                                catalog,
                                                names.pod_import,
                                                names.pod_export,
                                                1U,
                                                result.exports[0],
                                                result.pod_subscriber);
  const bool schema = run_value_clients<holoscan::schema::StringT>(result.runtime,
                                                                   catalog,
                                                                   names.schema_import,
                                                                   names.schema_export,
                                                                   2U,
                                                                   result.exports[1],
                                                                   result.schema_subscriber);
  const bool tensor = run_tensor_clients(result.runtime,
                                         catalog,
                                         names.tensor_import,
                                         names.tensor_export,
                                         result.exports[2],
                                         result.tensor_subscriber);
  if (!pod || !schema || !tensor) {
    return std::nullopt;
  }
  return result;
}

// FNV-1a hash over the canonical 8-byte Tensor payload (bytes 1..8).
// Must match the hash computed in normalize_sample<Tensor>.
[[nodiscard]] std::uint64_t expected_tensor_hash() noexcept {
  std::uint64_t hash = 1469598103934665603ULL;
  for (std::uint8_t byte = 1U; byte <= 8U; ++byte) {
    hash = (hash ^ byte) * 1099511628211ULL;
  }
  return hash;
}

// Returns true when all observable fields in observation match the expected
// values for the given payload type: hash, frame/trace ids, flags, payload
// kind, byte size, clock validity, and timestamp presence.
[[nodiscard]] bool observation_matches(MatrixPayload payload,
                                       const NormalizedObservation& observation,
                                       const bool expect_publish) {
  if (observation.publish_present != expect_publish) {
    return false;
  }
  switch (payload) {
    case MatrixPayload::kPod:
      return observation.payload_hash ==
                 MatrixTraits<MatrixPod>::hash(MatrixTraits<MatrixPod>::value()) &&
             observation.frame_id == 101U && observation.trace_id == 1001U &&
             observation.flags == holoscan::SampleFlags::kDegraded &&
             observation.payload_kind == holoscan::PayloadKind::kInlineTyped &&
             observation.byte_size == sizeof(MatrixPod) && observation.capture_present;
    case MatrixPayload::kSchema:
      if (const auto expected_byte_size = canonical_schema_byte_size(); expected_byte_size) {
        return observation.payload_hash == MatrixTraits<holoscan::schema::StringT>::hash(
                                               MatrixTraits<holoscan::schema::StringT>::value()) &&
               observation.frame_id == 102U && observation.trace_id == 1002U &&
               observation.flags == holoscan::SampleFlags::kDegraded &&
               observation.payload_kind == holoscan::PayloadKind::kSerializedBytes &&
               observation.byte_size == *expected_byte_size && observation.capture_present;
      }
      return false;
    case MatrixPayload::kTensor:
      return observation.payload_hash == expected_tensor_hash() && observation.frame_id == 103U &&
             observation.trace_id == 1003U &&
             observation.flags == holoscan::SampleFlags::kDegraded &&
             observation.payload_kind == holoscan::PayloadKind::kDescriptorBacked &&
             observation.byte_size == 8U && !observation.source_clock_id.valid() &&
             !observation.capture_present;
  }
  return false;
}

// Checks that MatrixEcho recorded the correct sample for the given payload.
// Also verifies that the source_clock_id was translated from the catalog
// namespace into the plan's clock namespace (non-trivial ID mapping).
//
// These samples were published by an external client and imported, so they carry no publication
// coordinate: that publication belongs to the peer, whose clock this plan cannot place.
[[nodiscard]] bool validate_graph_import(const MatrixGraph& matrix, MatrixPayload payload) {
  const auto pod = MatrixProbe<MatrixPod>::snapshot();
  const auto schema = MatrixProbe<holoscan::schema::StringT>::snapshot();
  const auto tensor = MatrixProbe<holoscan::Tensor>::snapshot();
  switch (payload) {
    case MatrixPayload::kPod:
      return pod && observation_matches(payload, *pod, false) &&
             pod->source_clock_id == matrix.plan.clock_catalog().root_default_clock;
    case MatrixPayload::kSchema:
      return schema && observation_matches(payload, *schema, false) &&
             schema->source_clock_id == matrix.plan.clock_catalog().root_default_clock;
    case MatrixPayload::kTensor:
      return tensor && observation_matches(payload, *tensor, false);
  }
  return false;
}

// Checks that the external subscriber received the correct observation at
// the graph's export boundary, by index into the ClientResult export array.
//
// Runtime published these samples, so each must carry a publication coordinate the client can
// resolve against its catalog. This is the assertion the external-client reproducer needed.
[[nodiscard]] bool validate_graph_exports(MatrixPayload payload, const ClientResult& result) {
  const std::size_t index = static_cast<std::size_t>(payload);
  return index < result.exports.size() && observation_matches(payload, result.exports[index], true);
}

// Verifies that the catalog enforces direction constraints: a subscriber
// created against an import endpoint name and a publisher against an export
// endpoint name must both fail with kInvalidArgument.
template <typename T>
[[nodiscard]] bool rejects_wrong_catalog_roles(holoscan::ChannelRuntime& runtime,
                                               const holoscan::ChannelEndpointCatalog& catalog,
                                               std::string_view import_name,
                                               std::string_view export_name) {
  auto wrong_subscriber = runtime.create_subscriber<T>(catalog, import_name);
  auto wrong_publisher = runtime.create_publisher<T>(catalog, export_name);
  return !wrong_subscriber && !wrong_publisher &&
         wrong_subscriber.error().code == holoscan::ErrorCode::kInvalidArgument &&
         wrong_publisher.error().code == holoscan::ErrorCode::kInvalidArgument;
}

// Full catalog validation: plan health, distinguishable clock IDs for all
// clocks, correct direction metadata for all 6 endpoints, and rejection of
// wrong-role attachments for all three payload types.
[[nodiscard]] bool validate_catalog(const MatrixGraph& matrix) {
  if (!matrix.plan.ok()) {
    std::cerr << "matrix plan rejected:";
    for (const auto& diagnostic : matrix.plan.diagnostics()) {
      std::cerr << ' ' << diagnostic.token;
    }
    std::cerr << '\n';
    return false;
  }
  const auto clocks = matrix.plan.clock_catalog();
  if (clocks.clocks.size() < 2U || !clocks.root_default_clock.valid() ||
      clocks.root_default_clock.value == 1U) {
    std::cerr << "matrix clock namespaces are not distinguishable\n";
    return false;
  }
  const std::array imports{
      matrix.names.pod_import, matrix.names.schema_import, matrix.names.tensor_import};
  const std::array exports{
      matrix.names.pod_export, matrix.names.schema_export, matrix.names.tensor_export};
  for (const auto& name : imports) {
    const auto endpoint = matrix.plan.endpoint_catalog().endpoint(name);
    if (!endpoint || endpoint->name() != name ||
        endpoint->direction() != holoscan::TopicDirection::kImport) {
      std::cerr << "invalid matrix import endpoint " << name
                << " catalog=" << matrix.plan.endpoint_catalog().json() << '\n';
      return false;
    }
  }
  for (const auto& name : exports) {
    const auto endpoint = matrix.plan.endpoint_catalog().endpoint(name);
    if (!endpoint || endpoint->name() != name ||
        endpoint->direction() != holoscan::TopicDirection::kExport) {
      std::cerr << "invalid matrix export endpoint " << name << '\n';
      return false;
    }
  }
  auto runtime = holoscan::ChannelRuntime::create();
  if (!runtime) {
    return false;
  }
  const auto& catalog = matrix.plan.endpoint_catalog();
  const bool pod_rejected = rejects_wrong_catalog_roles<MatrixPod>(
      *runtime, catalog, matrix.names.pod_import, matrix.names.pod_export);
  const bool schema_rejected = rejects_wrong_catalog_roles<holoscan::schema::StringT>(
      *runtime, catalog, matrix.names.schema_import, matrix.names.schema_export);
  const bool tensor_rejected = rejects_wrong_catalog_roles<holoscan::Tensor>(
      *runtime, catalog, matrix.names.tensor_import, matrix.names.tensor_export);
  const bool rejected = pod_rejected && schema_rejected && tensor_rejected;
  if (!rejected) {
    std::cerr << "matrix wrong-role attachment did not fail closed: pod=" << pod_rejected
              << ", schema=" << schema_rejected << ", tensor=" << tensor_rejected << '\n';
  }
  return rejected;
}

// Runs one matrix cell with both the compiled Graph and the external client
// in the same process. The Runtime is started first so the import publisher
// discovers the external subscriber at emission rather than at startup.
[[nodiscard]] bool run_same_process(const MatrixGraph& matrix, MatrixCell cell) {
  MatrixProbe<MatrixPod>::reset();
  MatrixProbe<holoscan::schema::StringT>::reset();
  MatrixProbe<holoscan::Tensor>::reset();
  MatrixRuntimeStartupProbe::reset();
  // Start Runtime before constructing any external client. In particular, the Graph-bound catalog
  // Tensor publisher must discover the later subscriber at the publication boundary rather than
  // relying on the session's initial connection snapshot.
  holoscan::RunSession session = holoscan::run_async(matrix.plan);
  if (!MatrixRuntimeStartupProbe::wait_for_all()) {
    session.request_stop();
    session.wait();
    std::cerr << "matrix Runtime startup barrier timed out\n";
    return false;
  }
  auto clients = run_clients(matrix.plan.endpoint_catalog(), matrix.names);
  const bool imported = validate_graph_import(matrix, cell.payload);
  const bool exported = clients && validate_graph_exports(cell.payload, *clients);
  const bool selected =
      clients && (cell.direction == MatrixDirection::kImport ? imported : exported);
  const bool completed = clients && wait_for_session_terminal(session);
  if (!completed && session.is_running()) {
    session.request_stop();
  }
  session.wait();
  const bool subscribers_closed = clients && close_client_subscribers(*clients);
  if (!selected || !completed || session.termination() != holoscan::RunTermination::kCompleted ||
      !subscribers_closed || !session.diagnostics().empty()) {
    std::cerr << "matrix same-process runtime: clients=" << static_cast<bool>(clients)
              << ", imported=" << imported << ", exported=" << exported << ", selected=" << selected
              << ", completed=" << completed << ", subscribers_closed=" << subscribers_closed
              << ", termination=" << static_cast<int>(session.termination()) << ", diagnostic="
              << (session.diagnostics().empty() ? "<none>" : session.diagnostics().front().token)
              << '\n';
  }
  return selected && completed && subscribers_closed &&
         session.termination() == holoscan::RunTermination::kCompleted &&
         session.diagnostics().empty();
}

// RAII wrapper for a forked child process. Tracks the PID and pipe file
// descriptors used for the ready/start handshake. The destructor kills and
// reaps any still-running child to prevent zombie leaks.
struct ChildProcess {
  ChildProcess() = default;
  ChildProcess(const ChildProcess&) = delete;
  ChildProcess& operator=(const ChildProcess&) = delete;

  ~ChildProcess() { terminate_and_reap(); }

  void close_parent_pipes() noexcept {
    if (ready_fd >= 0) {
      ::close(ready_fd);
      ready_fd = -1;
    }
    if (start_fd >= 0) {
      ::close(start_fd);
      start_fd = -1;
    }
  }

  void terminate_and_reap() noexcept {
    close_parent_pipes();
    if (pid <= 0) {
      return;
    }

    int status{};
    pid_t result{};
    do {
      result = ::waitpid(pid, &status, WNOHANG);
    } while (result < 0 && errno == EINTR);
    if (result == pid || (result < 0 && errno == ECHILD)) {
      pid = -1;
      return;
    }

    static_cast<void>(::kill(pid, SIGKILL));
    do {
      result = ::waitpid(pid, &status, 0);
    } while (result < 0 && errno == EINTR);
    pid = -1;
  }

  pid_t pid{-1};
  int ready_fd{-1};
  int start_fd{-1};
};

constexpr auto kChildIoTimeout = 30s;
constexpr auto kChildJoinTimeout = 5s;

void close_pipe(std::array<int, 2U>& pipe_fds) noexcept {
  for (int& fd : pipe_fds) {
    if (fd >= 0) {
      ::close(fd);
      fd = -1;
    }
  }
}

[[nodiscard]] bool wait_for_fd(int fd, short events,
                               std::chrono::steady_clock::time_point deadline) {
  while (true) {
    const auto now = std::chrono::steady_clock::now();
    if (now >= deadline) {
      return false;
    }
    const auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now);
    const auto bounded =
        std::min<std::int64_t>(remaining.count() + 1, std::numeric_limits<int>::max());
    pollfd descriptor{.fd = fd, .events = events, .revents = 0};
    const int result = ::poll(&descriptor, 1U, static_cast<int>(bounded));
    if (result > 0) {
      if ((descriptor.revents & events) != 0) {
        return true;
      }
      if ((descriptor.revents & (POLLERR | POLLHUP | POLLNVAL)) != 0) {
        return false;
      }
    } else if (result < 0 && errno != EINTR) {
      return false;
    }
  }
}

[[nodiscard]] bool write_byte(int fd, std::uint8_t value) {
  const auto deadline = std::chrono::steady_clock::now() + kChildIoTimeout;
  if (!wait_for_fd(fd, POLLOUT, deadline)) {
    return false;
  }
  while (true) {
    const ssize_t result = ::write(fd, &value, sizeof(value));
    if (result == static_cast<ssize_t>(sizeof(value))) {
      return true;
    }
    if (result < 0 && errno == EINTR && std::chrono::steady_clock::now() < deadline) {
      continue;
    }
    return false;
  }
}

[[nodiscard]] bool read_byte(int fd, std::uint8_t& value) {
  const auto deadline = std::chrono::steady_clock::now() + kChildIoTimeout;
  if (!wait_for_fd(fd, POLLIN, deadline)) {
    return false;
  }
  while (true) {
    const ssize_t result = ::read(fd, &value, sizeof(value));
    if (result == static_cast<ssize_t>(sizeof(value))) {
      return true;
    }
    if (result < 0 && errno == EINTR && std::chrono::steady_clock::now() < deadline) {
      continue;
    }
    return false;
  }
}

// Forks and execs the same binary with --child flags, passing endpoint names
// and pipe FDs as argv. Returns false if pipe creation or fork/exec fails.
[[nodiscard]] bool spawn_child(std::string_view executable, std::string_view catalog_path,
                               const MatrixNames& names, std::size_t cell_index,
                               ChildProcess& child) {
  std::array<int, 2U> ready_pipe{};
  std::array<int, 2U> start_pipe{};
  ready_pipe.fill(-1);
  start_pipe.fill(-1);
  if (::pipe(ready_pipe.data()) != 0) {
    return false;
  }
  if (::pipe(start_pipe.data()) != 0) {
    close_pipe(ready_pipe);
    return false;
  }
  const pid_t pid = ::fork();
  if (pid < 0) {
    close_pipe(ready_pipe);
    close_pipe(start_pipe);
    return false;
  }
  if (pid == 0) {
    ::close(ready_pipe[0]);
    ::close(start_pipe[1]);
    const std::string ready = std::to_string(ready_pipe[1]);
    const std::string start = std::to_string(start_pipe[0]);
    const std::string selected_cell = std::to_string(cell_index);
    ::execl(std::string(executable).c_str(),
            std::string(executable).c_str(),
            "--child",
            std::string(catalog_path).c_str(),
            ready.c_str(),
            start.c_str(),
            names.pod_import.c_str(),
            names.pod_export.c_str(),
            names.schema_import.c_str(),
            names.schema_export.c_str(),
            names.tensor_import.c_str(),
            names.tensor_export.c_str(),
            selected_cell.c_str(),
            static_cast<char*>(nullptr));
    _exit(127);
  }
  ::close(ready_pipe[1]);
  ::close(start_pipe[0]);
  child.pid = pid;
  child.ready_fd = ready_pipe[0];
  child.start_fd = start_pipe[1];
  return true;
}

// Waits up to kChildJoinTimeout for the child to exit normally and returns its
// exit status. Kills and reaps the child if the deadline expires.
[[nodiscard]] int join_child(ChildProcess& child) {
  child.close_parent_pipes();
  const auto deadline = std::chrono::steady_clock::now() + kChildJoinTimeout;
  int status{};
  while (std::chrono::steady_clock::now() < deadline) {
    const pid_t result = ::waitpid(child.pid, &status, WNOHANG);
    if (result == child.pid) {
      child.pid = -1;
      return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
    }
    if (result < 0 && errno != EINTR) {
      child.pid = -1;
      return -1;
    }
    std::this_thread::sleep_for(2ms);
  }
  child.terminate_and_reap();
  return -1;
}

// Runs one matrix cell where the external client executes in a separate
// process. The catalog is serialized to a temp file; the child loads it
// independently via load_json. A two-phase pipe handshake synchronizes
// startup: parent waits for child "ready" → parent starts Runtime →
// parent signals child "start" → child runs clients → child reports result.
[[nodiscard]] bool run_real_child(const MatrixGraph& matrix, std::string_view executable,
                                  MatrixCell cell, std::size_t cell_index) {
  MatrixProbe<MatrixPod>::reset();
  MatrixProbe<holoscan::schema::StringT>::reset();
  MatrixProbe<holoscan::Tensor>::reset();
  MatrixRuntimeStartupProbe::reset();
  const auto path =
      std::filesystem::temp_directory_path() / (matrix.names.pod_import + "-catalog.json");
  const auto saved = matrix.plan.endpoint_catalog().save_json(path);
  if (!saved) {
    return false;
  }
  const auto reloaded = holoscan::ChannelEndpointCatalog::load_json(path);
  if (!reloaded || reloaded->json() != matrix.plan.endpoint_catalog().json()) {
    std::filesystem::remove(path);
    return false;
  }
  ChildProcess child;
  if (!spawn_child(executable, path.string(), matrix.names, cell_index, child)) {
    std::filesystem::remove(path);
    return false;
  }
  std::uint8_t ready{};
  if (!read_byte(child.ready_fd, ready) || ready != 1U) {
    static_cast<void>(join_child(child));
    std::filesystem::remove(path);
    return false;
  }
  holoscan::RunSession session = holoscan::run_async(matrix.plan);
  const bool runtime_started = MatrixRuntimeStartupProbe::wait_for_all();
  const bool released = runtime_started && write_byte(child.start_fd, 1U);
  std::uint8_t result{};
  const bool reported = released && read_byte(child.ready_fd, result);
  const bool selected =
      cell.direction == MatrixDirection::kExport || validate_graph_import(matrix, cell.payload);
  const bool completed = reported && result == 0U && wait_for_session_terminal(session);
  if (!completed && session.is_running()) {
    session.request_stop();
  }
  session.wait();
  const bool cleanup_released = reported && result == 0U && write_byte(child.start_fd, 2U);
  const int child_exit = join_child(child);
  std::filesystem::remove(path);
  return reported && result == 0U && selected && cleanup_released && child_exit == 0 && completed &&
         session.termination() == holoscan::RunTermination::kCompleted &&
         session.diagnostics().empty();
}

// Entry point executed by the forked child. Loads the catalog from argv[2],
// waits for the parent's start signal, runs the client-side of the matrix
// cell, and reports success (0) or failure (1) back through the ready pipe.
[[nodiscard]] int run_child(int argc, char** argv) {
  if (argc != 12) {
    return 20;  // exit 20: child requires exactly 11 positional arguments after the binary name
  }
  auto catalog = holoscan::ChannelEndpointCatalog::load_json(argv[2]);
  if (!catalog) {
    return 21;  // exit 21: catalog JSON could not be loaded from the path in argv[2]
  }
  const int ready_fd = std::stoi(argv[3]);
  const int start_fd = std::stoi(argv[4]);
  const MatrixNames names{
      .pod_import = argv[5],
      .pod_export = argv[6],
      .schema_import = argv[7],
      .schema_export = argv[8],
      .tensor_import = argv[9],
      .tensor_export = argv[10],
  };
  char* cell_end = nullptr;
  const unsigned long parsed_cell = std::strtoul(argv[11], &cell_end, 10);
  if (cell_end == argv[11] || *cell_end != '\0' || parsed_cell >= kMatrixCells.size()) {
    return 25;  // exit 25: cell index in argv[11] is out of range or not a valid decimal integer
  }
  const MatrixCell cell = kMatrixCells[static_cast<std::size_t>(parsed_cell)];
  if (cell.locality != MatrixLocality::kRealChild) {
    return 26;  // exit 26: selected cell is not kRealChild — child invoked with wrong index
  }
  if (!write_byte(ready_fd, 1U)) {
    return 22;  // exit 22: could not signal readiness to parent over the ready pipe
  }
  std::uint8_t start{};
  if (!read_byte(start_fd, start) || start != 1U) {
    return 23;  // exit 23: timed out or received unexpected value waiting for parent start signal
  }
  auto result = run_clients(*catalog, names);
  const bool selected = result && (cell.direction == MatrixDirection::kImport ||
                                   validate_graph_exports(cell.payload, *result));
  const bool reported = write_byte(ready_fd, selected ? 0U : 1U);
  if (!selected || !reported) {
    return 24;  // exit 24: client round-trip or result report failed
  }
  std::uint8_t cleanup{};
  if (!read_byte(start_fd, cleanup) || cleanup != 2U) {
    return 27;  // exit 27: parent did not authorize local close after session completion
  }
  return close_client_subscribers(*result)
             ? 0
             : 28;  // exit 28: explicit local subscriber cleanup failed
}

// Builds a fresh MatrixGraph, validates its catalog, and runs the selected
// cell in the appropriate locality mode. Each cell is self-contained: it gets
// its own graph with unique endpoint names to avoid cross-cell interference.
[[nodiscard]] int run_matrix_cell(std::size_t index, std::string_view executable) {
  if (index >= kMatrixCells.size()) {
    return 1;  // exit 1: cell index out of range of kMatrixCells
  }
  const MatrixCell cell = kMatrixCells[index];
  const MatrixGraph matrix = make_matrix_graph();
  if (!validate_catalog(matrix)) {
    std::cerr << "baseline external-client matrix catalog failed for cell " << index << '\n';
    return 2;  // exit 2: compiled plan catalog failed endpoint or direction validation
  }
  const bool passed = cell.locality == MatrixLocality::kSameProcess
                          ? run_same_process(matrix, cell)
                          : run_real_child(matrix, executable, cell, index);
  if (!passed) {
    std::cerr << "baseline external-client matrix failed for cell " << index << '\n';
    return 3;  // exit 3: cell execution failed (client, import probe, or export probe)
  }
  return 0;
}

}  // namespace

// REQ: SYSTEM-001
// REQ: SYSTEM-002
// REQ: SYSTEM-003
// REQ: SYSTEM-006
int main(int argc, char** argv) {
  // Pipe peers are independently failing processes. Convert peer exit into EPIPE
  // so the parent can execute its bounded kill/reap cleanup instead of dying.
  if (::signal(SIGPIPE, SIG_IGN) == SIG_ERR) {
    return 1;  // exit 1: SIGPIPE disposition is required for pipe-based child coordination
  }
  if (argc > 1 && std::string_view(argv[1]) == "--child") {
    return run_child(argc, argv);
  }
  if (argc == 3 && std::string_view(argv[1]) == "--cell") {
    char* end = nullptr;
    const unsigned long parsed = std::strtoul(argv[2], &end, 10);
    if (end == argv[2] || *end != '\0') {
      return 1;  // exit 1: --cell argument is not a valid decimal integer
    }
    return run_matrix_cell(static_cast<std::size_t>(parsed), argv[0]);
  }
  if (argc != 1) {
    return 1;  // exit 1: unexpected positional arguments with no recognized mode flag
  }
  for (std::size_t index = 0U; index < kMatrixCells.size(); ++index) {
    const int result = run_matrix_cell(index, argv[0]);
    if (result != 0) {
      return result;  // exit N: first failing cell propagates its exit code
    }
  }
  std::cout << "baseline external-client 12-cell matrix passed\n";
  return 0;
}
