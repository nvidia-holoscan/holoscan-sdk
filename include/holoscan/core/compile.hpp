/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <holoscan/holoscan_core_export.h>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <holoscan/channel/endpoint_catalog.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_plan.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/core/rate.hpp>
#include <holoscan/threading/thread_placement.hpp>

namespace holoscan {

/// @brief Describe one exact generated endpoint identity admitted by deployment policy.
///
/// The binding copies all views before returning. A pin grants no wildcard,
/// prefix, current-version, or future-version authority.
struct GeneratedEndpointNamePin {
  std::string_view final_name;  ///< Exact final client lookup identity.
  std::string_view algorithm{"holoscan.auto_endpoint_name"};  ///< Exact algorithm id.
  std::uint32_t version{1};  ///< Exact observable algorithm version.
};

/// @brief Own one exact manual execution-partition placement binding.
///
/// The partition name resolves only against a root-Graph manual partition
/// declaration. Compilation rejects unknown names and duplicate placement
/// authority rather than guessing a partition or silently selecting one
/// request.
struct PartitionThreadPlacementBinding {
  std::string partition_name;   ///< Exact authored partition name.
  ThreadPlacement placement{};  ///< Requested affinity and scheduling policy.
};

/// @brief Own one CPU-affinity request for a Runtime support-thread role.
struct SupportThreadAffinityBinding {
  SupportThreadRole role{SupportThreadRole::kController};  ///< Exact closed Runtime role.
  ThreadPlacement placement{};  ///< Canonical internal affinity-only placement request.
};

/// @brief Deep-own one manual-partition Green Context placement request.
struct PartitionCudaGreenContextBinding {
  std::string partition_name;             ///< Exact owned manual partition name.
  CudaGreenContextPlacement placement{};  ///< Exact target-specific CUDA placement.
};

/// @brief Report read-only CUDA Green Context partition capability for one device.
///
/// This value is discovery evidence only. It neither reserves SMs nor authorizes a deployment.
struct CudaGreenContextCapability {
  DeviceId device;                           ///< Queried process-local CUDA ordinal.
  std::string deployment_device_id;          ///< Stable UUID/PCI deployment identity.
  std::uint32_t total_sm_count{};            ///< Total device SM resource count.
  std::uint32_t min_sm_partition_size{};     ///< Driver minimum structured group size.
  std::uint32_t sm_coscheduled_alignment{};  ///< Driver co-scheduling alignment.
  std::uint32_t structured_sm_quantum{};     ///< LCM used for exact group admission.
  /// Maximum driver-qualified minimum-size group count; may be below total_sm_count / quantum.
  std::uint32_t max_structured_context_count{};

  /// @brief Compare all target capability evidence.
  friend bool operator==(const CudaGreenContextCapability& lhs,
                         const CudaGreenContextCapability& rhs) noexcept = default;
};

/// @brief Query CUDA Green Context capability without reserving resources.
/// @param device Explicit nonnegative process-local CUDA ordinal; unresolved/negative values fail.
/// @return Driver-reported capability, or a typed argument/driver/toolkit error.
/// @note The result is target-specific discovery evidence, not deployment authority or a resource
/// reservation. Exact placement remains restricted to named manual execution partitions.
[[nodiscard]] HOLOSCAN_CORE_EXPORT expected<CudaGreenContextCapability, Error>
query_cuda_green_context_capability(DeviceId device);

/// @brief Own one exact generated endpoint-name admission pin.
struct GeneratedEndpointNamePinRecord {
  std::string final_name;   ///< Exact admitted client lookup identity.
  std::string algorithm;    ///< Exact generation algorithm identifier.
  std::uint32_t version{};  ///< Exact observable algorithm version.
};

/// @brief Place one exact Tensor or Tensor-bearing schema output on a CUDA device.
///
/// The operator path uses the canonical segment-encoded, no-leading-slash
/// OperatorPath grammar (for example, `source` at the root or
/// `pipeline/source` in an embedded Graph). The port name identifies one operator-owned
/// `Output<Tensor>` or an owning generated-schema output whose payload kind is
/// descriptor-backed. Compilation rejects unknown, unrelated payloads,
/// duplicates, or conflicting placements rather than selecting device zero or
/// inserting an implicit cross-device transfer.
struct TensorOutputDevicePlacement {
  std::string_view operator_path;  ///< Exact canonical Operator path.
  std::string_view output_port;    ///< Exact local Tensor-bearing output name.
  DeviceId device;                 ///< Explicit nonnegative CUDA device ordinal.
};

/// @brief Deep-own one Tensor output device-placement request.
struct TensorOutputDevicePlacementRecord {
  std::string operator_path;  ///< Owned canonical Operator path.
  std::string output_port;    ///< Owned local Tensor-bearing output name.
  DeviceId device;            ///< Requested CUDA device ordinal.
};

/// @brief Own cold deployment bindings supplied to Graph compilation.
///
/// The value deep-copies every string and placement. It contains no live
/// scheduler, process, thread, transport, or backend handle. Compilation
/// validates partition names, placement shape, and duplicates, then performs only
/// read-only host capability probes. Probe results and any profile-permitted
/// fallback become immutable plan evidence; compilation never applies affinity
/// or scheduling state. Runtime consumes the accepted request and evidence when
/// applying placement to executing threads.
class HOLOSCAN_CORE_EXPORT DeploymentBinding {
 public:
  /// @brief Bind one manual execution partition to an exact placement request.
  /// @param partition_name Exact authored partition name copied before return.
  /// @param placement Placement request copied before return.
  /// @return This binding for fluent cold-path authoring.
  /// @throws std::bad_alloc if owned storage allocation fails.
  /// @note Compile rejects unknown names and duplicate authority with
  /// `PARTITION_PLACEMENT_UNKNOWN_OR_DUPLICATE`.
  DeploymentBinding& bind_thread_placement(std::string_view partition_name,
                                           ThreadPlacement placement);

  /// @brief Bind one Runtime-owned support-thread role to an allowed CPU set.
  /// @param role Controller or optional process-local Channel pump.
  /// @param affinity Nonempty CPU set copied before return.
  /// @return This binding for fluent cold-path authoring.
  /// @throws std::bad_alloc if owned storage allocation fails.
  /// @note Compilation rejects unknown or duplicate roles and affinity requests that cannot be
  /// proven against the compiling thread's effective CPU set. This method never changes affinity.
  /// A controller binding also constrains inline Operator callbacks hosted by that controller;
  /// separately worker-backed execution lanes without explicit partition affinity retain the
  /// controller thread's pre-binding allowed CPU set.
  DeploymentBinding& bind_support_thread_affinity(SupportThreadRole role, CPUAffinity affinity);

  /// @brief Bind one manual execution partition to an exact CUDA Green Context placement.
  /// @param partition_name Exact authored manual partition name copied before return.
  /// @param placement Explicit target device and positive exact structured SM count.
  /// @return This binding for fluent cold-path authoring.
  /// @throws std::bad_alloc if owned storage allocation fails.
  /// @note Invalid devices, zero counts, unknown names, and duplicate authority are preserved for
  /// stable compile diagnostics. This call does not reserve CUDA resources.
  DeploymentBinding& bind_cuda_green_context(std::string_view partition_name,
                                             CudaGreenContextPlacement placement);

  /// @brief Allow one exact versioned generated endpoint name.
  /// @param pin Exact final name, algorithm identifier, and version.
  /// @return This binding for fluent cold-path authoring.
  /// @throws std::bad_alloc if owned storage allocation fails.
  /// @note Compile rejects invalid or conflicting duplicates; this call grants
  /// no wildcard or future-version authority.
  DeploymentBinding& allow_generated_endpoint_name(GeneratedEndpointNamePin pin);

  /// @brief Bind one exact Tensor output to a CUDA device.
  /// @param placement Stable output selector and explicit device ordinal;
  /// strings are copied before return.
  /// @return This binding for fluent cold-path authoring.
  /// @throws std::bad_alloc if owned storage allocation fails.
  /// @note This is deployment placement, not Tensor shape or allocation
  /// authorship. One Operator currently has one compute stream and therefore
  /// must resolve to at most one device across all of its Tensor obligations.
  DeploymentBinding& bind_tensor_output_device(TensorOutputDevicePlacement placement);

  /// @brief Return owned execution-partition placements in insertion order.
  /// @return Immutable bindings valid until this value is mutated or destroyed.
  [[nodiscard]] std::span<const PartitionThreadPlacementBinding> thread_placements()
      const noexcept {
    return thread_placements_;
  }

  /// @brief Return owned support-thread affinity bindings in insertion order.
  /// @return Immutable bindings valid until this value is mutated or destroyed.
  [[nodiscard]] std::span<const SupportThreadAffinityBinding> support_thread_affinities()
      const noexcept {
    return support_thread_affinities_;
  }

  /// @brief Return owned manual-partition Green Context placements in insertion order.
  /// @return Immutable bindings valid until this value is mutated or destroyed.
  [[nodiscard]] std::span<const PartitionCudaGreenContextBinding> cuda_green_contexts()
      const noexcept {
    return cuda_green_contexts_;
  }

  /// @brief Return owned generated endpoint pins in insertion order.
  /// @return Immutable pins valid until this value is mutated or destroyed.
  [[nodiscard]] std::span<const GeneratedEndpointNamePinRecord> generated_endpoint_names()
      const noexcept {
    return generated_endpoint_names_;
  }

  /// @brief Return owned Tensor output device placements in insertion order.
  /// @return Immutable placements valid until this value is mutated or destroyed.
  [[nodiscard]] std::span<const TensorOutputDevicePlacementRecord> tensor_output_devices()
      const noexcept {
    return tensor_output_devices_;
  }

 private:
  std::vector<PartitionThreadPlacementBinding> thread_placements_;
  std::vector<SupportThreadAffinityBinding> support_thread_affinities_;
  std::vector<PartitionCudaGreenContextBinding> cuda_green_contexts_;
  std::vector<GeneratedEndpointNamePinRecord> generated_endpoint_names_;
  std::vector<TensorOutputDevicePlacementRecord> tensor_output_devices_;
};

/// @brief Select the admitted storage strategy for Parameter generations.
enum class ParamStorageMode : std::uint8_t {
  kProfileDefault = 0,       ///< Resolve storage from profile and schema evidence.
  kDynamicColdPath = 1,      ///< Permit cold-path dynamic construction only.
  kFreezeBeforeArm = 2,      ///< Materialize all values before Runtime arming.
  kPreallocatedBounded = 3,  ///< Require finite preallocated generation storage.
  kInlineOnly = 4,           ///< Require complete inline bounded representation.
};

/// @brief Select behavior when admitted Parameter update storage is exhausted.
enum class ParamExhaustionPolicy : std::uint8_t {
  kRejectUpdate = 0,  ///< Reject the new update without changing the active generation.
};

/// @brief Configure Parameter lowering and bounded runtime update capacity.
struct ParameterCompileOptions {
  ParamStorageMode storage{ParamStorageMode::kProfileDefault};  ///< Requested storage strategy.
  ParamRuntimeUpdatePolicy runtime_updates{
      ParamRuntimeUpdatePolicy::kProfileDefault};     ///< Requested active-update policy.
  std::uint32_t generation_slots_per_operator{};      ///< Zero requests compiler derivation.
  std::uint32_t max_pending_updates_per_operator{4};  ///< Accepted pending-update bound.
  std::uint32_t audit_capacity{};      ///< Zero requests the selected profile default.
  std::uint32_t max_event_cursors{4};  ///< Independent retained Parameter observer bound.
  std::optional<std::size_t> max_parameter_arena_bytes;  ///< Optional hard arena ceiling.
  ParamExhaustionPolicy exhaustion{ParamExhaustionPolicy::kRejectUpdate};  ///< Full behavior.
  bool require_proven_schema_completeness{};  ///< Reject opaque or incomplete schemas.
  bool require_explicit_field_policies{};     ///< Reject fields using inferred policy.
};

/// @brief Opt one compile into bounded active lifecycle transactions.
///
/// Absence from \ref CompileOptions disables lifecycle-mutable-parameter,
/// external-control, and compute-origin active transactions. When present,
/// every capacity must be positive and at least the compiler-derived minimum.
struct LifecycleTransactionOptions {
  std::uint32_t max_pending_requests{};       ///< Accepted submissions before terminalization.
  std::uint32_t max_pending_transactions{};   ///< Active plus successor transaction bound.
  std::uint32_t max_receipts{};               ///< Retained per-submission receipt bound.
  std::uint32_t max_fenced_operators{};       ///< Maximum union-closure membership.
  std::uint32_t max_candidate_generations{};  ///< Private complete candidate bound.
  std::uint32_t max_events{};                 ///< Structural event retention capacity.
  std::uint32_t max_audit_records{};          ///< Required bounded audit capacity.
  std::uint32_t max_quiesce_records{};        ///< Quiesce-obligation journal bound.
  std::uint32_t max_rollback_records{};       ///< Completed-work rollback journal bound.
};

/// @brief Select deterministic compile-wide advisory queue-sizing headroom.
///
/// The exact multiplier applies only to the soft nominal-arrival recommendation. It is not a
/// burst, jitter, maximum-arrival, memory-safety, or scheduling guarantee.
struct AutomaticQueueSizingOptions {
  RateRatio headroom_ratio{2U, 1U};  ///< Exact multiplier, required to be at least one.
};

/// @brief Configure immutable CUDA execution-resource lowering.
struct CudaExecutionCompileOptions {
  /// Ordinary compute-stream sharing policy. By default, each active execution partition gets a
  /// device-local stream; select kSharedProcessDevice to minimize the number of ordinary streams.
  CudaComputeStreamPolicy compute_stream_policy{CudaComputeStreamPolicy::kPerExecutionPartition};
};

/// @brief Select compile admission for process-local opaque object payload edges.
enum class OpaqueObjectPayloadPolicy : std::uint8_t {
  kAllow = 0,   ///< Admit opaque object edges without a diagnostic.
  kWarn = 1,    ///< Admit opaque object edges with one advisory per edge.
  kReject = 2,  ///< Reject opaque object edges with one error per edge.
};

/// @brief Own all caller-supplied inputs to one Graph compilation.
///
/// Compilation snapshots this value and the Graph. No field supplies a live
/// runtime object or mutable policy service, and no environment variable may
/// alter fields other than the separately documented private partition seam.
struct CompileOptions {
  DeploymentBinding deployment;         ///< Owned partition, endpoint, and Tensor-device bindings.
  Profile profile{Profile::kStandard};  ///< Closed requested root profile.
  bool auto_insert_converters{true};    ///< False rejects until converter lowering is available.
  std::optional<ChannelEndpointCatalog> endpoint_catalog;  ///< Optional immutable authority.
  ParameterCompileOptions parameters;                      ///< Parameter storage/update policy.
  std::optional<LifecycleTransactionOptions> lifecycle_transactions;  ///< Active restart opt-in.
  AutomaticQueueSizingOptions automatic_queue_sizing;  ///< Advisory rate-derived depth policy.
  CudaExecutionCompileOptions cuda_execution;          ///< CUDA compute-stream policy.
  /// Admission policy for the fixed process-local Python object payload identity.
  OpaqueObjectPayloadPolicy opaque_object_payloads{OpaqueObjectPayloadPolicy::kAllow};
};

/// @brief Select the deployment artifact destination for explicit publication.
struct ArtifactEmitOptions {
  std::filesystem::path root;  ///< Explicit root, or empty for documented precedence.
  /// Portable single-component deployment identity.
  ///
  /// The value contains 1 through 128 ASCII characters, begins with an
  /// alphanumeric character, and otherwise contains only alphanumeric
  /// characters, period, hyphen, or underscore. It must not end in a period
  /// or use a Windows reserved-device basename. These restrictions keep the
  /// emitted bundle below the selected artifact root on every supported filesystem.
  std::string deployment_id{"default"};
};

/// @brief Report the files and content identity published for one execution plan.
struct ArtifactEmitResult {
  bool changed{};                       ///< Whether any official artifact content changed.
  std::filesystem::path catalog_path;   ///< Final endpoint-catalog path.
  std::filesystem::path manifest_path;  ///< Final commit-marker manifest path.
  std::string catalog_digest;           ///< Canonical `sha256:` catalog digest.
};

/// @brief Report an attempt to execute or require a rejected compile result.
class HOLOSCAN_CORE_EXPORT CompileError : public RuntimeError {
 public:
  /// @brief Construct a generic compile error from bounded text.
  /// @param what_arg Error text copied into the exception.
  explicit CompileError(std::string_view what_arg) : RuntimeError(ErrorCode::kFailure, what_arg) {}

  /// @brief Construct a compile error from a typed Core error.
  /// @param error Error category and bounded detail to preserve.
  explicit CompileError(const Error& error) : RuntimeError(error) {}
};

/// @brief Compile one Graph generation into an immutable inspectable plan.
/// @param graph Editable Graph whose composed generation is snapshotted.
/// @param options Owned profile, deployment, catalog, and capacity requests.
/// @return Accepted or rejected immutable plan with structural diagnostics.
/// @throws RuntimeError only when Graph composition itself cannot produce a
/// coherent snapshot, executable factory reconstruction cannot be retained, or
/// a process resource prevents construction of any plan result.
/// @note Admission failures are represented by a rejected \ref ExecutionPlan;
/// they do not throw and never mutate a previously returned plan.
[[nodiscard]] HOLOSCAN_CORE_EXPORT ExecutionPlan compile(Graph& graph, CompileOptions options = {});

/// @brief Publish one accepted plan's canonical deployment artifacts.
/// @param plan Immutable accepted plan; rejected plans are not publishable.
/// @param options Explicit root and deployment identity.
/// @return Publication paths, digest, and whether canonical content changed.
/// @retval ErrorCode::kInvalidArgument if the plan or deployment identity is invalid.
/// @retval ErrorCode::kPermissionDenied if generated endpoint names lack exact artifact authority.
/// @retval ErrorCode::kFailure if filesystem preparation, verification, or atomic replacement
/// fails.
/// @note This cold operation never starts execution or constructs transport endpoints. It writes
/// changed canonical files through verified temporary siblings and publishes the manifest last.
/// Existing destinations and staged files are read only through fixed-size comparisons bounded by
/// their expected canonical sizes; the manifest comparison uses its fixed schema-v1 maximum and
/// validates the update timestamp as one canonical signed 64-bit decimal token.
[[nodiscard]] HOLOSCAN_CORE_EXPORT expected<ArtifactEmitResult, Error> emit_artifacts(
    const ExecutionPlan& plan, ArtifactEmitOptions options = {});

}  // namespace holoscan
