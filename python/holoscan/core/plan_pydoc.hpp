/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

namespace holoscan::doc::plan {

inline constexpr const char* kPlan =
    "An immutable, inspectable result of compiling one Graph generation.";
inline constexpr const char* kCompile =
    "Compile a Graph into a Plan. Admission failures are retained as diagnostics.";
inline constexpr const char* kDiagnostic = "A stable structured compile or runtime diagnostic.";
inline constexpr const char* kDiagnosticOperation =
    "The Core operation that produced supplemental diagnostic evidence.";
inline constexpr const char* kDiagnosticFailureRole =
    "The failure's relationship to its initiating lifecycle or runtime operation.";
inline constexpr const char* kDiagnosticSource =
    "A bounded plan-local location associated with a diagnostic.";
inline constexpr const char* kDiagnosticError =
    "A read-only error category and bounded explanatory message retained by a diagnostic.";
inline constexpr const char* kDiagnosticMismatch =
    "One read-only provided-versus-required field mismatch retained by a diagnostic.";
inline constexpr const char* kDiagnosticDetail =
    "Read-only supplemental failure evidence retained by a diagnostic.";
inline constexpr const char* kCompileOptions =
    "Caller-owned inputs that control one Graph compilation.";
inline constexpr const char* kDeviceId = "An explicit deployment-device ordinal.";
inline constexpr const char* kCPUAffinity =
    "A normalized CPU-core selection whose cores property is an immutable tuple snapshot.";
inline constexpr const char* kSupportThreadRole =
    "A Runtime-owned support-thread role that can receive CPU affinity.";
inline constexpr const char* kDeadlineScheduling =
    "Linux deadline-scheduling runtime, deadline, and period values in nanoseconds.";
inline constexpr const char* kThreadScheduling =
    "A thread scheduling policy and its policy-specific parameters.";
inline constexpr const char* kThreadPlacement =
    "CPU affinity, scheduling, wake, interrupt-isolation, and probe policy for one execution lane.";
inline constexpr const char* kCudaGreenContextPlacement =
    "A requested CUDA Green Context device and streaming-multiprocessor count.";
inline constexpr const char* kCudaGreenContextCapability =
    "Read-only CUDA Green Context partitioning evidence for one device.";
inline constexpr const char* kCudaExecutionCompileOptions =
    "Options controlling immutable CUDA execution-resource lowering.";
inline constexpr const char* kQueryCudaGreenContextCapability =
    "Query CUDA Green Context partitioning evidence for one explicit device.";
inline constexpr const char* kDeploymentBinding =
    "Caller-owned deployment bindings applied while compiling a Graph.";
inline constexpr const char* kBindSupportThreadAffinity =
    "Bind a Runtime support-thread role to a copied, nonempty CPU affinity request.";
inline constexpr const char* kRateRatio = "A positive rational rate or sizing ratio.";
inline constexpr const char* kAutomaticQueueSizingOptions =
    "Options controlling automatic logical queue sizing.";
inline constexpr const char* kLifecycleTransactionOptions =
    "Explicit bounded capacities for lifecycle parameter transactions.";
inline constexpr const char* kEndpointCatalog = "An immutable catalog of external Graph endpoints.";
inline constexpr const char* kSaveEndpointCatalog =
    "Save canonical catalog JSON to a filesystem path if its bytes changed.";
inline constexpr const char* kPartitions =
    "A retained range and lookup view over the accepted execution partitions.";
inline constexpr const char* kPartition = "An immutable view of one accepted execution partition.";
inline constexpr const char* kTensorByteSpanDerivation =
    "Immutable compiler evidence for one derived Tensor output-pool slot capacity.";
inline constexpr const char* kFindTensorByteSpanDerivation =
    "Find derived Tensor capacity evidence by authored operator and output names.";
inline constexpr const char* kTensorMemoryKindDerivation =
    "Immutable compiler evidence for one Parameter-derived Tensor output memory kind.";
inline constexpr const char* kFindTensorMemoryKindDerivation =
    "Find derived Tensor memory-kind evidence by authored operator and output names.";

}  // namespace holoscan::doc::plan
