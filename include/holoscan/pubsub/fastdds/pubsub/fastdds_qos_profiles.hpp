/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOSCAN_PUBSUB_FASTDDS_PUBSUB_FASTDDS_QOS_PROFILES_HPP
#define HOLOSCAN_PUBSUB_FASTDDS_PUBSUB_FASTDDS_QOS_PROFILES_HPP

#include <algorithm>
#include <cstdint>
#include <limits>

#include <fastdds/dds/core/Time_t.hpp>
#include <fastdds/dds/publisher/qos/DataWriterQos.hpp>
#include <fastdds/dds/subscriber/qos/DataReaderQos.hpp>

#include <gxf/pubsub/qos_profile.hpp>

namespace holoscan {
namespace dds_qos {

/**
 * @brief Map a GXF QoSProfile struct to FastDDS DataWriterQos.
 *
 * Maps all QoSProfile fields to their DDS equivalents:
 * - reliability, durability, history, depth: core QoS policies
 * - priority: DDS TransportPriorityQosPolicy
 * - max_blocking_time_ns: DDS ReliabilityQosPolicy.max_blocking_time
 * - deadline_ns: DDS DeadlineQosPolicy.period
 * - lifespan_ns: DDS LifespanQosPolicy.duration
 *
 * DDS-specific resource_limits tuning is derived from history policy/depth
 * (no GXF-level equivalent).
 *
 * @param dds_qos The DataWriterQos to configure (modified in-place). Must be
 *        freshly default-initialized (e.g. ``DATAWRITER_QOS_DEFAULT``).
 *        Reusing a previously configured object may leave stale values for
 *        fields that this mapping does not set (deadline, lifespan,
 *        resource_limits).
 * @param qos The GXF QoSProfile to map from.
 *
 * @see docs/PUBSUB_DESIGN/DDS_BACKEND_ADOPT_NEW_GXF_APIS.md §3
 */
inline void apply_writer_qos(eprosima::fastdds::dds::DataWriterQos& dds_qos,
                             const nvidia::gxf::QoSProfile& qos) {
  using namespace eprosima::fastdds::dds;

  // --- Core fields (direct mapping from QoSProfile struct) ---
  dds_qos.reliability().kind = (qos.reliability == nvidia::gxf::ReliabilityPolicy::kReliable)
                                   ? RELIABLE_RELIABILITY_QOS
                                   : BEST_EFFORT_RELIABILITY_QOS;

  dds_qos.durability().kind = (qos.durability == nvidia::gxf::DurabilityPolicy::kTransientLocal)
                                  ? TRANSIENT_LOCAL_DURABILITY_QOS
                                  : VOLATILE_DURABILITY_QOS;

  dds_qos.history().kind = (qos.history == nvidia::gxf::HistoryPolicy::kKeepAll)
                               ? KEEP_ALL_HISTORY_QOS
                               : KEEP_LAST_HISTORY_QOS;

  dds_qos.history().depth = static_cast<int32_t>(qos.depth);

  // --- Priority ---
  dds_qos.transport_priority().value = static_cast<uint32_t>(qos.priority);

  // --- Max blocking time ---
  // Direct mapping from QoSProfile (default 100ms).
  // 0 = non-blocking, std::numeric_limits<uint64_t>::max() = infinite.
  if (qos.max_blocking_time_ns == std::numeric_limits<uint64_t>::max()) {
    dds_qos.reliability().max_blocking_time = eprosima::fastdds::dds::c_TimeInfinite;
  } else {
    auto secs = static_cast<int32_t>(qos.max_blocking_time_ns / 1'000'000'000ULL);
    auto nsecs = static_cast<uint32_t>(qos.max_blocking_time_ns % 1'000'000'000ULL);
    dds_qos.reliability().max_blocking_time = {secs, nsecs};
  }

  // --- Deadline ---
  if (qos.deadline_ns > 0) {
    if (qos.deadline_ns == std::numeric_limits<uint64_t>::max()) {
      dds_qos.deadline().period = eprosima::fastdds::dds::c_TimeInfinite;
    } else {
      auto secs = static_cast<int32_t>(qos.deadline_ns / 1'000'000'000ULL);
      auto nsecs = static_cast<uint32_t>(qos.deadline_ns % 1'000'000'000ULL);
      dds_qos.deadline().period = {secs, nsecs};
    }
  }

  // --- Lifespan ---
  if (qos.lifespan_ns > 0) {
    if (qos.lifespan_ns == std::numeric_limits<uint64_t>::max()) {
      dds_qos.lifespan().duration = eprosima::fastdds::dds::c_TimeInfinite;
    } else {
      auto secs = static_cast<int32_t>(qos.lifespan_ns / 1'000'000'000ULL);
      auto nsecs = static_cast<uint32_t>(qos.lifespan_ns % 1'000'000'000ULL);
      dds_qos.lifespan().duration = {secs, nsecs};
    }
  }

  // --- DDS-specific tuning (derived from QoS fields, no GXF equivalent) ---
  if (qos.history == nvidia::gxf::HistoryPolicy::kKeepAll) {
    // Keep-all needs resource limits to avoid unbounded memory growth.
    // Use depth as the max_samples cap.
    dds_qos.resource_limits().max_samples = static_cast<int32_t>(qos.depth);
    dds_qos.resource_limits().allocated_samples = std::min(10, static_cast<int>(qos.depth));
  } else if (qos.depth <= 3) {
    // Small history depth — conservative resource limits to bound memory
    dds_qos.resource_limits().max_samples = static_cast<int32_t>(qos.depth) + 2;
    dds_qos.resource_limits().allocated_samples = static_cast<int32_t>(qos.depth);
  }
  // Larger keep-last depths: leave resource_limits at FastDDS defaults
}

/**
 * @brief Map a GXF QoSProfile struct to FastDDS DataReaderQos.
 *
 * Maps the same fields as apply_writer_qos(), except:
 * - lifespan is not applicable to readers (DDS spec)
 * - max_blocking_time is not applicable to readers (writer-side only)
 * - transport_priority is not applicable to readers
 *
 * @param dds_qos The DataReaderQos to configure (modified in-place). Must be
 *        freshly default-initialized (e.g. ``DATAREADER_QOS_DEFAULT``).
 *        Reusing a previously configured object may leave stale values.
 * @param qos The GXF QoSProfile to map from.
 *
 * @see docs/PUBSUB_DESIGN/DDS_BACKEND_ADOPT_NEW_GXF_APIS.md §3
 */
inline void apply_reader_qos(eprosima::fastdds::dds::DataReaderQos& dds_qos,
                             const nvidia::gxf::QoSProfile& qos) {
  using namespace eprosima::fastdds::dds;

  // --- Core fields ---
  dds_qos.reliability().kind = (qos.reliability == nvidia::gxf::ReliabilityPolicy::kReliable)
                                   ? RELIABLE_RELIABILITY_QOS
                                   : BEST_EFFORT_RELIABILITY_QOS;

  dds_qos.durability().kind = (qos.durability == nvidia::gxf::DurabilityPolicy::kTransientLocal)
                                  ? TRANSIENT_LOCAL_DURABILITY_QOS
                                  : VOLATILE_DURABILITY_QOS;

  dds_qos.history().kind = (qos.history == nvidia::gxf::HistoryPolicy::kKeepAll)
                               ? KEEP_ALL_HISTORY_QOS
                               : KEEP_LAST_HISTORY_QOS;

  dds_qos.history().depth = static_cast<int32_t>(qos.depth);

  // --- Deadline ---
  if (qos.deadline_ns > 0) {
    if (qos.deadline_ns == std::numeric_limits<uint64_t>::max()) {
      dds_qos.deadline().period = eprosima::fastdds::dds::c_TimeInfinite;
    } else {
      auto secs = static_cast<int32_t>(qos.deadline_ns / 1'000'000'000ULL);
      auto nsecs = static_cast<uint32_t>(qos.deadline_ns % 1'000'000'000ULL);
      dds_qos.deadline().period = {secs, nsecs};
    }
  }

  // --- DDS-specific tuning (same resource_limits logic as writer) ---
  if (qos.history == nvidia::gxf::HistoryPolicy::kKeepAll) {
    dds_qos.resource_limits().max_samples = static_cast<int32_t>(qos.depth);
    dds_qos.resource_limits().allocated_samples = std::min(10, static_cast<int>(qos.depth));
  } else if (qos.depth <= 3) {
    dds_qos.resource_limits().max_samples = static_cast<int32_t>(qos.depth) + 2;
    dds_qos.resource_limits().allocated_samples = static_cast<int32_t>(qos.depth);
  }
}

}  // namespace dds_qos
}  // namespace holoscan

#endif  // HOLOSCAN_PUBSUB_FASTDDS_PUBSUB_FASTDDS_QOS_PROFILES_HPP
