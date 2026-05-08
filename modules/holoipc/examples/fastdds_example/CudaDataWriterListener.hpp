/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use it except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef HOLOIPC_EXAMPLES_FASTDDS_EXAMPLE_CUDA_DATA_WRITER_LISTENER_HPP
#define HOLOIPC_EXAMPLES_FASTDDS_EXAMPLE_CUDA_DATA_WRITER_LISTENER_HPP

#include <fastdds/dds/core/status/BaseStatus.hpp>
#include <fastdds/dds/core/status/DeadlineMissedStatus.hpp>
#include <fastdds/dds/core/status/IncompatibleQosStatus.hpp>
#include <fastdds/dds/publisher/DataWriterListener.hpp>
#include <fastdds/rtps/common/InstanceHandle.hpp>

namespace holoscan {
namespace ipc {
namespace dds_example {

/**
 * @brief DataWriterListener overrides for the example's sample DataWriter.
 *
 * Pointer acquire/release is handled by holoscan::ipc in the application; these callbacks
 * are no-ops aside from satisfying the Fast DDS listener interface.
 */
class CudaDataWriterListener : public eprosima::fastdds::dds::DataWriterListener {
 public:
  /**
   * @brief Constructor
   */
  CudaDataWriterListener();

  /**
   * @brief Destructor
   */
  virtual ~CudaDataWriterListener();

  /**
   * @brief Called when the DataWriter fails to meet a deadline
   * @param writer The DataWriter
   * @param status Deadline missed status information
   */
  void on_offered_deadline_missed(
      eprosima::fastdds::dds::DataWriter* writer,
      const eprosima::fastdds::dds::OfferedDeadlineMissedStatus& status) override;

  /**
   * @brief Called when a QoS incompatibility is detected
   * @param writer The DataWriter
   * @param status Incompatible QoS status information
   */
  void on_offered_incompatible_qos(
      eprosima::fastdds::dds::DataWriter* writer,
      const eprosima::fastdds::dds::OfferedIncompatibleQosStatus& status) override;

  /**
   * @brief Called when liveliness is lost
   * @param writer The DataWriter
   * @param status Liveliness lost status information
   */
  void on_liveliness_lost(eprosima::fastdds::dds::DataWriter* writer,
                          const eprosima::fastdds::dds::LivelinessLostStatus& status) override;

  /**
   * @brief Called when a sample is removed from the writer's history before being acknowledged.
   * @param writer The DataWriter
   * @param instance Instance handle of the removed sample
   */
  void on_unacknowledged_sample_removed(
      eprosima::fastdds::dds::DataWriter* writer,
      const eprosima::fastdds::rtps::InstanceHandle_t& instance) override;
};

}  // namespace dds_example
}  // namespace ipc
}  // namespace holoscan

#endif  // HOLOIPC_EXAMPLES_FASTDDS_EXAMPLE_CUDA_DATA_WRITER_LISTENER_HPP
