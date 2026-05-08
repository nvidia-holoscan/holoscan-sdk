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

#include "CudaDataWriterListener.hpp"

namespace holoscan {
namespace ipc {
namespace dds_example {

CudaDataWriterListener::CudaDataWriterListener() {}

CudaDataWriterListener::~CudaDataWriterListener() {}

void CudaDataWriterListener::on_offered_deadline_missed(
    eprosima::fastdds::dds::DataWriter* writer,
    const eprosima::fastdds::dds::OfferedDeadlineMissedStatus& status) {
  static_cast<void>(writer);
  static_cast<void>(status);
  // Default: no action needed for CUDA IPC
}

void CudaDataWriterListener::on_offered_incompatible_qos(
    eprosima::fastdds::dds::DataWriter* writer,
    const eprosima::fastdds::dds::OfferedIncompatibleQosStatus& status) {
  static_cast<void>(writer);
  static_cast<void>(status);
  // Default: no action needed for CUDA IPC
}

void CudaDataWriterListener::on_liveliness_lost(
    eprosima::fastdds::dds::DataWriter* writer,
    const eprosima::fastdds::dds::LivelinessLostStatus& status) {
  static_cast<void>(writer);
  static_cast<void>(status);
  // Default: no action needed for CUDA IPC
}

void CudaDataWriterListener::on_unacknowledged_sample_removed(
    eprosima::fastdds::dds::DataWriter* writer,
    const eprosima::fastdds::rtps::InstanceHandle_t& instance) {
  static_cast<void>(writer);
  static_cast<void>(instance);
  // Sample lifetime for this demo is managed in application code; no extra cleanup here.
}

}  // namespace dds_example
}  // namespace ipc
}  // namespace holoscan
