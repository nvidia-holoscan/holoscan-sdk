/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
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

#ifndef HOLOSCAN_IPC_DETAIL_FASTDDS_TRANSPORT_HPP
#define HOLOSCAN_IPC_DETAIL_FASTDDS_TRANSPORT_HPP

#include <memory>
#include <string>

#include <fastdds/dds/domain/DomainParticipant.hpp>
#include <fastdds/dds/topic/TypeSupport.hpp>

#include <holoscan/ipc/detail/transport_interface.hpp>
#include <holoscan/ipc/transport/fastdds/control_channel.hpp>
#include <holoscan/ipc/transport/fastdds/gen/pointer_descriptor.hpp>

namespace holoscan {
namespace ipc {
namespace transport {
namespace fastdds {

class FastDdsTransport : public TransportInterface {
 public:
  /** Wire-format descriptor type for this transport (DDS gen::PointerDescriptor). */
  using PointerDescriptorType = gen::PointerDescriptor;

  /** Wire-format handle type for this transport (DDS gen::HandleType). */
  using HandleType = gen::HandleType;

  explicit FastDdsTransport(::eprosima::fastdds::dds::DomainParticipant* participant);

  ~FastDdsTransport() override = default;

  void start(TransportInterface::Callback* callback) override;
  void stop() override;

  const std::string& get_inbox_topic_name() const override;
  std::shared_ptr<TransportInterface::ControlWriter> create_control_writer(
      const std::string& topic_name) override;
  PointerDescriptor from_wire_descriptor(const std::any& wire_descriptor) override;
  std::any to_wire_descriptor(const PointerDescriptor& descriptor) override;

 private:
  TransportInterface::Callback* callback_ = nullptr;
  ::eprosima::fastdds::dds::DomainParticipant* participant_ = nullptr;
  ::eprosima::fastdds::dds::TypeSupport type_support_;

  // Inbox topic name; used in descriptors and control messages so the peer knows where to send.
  std::string inbox_topic_name_;
  // Created in start(); torn down in stop().
  std::unique_ptr<ControlReader> control_reader_;
};

}  // namespace fastdds
}  // namespace transport
}  // namespace ipc
}  // namespace holoscan

#endif  // HOLOSCAN_IPC_DETAIL_FASTDDS_TRANSPORT_HPP
