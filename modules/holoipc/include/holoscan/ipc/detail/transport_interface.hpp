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

#ifndef HOLOSCAN_IPC_DETAIL_TRANSPORT_INTERFACE_HPP
#define HOLOSCAN_IPC_DETAIL_TRANSPORT_INTERFACE_HPP

#include <any>
#include <memory>
#include <string>
#include <vector>

#include <holoscan/ipc/detail/control_message.hpp>
#include <holoscan/ipc/detail/key.hpp>
#include <holoscan/ipc/detail/pointer_descriptor.hpp>

namespace holoscan {
namespace ipc {

/**
 * @brief Transport façade for the control channel and wire descriptor conversion.
 *
 * @par Context construction contract
 * When used as `Context<TransportType>`, `Context` calls `start(Callback*)` after `IpcCore` exists
 * so `on_control_message` can safely `post()` to the core. Implementations **must not** call
 * `Callback::on_control_message` or `Callback::on_control_disconnections` from inside the
 * transport's constructor — that would be undefined behavior. Register the callback in `start()`
 * and invoke it only from asynchronous paths (e.g. DDS reader/listener callbacks, reader thread)
 * after `start()` returns.
 */
class TransportInterface {
 public:
  /** Wire-format descriptor type; void at interface level. Implementations alias to their concrete
   * wire type. */
  using PointerDescriptorType = void;

  /** Wire-format handle type; void at interface level. Implementations alias to their concrete wire
   * type. */
  using HandleType = void;

  /**
   * @brief Interface for control channel callbacks from the transport.
   *
   * Implementations typically post the work to an IO context so handling runs on a dedicated
   * thread.
   */
  class Callback {
   public:
    virtual ~Callback() = default;

    virtual void on_control_message(const ControlMessage& msg,
                                    const PublicationHandle& publication_handle) = 0;

    virtual void on_control_disconnections(
        const std::vector<PublicationHandle>& currently_matched) = 0;
  };

  /**
   * @brief Interface for sending control messages (e.g. ACQUIRED, RELEASED, ACK, NACK).
   *
   * Obtained from the transport via create_control_writer(topic_name). Implementations are
   * responsible for serialization and delivery to the given topic.
   */
  class ControlWriter {
   public:
    virtual ~ControlWriter() = default;

    virtual const std::string& get_topic_name() const = 0;

    /** Send a control message. \a reply_to_topic_name is the sender inbox for ACQUIRED/RELEASED;
     * publisher inbox for ACK/NACK. */
    virtual bool send_message(const Key& key, ControlMessageType message_type,
                              const std::string& reply_to_topic_name) = 0;
  };

  virtual ~TransportInterface() = default;

  /**
   * @brief Begin delivering control callbacks to \a callback (e.g. start reader thread / DDS
   * reader).
   *
   * Called by `Context` after `IpcCore` is constructed so posted work is safe. \a callback must
   * remain valid until `stop()`. Must not be invoked from the transport constructor.
   */
  virtual void start(Callback* callback) = 0;

  /**
   * @brief Stop transport-owned background work before IpcCore is destroyed.
   *
   * `Context` calls this from its destructor before tearing down `IpcCore` so implementations that
   * deliver control callbacks from their own threads do not post to a destroyed core. Idempotent.
   */
  virtual void stop() = 0;

  virtual const std::string& get_inbox_topic_name() const = 0;

  /** Create a control writer for the given topic. Caller may cache the returned shared_ptr. */
  virtual std::shared_ptr<ControlWriter> create_control_writer(const std::string& topic_name) = 0;

  /**
   * @brief Convert a wire-format descriptor to transport-agnostic ipc::PointerDescriptor.
   *
   * The implementation type of \a wire_descriptor is transport-specific (e.g. for FastDDS
   * it holds gen::PointerDescriptor). From the interface perspective it is opaque std::any.
   * Passed by const reference to avoid copying the `std::any` payload (e.g. strings/vectors).
   * @return ipc::PointerDescriptor with the same logical fields.
   */
  virtual PointerDescriptor from_wire_descriptor(const std::any& wire_descriptor) = 0;

  /**
   * @brief Convert transport-agnostic ipc::PointerDescriptor to wire-format descriptor.
   *
   * The returned std::any holds the transport-specific wire type (e.g. for FastDDS,
   * gen::PointerDescriptor). From the interface perspective it is opaque std::any.
   * @return std::any holding the wire-format descriptor.
   */
  virtual std::any to_wire_descriptor(const PointerDescriptor& descriptor) = 0;

 protected:
  TransportInterface() = default;
  TransportInterface(const TransportInterface&) = default;
  TransportInterface& operator=(const TransportInterface&) = default;
  TransportInterface(TransportInterface&&) = default;
  TransportInterface& operator=(TransportInterface&&) = default;
};

}  // namespace ipc
}  // namespace holoscan

#endif  // HOLOSCAN_IPC_DETAIL_TRANSPORT_INTERFACE_HPP
