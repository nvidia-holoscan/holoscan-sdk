/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOSCAN_IPC_CONTEXT_HPP
#define HOLOSCAN_IPC_CONTEXT_HPP

#include <any>
#include <iomanip>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <typeinfo>
#include <utility>
#include <vector>

#include <holoscan/ipc/future.hpp>

#include <holoscan/ipc/detail/transport_interface.hpp>
// Include outside namespace so heavy deps (<thread>, etc.) are at global scope.
#include <holoscan/ipc/detail/ipc_core.hpp>

namespace holoscan {
namespace ipc {

/**
 * @brief Context for converting between std::shared_ptr<void> and the transport's wire descriptor.
 *
 * Uses the configured Transport for the lifecycle channel (ACQUIRED/RELEASED). Use share_pointer()
 * before publishing and acquire_pointer() (or acquire_pointer_eager()) after receiving a descriptor
 * when you need the subscriber-side representation of the shared resource (still exposed as
 * std::shared_ptr<void>; meaning is defined by handle_type).
 *
 * Must be created via make_context<TransportType>(...) and held as
 * std::shared_ptr<Context<TransportType>> so that ptr deleters can safely send RELEASED via the
 * context's implementation.
 *
 * @tparam TransportType Transport implementation type (e.g. a DDS-based transport). Must satisfy
 *            the **Context construction contract** on `TransportInterface`: do not call back into
 *            the `Context` (`Callback`) from the transport's constructor; `Context` calls
 *            `transport_->start(this)` after `IpcCore` exists (see `TransportInterface`).
 */
template <typename TransportType>
class Context : public TransportInterface::Callback {
 public:
  /** Wire-format descriptor type for this transport. */
  using PointerDescriptorType = typename TransportType::PointerDescriptorType;

  /** Wire-format handle type for this transport. */
  using HandleType = typename TransportType::HandleType;

  template <typename T, typename... Args>
  friend std::shared_ptr<Context<T>> make_context(Args&&... args);

  ~Context() override { transport_->stop(); }

  void on_control_message(const ControlMessage& msg,
                          const PublicationHandle& publication_handle) override {
    core_->async_control_reader_message_dispatcher(msg, publication_handle);
  }

  void on_control_disconnections(const std::vector<PublicationHandle>& currently_matched) override {
    core_->async_control_reader_disconnections_handler(currently_matched);
  }

  Context(const Context&) = delete;
  Context& operator=(const Context&) = delete;
  Context(Context&&) = delete;
  Context& operator=(Context&&) = delete;

  /**
   * @brief Convert a std::shared_ptr<void> to a shared descriptor (publisher side).
   *
   * Exports the handle via the backend for the given handle_type (e.g. HandleType::CUDA_IPC),
   * registers the ptr with the lifecycle channel, and returns a shared_ptr to a descriptor.
   * Hold this shared_ptr to keep the export registered; use *descriptor or descriptor->...
   * for DDS write. Application metadata (e.g. payload size) is not part of the IPC protocol
   * unless the application adds it (fixed layout, separate topic, or out-of-band).
   *
   * @param ptr Opaque pointer to the resource to share; meaning depends entirely on \a handle_type
   *            (e.g. CUDA device memory for CUDA_IPC).
   * @param handle_type How the handle is created and interpreted (e.g. HandleType::CUDA_IPC).
   * @return shared_ptr to the descriptor, or null on failure.
   */
  std::shared_ptr<PointerDescriptorType> share_pointer(const std::shared_ptr<void>& ptr,
                                                       HandleType handle_type) {
    std::shared_ptr<PointerDescriptor> ipc_desc = core_->share_pointer(
        ptr, static_cast<holoscan::ipc::HandleType>(static_cast<int32_t>(handle_type)));
    if (!ipc_desc) {
      return {};
    }
    std::any wired = transport_->to_wire_descriptor(*ipc_desc);
    auto* wire_ptr = std::any_cast<PointerDescriptorType>(&wired);
    if (wire_ptr == nullptr) {
      throw std::runtime_error(std::string("Context::share_pointer: to_wire_descriptor std::any "
                                           "type mismatch: expected ") +
                               typeid(PointerDescriptorType).name() + ", got " +
                               wired.type().name());
    }
    return std::shared_ptr<PointerDescriptorType>(new PointerDescriptorType(std::move(*wire_ptr)),
                                                  [ipc_desc](PointerDescriptorType* p) {
                                                    delete p;
                                                    // ipc_desc goes out of scope; when last wire
                                                    // ref is released, ipc_desc's deleter runs
                                                  });
  }

  /**
   * @brief Convert a PointerDescriptor to a std::shared_ptr<void> (subscriber side).
   *
   * Sends ACQUIRED and returns a Future. Call get() to block until ACK
   * is received; get() then opens the handle via the backend on the calling thread and returns
   * a std::shared_ptr<void> holding the local representation of the resource (or empty on failure).
   * The open runs on the caller's thread so the result is valid in that thread's context (e.g.
   * current CUDA context when the handle type is CUDA-based).
   *
   * @param descriptor The received descriptor (from take/read).
   * @return A Future<std::shared_ptr<void>>; call get() to obtain the imported resource as void*
   *         (reinterpret per handle_type / application contract).
   *         On validation failure (invalid key, empty reply_topic_name, unsupported handle_type,
   *         or wrong handle size), returns an invalid future (valid() == false); get() then
   *         returns an empty shared_ptr without blocking.
   */
  Future<std::shared_ptr<void>> acquire_pointer(const PointerDescriptorType& descriptor) {
    return core_->acquire_pointer(transport_->from_wire_descriptor(descriptor));
  }

  /**
   * @brief Eager acquire: opens the handle locally, then sends ACQUIRED (no wait for ACK).
   *
   * See docs/EAGER_ACQUIRE.md. If the publisher NACKs after a successful open,
   * `std::terminate()` is called. Successful imports use the same RELEASED-on-drop behavior
   * as `acquire_pointer`.
   */
  std::shared_ptr<void> acquire_pointer_eager(const PointerDescriptorType& descriptor) {
    return core_->acquire_pointer_eager(transport_->from_wire_descriptor(descriptor));
  }

  void set_on_last_release(typename IpcCore::ReleaseCompleteCallback callback) {
    core_->set_on_last_release(std::move(callback));
  }

 private:
  template <typename... Args>
  explicit Context(Args&&... args)
      : transport_(std::make_shared<TransportType>(std::forward<Args>(args)...)),
        core_(std::make_shared<IpcCore>(transport_.get())) {
    transport_->start(this);
  }

  std::shared_ptr<TransportType> transport_;
  std::shared_ptr<IpcCore> core_;
};

/**
 * @brief Create a context for the given transport, forwarding \a args to the context constructor.
 * @tparam TransportType Transport implementation type.
 * @param args Arguments forwarded to Context<TransportType>'s constructor.
 * @return A shared_ptr to the new context; never null.
 */
template <typename TransportType, typename... Args>
std::shared_ptr<Context<TransportType>> make_context(Args&&... args) {
  return std::shared_ptr<Context<TransportType>>(
      new Context<TransportType>(std::forward<Args>(args)...));
}

}  // namespace ipc
}  // namespace holoscan

#endif  // HOLOSCAN_IPC_CONTEXT_HPP
