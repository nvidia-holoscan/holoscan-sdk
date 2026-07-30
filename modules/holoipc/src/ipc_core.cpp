/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holoscan/ipc/detail/ipc_core.hpp>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstring>
#include <exception>
#include <functional>
#include <iomanip>
#include <iterator>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <holoscan/ipc/detail/control_message.hpp>
#include <holoscan/ipc/log.hpp>

#include "cuda_ipc_backend.hpp"
#include "io_context.hpp"
#include "ipc_backend_interface.hpp"

namespace holoscan {
namespace ipc {

namespace {

// Returns the default backends map (handle type -> backend implementation).
using BackendMap =
    std::unordered_map<HandleType, std::shared_ptr<holoscan::ipc::IpcBackendInterface>>;

BackendMap make_backends() {
  return {{HandleType::CUDA_IPC, std::make_shared<holoscan::ipc::CudaIpcBackend>()}};
}

// Builds a PointerDescriptor from key, reply topic name, handle type, and opaque handle bytes
// (backend-agnostic).
PointerDescriptor make_pointer_descriptor(const holoscan::ipc::Key& key,
                                          std::string reply_to_topic_name, HandleType handle_type,
                                          std::vector<uint8_t> handle_bytes) {
  PointerDescriptor pd;
  pd.version = kIpcProtocolVersion;
  pd.key = key.to_bytes();
  pd.handle_type = handle_type;
  pd.handle = std::move(handle_bytes);
  pd.reply_to_topic_name = std::move(reply_to_topic_name);
  return pd;
}

}  // namespace

IpcCore::IpcCore(TransportInterface* transport)
    : transport_(transport),
      backends_(make_backends()),
      io_context_(std::make_shared<IOContext>()),
      worker_thread_([this]() { io_context_->run(); }) {}

void IpcCore::async_control_reader_message_dispatcher(const ControlMessage& msg,
                                                      const PublicationHandle& publication_handle) {
  io_context_->post([this, msg, publication_handle]() {
    control_reader_message_dispatcher(msg, publication_handle);
  });
}

void IpcCore::async_control_reader_disconnections_handler(
    const std::vector<PublicationHandle>& currently_matched) {
  io_context_->post(
      [this, currently_matched]() { control_reader_disconnections_handler(currently_matched); });
}

void IpcCore::set_on_last_release(ReleaseCompleteCallback callback) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  on_last_release_ = std::move(callback);
}

IpcCore::~IpcCore() {
  // Clear reply promises so ~Promise() runs and sets broken_promise on each; any worker task
  // blocked in future.get() is unblocked. Must do this before stop/join so the worker can exit.
  {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    publishers_.clear();
  }
  io_context_->stop();
  worker_thread_.join();
  // Now no posted task can run. Clear control writers so DataWriters are destroyed.
  {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    control_writers_.clear();
  }
}

std::shared_ptr<PointerDescriptor> IpcCore::share_pointer(const std::shared_ptr<void>& ptr,
                                                          HandleType handle_type) {
  // Check pointer validity.
  if (!ptr) {
    HOLOSCAN_IPC_LOG_ERROR("[Context] {}: ptr has null device pointer", __FUNCTION__);
    return {};
  }
  // Check if the backend is supported.
  auto it = backends_.find(handle_type);
  if (it == backends_.end()) {
    HOLOSCAN_IPC_LOG_ERROR(
        "[Context] {}: no backend for handle_type {}", __FUNCTION__, static_cast<int>(handle_type));
    return {};
  }

  const Key key(handle_type, ptr.get());
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  // Check if the pointer descriptor already exists.
  auto pointer_descriptor_iter = pointer_descriptors_.find(key);
  if (pointer_descriptor_iter != pointer_descriptors_.end()) {
    return pointer_descriptor_iter->second.weak_ref_.lock();
  }

  // Export the handle via the backend.
  auto handle_bytes(it->second->export_handle(ptr.get()));
  if (handle_bytes.empty()) {
    HOLOSCAN_IPC_LOG_ERROR("[Context] {}: failed to export handle for handle_type {}",
                           __FUNCTION__,
                           static_cast<int>(handle_type));
    return {};
  }

  // When the deleter (lambda) is destroyed, the guard goes out of scope and releases the export
  // in its dtor.
  struct ExportGuard {
    std::weak_ptr<IpcBackendInterface> backend;
    IpcBackendInterface::HandleBytes bytes_copy;
    ExportGuard(std::weak_ptr<IpcBackendInterface> b, const IpcBackendInterface::HandleBytes& bytes)
        : backend(std::move(b)), bytes_copy(bytes) {}
    ~ExportGuard() {
      auto backend_ptr = backend.lock();
      if (backend_ptr && !bytes_copy.empty()) {
        backend_ptr->release_export(bytes_copy);
      }
    }
  };
  ExportGuard export_guard(it->second, handle_bytes);

  // Create a new pointer descriptor (version and handle_type set inside make_pointer_descriptor).
  PointerDescriptor pd = make_pointer_descriptor(
      key, transport_->get_inbox_topic_name(), handle_type, std::move(handle_bytes));
  std::shared_ptr<PointerDescriptor> descriptor_shared(
      new PointerDescriptor(std::move(pd)),
      [ptr, key, wp = weak_from_this(), guard = std::move(export_guard)](PointerDescriptor* p) {
        if (auto impl_ptr = wp.lock()) {
          std::lock_guard<std::recursive_mutex> lock(impl_ptr->mutex_);
          impl_ptr->pointer_descriptors_.erase(key);
        }
        delete p;
      });

  // Add the pointer descriptor to the pointer descriptors.
  pointer_descriptors_.emplace(key, PointerDescriptorEntry(descriptor_shared));
  return descriptor_shared;
}

Future<std::shared_ptr<void>> IpcCore::acquire_pointer(PointerDescriptor pointer_descriptor) {
  if (!validate_pointer_descriptor(pointer_descriptor)) {
    Promise<std::shared_ptr<void>> immediate;
    Future<std::shared_ptr<void>> f = immediate.get_future();
    immediate.set_value(std::shared_ptr<void>{});
    return f;
  }
  Key reply_key(pointer_descriptor.key);
  Promise<ControlMessageType> ack_promise;
  std::string publisher_inbox = pointer_descriptor.reply_to_topic_name;
  // Create a future with a continuation that opens the handle via the backend.
  // The reason for that is that we need the handle to be opened on the caller's thread.
  auto control_message_future = ack_promise.get_future().then(
      [this, descriptor_copy = std::move(pointer_descriptor)](ControlMessageType t) {
        if (t != ControlMessageType::ACK)
          return std::shared_ptr<void>{};
        return attach_release_deleter(open_handle(descriptor_copy), descriptor_copy);
      });

  // Send ACQUIRED to publisher's inbox; put our inbox in the message so publisher sends ACK to
  // us.
  HOLOSCAN_IPC_LOG_DEBUG("[Context] {}: posting ACQUIRED", __FUNCTION__);
  io_context_->post([this,
                     reply_key,
                     publisher_inbox = std::move(publisher_inbox),
                     ack_promise = std::move(ack_promise)]() mutable {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    bool reply_enqueued = false;
    try {
      // Register the promise before sending ACQUIRED so a failed map insert cannot leave
      // an ACQUIRED in flight with no matching reply slot; if send fails or throws, roll
      // back.
      push_reply_promise(publisher_inbox, reply_key, std::move(ack_promise), false);
      reply_enqueued = true;
      if (!get_control_writer_locked(publisher_inbox)
               .send_message(
                   reply_key, ControlMessageType::ACQUIRED, transport_->get_inbox_topic_name())) {
        throw std::runtime_error("ACQUIRED write failed");
      }
    } catch (...) {
      if (reply_enqueued) {
        abort_last_reply_promise(publisher_inbox, reply_key);
      }
      HOLOSCAN_IPC_LOG_ERROR("[Context] {}: exception in posted ACQUIRED key={} publisher_inbox={}",
                             __FUNCTION__,
                             reply_key,
                             publisher_inbox);
    }
  });

  HOLOSCAN_IPC_LOG_DEBUG("[Context] {}: returning future", __FUNCTION__);
  return control_message_future;
}

bool IpcCore::validate_pointer_descriptor(const PointerDescriptor& pointer_descriptor) {
  const Key key(pointer_descriptor.key);
  HOLOSCAN_IPC_LOG_DEBUG("[Context] {}: entry key={}", __FUNCTION__, key);
  if (pointer_descriptor.reply_to_topic_name.empty()) {
    HOLOSCAN_IPC_LOG_ERROR("[Context] {}: invalid pointer descriptor (empty reply_to_topic_name)",
                           __FUNCTION__);
    return false;
  }
  auto backend_it = backends_.find(pointer_descriptor.handle_type);
  if (backend_it == backends_.end()) {
    HOLOSCAN_IPC_LOG_ERROR("[Context] {}: unsupported handle_type {}",
                           __FUNCTION__,
                           static_cast<int>(pointer_descriptor.handle_type));
    return false;
  }
  if (!backend_it->second->is_valid_handle(pointer_descriptor.handle)) {
    HOLOSCAN_IPC_LOG_ERROR("[Context] {}: invalid descriptor for handle_type {}",
                           __FUNCTION__,
                           static_cast<int>(pointer_descriptor.handle_type));
    return false;
  }
  return true;
}

std::shared_ptr<void> IpcCore::acquire_pointer_eager(const PointerDescriptor& pointer_descriptor) {
  if (!validate_pointer_descriptor(pointer_descriptor)) {
    return {};
  }
  Key reply_key(pointer_descriptor.key);
  Promise<ControlMessageType> ack_promise;

  auto opened = open_handle(pointer_descriptor);
  if (!opened) {
    HOLOSCAN_IPC_LOG_ERROR("[Context] {}: open_handle failed (eager aborted)", __FUNCTION__);
    return {};
  }
  {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    bool reply_enqueued = false;
    try {
      push_reply_promise(
          pointer_descriptor.reply_to_topic_name, reply_key, std::move(ack_promise), true);
      reply_enqueued = true;
      if (!get_control_writer_locked(pointer_descriptor.reply_to_topic_name)
               .send_message(
                   reply_key, ControlMessageType::ACQUIRED, transport_->get_inbox_topic_name())) {
        throw std::runtime_error("ACQUIRED write failed (eager aborted)");
      }
    } catch (const std::exception& e) {
      if (reply_enqueued) {
        abort_last_reply_promise(pointer_descriptor.reply_to_topic_name, reply_key);
      }
      HOLOSCAN_IPC_LOG_ERROR(
          "[Context] {}: exception sending ACQUIRED (eager): {}", __FUNCTION__, e.what());
      return {};
    }
  }

  return attach_release_deleter(std::move(opened), pointer_descriptor);
}

void IpcCore::control_reader_message_dispatcher(const ControlMessage& msg,
                                                const PublicationHandle& publication_handle) {
  const Key key(msg.key);
  const ControlMessageType message_type = msg.message_type;
  HOLOSCAN_IPC_LOG_DEBUG(
      "[Context] {}: key={} type={}", __FUNCTION__, key, static_cast<int>(message_type));
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (msg.reply_to_topic_name.empty()) {
    HOLOSCAN_IPC_LOG_WARN(
        "[Context] {}: empty reply_to_topic_name; ignoring message key={}", __FUNCTION__, key);
    return;
  }

  publication_handle_to_inbox_topic_[publication_handle] = msg.reply_to_topic_name;

  switch (message_type) {
    case ControlMessageType::ACK:
    case ControlMessageType::NACK:
      handle_ack_nack_message(msg);
      return;
    case ControlMessageType::ACQUIRED:
      handle_acquired_message(msg);
      return;
    case ControlMessageType::RELEASED:
      handle_released_message(msg);
      return;
    default:
      HOLOSCAN_IPC_LOG_WARN("[Context] {}: Unknown message type {} for buffer {}",
                            __FUNCTION__,
                            static_cast<int>(message_type),
                            key);
      return;
  }
}

void IpcCore::control_reader_disconnections_handler(
    const std::vector<PublicationHandle>& currently_matched) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  // Convert the vector to a set for O(1) lookup.
  std::set<PublicationHandle> matched_set(currently_matched.begin(), currently_matched.end());
  // Pre-allocate the vector for the number of disconnected endpoints.
  using HandleTopicPair = std::pair<PublicationHandle, std::string>;
  std::vector<HandleTopicPair> disconnected_subscribers;
  disconnected_subscribers.reserve(publication_handle_to_inbox_topic_.size());

  // Find remote endpoints (by publication handle) that are no longer matched.
  for (const auto& [instance_handle, inbox_topic] : publication_handle_to_inbox_topic_) {
    if (matched_set.find(instance_handle) == matched_set.end()) {
      disconnected_subscribers.emplace_back(instance_handle, inbox_topic);
    }
  }

  // Clean up state and writers for each disconnected endpoint.
  for (const auto& [publication_handle, inbox_topic] : disconnected_subscribers) {
    subscribers_remove(inbox_topic);
    publishers_remove(inbox_topic);
    // Erase the publication handle -> inbox topic mapping.
    publication_handle_to_inbox_topic_.erase(publication_handle);
    // Erase the control writer for this inbox topic.
    control_writers_.erase(inbox_topic);
  }
}

void IpcCore::handle_ack_nack_message(const ControlMessage& msg) {
  const Key key(msg.key);  // Needed for logging.
  const std::string& publisher_inbox = msg.reply_to_topic_name;
  // Find the publisher entry for this publisher inbox topic name.
  auto pub_it = publishers_.find(publisher_inbox);
  if (pub_it == publishers_.end()) {
    HOLOSCAN_IPC_LOG_ERROR("[Context] {}: Could not find publisher entry for topic {} key {}",
                           __FUNCTION__,
                           publisher_inbox,
                           key);
    return;
  }
  // Find the reply promise for this key.
  auto reply_it = pub_it->second.reply_promises_.find(key);
  if (reply_it == pub_it->second.reply_promises_.end() || reply_it->second.empty()) {
    HOLOSCAN_IPC_LOG_ERROR(
        "[Context] {}: Could not find reply promise for key {}", __FUNCTION__, key);
    return;
  }
  // Fulfill the reply promise.
  HOLOSCAN_IPC_LOG_DEBUG("[Context] {}: fulfilling promise for key={}", __FUNCTION__, key);
  auto& fifo = reply_it->second;
  // The Future's continuation will continue with processing the message.
  // The reason for that is that we need the handle to be opened on the caller's thread.
  PendingAcquireReply pending = std::move(fifo.front());
  fifo.pop_front();
  if (fifo.empty()) {
    pub_it->second.reply_promises_.erase(reply_it);
  }
  // acquire_pointer_eager maps before ACK; NACK means the publisher did not grant the ref.
  // No safe recovery while holding a local mapping; see docs/EAGER_ACQUIRE.md and
  // PROTOCOL_SPEC.md.
  if (msg.message_type == ControlMessageType::NACK && pending.eager_acquire) {
    HOLOSCAN_IPC_LOG_ERROR(
        "[Context] {}: NACK after eager acquire (key={} publisher_inbox={}); terminating — "
        "see docs/EAGER_ACQUIRE.md",
        __FUNCTION__,
        key,
        publisher_inbox);
    std::terminate();
  }
  pending.promise.set_value(msg.message_type);
}

void IpcCore::handle_acquired_message(const ControlMessage& msg) {
  const Key key(msg.key);
  try {
    // Find the pointer descriptor for the device pointer key.
    auto pointer_descriptor_iter = pointer_descriptors_.find(key);
    if (pointer_descriptor_iter == pointer_descriptors_.end()) {
      HOLOSCAN_IPC_LOG_WARN("[Context] {}: Message for unknown buffer {}", __FUNCTION__, key);
      throw std::runtime_error("unknown buffer");  // A NACK will be sent.
    }

    // Record this ACQUIRED for the subscriber (increment ref count; topic -> entry).
    subscribers_add(msg.reply_to_topic_name, key);

    // Add a ref to the pointer descriptor.
    pointer_descriptors_add_refs(key, 1);
    // Send an ACK (subscriber_inbox_topic_name is the key for this entry).
    get_control_writer_locked(msg.reply_to_topic_name)
        .send_message(key, ControlMessageType::ACK, transport_->get_inbox_topic_name());
  } catch (const std::exception& e) {
    // Send a NACK to the publisher if the reply topic name is valid.
    HOLOSCAN_IPC_LOG_ERROR("[Context] {}: Failed to handle ACQUIRED message for buffer {}: {}",
                           __FUNCTION__,
                           Key(msg.key),
                           e.what());
    const std::string& reply_topic = msg.reply_to_topic_name;
    const bool nack_send_failed =
        reply_topic.empty() ||
        !get_control_writer_locked(reply_topic)
             .send_message(key, ControlMessageType::NACK, transport_->get_inbox_topic_name());
    if (nack_send_failed) {
      HOLOSCAN_IPC_LOG_ERROR("[Context] {}: Failed to send NACK to {}: write failed",
                             __FUNCTION__,
                             msg.reply_to_topic_name);
    }
  }
}

void IpcCore::handle_released_message(const ControlMessage& msg) {
  const Key key(msg.key);
  // Check if the pointer descriptor exists.
  auto pointer_descriptor_iter = pointer_descriptors_.find(key);
  if (pointer_descriptor_iter == pointer_descriptors_.end()) {
    HOLOSCAN_IPC_LOG_WARN("[Context] {}: Message for unknown buffer {}", __FUNCTION__, key);
    return;
  }

  auto subscriber_iter = subscribers_.find(msg.reply_to_topic_name);
  if (subscriber_iter == subscribers_.end()) {
    HOLOSCAN_IPC_LOG_WARN("[Context] {}: RELEASED for buffer {} from unknown subscriber; ignoring",
                          __FUNCTION__,
                          key);
    return;
  }

  // Remove a reference from the subscriber.
  if (!subscriber_iter->second.remove_ref(key)) {
    HOLOSCAN_IPC_LOG_WARN(
        "[Context] {}: RELEASED for buffer {} with no ref count for this subscriber; ignoring",
        __FUNCTION__,
        key);
    return;
  }
  // If no more references for this subscriber, erase subscriber entry. Handle->topic mapping is
  // cleared in control_reader_disconnections_handler on disconnection.
  if (subscriber_iter->second.ref_counts_.empty()) {
    subscribers_.erase(subscriber_iter);
  }

  if (pointer_descriptors_remove_refs(key, 1) && on_last_release_) {
    on_last_release_(key);
  }
}

void IpcCore::async_send_message(const std::string& topic_name, const Key& device_ptr_key,
                                 ControlMessageType message_type) {
  io_context_->post([this, topic_name, device_ptr_key, message_type]() {
    HOLOSCAN_IPC_LOG_DEBUG("[Context] {}: running posted task key={} type={}",
                           __FUNCTION__,
                           device_ptr_key,
                           static_cast<int>(message_type));
    get_control_writer(topic_name)
        .send_message(device_ptr_key, message_type, transport_->get_inbox_topic_name());
  });
}

std::shared_ptr<void> IpcCore::attach_release_deleter(std::shared_ptr<void> opened_ptr,
                                                      const PointerDescriptor& pointer_descriptor) {
  if (!opened_ptr) {
    return {};
  }
  return std::shared_ptr<void>(opened_ptr.get(),
                               [opened_ptr,
                                wp = weak_from_this(),
                                reply_to_topic_name = pointer_descriptor.reply_to_topic_name,
                                key = Key(pointer_descriptor.key)](void*) mutable {
                                 // The pointer is deleted indirectly by the shared_ptr.
                                 opened_ptr.reset();
                                 // Send a RELEASED message to the publisher.
                                 if (auto impl = wp.lock()) {
                                   impl->async_send_message(
                                       reply_to_topic_name, key, ControlMessageType::RELEASED);
                                 }
                               });
}

std::shared_ptr<void> IpcCore::open_handle(const PointerDescriptor& pointer_descriptor) {
  auto backend_it = backends_.find(pointer_descriptor.handle_type);
  if (backend_it == backends_.end()) {
    HOLOSCAN_IPC_LOG_ERROR("[Context] {}: no backend for handle_type {}",
                           __FUNCTION__,
                           static_cast<int>(pointer_descriptor.handle_type));
    return {};
  }
  return backend_it->second->open_handle(pointer_descriptor.handle);
}

TransportInterface::ControlWriter& IpcCore::get_control_writer(const std::string& topic_name) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  return get_control_writer_locked(topic_name);
}

TransportInterface::ControlWriter& IpcCore::get_control_writer_locked(
    const std::string& topic_name) {
  HOLOSCAN_IPC_LOG_DEBUG("[Context] {}: entry topic={}", __FUNCTION__, topic_name);
  auto it = control_writers_.find(topic_name);
  if (it == control_writers_.end()) {
    auto writer = transport_->create_control_writer(topic_name);
    it = control_writers_.emplace(topic_name, std::move(writer)).first;
    HOLOSCAN_IPC_LOG_DEBUG("[Context] {}: exit (new topic={})", __FUNCTION__, topic_name);
  } else {
    HOLOSCAN_IPC_LOG_DEBUG("[Context] {}: exit (cached topic={})", __FUNCTION__, topic_name);
  }
  return *it->second;
}

void IpcCore::PointerDescriptorEntry::add_refs(size_t n) {
  descriptor_ = weak_ref_.lock();
  ref_count_ += n;
}

bool IpcCore::PointerDescriptorEntry::remove_refs(size_t n) {
  if (n > ref_count_) {
    HOLOSCAN_IPC_LOG_ERROR("[Context] {}: n exceeds ref_count_, capping", __FUNCTION__);
    n = ref_count_;
  }

  ref_count_ -= n;
  bool was_last = (ref_count_ == 0);
  if (was_last) {
    descriptor_.reset();
  }
  return was_last;
}

void IpcCore::pointer_descriptors_add_refs(const Key& key, size_t n) {
  auto it = pointer_descriptors_.find(key);
  if (it != pointer_descriptors_.end()) {
    it->second.add_refs(n);
  }
}

bool IpcCore::pointer_descriptors_remove_refs(const Key& key, size_t n) {
  auto it = pointer_descriptors_.find(key);
  if (it == pointer_descriptors_.end()) {
    return false;
  }

  return it->second.remove_refs(n);
}

void IpcCore::SubscriberEntry::add_ref(const Key& key) {
  ref_counts_[key]++;
}

bool IpcCore::SubscriberEntry::remove_ref(const Key& key) {
  auto it = ref_counts_.find(key);
  if (it == ref_counts_.end()) {
    return false;
  }

  it->second--;
  // If no more references for this key, erase the entry.
  if (it->second == 0) {
    ref_counts_.erase(it);
  }
  return true;
}

void IpcCore::subscribers_add(const std::string& subscriber_inbox_topic_name, const Key& key) {
  subscribers_[subscriber_inbox_topic_name].add_ref(key);
}

void IpcCore::subscribers_remove(const std::string& subscriber_inbox_topic_name) {
  auto subscriber_iter = subscribers_.find(subscriber_inbox_topic_name);
  if (subscriber_iter == subscribers_.end()) {
    return;
  }
  for (const auto& [key, nrefs] : subscriber_iter->second.ref_counts_) {
    pointer_descriptors_remove_refs(key, nrefs);
  }
  subscribers_.erase(subscriber_iter);
}

IpcCore::PointerDescriptorEntry::PointerDescriptorEntry(
    std::shared_ptr<PointerDescriptor> descriptor)
    : descriptor_(std::move(descriptor)), weak_ref_(descriptor_), ref_count_(0) {}

void IpcCore::push_reply_promise(const std::string& publisher_inbox_topic_name, const Key& key,
                                 Promise<ControlMessageType> promise, bool eager_acquire) {
  publishers_[publisher_inbox_topic_name].reply_promises_[key].push_back(
      PendingAcquireReply{std::move(promise), eager_acquire});
}

void IpcCore::abort_last_reply_promise(const std::string& publisher_inbox_topic_name,
                                       const Key& key) noexcept {
  try {
    auto pub_it = publishers_.find(publisher_inbox_topic_name);
    if (pub_it == publishers_.end()) {
      return;
    }
    auto reply_it = pub_it->second.reply_promises_.find(key);
    if (reply_it == pub_it->second.reply_promises_.end() || reply_it->second.empty()) {
      return;
    }
    auto& fifo = reply_it->second;
    PendingAcquireReply dropped = std::move(fifo.back());
    fifo.pop_back();
    if (fifo.empty()) {
      pub_it->second.reply_promises_.erase(reply_it);
    }
    dropped.promise.set_value(ControlMessageType::NACK);
  } catch (...) {
  }
}

void IpcCore::publishers_remove(const std::string& publisher_inbox_topic_name) {
  publishers_.erase(publisher_inbox_topic_name);
}

}  // namespace ipc
}  // namespace holoscan
