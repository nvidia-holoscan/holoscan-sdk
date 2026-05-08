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

#ifndef HOLOSCAN_IPC_DETAIL_IPC_CORE_HPP
#define HOLOSCAN_IPC_DETAIL_IPC_CORE_HPP

// Transport-agnostic IPC core (share_pointer, acquire_pointer, acquire_pointer_eager, backends).
// Used as Context's pimpl.

#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <holoscan/ipc/detail/pointer_descriptor.hpp>

#include <holoscan/ipc/future.hpp>

#include <holoscan/ipc/detail/key.hpp>
#include <holoscan/ipc/detail/transport_interface.hpp>

namespace holoscan {
namespace ipc {

class IpcBackendInterface;
class IOContext;

/**
 * Transport-agnostic IPC core: share_pointer, acquire_pointer, acquire_pointer_eager,
 * backends, control message handling.
 */
class IpcCore : public std::enable_shared_from_this<IpcCore> {
 public:
  using ReleaseCompleteCallback = std::function<void(const Key&)>;

  explicit IpcCore(TransportInterface* transport);
  ~IpcCore();

  std::shared_ptr<PointerDescriptor> share_pointer(const std::shared_ptr<void>& ptr,
                                                   HandleType handle_type);

  /**
   * Sends ACQUIRED, completes after ACK with an opened pointer (RELEASED on drop); empty on
   * NACK or failure. */
  Future<std::shared_ptr<void>> acquire_pointer(PointerDescriptor pointer_descriptor);

  /**
   * Like acquire_pointer but opens handles locally first, then sends ACQUIRED (no wait for ACK).
   * NACK after a successful open calls std::terminate(). See docs/EAGER_ACQUIRE.md.
   */
  std::shared_ptr<void> acquire_pointer_eager(const PointerDescriptor& pointer_descriptor);

  void async_control_reader_message_dispatcher(const ControlMessage& msg,
                                               const PublicationHandle& publication_handle);
  void async_control_reader_disconnections_handler(
      const std::vector<PublicationHandle>& currently_matched);

  void set_on_last_release(ReleaseCompleteCallback callback);

 private:
  TransportInterface* transport_;

  // BackendsMap: HandleType -> IpcBackendInterface
  // Used to track the backends for each handle type.
  using BackendsMap = std::unordered_map<HandleType, std::shared_ptr<IpcBackendInterface>>;
  const BackendsMap backends_;

  // Control reader message handlers
  void control_reader_message_dispatcher(const ControlMessage& msg,
                                         const PublicationHandle& publication_handle);
  void control_reader_disconnections_handler(
      const std::vector<PublicationHandle>& currently_matched);

  // Message handlers for ACQUIRED/RELEASED and ACK/NACK messages. Preconditions: mutex_ is
  // held by the caller (control_reader_message_dispatcher).
  void handle_ack_nack_message(const ControlMessage& msg);
  void handle_acquired_message(const ControlMessage& msg);
  void handle_released_message(const ControlMessage& msg);

  // Sends a message to the given topic name asynchronously.
  void async_send_message(const std::string& topic_name, const Key& device_ptr_key,
                          ControlMessageType message_type);

  // Looks up backend and opens handle bytes. Returns {} if no backend or
  // open fails. Callers that own the acquisition call attach_release_deleter to send RELEASED
  // on drop. Strict acquire: open after ACK, then attach. Eager: open, send ACQUIRED,
  // then attach.
  std::shared_ptr<void> open_handle(const PointerDescriptor& descriptor);

  // Wraps a backend-opened pointer so destruction sends RELEASED to the publisher inbox.
  std::shared_ptr<void> attach_release_deleter(std::shared_ptr<void> opened_ptr,
                                               const PointerDescriptor& pointer_descriptor);

  // PublicationHandleToInboxTopicMap: opaque handle -> inbox topic name
  using PublicationHandleToInboxTopicMap = std::map<PublicationHandle, std::string>;
  PublicationHandleToInboxTopicMap publication_handle_to_inbox_topic_;

  // ControlWritersMap: topic_name -> shared_ptr to transport's ControlWriter (interface)
  // Used to cache control writer instances per topic so we can send ACK/NACK/RELEASED to remote
  // inboxes.
  using ControlWritersMap =
      std::unordered_map<std::string, std::shared_ptr<TransportInterface::ControlWriter>>;
  ControlWritersMap control_writers_;
  // Returns the ControlWriter for the given topic name, creates it via transport if it doesn't
  // exist. Thread-safe (acquires mutex_).
  TransportInterface::ControlWriter& get_control_writer(const std::string& topic_name);
  // Same as get_control_writer but caller must already hold mutex_ (non-reentrant paths only).
  TransportInterface::ControlWriter& get_control_writer_locked(const std::string& topic_name);

  // PointerDescriptorMap: Key -> PointerDescriptorEntry
  // Used to track shared pointer descriptors and their ref counts (one ref per ACQUIRED from a
  // subscriber).
  struct PointerDescriptorEntry {
    explicit PointerDescriptorEntry(std::shared_ptr<PointerDescriptor> descriptor);

    // Ensures descriptor_ is valid (restores from weak_ref_ if reset), then increments
    // ref_count_ by n.
    void add_refs(size_t n);
    // Decrements ref count by n (capped at ref_count_). Resets descriptor_ if ref_count_
    // becomes 0. Returns true if ref_count_ reached 0.
    bool remove_refs(size_t n);

    // Keeps the descriptor alive while ref_count_ > 0; reset when ref_count_ becomes 0.
    // Declaration order: descriptor_ before weak_ref_ so the ctor can initialize weak_ref_ from
    // descriptor_ (C++ initializes members in declaration order).
    std::shared_ptr<PointerDescriptor> descriptor_;
    // Used to obtain a shared_ptr when descriptor_ may have been reset (e.g. user still holds a
    // ref from share_pointer).
    std::weak_ptr<PointerDescriptor> weak_ref_;
    size_t ref_count_{0};
  };
  using PointerDescriptorMap = std::map<Key, PointerDescriptorEntry>;
  PointerDescriptorMap pointer_descriptors_;
  // Adds n refs to the entry for key. No-op if key is not found.
  void pointer_descriptors_add_refs(const Key& key, size_t n);
  // Removes n refs from the entry for key. No-op if key is not found. Returns true if ref_count_
  // reached 0. Map entry is removed by the share_pointer deleter when the descriptor is destroyed.
  bool pointer_descriptors_remove_refs(const Key& key, size_t n);

  // SubscribersMap: inbox topic name -> SubscriberEntry
  // Used to track ref counts per key for each subscriber.
  struct SubscriberEntry {
    // Key -> number of references to the pointer descriptor for this subscriber.
    using RefCountsMap = std::map<Key, size_t>;
    RefCountsMap ref_counts_;

    // Adds one ref for key. Does not send ACK; caller must send after this returns.
    void add_ref(const Key& key);

    // Decrements ref count for key; erases key from ref_counts_ if count reaches 0. Returns
    // true if key was found, false otherwise.
    bool remove_ref(const Key& key);
  };
  using SubscribersMap = std::map<std::string, SubscriberEntry>;
  SubscribersMap subscribers_;
  // Gets or creates the subscriber entry for subscriber_inbox_topic_name and adds one ref for
  // key. Publication handle -> inbox is updated in the dispatcher before this is called.
  void subscribers_add(const std::string& subscriber_inbox_topic_name, const Key& key);
  // Releases pointer descriptor refs for the subscriber at subscriber_inbox_topic_name and erases
  // its entry. Publication handle -> inbox is only cleared in
  // control_reader_disconnections_handler on disconnection. No-op if not found.
  void subscribers_remove(const std::string& subscriber_inbox_topic_name);

  // PublishersMap: publisher inbox topic name -> PublisherEntry
  // Used to track pending ACK/NACK promises for each publisher.
  struct PendingAcquireReply {
    Promise<ControlMessageType> promise;
    bool eager_acquire = false;
  };
  struct PublisherEntry {
    using ReplyPromisesMap = std::map<Key, std::deque<PendingAcquireReply>>;
    ReplyPromisesMap reply_promises_;
  };
  using PublishersMap = std::map<std::string, PublisherEntry>;
  PublishersMap publishers_;

  // Validates pointer_descriptor for a subscriber acquire: non-empty reply_to_topic_name,
  // handle_type backed by backends_, and handle bytes accepted by that backend
  // (is_valid_handle). Returns true if valid; on failure logs and returns false. Does not send
  // ACQUIRED or call push_reply_promise.
  bool validate_pointer_descriptor(const PointerDescriptor& pointer_descriptor);

  void push_reply_promise(const std::string& publisher_inbox_topic_name, const Key& key,
                          Promise<ControlMessageType> promise, bool eager_acquire);
  // Pops the last enqueued reply promise for key and fulfills it with NACK. Preconditions:
  // mutex_ held; queue for key was non-empty (typically after push_reply_promise + failed send).
  // noexcept: swallows internal exceptions so error-path callers need not wrap the call.
  void abort_last_reply_promise(const std::string& publisher_inbox_topic_name,
                                const Key& key) noexcept;
  // Erases the publisher entry.
  void publishers_remove(const std::string& publisher_inbox_topic_name);

  // Protects control_writers_, publication_handle_to_inbox_topic_, pointer_descriptors_,
  // subscribers_, publishers_.
  mutable std::recursive_mutex mutex_;
  std::shared_ptr<IOContext> io_context_;
  std::thread worker_thread_;
  ReleaseCompleteCallback on_last_release_;
};

}  // namespace ipc
}  // namespace holoscan

#endif  // HOLOSCAN_IPC_DETAIL_IPC_CORE_HPP
