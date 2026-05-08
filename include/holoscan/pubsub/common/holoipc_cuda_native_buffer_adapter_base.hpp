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

#ifndef PUBSUB_COMMON_INCLUDE_PUBSUB_HOLOIPC_CUDA_NATIVE_BUFFER_ADAPTER_BASE_HPP
#define PUBSUB_COMMON_INCLUDE_PUBSUB_HOLOIPC_CUDA_NATIVE_BUFFER_ADAPTER_BASE_HPP

#include <chrono>
#include <cstring>
#include <future>
#include <iterator>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include <gxf/core/expected.hpp>
#include <gxf/pubsub/cuda_ipc_descriptor.hpp>
#include <gxf/pubsub/cuda_ipc_eligibility.hpp>
#include <gxf/pubsub/pubsub_context.hpp>
#include <gxf/pubsub/pubsub_native_buffer.hpp>
#include <gxf/std/tensor.hpp>

#include <holoscan/ipc/context.hpp>
#include <holoscan/ipc/detail/control_message.hpp>
#include <holoscan/ipc/detail/key.hpp>

#include <holoscan/logger/logger.hpp>

#include <holoscan/pubsub/common/native_buffer_protocol.hpp>
#include <holoscan/pubsub/common/native_buffer_protocol_adapter.hpp>

namespace holoscan {

/// Shared CUDA IPC + holoipc lifecycle adapter core, parameterized on the holoipc transport.
///
/// Owns all transport-independent logic: GPU info caching, CudaTensorDescriptor conversion,
/// CUDA IPC export/import via GXF helpers, holoipc share_pointer/acquire_pointer lifecycle,
/// DNBX (Descriptor Native Buffer eXport) wire-format encode/decode, pending-export tracking,
/// and deferred destruction.
///
/// Subclasses provide only the initialize() signature and ipc::Context<Transport> creation.
///
/// Threading: three threads access this object --
///   - GXF scheduler (export_tensor, evict_stale_exports)
///   - sidecar dispatch (import_tensor)
///   - IpcCore io_context (handle_last_release callback)
/// exports_mutex_ protects pending_exports_, lifecycle_key_to_seq_, released_entries_.
///
/// ExportEntry destruction must happen outside both exports_mutex_ and the io_context
/// thread because the ipc_descriptor deleter takes IpcCore::mutex_ and may re-enter the
/// io_context. The solution is deferred destruction: handle_last_release queues entries
/// into released_entries_; the next export_tensor call drains them on the worker thread,
/// freeing GPU memory inline and moving ipc_descriptors to a background thread.
template <typename TransportType>
class HoloIpcCudaNativeBufferAdapterBase : public NativeBufferProtocolAdapter {
 public:
  using IpcContextType = ipc::Context<TransportType>;
  using PointerDescriptorType = typename IpcContextType::PointerDescriptorType;
  using HandleType = typename IpcContextType::HandleType;

  /// RAII holder for a lifecycle-managed CUDA IPC mapping.
  struct NativeBufferHolder {
    nvidia::gxf::CudaIpcDescriptor descriptor;
    std::shared_ptr<void> ipc_device_ptr;  // releasing this sends RELEASED to publisher

    void* device_ptr() const { return ipc_device_ptr.get(); }
    size_t byte_size() const { return descriptor.byte_size; }
  };

  /// Decoded DNBX payload: GXF CUDA IPC descriptor + holoipc lifecycle metadata.
  struct ExportedTensorDescriptor {
    nvidia::gxf::CudaIpcDescriptor gxf_descriptor;
    std::vector<uint8_t> lifecycle_key;         // holoipc correlation key
    std::string lifecycle_reply_to_topic_name;  // holoipc control-channel topic
    int32_t handle_type{0};                     // IpcHandleType (e.g. CUDA_IPC = 0)
    std::string ipc_protocol_version;           // holoipc protocol version (e.g. "1.0")
  };

  explicit HoloIpcCudaNativeBufferAdapterBase(std::string adapter_name)
      : adapter_name_(std::move(adapter_name)) {}

  ~HoloIpcCudaNativeBufferAdapterBase() override {
    if (initialized_) {
      shutdown();
    }
  }

  HoloIpcCudaNativeBufferAdapterBase(const HoloIpcCudaNativeBufferAdapterBase&) = delete;
  HoloIpcCudaNativeBufferAdapterBase& operator=(const HoloIpcCudaNativeBufferAdapterBase&) = delete;

  void shutdown() {
    if (!initialized_)
      return;
    cleanup_exports(kDefaultShutdownGrace);
    ipc_context_.reset();
    initialized_ = false;
    HOLOSCAN_LOG_DEBUG("{}: shut down", adapter_name_);
  }

  bool is_initialized() const override { return initialized_; }
  nvidia::gxf::NativeBufferPolicy policy() const { return policy_; }
  void set_acquire_timeout(std::chrono::milliseconds timeout) { acquire_timeout_ = timeout; }
  void set_export_ttl(std::chrono::milliseconds ttl) { export_ttl_ = ttl; }
  void set_use_eager_acquire(bool enabled) { use_eager_acquire_ = enabled; }

  const std::string& default_protocol_name() const override {
    static const std::string kCudaIpc = "cuda_ipc";
    return kCudaIpc;
  }

  bool supports_protocol(const std::string& protocol_name) const override {
    return protocol_name == "cuda_ipc";
  }

  bool can_export_tensor(const nvidia::gxf::Tensor& tensor) const override {
    return tensor.storage_type() == nvidia::gxf::MemoryStorageType::kDevice;
  }

  uint8_t descriptor_format_version() const override { return nvidia::gxf::kCudaIpcFormatVersion; }

  // --- Publisher side ---

  /// NativeBufferProtocolAdapter interface: converts NativeTensorMetadata to
  /// CudaTensorDescriptor and delegates to the CudaTensorDescriptor overload.
  nvidia::gxf::Expected<std::vector<uint8_t>> export_tensor(
      void* device_ptr, const std::shared_ptr<void>& device_ptr_owner,
      const NativeTensorMetadata& tensor_info, const std::string& protocol_name,
      uint32_t sequence_num) override {
    if (!supports_protocol(protocol_name)) {
      HOLOSCAN_LOG_ERROR(
          "{}::export_tensor: unsupported protocol '{}'", adapter_name_, protocol_name);
      return nvidia::gxf::Unexpected(GXF_NOT_IMPLEMENTED);
    }
    ensure_gpu_info_initialized();

    nvidia::gxf::CudaTensorDescriptor cuda_tensor_info;
    cuda_tensor_info.shape = tensor_info.shape;
    cuda_tensor_info.strides = tensor_info.strides;
    cuda_tensor_info.dtype = tensor_info.dtype;
    cuda_tensor_info.storage_type = static_cast<uint8_t>(tensor_info.storage_type);
    cuda_tensor_info.bytes_per_element = tensor_info.bytes_per_element;

    return export_tensor(device_ptr,
                         device_ptr_owner,
                         cuda_tensor_info,
                         gpu_device_uuid_,
                         gpu_device_id_,
                         sequence_num);
  }

  /// Export a GPU tensor for CUDA IPC transfer. Steps:
  ///   1. Drain deferred-destroy entries from prior releases (see threading note above)
  ///   2. Call gxf::export_cuda_ipc_descriptor() to get CudaIpcDescriptor + mem_handle
  ///   3. Register with holoipc via share_pointer() using pre-exported handle bytes
  ///   4. Encode into DNBX wire format (GXF descriptor + holoipc lifecycle metadata)
  ///   5. Track in pending_exports_ for lifecycle management
  nvidia::gxf::Expected<std::vector<uint8_t>> export_tensor(
      void* device_ptr, const std::shared_ptr<void>& device_ptr_owner,
      const nvidia::gxf::CudaTensorDescriptor& tensor_info, const std::string& device_uuid,
      int32_t device_id, uint32_t sequence_num) {
    if (!initialized_ || !ipc_context_) {
      HOLOSCAN_LOG_ERROR("{}::export_tensor: not initialized", adapter_name_);
      return nvidia::gxf::Unexpected(GXF_UNINITIALIZED_VALUE);
    }

    // Collect entries for deferred destruction.  This includes:
    //   - released_entries_ queued by handle_last_release on the io_context thread
    //   - stale exports past their TTL
    //
    // All destruction happens on THIS (worker) thread, outside both exports_mutex_
    // and the io_context, avoiding the two constraints documented in the header.
    std::vector<ExportEntry> deferred_destroy;
    {
      std::lock_guard<std::mutex> lock(exports_mutex_);
      if (!released_entries_.empty()) {
        deferred_destroy.insert(deferred_destroy.end(),
                                std::make_move_iterator(released_entries_.begin()),
                                std::make_move_iterator(released_entries_.end()));
        released_entries_.clear();
      }
      evict_stale_exports_locked(export_ttl_, deferred_destroy);
    }
    // Free GPU memory on the worker thread, then move ipc_descriptor shared_ptrs to a
    // background thread: their destructors acquire IpcCore::mutex_, which may be held
    // by the io_context thread during send_message. Blocking the worker here would
    // prevent new frames from being exported.
    if (!deferred_destroy.empty()) {
      HOLOSCAN_LOG_DEBUG("{}::export_tensor seq={}: freeing {} deferred entries",
                         adapter_name_,
                         sequence_num,
                         deferred_destroy.size());
      std::vector<std::shared_ptr<PointerDescriptorType>> bg_descriptors;
      bg_descriptors.reserve(deferred_destroy.size());
      for (auto& entry : deferred_destroy) {
        entry.device_ptr_owner.reset();
        bg_descriptors.push_back(std::move(entry.ipc_descriptor));
      }
      deferred_destroy.clear();
      std::thread([descs = std::move(bg_descriptors)]() mutable { descs.clear(); }).detach();
    }

    HOLOSCAN_LOG_DEBUG("{}::export_tensor: starting export seq={} (pending={})",
                       adapter_name_,
                       sequence_num,
                       pending_export_count());

    // 1. Export CUDA IPC descriptor via GXF helper
    auto maybe_desc = nvidia::gxf::export_cuda_ipc_descriptor(
        device_ptr, tensor_info, device_uuid, device_id, sequence_num, false);
    if (!maybe_desc) {
      HOLOSCAN_LOG_ERROR("{}::export_tensor: export_cuda_ipc_descriptor failed", adapter_name_);
      return nvidia::gxf::ForwardError(maybe_desc);
    }
    auto gxf_desc = std::move(maybe_desc.value());

    // 2. Register with holoipc lifecycle tracking.
    // TODO(grelee): Avoid the redundant cudaIpcGetMemHandle call that occurs here because
    // gxf_desc already contains the exported handle bytes.
    auto ipc_descriptor = ipc_context_->share_pointer(device_ptr_owner, HandleType::CUDA_IPC);
    if (!ipc_descriptor) {
      HOLOSCAN_LOG_ERROR("{}::export_tensor: share_pointer failed", adapter_name_);
      return nvidia::gxf::Unexpected(GXF_FAILURE);
    }

    // 3. Encode descriptor + lifecycle metadata
    auto maybe_bytes = encode_exported_tensor(gxf_desc, *ipc_descriptor);
    if (!maybe_bytes) {
      HOLOSCAN_LOG_ERROR("{}::export_tensor: encode_exported_tensor failed", adapter_name_);
      return nvidia::gxf::ForwardError(maybe_bytes);
    }

    // 4. Store in pending exports for lifecycle tracking
    {
      std::lock_guard<std::mutex> lock(exports_mutex_);
      const ipc::Key lifecycle_key(ipc_descriptor->key());

      // holoipc deduplicates share_pointer() by device address, so a recycled
      // buffer can produce the same lifecycle key as a still-pending export.
      // Evict the old entry now — the buffer was recycled so its data is gone
      // regardless.  This may briefly under-count pending exports (causing
      // PendingExportCondition to unblock slightly early), but that is benign
      // since the buffer pool already considers the memory available.
      auto old_it = lifecycle_key_to_seq_.find(lifecycle_key);
      if (old_it != lifecycle_key_to_seq_.end()) {
        auto old_entry_it = pending_exports_.find(old_it->second);
        if (old_entry_it != pending_exports_.end()) {
          HOLOSCAN_LOG_WARN(
              "{}::export_tensor seq={}: evicting prior export seq={} with same lifecycle key "
              "(buffer address reused before subscriber released)",
              adapter_name_,
              sequence_num,
              old_it->second);
          released_entries_.push_back(std::move(old_entry_it->second));
          pending_exports_.erase(old_entry_it);
        }
        lifecycle_key_to_seq_.erase(old_it);
      }

      lifecycle_key_to_seq_[lifecycle_key] = sequence_num;
      pending_exports_.emplace(sequence_num,
                               ExportEntry{ipc_descriptor,
                                           device_ptr_owner,
                                           gxf_desc,
                                           lifecycle_key,
                                           std::chrono::steady_clock::now()});
    }

    HOLOSCAN_LOG_DEBUG("{}::export_tensor: exported seq={}, size={} bytes, pending={}",
                       adapter_name_,
                       sequence_num,
                       gxf_desc.byte_size,
                       pending_export_count());

    return maybe_bytes.value();
  }

  // --- Subscriber side ---

  /// NativeBufferProtocolAdapter interface: delegates to import_tensor() and
  /// converts the NativeBufferHolder into a generic ImportedNativeTensor.
  nvidia::gxf::Expected<ImportedNativeTensor> import_tensor_generic(
      const std::vector<uint8_t>& descriptor_bytes, const std::string& protocol_name,
      std::chrono::milliseconds timeout = std::chrono::milliseconds{0}) override {
    if (!supports_protocol(protocol_name)) {
      HOLOSCAN_LOG_ERROR(
          "{}::import_tensor_generic: unsupported protocol '{}'", adapter_name_, protocol_name);
      return nvidia::gxf::Unexpected(GXF_NOT_IMPLEMENTED);
    }
    auto maybe_holder = import_tensor(descriptor_bytes, timeout);
    if (!maybe_holder) {
      return nvidia::gxf::ForwardError(maybe_holder);
    }

    ImportedNativeTensor imported;
    imported.mapped_ptr = std::move(maybe_holder.value().ipc_device_ptr);
    const auto& ti = maybe_holder.value().descriptor.tensor_info;
    imported.metadata.shape = ti.shape;
    imported.metadata.strides = ti.strides;
    imported.metadata.dtype = ti.dtype;
    imported.metadata.storage_type = static_cast<nvidia::gxf::MemoryStorageType>(ti.storage_type);
    imported.metadata.bytes_per_element = ti.bytes_per_element;
    return imported;
  }

  /// Decode a DNBX payload and acquire the CUDA IPC mapping via holoipc.
  /// In strict mode (default): sends ACQUIRED, waits for ACK, then opens handle.
  /// In eager mode: opens handle locally first, then sends ACQUIRED (no ACK wait).
  /// Returns an RAII NativeBufferHolder whose destruction sends RELEASED.
  nvidia::gxf::Expected<NativeBufferHolder> import_tensor(
      const std::vector<uint8_t>& descriptor_bytes,
      std::chrono::milliseconds timeout = std::chrono::milliseconds{0}) {
    if (!initialized_ || !ipc_context_) {
      HOLOSCAN_LOG_ERROR("{}::import_tensor: not initialized", adapter_name_);
      return nvidia::gxf::Unexpected(GXF_UNINITIALIZED_VALUE);
    }

    auto decoded = decode_exported_tensor(descriptor_bytes);
    if (!decoded) {
      HOLOSCAN_LOG_ERROR("{}::import_tensor: failed to decode exported descriptor", adapter_name_);
      return nvidia::gxf::ForwardError(decoded);
    }
    const auto& exported = decoded.value();

    PointerDescriptorType ipc_descriptor;
    ipc_descriptor.version(exported.ipc_protocol_version);
    ipc_descriptor.key(exported.lifecycle_key);
    ipc_descriptor.handle_type(static_cast<HandleType>(exported.handle_type));

    std::vector<uint8_t> handle_bytes(sizeof(exported.gxf_descriptor.mem_handle));
    std::memcpy(handle_bytes.data(),
                &exported.gxf_descriptor.mem_handle,
                sizeof(exported.gxf_descriptor.mem_handle));
    ipc_descriptor.handle(std::move(handle_bytes));
    ipc_descriptor.reply_to_topic_name(exported.lifecycle_reply_to_topic_name);

    if (use_eager_acquire_) {
      auto acquired_ptr = ipc_context_->acquire_pointer_eager(ipc_descriptor);
      if (!acquired_ptr) {
        HOLOSCAN_LOG_ERROR("{}::import_tensor: acquire_pointer_eager failed (seq={})",
                           adapter_name_,
                           exported.gxf_descriptor.sequence_number);
        return nvidia::gxf::Unexpected(GXF_PUBSUB_DESCRIPTOR_OPEN_FAIL);
      }

      NativeBufferHolder holder;
      holder.descriptor = exported.gxf_descriptor;
      holder.ipc_device_ptr = std::move(acquired_ptr);

      HOLOSCAN_LOG_DEBUG("{}::import_tensor: eagerly imported seq={}, size={} bytes, ptr={}",
                         adapter_name_,
                         holder.descriptor.sequence_number,
                         holder.descriptor.byte_size,
                         holder.device_ptr());
      return holder;
    }

    auto future = ipc_context_->acquire_pointer(ipc_descriptor);
    if (!future.valid()) {
      HOLOSCAN_LOG_ERROR("{}::import_tensor: acquire_pointer validation failed (seq={})",
                         adapter_name_,
                         exported.gxf_descriptor.sequence_number);
      return nvidia::gxf::Unexpected(GXF_PUBSUB_DESCRIPTOR_OPEN_FAIL);
    }

    const auto effective_timeout = timeout.count() > 0 ? timeout : acquire_timeout_;
    if (future.wait_for(effective_timeout) != std::future_status::ready) {
      HOLOSCAN_LOG_ERROR("{}::import_tensor: acquire_pointer timed out after {}ms (seq={})",
                         adapter_name_,
                         effective_timeout.count(),
                         exported.gxf_descriptor.sequence_number);
      // The ACQUIRED command may already have been sent. Drain the future in the
      // background so that when the ACK arrives, the returned shared_ptr is
      // destroyed and its deleter sends RELEASED, keeping lifecycle accounting
      // balanced.
      std::thread([f = std::move(future)]() mutable {
        try {
          (void)f.get();
        } catch (...) {
        }
      }).detach();
      return nvidia::gxf::Unexpected(GXF_PUBSUB_DESCRIPTOR_OPEN_FAIL);
    }

    std::shared_ptr<void> acquired_ptr;
    try {
      acquired_ptr = future.get();
    } catch (const std::exception& e) {
      HOLOSCAN_LOG_ERROR("{}::import_tensor: acquire_pointer failed (seq={}, error={})",
                         adapter_name_,
                         exported.gxf_descriptor.sequence_number,
                         e.what());
      return nvidia::gxf::Unexpected(GXF_PUBSUB_DESCRIPTOR_OPEN_FAIL);
    }

    if (!acquired_ptr) {
      HOLOSCAN_LOG_DEBUG("{}::import_tensor: descriptor expired before acquire completed (seq={})",
                         adapter_name_,
                         exported.gxf_descriptor.sequence_number);
      return nvidia::gxf::Unexpected(GXF_PUBSUB_DESCRIPTOR_EXPIRED);
    }

    NativeBufferHolder holder;
    holder.descriptor = exported.gxf_descriptor;
    holder.ipc_device_ptr = std::move(acquired_ptr);

    HOLOSCAN_LOG_DEBUG("{}::import_tensor: imported seq={}, size={} bytes, ptr={}",
                       adapter_name_,
                       holder.descriptor.sequence_number,
                       holder.descriptor.byte_size,
                       holder.device_ptr());
    return holder;
  }

  // --- Lifecycle ---

  size_t pending_export_count() const override {
    std::lock_guard<std::mutex> lock(exports_mutex_);
    return pending_exports_.size();
  }

  void set_on_pending_export_count_changed(PendingExportCountChangedCallback callback) override {
    std::lock_guard<std::mutex> lock(exports_mutex_);
    pending_export_count_changed_callback_ = std::move(callback);
  }

  size_t evict_stale_exports(std::chrono::milliseconds max_age) {
    std::vector<ExportEntry> deferred_destroy;
    size_t n;
    {
      std::lock_guard<std::mutex> lock(exports_mutex_);
      n = evict_stale_exports_locked(max_age, deferred_destroy);
    }
    deferred_destroy.clear();
    return n;
  }

  /// Parse DNBX wire format back into ExportedTensorDescriptor.
  /// Wire layout: magic | version | handle_type | ipc_version | hipc_blob | key | reply_topic
  static nvidia::gxf::Expected<ExportedTensorDescriptor> decode_exported_tensor(
      const std::vector<uint8_t>& descriptor_bytes) {
    return decode_exported_tensor(descriptor_bytes.data(), descriptor_bytes.size());
  }

  static nvidia::gxf::Expected<ExportedTensorDescriptor> decode_exported_tensor(const uint8_t* data,
                                                                                size_t size) {
    if (data == nullptr || size < sizeof(uint32_t) + sizeof(uint8_t) + sizeof(uint32_t) * 3) {
      HOLOSCAN_LOG_ERROR(
          "HoloIpcCudaNativeBufferAdapterBase::decode_exported_tensor: "
          "truncated descriptor wrapper");
      return nvidia::gxf::Unexpected(GXF_PUBSUB_SERIALIZATION_FAILED);
    }

    size_t offset = 0;
    auto read_field = [&data, size, &offset](auto* value) -> bool {
      using ValueType = std::decay_t<decltype(*value)>;
      if (offset + sizeof(ValueType) > size)
        return false;
      std::memcpy(value, data + offset, sizeof(ValueType));
      offset += sizeof(ValueType);
      return true;
    };

    uint32_t magic = 0;
    uint8_t version = 0;
    int32_t handle_type = 0;
    uint8_t ver_len = 0;
    uint32_t hipc_size = 0;
    uint32_t key_size = 0;
    uint32_t reply_size = 0;
    if (!read_field(&magic) || !read_field(&version)) {
      return nvidia::gxf::Unexpected(GXF_PUBSUB_SERIALIZATION_FAILED);
    }

    if (magic != kExportedTensorMagic || version != kExportedTensorVersion) {
      HOLOSCAN_LOG_ERROR(
          "HoloIpcCudaNativeBufferAdapterBase::decode_exported_tensor: "
          "invalid wrapper header (magic=0x{:08x}, version={})",
          magic,
          static_cast<int>(version));
      return nvidia::gxf::Unexpected(GXF_PUBSUB_SERIALIZATION_FAILED);
    }

    if (!read_field(&handle_type) || !read_field(&ver_len) || offset + ver_len > size) {
      return nvidia::gxf::Unexpected(GXF_PUBSUB_SERIALIZATION_FAILED);
    }
    std::string ipc_protocol_version(reinterpret_cast<const char*>(data + offset), ver_len);
    offset += ver_len;

    if (!read_field(&hipc_size) || offset + hipc_size > size) {
      return nvidia::gxf::Unexpected(GXF_PUBSUB_SERIALIZATION_FAILED);
    }
    auto maybe_desc = nvidia::gxf::deserialize_cuda_ipc_descriptor(data + offset, hipc_size);
    if (!maybe_desc) {
      return nvidia::gxf::ForwardError(maybe_desc);
    }
    offset += hipc_size;

    if (!read_field(&key_size) || offset + key_size > size) {
      return nvidia::gxf::Unexpected(GXF_PUBSUB_SERIALIZATION_FAILED);
    }

    ExportedTensorDescriptor result;
    result.gxf_descriptor = std::move(maybe_desc.value());
    result.handle_type = handle_type;
    result.ipc_protocol_version = std::move(ipc_protocol_version);
    result.lifecycle_key.assign(data + offset, data + offset + key_size);
    offset += key_size;
    if (!read_field(&reply_size) || offset + reply_size > size) {
      return nvidia::gxf::Unexpected(GXF_PUBSUB_SERIALIZATION_FAILED);
    }
    result.lifecycle_reply_to_topic_name.assign(reinterpret_cast<const char*>(data + offset),
                                                reply_size);

    if (result.lifecycle_key.empty() || result.lifecycle_reply_to_topic_name.empty()) {
      HOLOSCAN_LOG_ERROR(
          "HoloIpcCudaNativeBufferAdapterBase::decode_exported_tensor: "
          "missing lifecycle key or reply topic");
      return nvidia::gxf::Unexpected(GXF_PUBSUB_SERIALIZATION_FAILED);
    }
    return result;
  }

 protected:
  /// Called by derived-class initialize() after creating the ipc::Context.
  nvidia::gxf::Expected<void> initialize_common(std::shared_ptr<IpcContextType> ipc_context,
                                                nvidia::gxf::NativeBufferPolicy policy) {
    if (initialized_) {
      HOLOSCAN_LOG_WARN("{}::initialize: already initialized", adapter_name_);
      return nvidia::gxf::Expected<void>();
    }

    policy_ = policy;

    if (policy_ == nvidia::gxf::NativeBufferPolicy::kDisabled) {
      HOLOSCAN_LOG_INFO("{}: native buffer policy is disabled", adapter_name_);
      initialized_ = true;
      return nvidia::gxf::Expected<void>();
    }

    ipc_context_ = std::move(ipc_context);
    if (!ipc_context_) {
      HOLOSCAN_LOG_ERROR("{}::initialize: null ipc::Context", adapter_name_);
      return nvidia::gxf::Unexpected(GXF_FAILURE);
    }
    ipc_context_->set_on_last_release([this](const ipc::Key& key) { handle_last_release(key); });

    initialized_ = true;
    HOLOSCAN_LOG_INFO(
        "{}: initialized (policy={}, protocol=cuda_ipc, eager_acquire={})",
        adapter_name_,
        policy_ == nvidia::gxf::NativeBufferPolicy::kPreferred ? "preferred" : "required",
        use_eager_acquire_);
    return nvidia::gxf::Expected<void>();
  }

  std::string adapter_name_;

 private:
  /// Tracks a single exported tensor for lifecycle management.
  /// Holds the holoipc descriptor (controls RELEASED on drop), the GPU memory owner
  /// (prevents cudaFree while subscribers hold the mapping), and timing for TTL eviction.
  struct ExportEntry {
    std::shared_ptr<PointerDescriptorType> ipc_descriptor;
    std::shared_ptr<void> device_ptr_owner;
    nvidia::gxf::CudaIpcDescriptor gxf_descriptor;
    ipc::Key lifecycle_key;
    std::chrono::steady_clock::time_point export_time;
  };

  static constexpr uint32_t kExportedTensorMagic = 0x58424E44;  // "DNBX" little-endian
  static constexpr uint8_t kExportedTensorVersion = 1;
  static constexpr auto kShutdownCleanupSleep = std::chrono::milliseconds{10};
  static constexpr auto kDefaultShutdownGrace = std::chrono::milliseconds{250};

  template <typename T>
  static void append_bytes(std::vector<uint8_t>& out, const T& value) {
    const auto* bytes = reinterpret_cast<const uint8_t*>(&value);
    out.insert(out.end(), bytes, bytes + sizeof(T));
  }

  static void append_buffer(std::vector<uint8_t>& out, const uint8_t* data, size_t sz) {
    out.insert(out.end(), data, data + sz);
  }

  void ensure_gpu_info_initialized() {
    if (gpu_info_initialized_) {
      return;
    }

    int device_id = 0;
    const cudaError_t err = cudaGetDevice(&device_id);
    if (err == cudaSuccess) {
      gpu_device_id_ = device_id;
      gpu_device_uuid_ = nvidia::gxf::CudaDeviceIpcInfo::query(device_id).uuid;
    }
    gpu_info_initialized_ = true;
  }

  /// Serialize GXF CudaIpcDescriptor + holoipc lifecycle metadata into DNBX wire format.
  static nvidia::gxf::Expected<std::vector<uint8_t>> encode_exported_tensor(
      const nvidia::gxf::CudaIpcDescriptor& gxf_descriptor,
      const PointerDescriptorType& ipc_descriptor) {
    auto maybe_hipc = nvidia::gxf::serialize_cuda_ipc_descriptor(gxf_descriptor);
    if (!maybe_hipc) {
      return nvidia::gxf::ForwardError(maybe_hipc);
    }

    const auto& hipc_bytes = maybe_hipc.value();
    const auto& key = ipc_descriptor.key();
    const auto& reply_to_topic_name = ipc_descriptor.reply_to_topic_name();
    if (key.empty() || reply_to_topic_name.empty()) {
      HOLOSCAN_LOG_ERROR(
          "HoloIpcCudaNativeBufferAdapterBase::encode_exported_tensor: "
          "missing lifecycle metadata");
      return nvidia::gxf::Unexpected(GXF_PUBSUB_SERIALIZATION_FAILED);
    }

    const auto& version_str = std::string(holoscan::ipc::kIpcProtocolVersion);
    const int32_t handle_type = static_cast<int32_t>(ipc_descriptor.handle_type());

    std::vector<uint8_t> out;
    out.reserve(sizeof(kExportedTensorMagic) + sizeof(kExportedTensorVersion) + sizeof(int32_t) +
                sizeof(uint8_t) + version_str.size() + sizeof(uint32_t) * 3 + hipc_bytes.size() +
                key.size() + reply_to_topic_name.size());
    append_bytes(out, kExportedTensorMagic);
    append_bytes(out, kExportedTensorVersion);

    append_bytes(out, handle_type);
    const uint8_t ver_len = static_cast<uint8_t>(version_str.size());
    append_bytes(out, ver_len);
    append_buffer(out, reinterpret_cast<const uint8_t*>(version_str.data()), version_str.size());

    const uint32_t hipc_size = static_cast<uint32_t>(hipc_bytes.size());
    const uint32_t key_size = static_cast<uint32_t>(key.size());
    const uint32_t reply_size = static_cast<uint32_t>(reply_to_topic_name.size());

    append_bytes(out, hipc_size);
    append_buffer(out, hipc_bytes.data(), hipc_bytes.size());
    append_bytes(out, key_size);
    append_buffer(out, key.data(), key.size());
    append_bytes(out, reply_size);
    append_buffer(out,
                  reinterpret_cast<const uint8_t*>(reply_to_topic_name.data()),
                  reply_to_topic_name.size());

    return out;
  }

  /// holoipc on_last_release callback (runs on io_context thread).
  /// Moves the released entry to released_entries_ for deferred destruction;
  /// must not destroy ExportEntry here (would deadlock the io_context).
  void handle_last_release(const ipc::Key& key) {
    PendingExportCountChangedCallback callback;
    size_t pending_after = 0;
    bool pending_changed = false;
    {
      std::lock_guard<std::mutex> lock(exports_mutex_);
      auto idx_it = lifecycle_key_to_seq_.find(key);
      if (idx_it != lifecycle_key_to_seq_.end()) {
        auto entry_it = pending_exports_.find(idx_it->second);
        if (entry_it != pending_exports_.end()) {
          HOLOSCAN_LOG_DEBUG("{}: RELEASED seq={} (pending_after={}) [deferred to worker]",
                             adapter_name_,
                             idx_it->second,
                             pending_exports_.size() - 1);
          released_entries_.push_back(std::move(entry_it->second));
          pending_exports_.erase(entry_it);
          pending_after = pending_exports_.size();
          callback = pending_export_count_changed_callback_;
          pending_changed = true;
        }
        lifecycle_key_to_seq_.erase(idx_it);
      } else {
        HOLOSCAN_LOG_DEBUG("{}: RELEASED for unknown key (already evicted?)", adapter_name_);
      }
    }
    if (pending_changed && callback) {
      callback(pending_after);
    }
  }

  /// Move exports older than max_age into deferred_destroy. Caller holds exports_mutex_.
  size_t evict_stale_exports_locked(std::chrono::milliseconds max_age,
                                    std::vector<ExportEntry>& deferred_destroy) {
    auto now = std::chrono::steady_clock::now();
    size_t evicted = 0;

    for (auto it = pending_exports_.begin(); it != pending_exports_.end();) {
      auto age =
          std::chrono::duration_cast<std::chrono::milliseconds>(now - it->second.export_time);
      if (age > max_age) {
        HOLOSCAN_LOG_WARN(
            "{}: evicting stale export seq={} (age={}ms) -- "
            "if this happens frequently, increase export_ttl or investigate "
            "subscriber disconnects",
            adapter_name_,
            it->first,
            age.count());
        lifecycle_key_to_seq_.erase(it->second.lifecycle_key);
        deferred_destroy.push_back(std::move(it->second));
        it = pending_exports_.erase(it);
        ++evicted;
      } else {
        ++it;
      }
    }

    if (evicted > 0) {
      HOLOSCAN_LOG_DEBUG("{}: evicted {} stale exports", adapter_name_, evicted);
    }
    return evicted;
  }

  /// Drain all pending exports during shutdown, waiting up to shutdown_grace_period
  /// for subscribers to send RELEASED before force-evicting.
  void cleanup_exports(std::chrono::milliseconds shutdown_grace_period) {
    const auto deadline = std::chrono::steady_clock::now() + shutdown_grace_period;
    while (true) {
      std::vector<ExportEntry> deferred_destroy;
      size_t remaining = 0;
      {
        std::lock_guard<std::mutex> lock(exports_mutex_);
        if (!released_entries_.empty()) {
          deferred_destroy.insert(deferred_destroy.end(),
                                  std::make_move_iterator(released_entries_.begin()),
                                  std::make_move_iterator(released_entries_.end()));
          released_entries_.clear();
        }
        evict_stale_exports_locked(export_ttl_, deferred_destroy);
        remaining = pending_exports_.size();
        if (remaining == 0 || std::chrono::steady_clock::now() >= deadline) {
          if (remaining > 0) {
            HOLOSCAN_LOG_WARN(
                "{}::shutdown: {} pending exports still outstanding", adapter_name_, remaining);
          }
          for (auto& entry : pending_exports_) {
            deferred_destroy.push_back(std::move(entry.second));
          }
          pending_exports_.clear();
          lifecycle_key_to_seq_.clear();
        }
      }
      // Same background-thread pattern as export_tensor: free GPU mem inline,
      // defer ipc_descriptor destruction to avoid blocking on IpcCore::mutex_.
      {
        std::vector<std::shared_ptr<PointerDescriptorType>> bg_descriptors;
        bg_descriptors.reserve(deferred_destroy.size());
        for (auto& entry : deferred_destroy) {
          entry.device_ptr_owner.reset();
          bg_descriptors.push_back(std::move(entry.ipc_descriptor));
        }
        deferred_destroy.clear();
        if (!bg_descriptors.empty()) {
          std::thread([d = std::move(bg_descriptors)]() mutable { d.clear(); }).detach();
        }
      }
      if (remaining == 0 || std::chrono::steady_clock::now() >= deadline) {
        return;
      }
      std::this_thread::sleep_for(kShutdownCleanupSleep);
    }
  }

  std::shared_ptr<IpcContextType> ipc_context_;
  nvidia::gxf::NativeBufferPolicy policy_{nvidia::gxf::NativeBufferPolicy::kPreferred};
  bool initialized_ = false;
  std::string gpu_device_uuid_;
  int32_t gpu_device_id_{0};
  bool gpu_info_initialized_{false};
  std::chrono::milliseconds acquire_timeout_{500};
  bool use_eager_acquire_{false};
  // TTL is a safety net for error recovery (e.g. subscriber disconnected without
  // sending RELEASED). Normal cleanup is via the on_last_release callback.
  std::chrono::milliseconds export_ttl_{20000};
  std::unordered_map<uint32_t, ExportEntry> pending_exports_;  // seq_num -> entry
  std::map<ipc::Key, uint32_t> lifecycle_key_to_seq_;          // reverse index for releases
  std::vector<ExportEntry> released_entries_;  // queued by io_context, drained by worker
  PendingExportCountChangedCallback pending_export_count_changed_callback_;
  mutable std::mutex exports_mutex_;
};

}  // namespace holoscan

#endif /* PUBSUB_COMMON_INCLUDE_PUBSUB_HOLOIPC_CUDA_NATIVE_BUFFER_ADAPTER_BASE_HPP */
