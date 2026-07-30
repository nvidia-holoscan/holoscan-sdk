/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOSCAN_CORE_GXF_ENTITY_HPP
#define HOLOSCAN_CORE_GXF_ENTITY_HPP

#include <cuda_runtime_api.h>

#include <memory>
#include <mutex>
#include <optional>
#include <utility>

// Entity definition
// Since it has code that causes a warning as an error, we disable it here.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#include <gxf/core/entity.hpp>
#pragma GCC diagnostic pop

#include <gxf/multimedia/video.hpp>
#include <gxf/std/tensor.hpp>
#include <holoscan/core/gxf/gxf_utils.hpp>
#include <holoscan/core/type_traits.hpp>

// Forward declaration
namespace holoscan {
class ExecutionContext;
}

namespace holoscan::gxf {

/**
 * @brief Class to wrap GXF Entity (`nvidia::gxf::Entity`).
 *
 */
class Entity : public nvidia::gxf::Entity {
 public:
  Entity() = default;
  explicit Entity(const nvidia::gxf::Entity& other) : nvidia::gxf::Entity(other) {}
  explicit Entity(nvidia::gxf::Entity&& other) : nvidia::gxf::Entity(std::move(other)) {}

  // Creates a new entity
  static Entity New(ExecutionContext* context);

  /**
   * @brief Return true if the entity is not null.
   *
   * Calling this method on an entity object from {cpp:func}`holoscan::IOContext::receive` will
   * return false if there is no entity to receive.
   *
   * @return true if the entity is not null. Otherwise, false.
   */
  operator bool() const { return !is_null(); }

  // Gets a component by type. Asserts if no such component.
  template <typename DataT,
            typename = std::enable_if_t<!holoscan::is_vector_v<DataT> &&
                                        holoscan::is_one_of_v<DataT, holoscan::Tensor>>>
  std::shared_ptr<DataT> get(const char* name = nullptr, bool log_errors = true) const {
    // We should use nullptr as a default name because In GXF, 'nullptr' should be used with
    // GxfComponentFind() if we want to get the first component of the given type.

    // Use cached type ID for nvidia::gxf::Tensor to avoid repeated lookups (thread-safe)
    static std::once_flag tid_init_flag;
    static gxf_tid_t tid;
    static gxf_result_t tid_init_result = GXF_SUCCESS;
    std::call_once(tid_init_flag, [this]() {
      tid_init_result =
          GxfComponentTypeId(context(), nvidia::TypenameAsString<nvidia::gxf::Tensor>(), &tid);
    });
    if (tid_init_result != GXF_SUCCESS) {
      if (log_errors) {
        HOLOSCAN_LOG_ERROR("Unable to get component type id from 'nvidia::gxf::Tensor' (error: {})",
                           GxfResultStr(tid_init_result));
      }
      return nullptr;
    }

    gxf_uid_t cid;
    auto cid_result = GxfComponentFind(context(), eid(), tid, name, nullptr, &cid);
    if (cid_result != GXF_SUCCESS) {
      if (log_errors) {
        HOLOSCAN_LOG_ERROR("Unable to find component from the name '{}' (error: {})",
                           name == nullptr ? "" : name,
                           GxfResultStr(cid_result));
      }
      return nullptr;
    }

    // Create a holoscan::Tensor object from the newly constructed GXF Tensor object. (~680 ns)
    auto handle = nvidia::gxf::Handle<nvidia::gxf::Tensor>::Create(context(), cid);

    auto maybe_dl_ctx = (*handle->get()).toDLManagedTensorContext();
    if (!maybe_dl_ctx) {
      HOLOSCAN_LOG_ERROR(
          "Failed to get std::shared_ptr<DLManagedTensorContext> from nvidia::gxf::Tensor");
      return nullptr;
    }
    auto dl_ctx = maybe_dl_ctx.value();
    // Get MemoryBuffer pointer for stream-aware deallocation support
    auto* mem_buf_ptr = static_cast<nvidia::gxf::MemoryBuffer*>(dl_ctx->memory_ref.get());
    std::shared_ptr<Tensor> tensor = std::make_shared<Tensor>(dl_ctx, mem_buf_ptr);
    return tensor;
  }

  /**
   * @brief Adds a tensor component to the entity.
   *
   * @tparam DataT The data type (must be holoscan::Tensor).
   * @param data Shared pointer to the tensor data.
   * @param name Optional name for the component.
   * @param stream Optional CUDA stream for stream-aware memory deallocation. When provided,
   *               the stream is set on the tensor's memory buffer, enabling allocators like
   *               BlockMemoryPool to defer memory reuse until GPU operations complete.
   */
  template <typename DataT,
            typename = std::enable_if_t<!holoscan::is_vector_v<DataT> &&
                                        holoscan::is_one_of_v<DataT, holoscan::Tensor>>>
  void add(const std::shared_ptr<DataT>& data, const char* name = nullptr,
           std::optional<cudaStream_t> stream = std::nullopt) {
    // Use cached type ID for nvidia::gxf::Tensor to avoid repeated lookups (thread-safe)
    static std::once_flag tid_init_flag;
    static gxf_tid_t tid;
    std::call_once(tid_init_flag, [this]() {
      HOLOSCAN_GXF_CALL_FATAL(
          GxfComponentTypeId(context(), nvidia::TypenameAsString<nvidia::gxf::Tensor>(), &tid));
    });

    gxf_uid_t cid;
    HOLOSCAN_GXF_CALL_FATAL(GxfComponentAdd(context(), eid(), tid, name, &cid));

    auto handle = nvidia::gxf::Handle<nvidia::gxf::Tensor>::Create(context(), cid);
    nvidia::gxf::Tensor* tensor_ptr = handle->get();

    // Copy the member data (std::shared_ptr<DLManagedTensorContext>) from the Tensor to the
    // nvidia::gxf::Tensor
    *tensor_ptr = nvidia::gxf::Tensor(data->dl_ctx());

    // Set stream on memory buffer for stream-aware deallocation if provided
    if (stream.has_value()) {
      tensor_ptr->memory_buffer().setStream(static_cast<void*>(stream.value()));
    }
  }
};

// Modified version of the Tensor version of gxf::Entity::get
// Retrieves a VideoBuffer instead
// TODO(unknown): Support gxf::VideoBuffer natively in Holoscan
nvidia::gxf::Handle<nvidia::gxf::VideoBuffer> get_videobuffer(Entity entity,
                                                              const char* name = nullptr);

}  // namespace holoscan::gxf

#endif /* HOLOSCAN_CORE_GXF_ENTITY_HPP */
