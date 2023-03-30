/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef HOLOSCAN_CORE_GXF_GXF_EXTENSION_REGISTRAR_HPP
#define HOLOSCAN_CORE_GXF_GXF_EXTENSION_REGISTRAR_HPP

#include <gxf/core/gxf.h>

#include <random>
#include <set>
#include <memory>

#include <gxf/std/default_extension.hpp>
#include "../common.hpp"

namespace holoscan::gxf {

/**
 * @brief Class to register a GXF extension.
 *
 * This class is a helper class to register a GXF extension.
 * `GXFLoadExtensionFromPointer()` API is used to register the extension programmatically.
 */
class GXFExtensionRegistrar {
 public:
  /**
   * @brief Kind of the Type.
   */
  enum class TypeKind {
    kExtension,  ///< Extension
    kComponent,  ///< Component
  };

  GXFExtensionRegistrar() = delete;

  /**
   * @brief Construct a new GXFExtensionRegistrar object.
   *
   * If `tid` is not provided, a random `tid` is generated and used to register the extension.
   *
   * @param context The pointer to the GXF context.
   * @param extension_name The name of the extension.
   * @param tid The type ID of the extension to use.
   */
  explicit GXFExtensionRegistrar(gxf_context_t context, const char* extension_name,
                                 const char* extension_description = "", gxf_tid_t tid = {0, 0}) {
    reset(context, extension_name, extension_description, tid);
  }

  /**
   * @brief Create a random tid object.
   *
   * Generate a sudo random tid using std::mt19937_64.
   * This implementation does not guarantee that the tid is unique.
   *
   * @return A random tid.
   */
  static gxf_tid_t create_random_tid() {
    std::random_device device;
    std::mt19937_64 rng(device());
    std::uniform_int_distribution<uint64_t> dist;
    gxf_tid_t tid = {dist(rng), dist(rng)};
    return tid;
  }

  /**
   * @brief Check if the given tid is already allocated.
   *
   * @param tid The tid to check.
   * @param kind The kind of the tid to check.
   * @return true If the tid is already allocated. Otherwise, false.
   */
  bool is_allocated(gxf_tid_t tid, TypeKind kind) const {
    switch (kind) {
      case TypeKind::kExtension: {
        gxf_extension_info_t extension_info;
        auto result = GxfExtensionInfo(context_, tid, &extension_info);
        if (!result) { return false; }
        break;
      }
      case TypeKind::kComponent: {
        gxf_component_info_t component_info;
        auto result = GxfComponentInfo(context_, tid, &component_info);
        if (!result) { return false; }
        break;
      }
    }

    return allocated_tids_.find(tid) != allocated_tids_.end();
  }

  /**
   * @brief Create a random tid that is not allocated.
   *
   * @param kind The kind of the tid to create.
   * @return The allocated tid.
   */
  gxf_tid_t allocate_tid(TypeKind kind) {
    gxf_tid_t tid = create_random_tid();
    while (is_allocated(tid, kind)) { tid = create_random_tid(); }
    return tid;
  }

  /**
   * @brief Add a component to the extension.
   *
   * If `tid` is not provided, a random `tid` is generated and used to register the component.
   *
   * @tparam T The type of the component.
   * @tparam Base The base type of the component.
   * @param description The description of the component.
   * @param tid The type ID of the component to use.
   * @return true If the component is added successfully. Otherwise, false.
   */
  template <typename T, typename Base>
  bool add_component(const char* description = "", gxf_tid_t tid = {0, 0}) {
    if (tid == GxfTidNull() || is_allocated(tid, TypeKind::kComponent)) {
      tid = allocate_tid(TypeKind::kComponent);
    }

    allocated_tids_.insert(tid);

    const nvidia::gxf::Expected<void> result = factory_->add<T, Base>(tid, description);
    if (!result) {
      HOLOSCAN_LOG_ERROR("Unable to add component to the GXF extension: {}", result.error());
      return false;
    }

    return true;
  }

  /**
   * @brief Add a type to the extension.
   *
   * If `tid` is not provided, a random `tid` is generated and used to register the type.
   *
   * @tparam T The type to add.
   * @param description The description of the type.
   * @param tid The type ID of the type to use.
   * @return true If the type is added successfully. Otherwise, false.
   */
  template <typename T>
  bool add_type(const char* description = "", gxf_tid_t tid = {0, 0}) {
    if (tid == GxfTidNull() || is_allocated(tid, TypeKind::kComponent)) {
      tid = allocate_tid(TypeKind::kComponent);
    }

    allocated_tids_.insert(tid);

    const nvidia::gxf::Expected<void> result = factory_->add<T>(tid, description);
    if (!result) {
      HOLOSCAN_LOG_ERROR("Unable to add type to the GXF extension: {}", result.error());
      return false;
    }

    return true;
  }

  /**
   * @brief Register the extension.
   *
   * @param out_extension_ptr If provided, the pointer to the extension is set to this pointer.
   * @return true If the extension is registered successfully. Otherwise, false.
   */
  bool register_extension(nvidia::gxf::Extension** out_extension_ptr = nullptr) {
    if (!factory_) {
      HOLOSCAN_LOG_ERROR("GXF Extension factory is not initialized");
      return false;
    }

    auto check_result = factory_->checkInfo();
    if (!check_result) {
      HOLOSCAN_LOG_ERROR("Failed to check the GXF extension information: {}", check_result.error());
      return false;
    }

    nvidia::gxf::Extension* extension = factory_.release();

    // Set the extension pointer if provided.
    if (out_extension_ptr != nullptr) {
      if (extension != nullptr) {
        *out_extension_ptr = extension;
      } else {
        *out_extension_ptr = nullptr;
      }
    }

    gxf_result_t result = GxfLoadExtensionFromPointer(context_, extension);
    if (result != GXF_SUCCESS) {
      HOLOSCAN_LOG_ERROR("Unable to register the GXF extension: {}", GxfResultStr(result));
      return false;
    }
    return true;
  }

  /**
   * @brief Reset the GXFExtensionRegistrar object.
   *
   * If `tid` is not provided, a random `tid` is generated and used to register the extension.
   *
   * @param context The pointer to the GXF context.
   * @param extension_name The name of the extension.
   * @param tid The type ID of the extension to use.
   */
  void reset(gxf_context_t context, const char* extension_name,
            const char* extension_description = "", gxf_tid_t tid = {0, 0}) {
    context_ = context;
    factory_ = std::make_unique<nvidia::gxf::DefaultExtension>();
    allocated_tids_.clear();

    if (tid == GxfTidNull() || is_allocated(tid, TypeKind::kExtension)) {
      tid = allocate_tid(TypeKind::kExtension);
    }

    allocated_tids_.insert(tid);
    extension_tid_ = tid;

    if (!factory_) {
      HOLOSCAN_LOG_ERROR("Error creating GXF extension factory");
      return;
    }

    // Set the extension information.
    const nvidia::gxf::Expected<void> result = factory_->setInfo(extension_tid_,
                                                                 extension_name,
                                                                 extension_description,
                                                                 "NVIDIA",
                                                                 "1.0.0",
                                                                 "Apache 2.0");
    if (!result) {
      HOLOSCAN_LOG_ERROR("Unable to set the GXF extension information: {}", result.error());
      return;
    }
  }

 private:
  gxf_context_t context_ = nullptr;
  std::unique_ptr<nvidia::gxf::DefaultExtension> factory_;
  std::set<gxf_tid_t> allocated_tids_;
  gxf_tid_t extension_tid_ = {0, 0};
};

}  // namespace holoscan::gxf

#endif /* HOLOSCAN_CORE_GXF_GXF_EXTENSION_REGISTRAR_HPP */
