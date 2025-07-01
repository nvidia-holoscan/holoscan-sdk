/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef HOLOSCAN_CORE_FRAGMENT_SERVICE_HPP
#define HOLOSCAN_CORE_FRAGMENT_SERVICE_HPP

#include <memory>
#include <string>
#include <typeindex>  // for std::type_index

namespace holoscan {

// Forward declaration
class Resource;

/**
 * @brief Base interface for services that enable sharing of resources and functionality between
 * operators.
 */
class FragmentService {
 public:
  FragmentService() = default;
  virtual ~FragmentService() = default;

  /**
   * @brief Get the underlying resource.
   * @return Shared pointer to the resource.
   */
  virtual std::shared_ptr<Resource> resource() const = 0;

  /**
   * @brief Set the underlying resource.
   */
  virtual void resource(const std::shared_ptr<Resource>& resource) = 0;
};

/**
 * @brief Base class for services to be registered in the fragment service registry.
 *
 * Fragment services provide a way to share resources and functionality across
 * operators within a fragment or application.
 */
class DefaultFragmentService : public FragmentService {
 public:
  DefaultFragmentService() = default;
  explicit DefaultFragmentService(const std::shared_ptr<Resource>& resource);
  ~DefaultFragmentService() = default;

  // Explicitly delete copy operations to prevent accidental copying
  DefaultFragmentService(const DefaultFragmentService&) = delete;
  DefaultFragmentService& operator=(const DefaultFragmentService&) = delete;

  // Allow move operations
  DefaultFragmentService(DefaultFragmentService&&) = default;
  DefaultFragmentService& operator=(DefaultFragmentService&&) = default;

  /**
   * @brief Get the resource cast to a specific type.
   * @tparam ResourceT The type to cast the resource to.
   * @return Shared pointer to the resource cast to ResourceT, or nullptr if cast fails.
   */
  template <typename ResourceT>
  std::shared_ptr<ResourceT> resource() const {
    return std::dynamic_pointer_cast<ResourceT>(resource_);
  }

  /**
   * @brief Get the underlying resource.
   * @return Shared pointer to the resource.
   */
  std::shared_ptr<Resource> resource() const override { return resource_; }

  /**
   * @brief Set the underlying resource.
   */
  void resource(const std::shared_ptr<Resource>& resource) override { resource_ = resource; }

 protected:
  friend class Fragment;
  std::shared_ptr<Resource> resource_;
};

/**
 * @brief Key structure for service registry that combines type and identifier.
 */
struct ServiceKey {
  std::type_index type;  ///< Type of the service
  std::string id;        ///< Service identifier (empty string for default instance)

  /**
   * @brief Equality comparison operator.
   */
  bool operator==(const ServiceKey& other) const noexcept {
    return type == other.type && id == other.id;
  }
};

/**
 * @brief Hash function for ServiceKey to enable use in hash-based containers.
 */
struct ServiceKeyHash {
  /**
   * @brief Compute hash value for a ServiceKey.
   */
  std::size_t operator()(const ServiceKey& key) const noexcept {
    return std::hash<std::type_index>{}(key.type) ^ std::hash<std::string>{}(key.id);
  }
};

}  // namespace holoscan

#endif /* HOLOSCAN_CORE_FRAGMENT_SERVICE_HPP */
