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
#include <string_view>
#include <typeindex>  // for std::type_index

namespace holoscan {

// Forward declaration
class Resource;

/**
 * @brief Base interface for services that enable sharing of resources and functionality between
 * operators within a fragment or across distributed fragments.
 *
 * FragmentService provides a common interface for services that manage shared resources
 * accessible to multiple operators. Services implementing this interface can be registered
 * with a Fragment and accessed by operators through the Fragment's service registry.
 *
 * This interface is typically implemented by resource managers, shared data structures,
 * or other facilities that need to be shared across operators while maintaining a single
 * instance per fragment.
 */
class FragmentService {
 public:
  FragmentService() = default;
  virtual ~FragmentService() = default;

  /**
   * @brief Get the underlying resource managed by this service.
   *
   * @return Shared pointer to the managed resource.
   */
  virtual std::shared_ptr<Resource> resource() const = 0;

  /**
   * @brief Set the underlying resource managed by this service.
   *
   * @param resource Shared pointer to the resource to be managed.
   */
  virtual void resource(const std::shared_ptr<Resource>& resource) = 0;
};

/**
 * @brief Default implementation of FragmentService for basic resource sharing.
 *
 * DefaultFragmentService provides a straightforward implementation of the FragmentService
 * interface. It manages a single Resource instance and provides type-safe access methods.
 *
 * This class is designed to be used directly for simple resource sharing scenarios or
 * as a base class for more specialized service implementations.
 *
 * @note This class is non-copyable to prevent accidental duplication of the managed resource.
 * Move operations are supported to allow transfer of ownership.
 */
class DefaultFragmentService : public FragmentService {
 public:
  DefaultFragmentService() = default;

  /**
   * @brief Construct a service managing the specified resource.
   *
   * @param resource The resource to be managed by this service.
   */
  explicit DefaultFragmentService(const std::shared_ptr<Resource>& resource);
  ~DefaultFragmentService() = default;

  // Explicitly delete copy operations to prevent accidental copying
  DefaultFragmentService(const DefaultFragmentService&) = delete;
  DefaultFragmentService& operator=(const DefaultFragmentService&) = delete;

  // Allow move operations for ownership transfer
  DefaultFragmentService(DefaultFragmentService&&) = default;
  DefaultFragmentService& operator=(DefaultFragmentService&&) = default;

  /**
   * @brief Get the resource cast to a specific type.
   *
   * This templated method provides type-safe access to the managed resource.
   * It attempts to cast the resource to the requested type using dynamic_pointer_cast.
   *
   * @tparam ResourceT The type to cast the resource to. Must be derived from Resource.
   * @return Shared pointer to the resource cast to ResourceT, or nullptr if the cast fails.
   */
  template <typename ResourceT>
  std::shared_ptr<ResourceT> resource() const {
    return std::dynamic_pointer_cast<ResourceT>(resource_);
  }

  /**
   * @brief Get the underlying resource without type casting.
   *
   * @return Shared pointer to the managed resource.
   */
  std::shared_ptr<Resource> resource() const override { return resource_; }

  /**
   * @brief Set the underlying resource.
   *
   * @param resource The resource to be managed by this service.
   */
  void resource(const std::shared_ptr<Resource>& resource) override { resource_ = resource; }

 protected:
  friend class Fragment;
  std::shared_ptr<Resource> resource_;  ///< The managed resource instance
};

/**
 * @brief Key structure for service registry that combines type and identifier.
 *
 * ServiceKey uniquely identifies a service in the fragment's service registry by
 * combining the type information with an optional string identifier. This allows
 * multiple instances of the same service type to coexist with different identifiers.
 */
struct ServiceKey {
  std::type_index type;  ///< Type information of the service
  std::string id;        ///< Service identifier (empty string indicates default instance)

  /**
   * @brief Equality comparison operator.
   *
   * Two ServiceKeys are equal if both their type and id match.
   *
   * @param other The ServiceKey to compare with.
   * @return true if the keys are equal, false otherwise.
   */
  bool operator==(const ServiceKey& other) const noexcept {
    return type == other.type && id == other.id;
  }
};

/**
 * @brief Hash function for ServiceKey to enable use in hash-based containers.
 *
 * This functor provides a hash function for ServiceKey objects, allowing them
 * to be used as keys in std::unordered_map and similar containers.
 */
struct ServiceKeyHash {
  /**
   * @brief Compute hash value for a ServiceKey.
   *
   * The hash is computed by XORing the hashes of the type_index and string id.
   *
   * @param key The ServiceKey to hash.
   * @return Hash value for the key.
   */
  std::size_t operator()(const ServiceKey& key) const noexcept {
    return std::hash<std::type_index>{}(key.type) ^ std::hash<std::string>{}(key.id);
  }
};

namespace distributed {

/**
 * @brief Interface for services that act as driver endpoints in distributed applications.
 *
 * ServiceDriverEndpoint defines the interface for services that need to coordinate
 * distributed operations from a driver fragment. The driver typically acts as a
 * central coordinator for distributed services across multiple worker fragments.
 */
class ServiceDriverEndpoint {
 public:
  ServiceDriverEndpoint() = default;
  virtual ~ServiceDriverEndpoint() = default;

  /**
   * @brief Start the driver endpoint for distributed coordination.
   *
   * This method is called by the framework on the driver fragment when the
   * distributed application starts. Implementations should initialize any
   * necessary resources for coordinating with worker fragments.
   *
   * @param driver_ip The IP address of the driver fragment.
   */
  virtual void driver_start(std::string_view driver_ip) = 0;

  /**
   * @brief Shutdown the driver endpoint.
   *
   * This method is called by the framework when the distributed application
   * is shutting down. Implementations should clean up resources and notify
   * any connected workers of the shutdown.
   */
  virtual void driver_shutdown() = 0;
};

/**
 * @brief Interface for services that act as worker endpoints in distributed applications.
 *
 * ServiceWorkerEndpoint defines the interface for services that participate as
 * workers in a distributed application. Workers typically connect to a driver
 * fragment to participate in distributed coordination.
 */
class ServiceWorkerEndpoint {
 public:
  ServiceWorkerEndpoint() = default;
  virtual ~ServiceWorkerEndpoint() = default;

  /**
   * @brief Connect the worker endpoint to the driver.
   *
   * This method is called by the framework on worker fragments to establish
   * a connection with the driver fragment. Implementations should connect to
   * the driver and prepare to participate in distributed operations.
   *
   * @param driver_ip The IP address of the driver fragment to connect to.
   */
  virtual void worker_connect(std::string_view driver_ip) = 0;

  /**
   * @brief Disconnect the worker endpoint from the driver.
   *
   * This method is called by the framework when the worker needs to disconnect
   * from the driver. Implementations should clean up the connection and any
   * associated resources.
   */
  virtual void worker_disconnect() = 0;
};
}  // namespace distributed

/**
 * @brief Composite service interface for distributed fragment services.
 *
 * DistributedAppService combines the FragmentService interface with distributed endpoint
 * capabilities, allowing a single service implementation to:
 * - Manage shared resources within a fragment (FragmentService)
 * - Act as a driver coordinator in distributed applications (ServiceDriverEndpoint)
 * - Act as a worker participant in distributed applications (ServiceWorkerEndpoint)
 *
 * This composite interface is particularly useful for services that need to
 * synchronize state or coordinate operations across multiple fragments in a
 * distributed Holoscan application. When registered with the application using
 * `register_service()`, the framework automatically calls the appropriate
 * driver/worker methods based on each fragment's role.
 *
 * @note In single-fragment applications, only the FragmentService interface is used.
 * The distributed endpoint methods are only called in multi-fragment distributed applications.
 */
class DistributedAppService : public FragmentService,
                              public distributed::ServiceDriverEndpoint,
                              public distributed::ServiceWorkerEndpoint {
 public:
  /**
   * @brief Inherit constructors from base classes.
   *
   * This using declaration enables construction of DistributedAppService using the constructors
   * of FragmentService, ServiceDriverEndpoint, and ServiceWorkerEndpoint, providing
   * flexibility in how derived classes can be initialized.
   */
  using FragmentService::FragmentService;
  using ServiceDriverEndpoint::ServiceDriverEndpoint;
  using ServiceWorkerEndpoint::ServiceWorkerEndpoint;
};

}  // namespace holoscan

#endif /* HOLOSCAN_CORE_FRAGMENT_SERVICE_HPP */
