/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOSCAN_CORE_FRAGMENT_SERVICE_PROVIDER_HPP
#define HOLOSCAN_CORE_FRAGMENT_SERVICE_PROVIDER_HPP

#include <memory>
#include <string_view>
#include <typeinfo>  // Required for std::type_info
#include <vector>

namespace holoscan {

// Forward declarations
class FragmentService;
class Resource;

/**
 * @brief Interface for classes that can provide access to registered fragment services.
 *
 * This interface is used by ComponentBase to decouple component service access from the
 * concrete Fragment implementation, enabling better testability and modularity while
 * still allowing service retrieval.
 */
class FragmentServiceProvider {
 public:
  virtual ~FragmentServiceProvider() = default;

  /**
   * @brief Retrieves a service using type erasure.
   *
   * @param service_type The std::type_info of the service to retrieve.
   * @param id The identifier of the service instance.
   * @return A shared_ptr to FragmentService if found, otherwise nullptr.
   */
  virtual std::shared_ptr<FragmentService> get_service_erased(const std::type_info& service_type,
                                                              std::string_view id) const = 0;

  /**
   * @brief Retrieve a resource registered as a fragment service by name.
   *
   * @param id The service id (name) used during service registration.
   * @return A shared_ptr to the service resource, or nullptr if not found.
   */
  virtual std::shared_ptr<Resource> get_service_resource_by_name(std::string_view id) const = 0;

  /**
   * @brief Retrieve all fragment services with a matching id, regardless of registered type.
   *
   * This method is used as a fallback when exact type lookup fails, enabling retrieval
   * of services registered with a derived type when looking up by a base type.
   *
   * @param id The service id (name) used during service registration.
   * @return A vector of shared_ptrs to matching FragmentServices. Empty if none found.
   */
  virtual std::vector<std::shared_ptr<FragmentService>> get_services_by_id(
      [[maybe_unused]] std::string_view id) const {
    return {};
  }
};

}  // namespace holoscan

#endif  // HOLOSCAN_CORE_FRAGMENT_SERVICE_PROVIDER_HPP
