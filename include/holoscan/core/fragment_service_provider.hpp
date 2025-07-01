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

#ifndef HOLOSCAN_CORE_FRAGMENT_SERVICE_PROVIDER_HPP
#define HOLOSCAN_CORE_FRAGMENT_SERVICE_PROVIDER_HPP

#include <memory>
#include <string_view>
#include <typeinfo>  // Required for std::type_info

namespace holoscan {

// Forward declaration
class FragmentService;

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
};

}  // namespace holoscan

#endif  // HOLOSCAN_CORE_FRAGMENT_SERVICE_PROVIDER_HPP
