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

#ifndef PYHOLOSCAN_CORE_FRAGMENT_SERVICE_HPP
#define PYHOLOSCAN_CORE_FRAGMENT_SERVICE_HPP

#include <pybind11/pybind11.h>

#include <memory>

#include "holoscan/core/fragment_service.hpp"

namespace py = pybind11;

namespace holoscan {

// Forward definition
class Resource;

void init_fragment_service(py::module_&);

/**
 * @brief Trampoline class for FragmentService.
 *
 * This class is used to allow Python classes to inherit from holoscan::FragmentService
 * and override its virtual methods.
 */
class PYBIND11_EXPORT PyFragmentService : public FragmentService {
 public:
  /* Inherit the constructors */
  using FragmentService::FragmentService;

  std::shared_ptr<Resource> resource() const override;
  void resource(const std::shared_ptr<Resource>& resource) override;
};

/**
 * @brief Trampoline class for DefaultFragmentService.
 *
 * This class is used to allow Python classes to inherit from holoscan::DefaultFragmentService
 * and override its virtual methods.
 */
class PYBIND11_EXPORT PyDefaultFragmentService : public DefaultFragmentService {
 public:
  /* Inherit the constructors */
  using DefaultFragmentService::DefaultFragmentService;

  std::shared_ptr<Resource> resource() const override;
  void resource(const std::shared_ptr<Resource>& resource) override;
};

}  // namespace holoscan

#endif /* PYHOLOSCAN_CORE_FRAGMENT_SERVICE_HPP */
