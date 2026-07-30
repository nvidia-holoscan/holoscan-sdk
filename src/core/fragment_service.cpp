/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <memory>

#include <holoscan/core/fragment_service.hpp>

namespace holoscan {

DefaultFragmentService::DefaultFragmentService(const std::shared_ptr<Resource>& resource)
    : resource_(resource) {}

}  // namespace holoscan
