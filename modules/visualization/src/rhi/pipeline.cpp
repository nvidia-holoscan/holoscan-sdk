/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "holoviz/rhi/pipeline.hpp"

#include <memory>
#include <string>
#include <utility>

#include "holoviz/rhi/descriptor.hpp"

#include "detail/descriptor_impl.hpp"
#include "detail/handle_errors.hpp"
#include "detail/pipeline_impl.hpp"

namespace holoviz::rhi {

Pipeline::Pipeline(std::unique_ptr<detail::PipelineImpl> impl) : impl_(std::move(impl)) {}
Pipeline::Pipeline() = default;
Pipeline::Pipeline(Pipeline&&) noexcept = default;
Pipeline& Pipeline::operator=(Pipeline&&) noexcept = default;
Pipeline::~Pipeline() = default;

expected<void> Pipeline::destroy(std::chrono::nanoseconds timeout) {
  if (!impl_) {
    return {};  // empty handle: nothing to free
  }
  expected<void> result = impl_->destroy(timeout);
  if (result || result.error().code != ErrorCode::kBackpressured) {
    impl_.reset();
  }
  return result;
}

expected<Descriptor> Pipeline::create_descriptor() {
  if (!impl_) {
    return detail::null_impl_error("Pipeline::create_descriptor");
  }
  return impl_->create_descriptor().map(
      [](std::unique_ptr<detail::DescriptorImpl> impl) { return Descriptor{std::move(impl)}; });
}

expected<uint32_t> Pipeline::binding_name_to_index(const std::string& name) const {
  if (!impl_) {
    return detail::null_impl_error("Pipeline::binding_name_to_index");
  }
  return impl_->binding_name_to_index(name);
}

expected<NativeObject> Pipeline::native_object(NativeObjectType type) const {
  if (!impl_) {
    return detail::null_impl_error("Pipeline::native_object");
  }
  return impl_->native_object(type);
}

}  // namespace holoviz::rhi
