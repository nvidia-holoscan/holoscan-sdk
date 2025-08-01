/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "layer.hpp"

#include <vector>
#include <stdexcept>

namespace holoscan::viz {

struct Layer::Impl {
  int32_t priority_ = 0;
  float opacity_ = 1.F;
  std::vector<View> views_;
};

Layer::Layer(Type type) : type_(type), impl_(new Impl) {}

Layer::~Layer() {}

Layer::Type Layer::get_type() const {
  return type_;
}

bool Layer::can_be_reused(Layer& other) const {
  return (type_ == other.type_);
}

int32_t Layer::get_priority() const {
  return impl_->priority_;
}

void Layer::set_priority(int32_t priority) {
  impl_->priority_ = priority;
}

float Layer::get_opacity() const {
  return impl_->opacity_;
}

void Layer::set_opacity(float opacity) {
  if ((opacity < 0.F) || (opacity > 1.F)) {
    throw std::invalid_argument("Layer opacity should be in the range [0.0 ... 1.0]");
  }
  impl_->opacity_ = opacity;
}

const std::vector<Layer::View>& Layer::get_views() const {
  return impl_->views_;
}

void Layer::set_views(const std::vector<View>& views) {
  impl_->views_ = views;
}

void Layer::add_view(const View& view) {
  if (view.height == 0) {
    throw std::invalid_argument("Layer view height should not be zero");
  }
  if (view.width <= 0) {
    throw std::invalid_argument("Layer view width should not be less than or equal to zero");
  }
  impl_->views_.push_back(view);
}

}  // namespace holoscan::viz
