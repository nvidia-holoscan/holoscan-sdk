/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

namespace holoscan::viz {

struct Layer::Impl {
  int32_t priority_ = 0;
  float opacity_ = 1.f;
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
  impl_->opacity_ = opacity;
}

}  // namespace holoscan::viz
