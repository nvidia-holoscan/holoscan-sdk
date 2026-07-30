/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "layer.hpp"

#include <stdexcept>
#include <vector>

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
