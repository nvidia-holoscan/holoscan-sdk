/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef SAMPLE_PING_TX_HPP
#define SAMPLE_PING_TX_HPP

#include <cinttypes>
#include <string>
#include <vector>

#include "gxf/std/allocator.hpp"
#include "gxf/std/clock.hpp"
#include "gxf/std/codelet.hpp"
#include "gxf/std/parameter_parser_std.hpp"
#include "gxf/std/transmitter.hpp"

namespace nvidia {
namespace holoscan {
namespace sample {

// Sample codelet implementation to send an entity
class PingTx : public gxf::Codelet {
 public:
  virtual ~PingTx() = default;

  gxf_result_t registerInterface(gxf::Registrar* registrar) override;
  gxf_result_t start() override { return GXF_SUCCESS; }
  gxf_result_t tick() override;
  gxf_result_t stop() override { return GXF_SUCCESS; }

 private:
  gxf::Parameter<gxf::Handle<gxf::Transmitter>> signal_;
  gxf::Parameter<gxf::Handle<gxf::Clock>> clock_;
  gxf::Parameter<std::vector<gxf::Handle<gxf::Transmitter>>> signal_vector_;
  gxf::Parameter<std::vector<int32_t>> signal_data_;
  gxf::Parameter<gxf::Handle<gxf::Allocator>> pool_;
  int count = 1;
};

}  // namespace sample
}  // namespace holoscan
}  // namespace nvidia

#endif /* SAMPLE_PING_TX_HPP */
