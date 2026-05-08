/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef HOLOSCAN_OPERATORS_VIDEO_IO_VIDEO_TRANSMISSION_OPERATOR_HPP
#define HOLOSCAN_OPERATORS_VIDEO_IO_VIDEO_TRANSMISSION_OPERATOR_HPP

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <yaml-cpp/yaml.h>

#include "holoscan/core/gxf/entity.hpp"
#include "holoscan/core/operator.hpp"
#include "holoscan/core/parameter.hpp"

#include "./video_io_capabilities.hpp"

namespace holoscan::ops {

/**
 * @brief Base class for video transmission (output) operators.
 *
 * Declares a standard **signal** input carrying `holoscan::gxf::Entity`, aligned with acquisition
 * and passthrough operators for multi-input / single-output compositions.
 *
 * Subclasses must implement compute(). Override query_capabilities() for hardware-specific
 * enumeration without starting output.
 *
 * ==Dynamic Port Allocation==
 *
 * The number of input streams is set via the @p num_streams constructor parameter (default 1).
 * Because `make_operator` calls the constructor *before* `setup()`, the stream count is available
 * at port-registration time — only the requested number of ports are created. All registered
 * ports get proper backpressure scheduling (no dormant ports with `ConditionType::kNone`).
 *
 * ==num_streams vs channel_indices==
 *
 * @p num_streams controls **port allocation** (how many input ports `setup()` registers).
 * @p channel_indices is a **capability-reporting** parameter that describes which hardware
 * channels appear in the `VideoTransmitCapabilities` snapshot. These are intentionally
 * independent.
 *
 * ==Named Inputs==
 *
 * - **signal** : stream 0 (`std::shared_ptr<holoscan::gxf::Entity>`), always registered.
 * - **signal_1** … **signal_N** : additional streams, registered only when `num_streams > 1`.
 *
 * ==Parameters==
 *
 * Same common channel / URI / format parameters as `VideoAcquisitionOperator`.
 */
class VideoTransmissionOperator : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(VideoTransmissionOperator)

  VideoTransmissionOperator() = default;

  /**
   * @brief Construct with a specific number of input streams.
   *
   * @param num_streams Number of input ports to register (1 .. kVideoIoMaxStreams).
   *        Clamped to [1, kVideoIoMaxStreams].
   */
  explicit VideoTransmissionOperator(uint32_t num_streams);

  HOLOSCAN_OPERATOR_FORWARD_TEMPLATE()
  explicit VideoTransmissionOperator(uint32_t num_streams, ArgT&& arg, ArgsT&&... args)
      : num_streams_(std::max(1u, std::min(num_streams, video_io::kVideoIoMaxStreams))) {
    init_port_names();
    add_arg(std::forward<ArgT>(arg));
    (add_arg(std::forward<ArgsT>(args)), ...);
  }

  void setup(OperatorSpec& spec) override;
  void initialize() override;

  [[nodiscard]] virtual video_io::VideoTransmitCapabilities query_transmit_capabilities() const;

  [[nodiscard]] video_io::VideoDeviceCapabilities query_capabilities() const;

  [[nodiscard]] uint64_t dropped_frame_count() const {
    return dropped_frames_.load(std::memory_order_relaxed);
  }
  [[nodiscard]] uint64_t transmitted_frame_count() const {
    return transmitted_frames_.load(std::memory_order_relaxed);
  }

  /** @return true if @p stream_index < num_streams(). */
  [[nodiscard]] bool is_transmit_stream_enabled(uint32_t stream_index) const;

  /** @return Number of input streams registered in setup(). */
  [[nodiscard]] uint32_t num_streams() const { return num_streams_; }

 protected:
  void note_dropped_frame() { dropped_frames_.fetch_add(1, std::memory_order_relaxed); }
  void note_transmitted_frame() { transmitted_frames_.fetch_add(1, std::memory_order_relaxed); }

  /**
   * @brief Return the input port name for a given stream index.
   *
   * Index 0 returns `"signal"` (not `"signal_0"`) for backward compatibility with single-stream
   * operators. Indices 1+ return `"signal_1"`, `"signal_2"`, etc.
   * Validates against `kVideoIoMaxStreams` (global max), not the per-instance `num_streams()`.
   */
  static std::string transmit_input_port_name(uint32_t stream_index);

  /**
   * @brief Receive an entity from the given transmit stream port.
   *
   * Symmetric counterpart to `VideoAcquisitionOperator::emit_capture_stream()`.
   * Throws `std::out_of_range` if @p stream_index >= num_streams().
   *
   * @return The received entity, or an empty shared_ptr if nothing was available.
   */
  std::shared_ptr<holoscan::gxf::Entity> receive_transmit_stream(InputContext& op_input,
                                                                 uint32_t stream_index);

  virtual video_io::VideoTransmitCapabilities build_transmit_capabilities_from_parameters() const;

  void compute(InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override = 0;

  Parameter<std::string> backend_id_;
  Parameter<uint32_t> channel_index_;
  Parameter<std::vector<uint32_t>> channel_indices_;
  Parameter<std::string> uri_;
  Parameter<uint32_t> width_;
  Parameter<uint32_t> height_;
  Parameter<float> frame_rate_;
  Parameter<std::string> pixel_format_;
  Parameter<std::string> color_space_;
  Parameter<std::string> transport_;
  Parameter<YAML::Node> vendor_extensions_;

 private:
  void init_port_names() {
    port_names_.clear();
    port_names_.reserve(num_streams_);
    for (uint32_t i = 0; i < num_streams_; ++i) {
      port_names_.push_back(transmit_input_port_name(i));
    }
  }

  uint32_t num_streams_ = 1;
  std::vector<std::string> port_names_{"signal"};
  std::atomic<uint64_t> dropped_frames_{0};
  std::atomic<uint64_t> transmitted_frames_{0};
};

}  // namespace holoscan::ops

#endif  // HOLOSCAN_OPERATORS_VIDEO_IO_VIDEO_TRANSMISSION_OPERATOR_HPP
