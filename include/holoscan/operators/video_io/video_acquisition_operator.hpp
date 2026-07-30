/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOSCAN_OPERATORS_VIDEO_IO_VIDEO_ACQUISITION_OPERATOR_HPP
#define HOLOSCAN_OPERATORS_VIDEO_IO_VIDEO_ACQUISITION_OPERATOR_HPP

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
 * @brief Base class for video acquisition (capture) operators.
 *
 * Declares a standard **signal** output carrying `holoscan::gxf::Entity` (VideoBuffer or Tensor
 * payload), matching `V4L2VideoCaptureOp` and composable with format conversion and Holoviz.
 *
 * Subclasses must implement compute(). They may override query_capabilities() to query vendor
 * SDKs without starting capture; the default implementation derives a minimal snapshot from
 * configured parameters.
 *
 * ==Dynamic Port Allocation==
 *
 * The number of output streams is set via the @p num_streams constructor parameter (default 1).
 * Because `make_operator` calls the constructor *before* `setup()`, the stream count is available
 * at port-registration time — only the requested number of ports are created. All registered
 * ports get proper backpressure scheduling (no dormant ports with `ConditionType::kNone`).
 *
 * ==num_streams vs channel_indices==
 *
 * @p num_streams controls **port allocation** (how many output ports `setup()` registers).
 * @p channel_indices is a **capability-reporting** parameter that describes which hardware
 * channels appear in the `VideoCaptureCapabilities` snapshot. These are intentionally
 * independent: a singleton operator wrapping a vendor SDK that manages its own multiplexing
 * may have `num_streams=1` but `channel_indices={0,1,2,3}` to expose all channels the SDK
 * reports. Conversely, `num_streams=4` with empty `channel_indices` reports a single
 * default channel in the capabilities.
 *
 * ==Named Outputs==
 *
 * - **signal** : stream 0 (`std::shared_ptr<holoscan::gxf::Entity>`), always registered.
 * - **signal_1** … **signal_N** : additional streams, registered only when `num_streams > 1`.
 *   Subclasses call `emit_capture_stream()` with index 0 … `num_streams() - 1`.
 *
 * ==Parameters==
 *
 * - **backend_id**: Logical backend name for capability reporting (e.g. "v4l2", "vendor.example").
 * - **channel_index**: Zero-based channel index when one channel per operator instance.
 * - **channel_indices**: Optional list of channel indices for multi-channel singleton-SDK fallback
 *   When non-empty, operators may treat this as the authoritative channel set.
 * - **uri**: Device path, device index string, or stream URI.
 * - **width**, **height**: Requested frame size; `0` means device default / unspecified.
 * - **frame_rate**: Requested frame rate; `0.f` means unspecified.
 * - **pixel_format**: Requested pixel format token (fourcc or vendor-specific label).
 * - **color_space**: Requested color space (`auto`, `bt709`, ...).
 * - **transport**: Hint for `VideoTransport` (`auto`, `sdi`, `hdmi`, `ethernet`, `v4l2`, ...).
 * - **vendor_extensions**: Map of vendor-specific keys (`vendor.<name>.<param>`) as a YAML map
 *
 */
class VideoAcquisitionOperator : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(VideoAcquisitionOperator)

  VideoAcquisitionOperator() = default;

  /**
   * @brief Construct with a specific number of output streams.
   *
   * @param num_streams Number of output ports to register (1 .. kVideoIoMaxStreams).
   *        Clamped to [1, kVideoIoMaxStreams]. Passed as a constructor argument so that
   *        `setup()` can read it before port registration (make_operator lifecycle).
   */
  explicit VideoAcquisitionOperator(uint32_t num_streams);

  /**
   * @brief Construct with a specific stream count and forwarded Arg / ArgList arguments.
   *
   * Enables `make_operator<Derived>("name", num_streams, Arg("key", val), ...)`.
   */
  HOLOSCAN_OPERATOR_FORWARD_TEMPLATE()
  explicit VideoAcquisitionOperator(uint32_t num_streams, ArgT&& arg, ArgsT&&... args)
      : num_streams_(std::max(1u, std::min(num_streams, video_io::kVideoIoMaxStreams))) {
    init_port_names();
    add_arg(std::forward<ArgT>(arg));
    (add_arg(std::forward<ArgsT>(args)), ...);
  }

  void setup(OperatorSpec& spec) override;
  void initialize() override;

  [[nodiscard]] virtual video_io::VideoCaptureCapabilities query_capture_capabilities() const;

  [[nodiscard]] video_io::VideoDeviceCapabilities query_capabilities() const;

  [[nodiscard]] uint64_t dropped_frame_count() const {
    return dropped_frames_.load(std::memory_order_relaxed);
  }
  [[nodiscard]] uint64_t acquired_frame_count() const {
    return acquired_frames_.load(std::memory_order_relaxed);
  }

  /** @return true if @p stream_index < num_streams(). */
  [[nodiscard]] bool is_capture_stream_enabled(uint32_t stream_index) const;

  /** @return Number of output streams registered in setup(). */
  [[nodiscard]] uint32_t num_streams() const { return num_streams_; }

 protected:
  void note_dropped_frame() { dropped_frames_.fetch_add(1, std::memory_order_relaxed); }
  void note_acquired_frame() { acquired_frames_.fetch_add(1, std::memory_order_relaxed); }

  /**
   * @brief Return the output port name for a given stream index.
   *
   * Index 0 returns `"signal"` (not `"signal_0"`) for backward compatibility with single-stream
   * operators such as `V4L2VideoCaptureOp`. Indices 1+ return `"signal_1"`, `"signal_2"`, etc.
   * Validates against `kVideoIoMaxStreams` (global max), not the per-instance `num_streams()`.
   */
  static std::string capture_output_port_name(uint32_t stream_index);

  /**
   * @brief Emit an entity on the given capture stream port.
   *
   * The entity remains valid after this call (the underlying GXF handle is shared, not moved).
   * Throws `std::out_of_range` if @p stream_index >= num_streams().
   */
  void emit_capture_stream(OutputContext& op_output, uint32_t stream_index,
                           holoscan::gxf::Entity& entity);

  virtual video_io::VideoCaptureCapabilities build_capture_capabilities_from_parameters() const;

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
      port_names_.push_back(capture_output_port_name(i));
    }
  }

  uint32_t num_streams_ = 1;
  std::vector<std::string> port_names_{"signal"};
  std::atomic<uint64_t> dropped_frames_{0};
  std::atomic<uint64_t> acquired_frames_{0};
};

}  // namespace holoscan::ops

#endif  // HOLOSCAN_OPERATORS_VIDEO_IO_VIDEO_ACQUISITION_OPERATOR_HPP
