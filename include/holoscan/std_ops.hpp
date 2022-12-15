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

#ifndef HOLOSCAN_STD_OPS_HPP
#define HOLOSCAN_STD_OPS_HPP

#include "holoscan/operators/aja_source/aja_source.hpp"
#include "holoscan/operators/bayer_demosaic/bayer_demosaic.hpp"
#include "holoscan/operators/custom_lstm_inference/lstm_tensor_rt_inference.hpp"
#if HOLOSCAN_BUILD_EMERGENT == 1
  #include "holoscan/operators/emergent_source/emergent_source.hpp"
#endif
#include "holoscan/operators/format_converter/format_converter.hpp"
#include "holoscan/operators/holoviz/holoviz.hpp"
#include "holoscan/operators/multiai_inference/multiai_inference.hpp"
#include "holoscan/operators/multiai_postprocessor/multiai_postprocessor.hpp"
#include "holoscan/operators/segmentation_postprocessor/segmentation_postprocessor.hpp"
#include "holoscan/operators/stream_playback/video_stream_recorder.hpp"
#include "holoscan/operators/stream_playback/video_stream_replayer.hpp"
#include "holoscan/operators/tensor_rt/tensor_rt_inference.hpp"
#include "holoscan/operators/tool_tracking_postprocessor/tool_tracking_postprocessor.hpp"
#include "holoscan/operators/visualizer_icardio/visualizer_icardio.hpp"
#include "holoscan/operators/visualizer_tool_tracking/visualizer_tool_tracking.hpp"

#endif /* HOLOSCAN_STD_OPS_HPP */
