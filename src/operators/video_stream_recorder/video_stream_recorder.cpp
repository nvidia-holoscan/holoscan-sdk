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

#include "holoscan/operators/video_stream_recorder/video_stream_recorder.hpp"

#include <chrono>
#include <string>

#include "gxf/core/expected.hpp"
#include "gxf/serialization/entity_serializer.hpp"

#include "holoscan/core/execution_context.hpp"
#include "holoscan/core/fragment.hpp"
#include "holoscan/core/io_context.hpp"
#include "holoscan/core/gxf/entity.hpp"
#include "holoscan/core/operator_spec.hpp"

#include "holoscan/core/conditions/gxf/boolean.hpp"
#include "holoscan/core/conditions/gxf/message_available.hpp"
#include "holoscan/core/resources/gxf/video_stream_serializer.hpp"

namespace holoscan::ops {

void VideoStreamRecorderOp::setup(OperatorSpec& spec) {
  auto& input = spec.input<gxf::Entity>("input");

  spec.param(receiver_, "receiver", "Entity receiver", "Receiver channel to log", &input);
  spec.param(entity_serializer_,
             "entity_serializer",
             "Entity serializer",
             "Serializer for serializing entities");
  spec.param(directory_, "directory", "Directory path", "Directory path for storing files");
  spec.param(basename_, "basename", "Base file name", "User specified file name without extension");
  spec.param(flush_on_tick_,
             "flush_on_tick",
             "Flush on tick",
             "Flushes output buffer on every tick when true",
             false);
}

void VideoStreamRecorderOp::initialize() {
  // Set up prerequisite parameters before calling GXFOperator::initialize()
  auto frag = fragment();
  auto entity_serializer =
      frag->make_resource<holoscan::VideoStreamSerializer>("entity_serializer");
  add_arg(Arg("entity_serializer") = entity_serializer);

  // Operator::initialize must occur after all arguments have been added
  Operator::initialize();

  // Create path by appending receiver name to directory path if basename is not provided
  std::string path = directory_.get() + '/';

  // Note: basename was optional in the GXF operator, but not yet in the native operator,
  //       so in practice, this should always have a value.
  if (basename_.has_value()) {
    path += basename_.get();
  } else {
    path += receiver_.get()->name();
  }

  // Initialize index file stream as write-only
  index_file_stream_ =
      nvidia::gxf::FileStream("", path + nvidia::gxf::FileStream::kIndexFileExtension);

  // Initialize binary file stream as write-only
  binary_file_stream_ =
      nvidia::gxf::FileStream("", path + nvidia::gxf::FileStream::kBinaryFileExtension);

  // Open index file stream
  nvidia::gxf::Expected<void> result = index_file_stream_.open();
  if (!result) {
    auto code = nvidia::gxf::ToResultCode(result);
    throw std::runtime_error(fmt::format("Failed to open index_file_stream_ with code: {}", code));
  }

  // Open binary file stream
  result = binary_file_stream_.open();
  if (!result) {
    auto code = nvidia::gxf::ToResultCode(result);
    throw std::runtime_error(fmt::format("Failed to open binary_file_stream_ with code: {}", code));
  }
  binary_file_offset_ = 0;
}

VideoStreamRecorderOp::~VideoStreamRecorderOp() {
  // for the GXF codelet, this code is in a deinitialize() method

  // Close binary file stream
  nvidia::gxf::Expected<void> result = binary_file_stream_.close();
  if (!result) {
    auto code = nvidia::gxf::ToResultCode(result);
    HOLOSCAN_LOG_ERROR("Failed to close binary_file_stream_ with code: {}", code);
  }

  // Close index file stream
  result = index_file_stream_.close();
  if (!result) {
    auto code = nvidia::gxf::ToResultCode(result);
    HOLOSCAN_LOG_ERROR("Failed to close index_file_stream_ with code: {}", code);
  }
}

void VideoStreamRecorderOp::compute(InputContext& op_input, OutputContext& op_output,
                                    ExecutionContext& context) {
  // avoid warning about unused variable
  (void)op_output;

  auto entity = op_input.receive<gxf::Entity>("input");

  // dynamic cast from holoscan::Resource to holoscan::VideoStreamSerializer
  auto vs_serializer =
      std::dynamic_pointer_cast<holoscan::VideoStreamSerializer>(entity_serializer_.get());
  // get the Handle to the underlying GXF EntitySerializer
  auto entity_serializer = nvidia::gxf::Handle<nvidia::gxf::EntitySerializer>::Create(
      context.context(), vs_serializer.get()->gxf_cid());
  nvidia::gxf::Expected<size_t> size =
      entity_serializer.value()->serializeEntity(entity, &binary_file_stream_);
  if (!size) {
    auto code = nvidia::gxf::ToResultCode(size);
    throw std::runtime_error(fmt::format("Failed to serialize entity with code {}", code));
  }

  // Create entity index
  nvidia::gxf::EntityIndex index;
  index.log_time = std::chrono::system_clock::now().time_since_epoch().count();
  index.data_size = size.value();
  index.data_offset = binary_file_offset_;

  // Write entity index to index file
  nvidia::gxf::Expected<size_t> result = index_file_stream_.writeTrivialType(&index);
  if (!result) {
    auto code = nvidia::gxf::ToResultCode(size);
    throw std::runtime_error(fmt::format("Failed writing to index file stream with code {}", code));
  }
  binary_file_offset_ += size.value();

  if (flush_on_tick_) {
    // Flush binary file output stream
    nvidia::gxf::Expected<void> result = binary_file_stream_.flush();
    if (!result) {
      auto code = nvidia::gxf::ToResultCode(size);
      throw std::runtime_error(
          fmt::format("Failed writing to index file stream with code {}", code));
    }

    // Flush index file output stream
    result = index_file_stream_.flush();
    if (!result) {
      auto code = nvidia::gxf::ToResultCode(size);
      throw std::runtime_error(
          fmt::format("Failed writing to index file stream with code {}", code));
    }
  }
}

}  // namespace holoscan::ops
