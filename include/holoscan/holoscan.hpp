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

#ifndef HOLOSCAN_HOLOSCAN_HPP
#define HOLOSCAN_HOLOSCAN_HPP

#include "./core/common.hpp"

#include "./core/application.hpp"
#include "./core/arg.hpp"
#include "./core/condition.hpp"
#include "./core/config.hpp"
#include "./core/execution_context.hpp"
#include "./core/executor.hpp"
#include "./core/fragment.hpp"
#include "./core/graph.hpp"
#include "./core/io_context.hpp"
#include "./core/message.hpp"
#include "./core/operator.hpp"
#include "./core/resource.hpp"

// Domain objects
#include "./core/gxf/entity.hpp"

// Conditions
#include "./core/conditions/gxf/boolean.hpp"
#include "./core/conditions/gxf/count.hpp"
#include "./core/conditions/gxf/downstream_affordable.hpp"
#include "./core/conditions/gxf/message_available.hpp"

// Resources
#include "./core/resources/gxf/block_memory_pool.hpp"
#include "./core/resources/gxf/cuda_stream_pool.hpp"
#include "./core/resources/gxf/std_component_serializer.hpp"
#include "./core/resources/gxf/unbounded_allocator.hpp"
#include "./core/resources/gxf/video_stream_serializer.hpp"

// Operators
#include "./core/gxf/gxf_operator.hpp"

#endif /* HOLOSCAN_HOLOSCAN_HPP */
