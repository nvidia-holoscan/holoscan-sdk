/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef HOLOSCAN_CORE_FLOW_TRACKING_ANNOTATION_HPP
#define HOLOSCAN_CORE_FLOW_TRACKING_ANNOTATION_HPP

#include "holoscan/core/gxf/gxf_utils.hpp"
#include "holoscan/core/operator.hpp"

namespace holoscan {

/**
 * @brief This function annotates a message with a MessageLabel timestamp.
 *
 * @param uid The entity ID of the message.
 * @param context The GXF context.
 * @param op The operator that is transmitting the message.
 * @param transmitter_name The name of the transmitter from which the message is being published.
 * @return gxf_result_t The result of the annotation.
 */
gxf_result_t annotate_message(gxf_uid_t uid, const gxf_context_t& context, Operator* op,
                              const char* transmitter_name);

/**
 * @brief This function de-annotates a message and extracts the MessageLabel timestamp. It then
 * updates necessary data flow tracking information in DataFlowTracker object.
 *
 * @param uid The entity ID of the message.
 * @param context The GXF context.
 * @param op The operator that is receiving the message.
 * @param receiver_name The name of the receiver which is receiving the message.
 */
gxf_result_t deannotate_message(gxf_uid_t* uid, const gxf_context_t& context, Operator* op,
                                const char* receiver_name);

}  // namespace holoscan

#endif /* HOLOSCAN_CORE_FLOW_TRACKING_ANNOTATION_HPP */
