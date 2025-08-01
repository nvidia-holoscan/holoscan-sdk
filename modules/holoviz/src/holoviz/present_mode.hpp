/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef MODULES_HOLOVIZ_SRC_HOLOVIZ_PRESENT_MODE_HPP
#define MODULES_HOLOVIZ_SRC_HOLOVIZ_PRESENT_MODE_HPP

namespace holoscan::viz {

/// The present mode determines how the rendered result will be presented on the screen.
enum class PresentMode {
  AUTO,  ///< automatically select present mode depending on available modes, selection priority:
         ///<  1. MAILBOX
         ///<  2. IMMEDIATE
         ///<  3. FIFO
  FIFO,  ///< the presentation engine waits for the next vertical blanking period to update the
         ///< current image. Tearing cannot be observed. An internal queue is used to hold pending
         ///< presentation requests. New requests are appended to the end of the queue, and one
         ///< request is removed from the beginning of the queue and processed during each vertical
         ///< blanking period in which the queue is non-empty.
  IMMEDIATE,  ///< the presentation engine does not wait for a vertical blanking period to update
              ///< the current image, meaning this mode may result in visible tearing. No internal
              ///< queuing of presentation requests is needed, as the requests are applied
              ///< immediately.
  MAILBOX     ///< the presentation engine waits for the next vertical blanking period to update the
           ///< current image. Tearing cannot be observed. An internal single-entry queue is used to
           ///< hold pending presentation requests. If the queue is full when a new presentation
           ///< request is received, the new request replaces the existing entry, and any images
           ///< associated with the prior entry become available for reuse by the application. One
           ///< request is removed from the queue and processed during each vertical blanking period
           ///< in which the queue is non-empty.
};

}  // namespace holoscan::viz

#endif /* MODULES_HOLOVIZ_SRC_HOLOVIZ_PRESENT_MODE_HPP */
