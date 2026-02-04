/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
         ///<  3. FIFO_LATEST_READY
         ///<  4. FIFO
  FIFO,  ///< the presentation engine waits for the next vertical blanking period to update the
         ///< current image. Tearing cannot be observed. An internal queue is used to hold pending
         ///< presentation requests. New requests are appended to the end of the queue, and one
         ///< request is removed from the beginning of the queue and processed during each vertical
         ///< blanking period in which the queue is non-empty.
  IMMEDIATE,  ///< the presentation engine does not wait for a vertical blanking period to update
              ///< the current image, meaning this mode may result in visible tearing. No internal
              ///< queuing of presentation requests is needed, as the requests are applied
              ///< immediately.
  MAILBOX,    ///< the presentation engine waits for the next vertical blanking period to update the
              ///< current image. Tearing cannot be observed. An internal single-entry queue is used
              ///< to hold pending presentation requests. If the queue is full when a new
              ///< presentation request is received, the new request replaces the existing entry,
              ///< and any images associated with the prior entry become available for reuse by the
              ///< application. One request is removed from the queue and processed during each
              ///< vertical blanking period in which the queue is non-empty.
  SHARED_DEMAND_REFRESH,      ///< This mode is only supported in exclusive mode.
                              ///< Specifies that the presentation engine and application have
                              ///< concurrent access to a single image, which is referred to as a
                              ///< shared presentable image. The presentation engine is only
                              ///< required to update the current image after a new presentation
                              ///< request is received. Therefore the application must make a
                              ///< presentation request whenever an update is required. However, the
                              ///< presentation engine may update the current image at any point,
                              ///< meaning this mode may result in visible tearing.
  SHARED_CONTINUOUS_REFRESH,  ///< This mode is only supported in exclusive mode.
                              ///< Specifies that the presentation engine and application have
                              ///< concurrent access to a single image, which is referred to as
                              ///< a shared presentable image. The presentation engine periodically
                              ///< updates the current image on its regular refresh cycle. The
                              ///< application is only required to make one initial presentation
                              ///< request, after which the presentation engine must update the
                              ///< current image without any need for further presentation requests.
                              ///< The application can indicate the image contents have been updated
                              ///< by making a presentation request, but this does not guarantee the
                              ///< timing of when it will be updated. This mode may result in
                              ///< visible tearing if rendering to the image is not timed correctly.
  FIFO_LATEST_READY,  ///< specifies that the presentation engine waits for the next vertical
                      ///< blanking period to update the current image. Tearing cannot be observed.
                      ///< An internal queue is used to hold pending presentation requests. New
                      ///< requests are appended to the end of the queue. At each vertical blanking
                      ///< period, the presentation engine dequeues all successive requests that are
                      ///< ready to be presented from the beginning of the queue.
  FIFO_RELAXED,  ///< specifies that the presentation engine generally waits for the next vertical
                 ///< blanking period to update the current image. If a vertical blanking period has
                 ///< already passed since the last update of the current image then the
                 ///< presentation engine does not wait for another vertical blanking period for the
                 ///< update, meaning this mode may result in visible tearing in this case. This
                 ///< mode is useful for reducing visual stutter with an application that will
                 ///< mostly present a new image before the next vertical blanking period, but may
                 ///< occasionally be late, and present a new image just after the next vertical
                 ///< blanking period. An internal queue is used to hold pending presentation
                 ///< requests. New requests are appended to the end of the queue, and one request
                 ///< is removed from the beginning of the queue and processed during or after each
                 ///< vertical blanking period in which the queue is non-empty.
};

}  // namespace holoscan::viz

#endif /* MODULES_HOLOVIZ_SRC_HOLOVIZ_PRESENT_MODE_HPP */
