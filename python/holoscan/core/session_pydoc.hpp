/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

namespace holoscan::doc::session {

inline constexpr const char* kRunSession = R"doc(
The sole owner of one asynchronous run attempt.

Use this object as a context manager. Exiting requests cooperative stop and waits for cleanup. Python
runs leave process signals entirely caller-managed and never install an SDK signal handler.
)doc";

inline constexpr const char* kRun = R"doc(
Run a Plan to terminal completion and return its inspectable RunSession.

The call releases the GIL while waiting. Process signals remain caller-managed; no SDK signal handler
is installed.
)doc";

inline constexpr const char* kRunAsync = R"doc(
Start a Plan and return its RunSession without waiting.

Process signals remain caller-managed; no SDK signal handler is installed. Prefer ``with
run_async(plan) as session`` so scope exit always requests stop and waits.
)doc";

inline constexpr const char* kWait = R"doc(
Wait for terminal state and return True, or return False when the timeout expires.

``timeout`` is None for an unbounded wait or a nonnegative number of seconds. An unbounded successful
wait also joins the session worker. A bounded successful wait reports frozen terminal state while
the session retains strict ownership until a later unbounded wait or cleanup. This operation does
not request stop.
)doc";

inline constexpr const char* kSummary = R"doc(
Return the bounded human-readable report for this session's runtime diagnostics.

Errors render before warnings while preserving publication order within each severity. Use the
structured ``diagnostics`` values, rather than parsing this text, for program logic.
)doc";

}  // namespace holoscan::doc::session
