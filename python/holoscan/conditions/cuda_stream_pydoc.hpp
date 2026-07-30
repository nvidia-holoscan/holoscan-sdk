/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef PYHOLOSCAN_CONDITIONS_CUDA_STREAM_PYDOC_HPP
#define PYHOLOSCAN_CONDITIONS_CUDA_STREAM_PYDOC_HPP

#include <string>

#include "../macros.hpp"

namespace holoscan::doc {

namespace CudaStreamCondition {

PYDOC(CudaStreamCondition, R"doc(
Native condition class for CUDA stream synchronization with multi-message support.

This condition supports:

- Multiple messages in the input queue (queue_size > 1)
- Multiple CudaStreamId components per message
- Multiple receiver ports (both regular and IOSpec::kAnySize multi-receiver inputs)

By default, this condition examines ALL messages in all receiver queues and waits for
GPU work on ALL associated CUDA streams to complete before allowing the operator
to execute. This behavior can be changed by setting ``check_all_messages=False``.

The condition uses ``cudaLaunchHostFunc()`` to register callbacks that fire when
GPU work on each stream completes. It returns ``WAIT_EVENT`` status while waiting
for callbacks, then transitions to ``READY`` when all callbacks have fired.

**Note**: This condition does NOT consume messages - it only peeks at them.
The operator's compute() method is responsible for actually receiving the messages.

Parameters
----------
fragment : holoscan.core.Fragment or holoscan.core.Subgraph
    The fragment (or subgraph) the condition will be associated with.
receiver : str, optional
    **DEPRECATED** - Use ``receivers`` instead. Legacy API for a single input port
    to monitor for CUDA streams. Cannot be used together with ``receivers``.
    Using this parameter will log a deprecation warning.
receivers : str or list of str, optional
    Name(s) of input port(s) to monitor for CUDA streams. Can be a single string
    like ``"input"`` or a list like ``["input1", "input2"]``. Works with both:

    - Regular ports: specified by exact name (e.g., ``"input"``)
    - Multi-receiver ports (IOSpec::kAnySize): specified by base name (e.g.,
      ``"receivers"``), which automatically expands to find all ports matching
      the pattern (e.g., "receivers:0", "receivers:1", etc.)

    Cannot be used together with ``receiver``.
check_all_messages : bool, optional
    If True (default), checks ALL messages in the queue(s) for CudaStreamId components
    and waits for all associated streams. If False, only checks the first message
    per receiver.
name : str, optional
    The name of the condition.

Examples
--------
>>> # Single regular input port (legacy API)
>>> stream_cond = CudaStreamCondition(fragment, receiver="input")
>>> # Single regular input port (new API)
>>> stream_cond = CudaStreamCondition(fragment, receivers="input")
>>> # Multiple regular input ports
>>> stream_cond = CudaStreamCondition(fragment, receivers=["input1", "input2"])
>>> # Multi-receiver port (e.g., HolovizOp's "receivers")
>>> stream_cond = CudaStreamCondition(fragment, receivers="receivers")
>>> # Check only first message per receiver
>>> stream_cond = CudaStreamCondition(fragment, receivers="input", check_all_messages=False)
>>> op = MyOperator(fragment, stream_cond, name="my_op")
)doc")

PYDOC(receiver, R"doc(
**DEPRECATED** - The single receiver associated with the condition (legacy API).
Use ``receivers`` instead.
)doc")

PYDOC(receivers, R"doc(
The list of receivers associated with the condition.
)doc")

PYDOC(check_all_messages, R"doc(
Whether to check all messages in the queue(s) (True) or only the first per receiver (False).
)doc")

}  // namespace CudaStreamCondition

}  // namespace holoscan::doc

#endif /* PYHOLOSCAN_CONDITIONS_CUDA_STREAM_PYDOC_HPP */
