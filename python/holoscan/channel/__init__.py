# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Standalone Channel clients for processes that do not host a Graph.

``Publisher.emit`` retries only pre-submit ``NotReadyError`` and
``BackpressuredError`` when given a timeout. ``HoloscanTimeoutError`` means the
one call deadline expired; ``HoloscanInterruptedError`` is call-scoped and
leaves the subscriber usable; ``ClosedError`` is terminal and latched after
producer EOS; ``StaleEndpointError`` requires rebuilding the subscriber;
``NotSupportedError`` means the requested endpoint capability is absent;
``InvalidArgumentError`` reports misuse such as a second concurrent waiter;
and ``ResourceExhaustedError`` reports a finite resource limit.

Call ``Publisher.close()`` to publish ordered EOS. A normal ``Runtime`` context
exit or explicit ``Runtime.close()`` closes live publishers in the same way.
An exceptional context exit, object destruction, and process exit abandon live
publishers without publishing EOS. ``Subscriber.close()`` only retires the
local endpoint.

``TensorLoan`` is one-shot publication authority: every ``Publisher.emit``
attempt consumes it, including an attempt that returns an error. Async
publication therefore rejects a non-``None`` timeout before consuming a loan.
Writable NumPy and DLPack views must be released before the write context exits;
the context rejects transfer while a view is still live. A CUDA stream provider
passed to ``TensorLoan.write`` must remain alive through context exit.
"""

from __future__ import annotations as _annotations

from .._public_api import export_public_names as _export_public_names

__all__ = _export_public_names(__name__, globals())  # noqa: PLE0605

from . import aio as aio  # noqa: E402

__all__ += ["aio"]
