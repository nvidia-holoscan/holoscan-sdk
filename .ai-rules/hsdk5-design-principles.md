---
description: Use when designing or reviewing public APIs, runtime behavior, operators, or other changes that shape how users build Holoscan applications.
alwaysApply: false
---

<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# HSDK 5 Design Principles

These principles guide design decisions in the SDK. Apply them when choosing
between designs; the coding conventions in
[hsdk5-coding-conventions.md](hsdk5-coding-conventions.md) govern how the chosen
design is written.

- **Design for the user.** Prefer ergonomic, discoverable APIs and sensible
  defaults for a first-time user. Report mistakes as early as possible, ideally
  at compile time or when the graph is compiled, with a diagnostic that names
  the operator, port, or parameter involved and suggests a fix. If a feature
  needs a tutorial to be used safely, revisit its design.
- **Keep latency predictable.** Treat operator `compute()` and the runtime
  dispatch path as hot paths: avoid hidden allocations, copies, and
  synchronization there. Prefer zero-copy, device-resident data flow, and move
  expensive setup to graph compilation or operator `start()`.
- **Keep execution deterministic.** Do not introduce nondeterministic ordering,
  hidden threads, or background work. Behavior must remain reproducible under
  deterministic and simulated-time testing.
- **Compose instead of adding parallel concepts.** Operators do one thing and
  graphs compose them. Express new capabilities through the existing
  compose → compile → run model (operators, typed ports, temporal contracts,
  buffer policies, and subgraphs) rather than adding a separate mechanism.
- **Treat C++ and Python as equal citizens.** Consider both languages when
  adding a feature; the Python API should feel Pythonic rather than
  transliterated from C++.
- **Make interoperability a feature.** Tensors use DLPack so NumPy, CuPy,
  PyTorch, and JAX can share data without copies. Do not lock data into
  framework-specific containers.
- **Choose the simplest design that works.** Add complexity only when it is
  required. A few similar lines are better than a premature abstraction.
- **Keep domain-specific operators out of the core SDK.** Contribute them to
  [HoloHub](https://github.com/nvidia-holoscan/holohub) instead.
