# Holoscan SDK roadmap

Holoscan SDK is developed internally and released as open source software.
Direction is set by the NVIDIA Holoscan team through a monthly release process
that includes internal quality assurance. This page explains how that planning
is published, and where to look for what is coming next.

Nothing here is a commitment to a date, API, or feature. The live trackers
linked below are authoritative when they differ from this overview.

## How we publish plans

- **[GitHub Milestones](https://github.com/nvidia-holoscan/holoscan-sdk/milestones)**
  are the working view of the next monthly releases. Issues and pull requests
  that we expect to pursue are added to the approximate release milestone.
- Issues labeled `needs triage` are not on a milestone yet. They are revisited
  each month when priorities are set.
- **[GitHub Releases](https://github.com/nvidia-holoscan/holoscan-sdk/releases)**
  record what actually shipped.
- The [Contributing Guide](./CONTRIBUTING.md) describes how community issues
  and pull requests enter this process.

## 5.x direction

Holoscan SDK 5.x is an early-access redesign of the runtime. Most public
surfaces are **Beta** or **Prototype**; see
[API stability](./docs/api_stability.mdx)
for the current stage of each area and the capabilities planned for upcoming
releases. In summary, the 5.x series grows in stages:

- **Now (early access):** the core compose → compile → run programming model,
  typed ports, temporal contracts, explicit buffer policies, reusable subgraphs,
  lifecycle and restart control, parameter engines, deadlines, memory-kind
  awareness, and deterministic / simulated-time testing. This development branch
  also includes an [operator testing harness](./docs/testing.mdx).
- **Upcoming releases (order may change):** full pre-execution graph validation, rate-aware outputs, automatic host↔device
  transfer, fault-handling policies, GPU-resident execution, deeper runtime
  observability, capture/replay tooling, Windows support, and language bindings
  beyond C++ and Python.

Prototype APIs may change or be removed. Isolate them behind your own
interfaces if you depend on them.

## What belongs in this repository vs HoloHub

New application operators and domain samples typically land in
[HoloHub](https://github.com/nvidia-holoscan/holohub), not in this SDK.
Propose SDK changes that affect the core runtime, modules, packaging, or
documentation through an issue first, as described in the
[Contributing Guide](./CONTRIBUTING.md).

## Feedback

Use [GitHub Issues](https://github.com/nvidia-holoscan/holoscan-sdk/issues)
or the [NVIDIA Support Forum](https://forums.developer.nvidia.com/c/robotics-edge-computing/holoscan/757)
to ask about direction or to request a change. Feedback on Beta and Prototype
surfaces directly shapes what becomes Stable.
