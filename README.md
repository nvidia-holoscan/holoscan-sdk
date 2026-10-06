# Holoscan SDK

The **Holoscan SDK** is part of [NVIDIA Holoscan](https://developer.nvidia.com/holoscan-sdk), the AI sensor processing platform that combines hardware systems for low-latency sensor and network connectivity, optimized libraries for data processing and AI, and core microservices to run streaming, imaging, and other applications, from embedded to edge to cloud. It can be used to build streaming AI pipelines for a variety of domains, including Medical Devices, High Performance Computing at the Edge, Industrial Inspection and more.

## Table of Contents

- [Getting Started](#getting-started)
- [Obtaining the Holoscan SDK](#obtaining-the-holoscan-sdk)
- [Roadmap](#roadmap)
- [AI Coding Assistants](#ai-coding-assistants)
- [Troubleshooting and Feedback](#troubleshooting-and-feedback)
- [Contributing to Holoscan SDK](#contributing-to-holoscan-sdk)
- [Repository structure](#repository-structure)
- [License](#license)

## Getting Started

Start with a prebuilt [NGC container](https://docs.nvidia.com/holoscan/sdk-user-guide/setup/sdk-installation#ngc-container),
[Debian package](https://docs.nvidia.com/holoscan/sdk-user-guide/setup/sdk-installation#debian-package),
or [Python wheel](https://docs.nvidia.com/holoscan/sdk-user-guide/setup/sdk-installation#python-wheel),
then run a sample application from the user guide. Select the documentation version that
matches your installed SDK. For this 5.x development branch, follow the checked-in
[Getting Started guide](./docs/getting_started.mdx).

Building from source is documented in [DEVELOP.md](./DEVELOP.md). Running the SDK
test suite with `./run test` requires GPU access through the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

The Holoscan User Guide includes:

- An introduction to the NVIDIA Holoscan platform, including the Holoscan C++/Python SDK;
- Requirements and setup steps;
- Detailed SDK documentation, including a developer introduction, examples, and API details.

We also recommend visiting [NVIDIA HoloHub](https://github.com/nvidia-holoscan/holohub) to view
community projects and reusable components available for your Holoscan project.

## Obtaining the Holoscan SDK

The Holoscan User Guide documents several options to install and run the Holoscan SDK:

- As an [NGC Container 🐋](https://docs.nvidia.com/holoscan/sdk-user-guide/setup/sdk-installation#ngc-container)
- As a [Debian Package 📦️](https://docs.nvidia.com/holoscan/sdk-user-guide/setup/sdk-installation#debian-package)
- As a [Python Wheel 🐍](https://docs.nvidia.com/holoscan/sdk-user-guide/setup/sdk-installation#python-wheel)

Visit the [Holoscan User Guide](https://docs.nvidia.com/holoscan/sdk-user-guide/setup/sdk-installation#not-sure-what-to-choose) for
guidance to help choose which installation option may be right for your use case.

If the options above do not support your use case, you may prefer to [build the SDK from source](./DEVELOP.md).

Please review [Holoscan SDK prerequisites](https://docs.nvidia.com/holoscan/sdk-user-guide/setup/sdk-installation#prerequisites)
before getting started.

## Roadmap

See [ROADMAP.md](./ROADMAP.md) for how monthly releases are planned, what the
5.x series is aiming at, and where to follow live milestones.
Release planning is tracked with [GitHub Milestones](https://github.com/nvidia-holoscan/holoscan-sdk/milestones).
Issues labeled `needs triage` sit outside a milestone and are revisited each month.
See the [GitHub Releases](https://github.com/nvidia-holoscan/holoscan-sdk/releases) page for what shipped in recent versions.

## AI Coding Assistants

The Holoscan SDK user guide includes
[AI Coding Assistants](./docs/ai_coding_agents.mdx)
guidance for grounding AI coding assistants in the 5.x API, examples, and
compile-time diagnostics.

Holoscan also publishes coding-assistant skills in [`skills/`](./skills/README.md).
Install them from the NVIDIA catalog with the `skills` CLI:

```bash
npx skills add nvidia/skills
```

Check each skill's supported SDK versions before using it with this development
branch. See [skills/README.md](./skills/README.md) for install options, supported
workflows, and contribution limits.

## Troubleshooting and Feedback

We appreciate community discussion and feedback in support of Holoscan platform users and developers. We ask that users:

- Review the [Holoscan SDK Frequently Asked Questions](docs/hsdk_faq.mdx) document for common solutions and workarounds.
- Direct questions to the [NVIDIA Support Forum](https://forums.developer.nvidia.com/c/robotics-edge-computing/holoscan/757).
- Enter SDK issues on the [SDK GitHub Issues board](https://github.com/nvidia-holoscan/holoscan-sdk/issues).

## Contributing to Holoscan SDK

Holoscan SDK is developed internally and released as open source software. We welcome community contributions
and may include them in Holoscan SDK releases at our discretion. Please refer to the Holoscan SDK
[Contributing Guide](./CONTRIBUTING.md) for more information.

## Repository structure

The repository is organized as such:

- `benchmark/`: SDK performance coverage
- `ci/`: ABI export-macro allowlists and related CI artifacts
- `cmake/`: CMake configuration files
- `data/`: created during the build when dataset download is enabled
- `docs/`: user-guide MDX sources and images
- `examples/`: source code for the examples
- `include/`: source code for the holoscan SDK core
- `modules/`: source code for optional SDK modules, including Inference, Vision, and Visualization
- `python/`: python bindings for the holoscan SDK
- `schemas/`: FlatBuffers schema definitions
- `scripts/`: utility scripts
- `skills/`: agent-readable skills for installing and using the SDK from an AI coding assistant
- `src/`: source code for the holoscan SDK core
- `tests/`: tests for the holoscan SDK

## License

Holoscan SDK is licensed under the [Apache License, Version 2.0](./LICENSE.txt).
Third-party notices are listed in [NOTICE.txt](./NOTICE.txt).
