# Security Policy: Holoscan SDK

## Reporting a Vulnerability

If you discover a potential security vulnerability in the Holoscan SDK, please **do not open a public issue** or discuss it in public forums.

Report it through one of the following channels:

- **NVIDIA Vulnerability Disclosure Program** (preferred): [NVIDIA Product Security](https://www.nvidia.com/en-us/security/)
- **Email:** [psirt@nvidia.com](mailto:psirt@nvidia.com)
  We encourage you to encrypt sensitive details using the [NVIDIA public PGP key](https://www.nvidia.com/en-us/security/pgp-key).
- **GitHub:** On [nvidia-holoscan/holoscan-sdk](https://github.com/nvidia-holoscan/holoscan-sdk), use **Security** → **Report a vulnerability** to submit a private advisory.

Please include:

- Product name (**Holoscan SDK**), version or branch (see `VERSION` at the repository root when reporting from source), and how it was obtained (e.g., Debian package, PyPI wheel, NGC container, or built from source).
- Type of vulnerability (e.g., memory corruption, insecure defaults, logic flaw).
- Step-by-step reproduction instructions.
- Proof-of-concept code or minimal reproducer, if available.
- Impact assessment (confidentiality, integrity, availability, scope).

Detailed reports help NVIDIA PSIRT validate severity, develop fixes, and publish security bulletins when appropriate. PSIRT will acknowledge receipt and work with engineering on remediation.

## Security Architecture & Context

The **Holoscan SDK** is a C++/Python software development kit for building low-latency streaming and AI sensor processing applications. It orchestrates operators in graphs (including integration with NVIDIA GXF), uses the GPU (CUDA) for compute, and ships optional modules for visualization (e.g., Vulkan-backed Holoviz), inference (Holoinfer with ONNX Runtime and optional LibTorch), inter-process communication (optional `holoscan::ipc` and Fast DDS-based pub/sub), and **multi-node distributed execution** coordinated over gRPC between an application driver and workers.

**Classification:** Library / SDK (with optional long-running services when distributed execution or network transports are enabled).

**Primary security responsibilities:**

- **Safe composition of native code:** Correctness and robustness of the core runtime, executors, and bindings so that application-defined graphs and parameters do not lead to exploitable memory corruption or undefined behavior beyond what the underlying platform already exposes.
- **Controlled use of privileged resources:** Appropriate use of CUDA, GPU memory, filesystem, and OS APIs when executing user-supplied operator code and configuration.
- **Network and IPC surfaces (when enabled):** Behavior of optional gRPC control/data paths for distributed apps, DDS-backed pub/sub, and IPC transports that exchange application data between processes or hosts.

**Key boundaries and interfaces:**

- **Application configuration:** YAML-based application and operator configuration loaded from files or strings (`holoscan::Config` and related paths).
- **User code and extensions:** Application operators, GXF extensions, and Python code run in the same trust domain as the host process unless the deployer isolates them (containers, separate users, etc.).
- **Distributed execution:** gRPC services and clients under `public/src/core/distributed/` (e.g., `AppDriverServer`, `AppWorkerServer`, `AppDriverClient`, `AppWorkerClient`) exchange allocation, execution, and lifecycle messages; workers listen on all interfaces when started in the default distributed layout.
- **Inference:** Holoinfer loads ONNX models from caller-supplied paths and uses ONNX Runtime (and optionally LibTorch) as configured by the application.
- **Pub/sub and IPC (optional CMake options):** Fast DDS and related code paths handle discovery and messaging when those features are built and used.
- **Build and packaging:** CMake, CPack, containers, and scripts may download dependencies or datasets when options such as dataset download are enabled.

### Threat Model

The following scenarios reflect the main security concerns for this codebase, including optional and auxiliary paths:

1. **Unauthorized control of distributed execution (gRPC):** An attacker on the network who can reach driver or worker gRPC ports could invoke allocation, execution, shutdown, or health-related RPCs implemented in `app_driver/service_impl.cpp`, `app_worker/server.cpp`, and related clients, potentially disrupting pipelines or causing unintended execution if the deployment does not restrict access. The distributed stack uses insecure gRPC credentials (`grpc::InsecureServerCredentials` / `grpc::InsecureChannelCredentials`) in the current implementation, so confidentiality and integrity of control-plane traffic depend entirely on network isolation or external protections.

2. **Confidentiality and integrity of pub/sub and IPC traffic (Fast DDS / holo_ipc):** When Fast DDS or IPC transports are enabled, discovery and message exchange may expose application payloads to other participants on the same logical network or domain unless DDS security, TLS, VLANs, or equivalent controls are configured outside this repository’s defaults.

3. **Malicious or untrusted application configuration (YAML):** Loading attacker-controlled YAML via `YAML::LoadAllFromFile` and related paths in `core/config.cpp`, operator configuration (e.g., Holoviz YAML parsing), or tests’ `YAML::LoadFile` can lead to denial of service, unexpected operator wiring, or unsafe parameters if the deployer does not restrict who can write configuration files.

4. **Untrusted ONNX or Torch models and inference inputs:** Loading models from paths handled in Holoinfer’s ONNX stack (`modules/holoinfer/src/infer/onnx/core.cpp` and related code) can trigger vulnerabilities in the inference runtime or unsafe custom operators if model provenance is not controlled. Attackers with write access to model directories can replace models or engine caches.

5. **Supply chain and build-time behavior:** CMake options that download datasets or dependencies, third-party vendored or fetched components under `cmake/`, `swipat/`, and container build contexts increase exposure to compromised mirrors or tampered archives if builds are not pinned, verified, and run in trusted environments.

6. **Sensitive data in logs and telemetry:** Components such as `basic_console_logger`, async data loggers, and system resource reporting used in distributed registration can emit fragment names, addresses, or resource metrics. In regulated or multi-tenant environments, verbose logging can leak operational or patient-/process-sensitive metadata if logs are not access-controlled.

7. **GPU / native code faults affecting isolation:** Bugs in CUDA operators, GXF codelets, Vulkan presentation paths, or Python/C++ boundary code could corrupt memory or affect process stability; impact is generally bounded by the host OS process unless the platform provides stronger isolation.

### Critical Security Assumptions

- **Trusted operators and application code:** The SDK does not sandbox user-written operators or Python code; it assumes the application author and runtime operator are trusted, or that the deployer enforces isolation (e.g., separate containers or machines).

- **Trusted configuration and model artifacts:** Configuration files, ONNX/Torch models, shared libraries, and GXF extension binaries are assumed to come from trusted sources or to be integrity-checked by the deployer; the SDK does not implement a full code-signing or model attestation workflow.

- **Network placement for optional server features:** Distributed gRPC endpoints and DDS domains are assumed to be reachable only from trusted networks, or fronted by VPNs, mTLS proxies, or firewalls that compensate for use of insecure gRPC credentials inside the SDK.

- **Correct and patched platform stack:** The SDK assumes a patched OS, NVIDIA GPU drivers, CUDA, Vulkan loader/validation layers (as appropriate), and third-party runtimes (ONNX Runtime, gRPC, Fast DDS, etc.) as supplied by the platform or container image maintainer.

- **Host OS enforcement:** Process isolation and file permissions are assumed to be set correctly so that non-privileged users cannot modify binaries, models, or configs used by production pipelines. (`HOLOSCAN_ALLOW_SYSTEM_INSTALL` is a CMake build-time packaging option that affects installation paths, not a runtime security control.)

- **Physical and administrative security:** Devices running Holoscan in embedded or edge scenarios are assumed to be physically secured and administratively controlled where loss of the device would imply compromise of local data regardless of SDK behavior.

---

For general bug reports and feature requests that are **not** security-sensitive, use the [Holoscan SDK GitHub Issues](https://github.com/nvidia-holoscan/holoscan-sdk/issues) as described in the project README.
