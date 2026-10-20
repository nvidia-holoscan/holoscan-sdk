# Borrowed Image CUDA lifetime qualification

`test_borrowed_descriptor_cuda.cpp` exercises public `selected_sample()` inside real SDK compute
callbacks. A CUDA kernel reads the borrowed input Tensor and writes a separate output Tensor loan
on `ExecutionContext::cuda_stream()`. A bounded host-callback gate holds that stream until the test
releases it, while CPU compute is allowed to return. The kernel is loaded during cold test setup:
CUDA lazy module loading can synchronize a context, so putting the first launch behind the gate
would test a cold-load deadlock rather than Runtime retirement. No synchronization was added to
the production borrow, compute callbacks, or per-message path.

The source continually probes its unchanged compiler-admitted pool. Its first published slot
cannot retire or be handed out again while the delayed read is pending; it must become reusable
after completion. A consumer read counts as pending until its completion event has been recorded
and has completed, because CUDA reports a never-recorded event as complete. A fanout case runs a
second consumer on a distinct admitted stream and checks its pixel result independently before
releasing the delayed reader. Other cases submit work then fail the writer callback, reject
publication, cancel the output loan, stop, restart, or request whole-run completion. Each checks
the GPU result and eventual destruction of Runtime CUDA events and streams. Borrowed descriptors
are never retained between callbacks.

The kernel also stores each output value in a small pinned host buffer, and the test checks those
values on the host while the run is still active: the delayed reader's stream is held by the gate
and the source keeps submitting fills. Those values are not kept in managed memory. On devices
where `cudaDevAttrConcurrentManagedAccess` is 0, such as Jetson Orin, the CPU may not access any
managed allocation while GPU work is pending, and such an access faults and ends the test process
without a GTest failure. The Tensor ports stay `MemoryKind::kCudaManaged` because only the GPU
touches them: the source fills its slots with `cudaMemsetAsync` and the kernel reads and writes
them on the stream.

Distinct CUDA streams do not guarantee independent forward progress. The driver can map several
streams onto one hardware work queue (see `CUDA_DEVICE_MAX_CONNECTIONS`), and work submitted to
that queue after the blocked host callback waits for the gate. The fanout case therefore installs
the gate only after the sibling has submitted its kernel and completion event. The gate's own
60-second bound outlasts the test's 30-second polls, so a stalled observation is reported while
the gate is still held.

If a CUDA event poll times out, the failure reports the device, driver, integrated-GPU and
concurrent-managed-access attributes, `CUDA_DEVICE_MAX_CONNECTIONS` and `CUDA_MODULE_LOADING`
values, each observed stream's priority, stream and event query results, gate and ordering state,
and the source pool's CUDA readiness and wait counters. For the sibling poll, it then releases the
gate and reports whether the sibling completes. Completion shows that the sibling depended on the
gated stream, for example through a shared hardware queue or a cross-stream wait; a continued stall
points to a different dependency.

## Recorded configuration

- 2026-09-19, NVIDIA RTX A6000, driver 595.71.05, CUDA 13 SDK development container.
- Shared Release (`build-x86_64`) and Debug/static (`build-borrowed-debug-static`), GCC 13.3.0.
- All new GPU cases executed; none were skipped. Release stress ran both tests ten times.
- CUDA Compute Sanitizer 2026.1.1.0 from the host CUDA 13.2 installation was mounted read-only
  into the same SDK container; the development image itself does not include that tool.
- Memcheck with full leak checking: zero errors and zero leaked bytes/allocations.
- Racecheck: zero displayed hazards, errors, or warnings. Racecheck's shared-memory coverage is
  not a proof against arbitrary cross-stream global-memory races; the deterministic pool/event
  assertions supply the specific buffer-reuse evidence here.

The deliberate callback/publication failure cases log expected failed RunSession diagnostics.
Their GTest results and sanitizer exit codes must still succeed.

## Reproduce

Use the top-level SDK build, with a matching development image and a CUDA-capable host:

```bash
HOLOSCAN_CUDA_MAJOR_VERSION=13 ./run launch --standalone build-x86_64 \
  --run-cmd "cmake --build . --target runtime_test -j 4 && \\
    ctest -R '^RUNTIME.BorrowedDescriptorCuda' --repeat until-fail:10 \\
      --output-on-failure --timeout 120"
```

To exercise the fanout submission order with every stream on one hardware work queue:

```bash
HOLOSCAN_CUDA_MAJOR_VERSION=13 ./run launch --standalone build-x86_64 \
  --run-cmd "CUDA_DEVICE_MAX_CONNECTIONS=1 tests/runtime/runtime_test \\
    --gtest_filter=BorrowedDescriptorCuda.DelayedKernelAndFanoutPreventPrematureReuse \\
    --gtest_repeat=100"
```

For the recorded sanitizer installation:

```bash
HOLOSCAN_CUDA_MAJOR_VERSION=13 ./run launch --standalone build-x86_64 \
  --mount type=bind,src=/usr/local/cuda-13.2/compute-sanitizer,dst=/opt/borrowed-compute-sanitizer,readonly \
  --run-cmd "/opt/borrowed-compute-sanitizer/compute-sanitizer \\
    --tool memcheck --error-exitcode 86 --leak-check full \\
    tests/runtime/runtime_test --gtest_filter=BorrowedDescriptorCuda.* && \\
    /opt/borrowed-compute-sanitizer/compute-sanitizer \\
    --tool racecheck --error-exitcode 87 \\
    tests/runtime/runtime_test --gtest_filter=BorrowedDescriptorCuda.*"
```

Replace the mount source if the local toolkit is installed elsewhere. A missing device or tool
must be reported explicitly; a skipped GPU test does not establish this qualification.

This supports the existing Runtime-admitted stream path only. It does not authorize unregistered
side streams, mutable borrowed metadata, escaped host descriptor pointers, or use after session
destruction. It also does not establish allocation behavior of CV-CUDA/NVCV wrappers.
