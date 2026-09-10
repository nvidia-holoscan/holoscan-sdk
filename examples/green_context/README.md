# CUDA Green Context placement (`green_context`)

This example assigns one manually authored CUDA execution partition to a
dedicated, driver-structured SM group. A second CUDA partition is left unbound
and therefore uses the structured residual Green Context.

```text
producer --(CUDA Tensor)--> transform --(CUDA Tensor)--> sink
critical partition          background partition             background partition
dedicated Green Context     shared residual Green Context    shared residual Green Context
```

The program deliberately keeps the workload small. Its purpose is to show the
configuration sequence clearly:

1. Query the exact deployment device with
   `query_cuda_green_context_capability()`.
2. Bind both Tensor outputs to an explicit CUDA device.
3. Create named execution partitions.
4. Request one `structured_sm_quantum` for the `critical` partition.
5. Compile and inspect the resulting dedicated and residual resources.
6. Launch all application CUDA work through `ExecutionContext::cuda_stream()`.

The ordinary `receive()` call in the background operator automatically orders
its residual stream after the producer's dedicated stream. The example does not
receive a CUDA stream separately or synchronize the host between the two
kernels.

## Prerequisites

- A CUDA 13 deployment supported by the Holoscan SDK Green Context preview.
- A CUDA device and driver that report at least two structured SM units: one
  for the dedicated partition and one for residual HSDK CUDA work.

Unsupported targets print a short explanation and exit with code 77.

## Build and run

From the `cpp` directory, build against an installed SDK:

```bash
cmake -B build -S .
cmake --build build
./build/green_context
```

Representative output resembles:

```text
device: cuda.uuid...; total SMs: 132; structured quantum: 8
structured capacity: 120 SMs; unprovisioned topology tail: 12 SMs
resource: dedicated, 8 SMs
resource: shared residual, 112 SMs
partition critical -> cuda-green-context...partition.critical
partition background -> cuda-green-context...residual
green_context example complete
```

The exact counts and identifiers are target- and plan-specific. In particular,
the driver-qualified structured capacity can be smaller than
`total_sm_count / structured_sm_quantum`; the example uses the capability value
rather than reconstructing that capacity itself.

For the full guarantee and deployment limitations, see the
[CUDA Green Contexts user guide](https://docs.nvidia.com/holoscan/sdk-user-guide/advanced/cuda-green-contexts).
