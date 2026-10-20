# CUDA Graph region

This example runs a normal CPU-scheduled source and sink around two operators captured as one CUDA
Graph:

```text
DeviceSource -> [Multiply -> Add] -> ResultSink
                CUDA Graph region
```

The operators keep their normal `compute()` implementation. Calling
`cuda_graph_capture_eligible()` on each operator and applying `RegionConstraint::cuda_graph()` to
their region allows the backend to capture those calls once and launch the fused CUDA Graph when
the region source becomes ready.

Run either execution path with the same operators:

```bash
./cuda_graph_region --normal
./cuda_graph_region --cuda-graph
```

The first command uses normal per-operator dispatch. The second applies the CUDA Graph region
constraint and replaces the two transform operators with one fused CUDA Graph region operator.

The CUDA Graph path has a deliberately narrow first-release contract:

- Region members form one linear chain with at most one ingress and one egress.
- Every captured Tensor output uses Runtime-pool storage, an exact declared shape, queue depth one,
  and one outstanding loan.
- All members and both boundary endpoints stay in one process and CUDA device. Replay uses the
  source operator's planned CUDA stream; startup capture uses a private stream in the same context.
- The outside producer retries `kResourceExhausted` or `kBackpressured` while the single fixed
  ingress slot is still owned.
- Member CPU code runs once during startup capture. Keep changing metadata, control requests,
  runtime parameter updates, lifecycle restart, and other per-frame CPU work outside the region.

Egress is published after graph submission with normal Tensor CUDA readiness. The CPU does not wait
for the graph before publication, and downstream Tensor access performs the required ordering. Use
`--normal` as the correctness comparison when diagnosing asynchronous CUDA failures. Measure both
modes on your application; fusion reduces launch overhead but one-slot ownership can limit pipeline
overlap when a downstream consumer retains the output.
