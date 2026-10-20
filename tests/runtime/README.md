# Runtime test modes

baseline runtime evidence uses two deliberately distinct modes. Test names and
failure reports must identify the mode so a low-level substrate test cannot be
mistaken for a complete accepted-plan execution.

| Mode | Required path | What it may prove | What it does not prove |
|---|---|---|---|
| `synthetic-callback` | A private runtime class or callback adapter is invoked directly with preallocated records. | One bounded algorithm, state transition, error precedence, or allocation property. | Graph compilation, plan replay, lifecycle ordering, Dispatcher selection, user callback execution, or coordinated session cleanup. |
| `direct-graph-runtime` | `Graph` compile, accepted `ExecutionPlan`, fresh runtime reconstruction, lifecycle, private non-installed `PortBinding`, Dispatcher, observed Operator callback, terminal cleanup, and `RunSession` result. | The common process-local runtime path and its bounded direct witness semantics. | Channel/HoloMQ payload retention, typed selection claims, Tensor/CUDA lifetime, external transport, or distributed execution. |

The private direct binding is test and measurement infrastructure. It uses the
same internal binding contract as physical adapters, but it is not installed,
is not a production transport, and never adds input/output queues or standalone
execution methods to the public Operator API. Channel-backed runtime evidence
is owned by HSDK5-012 and must be labeled separately when it lands.

Runtime component validation includes both focused GoogleTest cases and the
`RUNTIME_SMOKE.fake_input_and_synthetic_clock` system test. A process exit alone
is insufficient: direct-runtime tests also inspect callback ordering, terminal
state, stable diagnostics where applicable, and coordinated teardown.

## Borrowed descriptor production graphs

`BorrowedDescriptorGraph` uses compiled SDK graphs and public compute callbacks, with direct
retained and forced HoloMQ delivery. It checks descriptor allocations on the actual worker,
metadata/provenance, fanout, bounded pools, and rejected-copy/publication retry.

`BorrowedDescriptorCuda` separately qualifies delayed GPU reads after CPU callback return,
fanout, failure, cancellation, stop, restart, and completion. It uses existing Runtime Tensor
read leases and registered streams, not synthetic CUDA handles. See the maintained
[CUDA qualification and reproduction notes](borrowed_descriptor_cuda.md).
