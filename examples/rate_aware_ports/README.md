# Rate-aware input and output ports

> **User guides:** [Tensors in depth](https://docs.nvidia.com/holoscan/sdk-user-guide/data-and-payloads/tensors-in-depth#size-queue-entries-from-nominal-rates) and [Lifecycle callbacks](https://docs.nvidia.com/holoscan/sdk-user-guide/building-operators/lifecycle-callbacks#bind-a-lifecycle-selected-output-rate).

This installed example is an executable tutorial for rate declaration, propagation, automatic
queue sizing, lifecycle selection, and restart reuse. It uses scalar samples where only logical
rate and queue evidence matters, then switches to a direct-host runtime-owned Tensor pool where
physical allocation identity matters. Every scenario checks the compiled or published evidence it
describes; it does not infer success from elapsed time or console output alone.

## Mental model

Rate-aware ports cross four distinct states:

```text
compose                      compile                 run / restart
-------------------------    --------------------    -------------------------------
author rate relationships -> immutable recipes   -> candidate rate at stage barrier
author consumption policy -> logical queue plan  -> one atomic published generation
author allocation bounds  -> admitted capacity  -> active limit inside that capacity
```

Keep rate metadata and scheduling separate. A declaration such as
`nominal_rate(Rate::hz(60))` tells the compiler and downstream consumers what rate to size and
reason about. It does not make the operator run at 60 Hz. `contract()` still decides when the
operator becomes activatable (`OnClock`, `OnEach`, and so on) and how it consumes each input
(`Exact`, `DrainAll`, and related selectors).

Lifecycle-selected rates add one candidate state:

```text
bind_nominal_rate() -> pending candidate -> global stage barrier -> published generation
                              |                       |
                     same-stage query          later-stage/compute query
                         is pending                  is resolved
```

Nothing downstream observes a half-propagated candidate. If validation or materialization fails,
the candidate is rejected and the previously published generation remains authoritative.

## Choose the authoring API

| What the application knows | API to author | Meaning |
|---|---|---|
| A producer has a fixed nominal rate | `output(...).nominal_rate(rate)` | Strong output declaration used for propagation and sizing. |
| An output rate is an exact rational multiple of an input | `output(...).follows(input, ratio)` | Compile-time relationship; it does not consume the input or schedule the operator. |
| Hardware or configuration selects the rate during lifecycle work | `lifecycle_nominal_rate(stage)`, then `bind_nominal_rate(...)` | Candidate selected at the named barrier. Add `max_nominal_rate(...)` when later allocation or provider geometry needs a finite admitted ceiling. |
| A connected ingress genuinely has no usable producer rate | `input(...).expected_rate(rate)` | Weak fallback evidence. It never overrides a valid connected-producer derivation. |
| The application requires one exact queue depth | `queue_depth(n)` or `ConnectionOptions{.queue_depth = n}` | Authoritative logical depth. The compiler does not replace it with a recommendation or provider ring size. |
| The application needs a trigger or consumption rule | `contract()` | Scheduling/selection policy, deliberately separate from nominal-rate metadata. |

When depth is omitted, the compiler derives it only when rate and consumption evidence close the
sizing problem. An otherwise unresolved legacy queue retains the depth-one fallback. An explicit
depth is exact even when the compiler can calculate a different recommendation.

## Walk through the scenarios

Run the executable without arguments to execute every row, or select one row with `--mode`:

| Mode | Executed evidence | Important boundary |
|---|---|---|
| `declarations` | A fixed 60 Hz output, a `follows(input)` 1:1 output, and a `RateRatio{1, 4}` output resolve to 60, 60, and 15 Hz in a published generation. | `follows()` records rate metadata; it does not schedule, consume, or emit. |
| `fusion` | A 60 Hz camera drives `OnEach`; a 1 kHz IMU is selected with `DrainAll`; omitted depth compiles to 34 logical entries under 2/1 headroom. A second graph proves an explicit depth stays exactly 48. | A backend might choose a 64-entry physical ring for the logical depth 34, but this example has no provider report for that scalar queue and does not claim that it did. |
| `fallback` | A connected producer with genuinely undeclared rate metadata lets a 60 Hz input expectation seed depth 120 over a one-second horizon, with `input_expectation_fallback` provenance. | This is the “otherwise unresolved ingress” case, not a simulated external transport. A failed or invalid upstream derivation may not use this fallback. |
| `discover` | A camera binds 60 Hz at `kDiscover` under a 120 Hz maximum. Its same-stage query remains pending; a downstream input and 1:1 output are resolved by `kAllocate`; lifecycle and compute queries agree. | Stage-barrier publication is atomic. The downstream `expected_rate(60 Hz)` remains weaker than the connected producer. |
| `late-binding` | Separate 100 Hz selections at `kWarmUp` and `kArm` publish over finite-maximum Tensor backing admitted before those stages. A 144 Hz Arm selection is rejected against the 120 Hz maximum. | WarmUp and Arm change only the active logical limit; they cannot allocate or grow backing. The rejection deliberately emits an SDK error log while the example verifies the typed diagnostic. |
| `restart` | A Discover-bound direct-host Tensor source restarts from 60 to 30 Hz. Active pool entries change from 14 to 10 inside the admitted ceiling of 22; the allocation generation is retained and a fresh storage-binding generation is published. | `RunGenerationId` and physical allocation stay stable; `RateGenerationId` and binding identity change. Larger backing does not enlarge the new active queue limit. |

The fusion arithmetic is exact:

```text
camera interarrival horizon = 1 / 60 s
nominal IMU arrivals        = ceil(1000 / 60) = 17
default recommendation      = ceil(17 × 2 / 1) = 34 logical entries
```

In general, this example's `DrainAll` sizing path first closes the integral number of arrivals in
one activation horizon, then applies headroom:

```text
arrivals       = ceil(producer rate × activation horizon)
logical depth  = ceil(arrivals × headroom numerator / headroom denominator)
```

The 34-entry value is a release-default heuristic, not a burst-safety guarantee. The explicit
48-entry example is application authority: the compiler can warn if it is below a nominal
recommendation, but it does not silently replace it with that recommendation or a backend ring
size.

The restart row includes four retained-history entries, one producer loan, and one retirement
entry. Its pool arithmetic is:

```text
startup 60 Hz active = history 4 + queue 8  + producer loan 1 + retirement 1 = 14
maximum 120 Hz       = history 4 + queue 16 + producer loan 1 + retirement 1 = 22
restart 30 Hz active = history 4 + queue 4  + producer loan 1 + retirement 1 = 10
```

The restart therefore changes logical state without pretending that physical storage changed:

```text
startup : run=R  rate=A  allocation=P  binding=X  active=14  admitted=22
restart : run=R  rate=B  allocation=P  binding=Y  active=10  admitted=22
                 changed              changed
                         retained
```

`RunGenerationId` and allocation generation remain equal (`R`, `P`), while
`RateGenerationId` and storage-binding generation change (`A` to `B`, `X` to `Y`). The larger
compatible allocation remains retained, but its presence never raises the new active limit above
10.

## Read the source progressively

The C++ file is organized as six `run_*` lessons in the same order as the table:

1. `run_declarations()` inspects fixed and `follows()` rate origins.
2. `run_fusion()` compares compiler-derived depth with an exact explicit override.
3. `run_fallback()` proves `expected_rate()` retained fallback provenance.
4. `run_discover()` observes pending, later-barrier, and compute-time rate views.
5. `run_late_binding()` checks finite preallocation and an above-maximum rejection.
6. `run_restart()` compares published generations, provider action, pool entries, and identities.

Start with `declarations`, then add lifecycle and physical-allocation concepts in that order. The
small `--self-check-restart-evidence provider|pool` paths are registered as test-only guards. Each
deliberately supplies one missing record, verifies a clean nonzero exit, and proves that incomplete
evidence is reported rather than dereferenced.

## Build

From the `cpp/` directory against an installed SDK:

```bash
cmake -B build -S . -Dholoscan_DIR=/opt/nvidia/holoscan/lib/cmake/holoscan
cmake --build build
```

## Run

```bash
./build/rate_aware_ports
./build/rate_aware_ports --mode declarations
./build/rate_aware_ports --mode fusion
./build/rate_aware_ports --mode fallback
./build/rate_aware_ports --mode discover
./build/rate_aware_ports --mode late-binding
./build/rate_aware_ports --mode restart
```

The all-mode run ends with:

```text
rate_aware_ports declarations: PASS fixed=60/1 follows=60/1 quarter=15/1 scheduling=contract-only
rate_aware_ports fusion: PASS camera=60/1 imu=1000/1 automatic_logical_depth=34 explicit_logical_depth=48 physical_ring=provider-specific
rate_aware_ports fallback: PASS upstream=undeclared expected=60/1 origin=input_expectation_fallback depth=120
rate_aware_ports discover: PASS selected=60/1 maximum=120/1 same_stage=pending downstream_allocate=resolved compute_query=resolved
rate_aware_ports late-binding: PASS warmup=100/1 arm=100/1 maximum=120/1 above_maximum=144/1-rejected geometry=compile-concrete
rate_aware_ports restart: PASS initial_rate=60/1 restart_rate=30/1 entries=14->10 admitted=22 allocation=reused-larger binding=fresh
rate_aware_ports example complete
```

The above-maximum row is an intentional failed startup candidate, so the SDK also writes its normal
`PORT_RATE_BINDING_ABOVE_MAXIMUM` error log. The executable succeeds only after checking that exact
rejected-candidate diagnostic.
