# Maximum start delay and violation policies (`contract_max_start_delay`)

> **User guide:** [Temporal Contracts](https://docs.nvidia.com/holoscan/sdk-user-guide/building-operators/temporal-contracts) —
> this example is the cited program
> for `max_start_delay()` / `on_violation()`: the latest permitted compute start and all three
> `OnViolation` policies (`kSafeDefault`, `kEmitWitness`, `kFailRun`).

A 5 ms tick source fans out to an overrun stage plus two consumers that share
one contract shape — `max_start_delay(2ms)` with different violation policies. The overrun
stage (no start-delay budget of its own) sleeps 25 ms on every 6th tick; because it is dispatched
first on the shared serial worker lane, both consumers' already-eligible activations
for that tick start ~25 ms late, and the post-run summary labels each activation
on-time vs LATE. A `--fail-run` flag swaps the consumers for a single `kFailRun`
consumer whose first late candidate fails the whole run with a diagnostics dump.

```mermaid
flowchart LR
    ticks["ticks<br/>TickSource — OnClock{.period = 5ms}"]
    overrun["a-overrun-stage<br/>no start-delay budget; sleeps 25ms on every 6th tick"]
    safe["safe-default<br/>max_start_delay(2ms) + kSafeDefault"]
    witness["emit-witness<br/>max_start_delay(2ms) + kEmitWitness"]
    ticks --> overrun
    ticks --> safe
    ticks --> witness
```

With `--fail-run`, the consumers are replaced by one `fail-run` operator that authors
`max_start_delay(2ms).on_violation(OnViolation::kFailRun)`; the same overrun stage delays its
6th activation past the budget and the run fails closed.

## What you will learn

- How to author a maximum start delay in a temporal contract with
  `Contract::max_start_delay(2ms)` chained with
  `Contract::on_violation(holoscan::OnViolation::...)` — the two must be authored
  together; a maximum start delay without a policy (or vice versa) is rejected.
- What the maximum start delay means: a **latest-permitted-compute-start budget**
  measured from the trigger's frozen source coordinate. Transport and waiting
  behind prior work on the execution lane, including earlier invocations of
  the same operator, count toward the budget. The boundary is inclusive:
  entering compute exactly at `source_coordinate + budget` is still nominal.
  Callback duration and return time do not change the disposition.
- How a late `kSafeDefault` / `kEmitWitness` activation enters `compute()` exactly once
  with a frozen `holoscan::ActivationStatus`, read via `ctx.activation()`
  (`.reason == ActivationReason::kMaxStartDelayExceeded`, `.policy` = the compiled action),
  and how the operator branches to its own degraded path.
- What `OnViolation::kFailRun` does: the late candidate never enters `compute()`, the
  session ends with `RunTermination::kFailed`, and `session.diagnostics()` retains the
  violation evidence: warning `MAX_START_DELAY_MISS` records the observed strict miss,
  and error `RUNTIME_MAX_START_DELAY_FAIL_RUN` records the fatal policy decision.
- The standard bounded-run idiom: `compile()` → `run_async()` → observation object →
  `request_stop()` → `wait()` → print results from `main()` only after shutdown.

## Prerequisites

- Previous step in the contract family:
  [`temporal_contracts/missing_data`](../missing_data/README.md) (E10) — missing-data
  policies decide what happens when *data* is not there in time; `max_start_delay`
  decides what happens when *compute* does not start in time.
- A built or installed HSDK 5 SDK providing the `holoscan::core` CMake target.
- No GPU is required; the run is bounded (well under a second per variant).

## Build

From this `cpp/` folder, as part of the SDK examples tree the example builds with the
parent `examples/` project. As a standalone consumer, copy `CMakeLists.min.txt` over
`CMakeLists.txt`, then:

```bash
cmake -B build -S .
cmake --build build
```

The `-Dholoscan_DIR=` hint is only needed when the SDK is not in one of the default
`find_package` search prefixes listed in `CMakeLists.min.txt`.

## Run

```bash
./build/contract_max_start_delay             # kSafeDefault + kEmitWitness, exits 0
./build/contract_max_start_delay --fail-run  # kFailRun demo, exits nonzero by design
```

## Expected output

Default run (the following is one representative 12-activation run). Stop is
cooperative, so a real run may report 13 or more activations, and heavy host
scheduling can add LATE labels after values 6 and 12:

```text
[max_start_delay] config: source_period=5ms max_start_delay=2ms overrun_stall=25ms stall_every=6
[max_start_delay] safe_default: activations=12 late=2 policy=kSafeDefault
[max_start_delay] safe_default timeline: 1:on-time 2:on-time 3:on-time 4:on-time 5:on-time 6:LATE 7:on-time 8:on-time 9:on-time 10:on-time 11:on-time 12:LATE
[max_start_delay] safe_default: first_late value=6 reason=kMaxStartDelayExceeded policy=kSafeDefault action=degraded-return
[max_start_delay] emit_witness: activations=12 late=2 policy=kEmitWitness
[max_start_delay] emit_witness timeline: 1:on-time 2:on-time 3:on-time 4:on-time 5:on-time 6:LATE 7:on-time 8:on-time 9:on-time 10:on-time 11:on-time 12:LATE
[max_start_delay] emit_witness: witness value=6 reason=kMaxStartDelayExceeded policy=kEmitWitness
contract max_start_delay complete
```

`--fail-run` run (exit code 3):

```text
[max_start_delay] fail-run: max_start_delay=2ms on_violation=kFailRun overrun_stall=25ms
[max_start_delay] fail-run: on_time_activations=5 late_compute_callbacks=0
[max_start_delay] fail-run: termination=kFailed
[max_start_delay] fail-run diagnostic[0]: token=MAX_START_DELAY_MISS severity=warning
[max_start_delay] fail-run diagnostic[1]: token=RUNTIME_MAX_START_DELAY_FAIL_RUN severity=error
contract max_start_delay fail-run complete
```

Annotations: on tick 6 the overrun stage — dispatched first for each tick — sleeps
25 ms inside `compute()`. Both consumers' activations for tick 6 were already frozen
eligible when the stall began, so their compute starts miss the 2 ms latest-start
budget and enter their policy paths; the degraded path returns immediately. This graph
uses cross-operator interference because it is a deterministic witness. Any backlog
whose original source decision was already frozen can likewise consume the budget.
In the fail-run variant `late_compute_callbacks=0` is the proof that `kFailRun` never
invokes `compute()` for a late candidate: activations 1–5 are nominal and the delayed
6th candidate fails the run instead of entering user code.

## Code walkthrough

### Why the maximum start delay is a latest-*start* budget

`Contract::max_start_delay(budget)` declares a positive budget from the trigger's
frozen source coordinate to the latest permitted *compute entry*. Runtime
forms `latest_start = source_coordinate + budget` with checked arithmetic in
the admitted comparison clock. It samples that clock at the final invocation
gate, after selection and any predicate decision, immediately before entering
user code. Entry exactly at `latest_start` is nominal; a strictly later entry
follows the compiled policy. Compute completion is not bounded or preempted.

The origin depends on the trigger:

| Trigger | Frozen source coordinate |
|---|---|
| `OnEach`, `OnAnyInput` | Trigger sample capture time |
| `OnClock` | Scheduled tick |
| `OnReadable`, `OnNotified`, `OnDoorbell`, `OnSpin` | Accepted normalized readiness witness |
| `OnAligned` | Complete tuple's highest capture coordinate, or comparable closure proof |
| `OnWindow` | Terminal window-close decision |
| Input-driven `OnPredicate` | Accepted predicate-driver sample's capture coordinate |
| Clock-driven `OnPredicate` | Selected scheduled tick |

This example uses `OnEach`, so its budget begins at the triggering sample's
capture time. Transport and queueing consume that budget. A producer clock
different from the reference clock needs an admitted conversion. EOS has no
capture instant and is not evaluated against sample- or predicate-driver
budgets; when it proves an aligned or window closure, the compiled closure
coordinate supplies the origin instead.

Two subtleties matter when manufacturing violations honestly:

- **The activation already running cannot retroactively become late.** Its compute
  started on time; `max_start_delay()` is not a compute-duration budget.
- **Waiting after the original source decision is frozen can make a start late.** Here
  the overrun stage is dispatched
  first (its `a-` name sorts first under serial alphabetic dispatch) and sleeps 25 ms
  while both consumers' activations for the same tick are already frozen eligible;
  when the shared worker lane frees, those starts are ~25 ms past a 2 ms budget. This
  demonstrates a slow neighbor, but the same rule applies to a same-operator backlog
  whose source decision was frozen before the earlier activation completed.

### Why maximum start delay and policy are authored together

`max_start_delay(2ms)` without `.on_violation(...)` is invalid (and vice versa). The design
forces the author to answer "and then what?" at authoring time, from a closed set of
three dispositions, instead of leaving lateness a silent log line or an unbounded
callback hook. The policy is compiled into the plan; at runtime it is visible read-only
as `ctx.activation().policy`.

### Why the degraded path is operator-authored

Under `kSafeDefault` the Runtime does not fabricate output or skip the operator — it
invokes `compute()` exactly once with `ActivationStatus{.reason = kMaxStartDelayExceeded,
.policy = kSafeDefault}`. Only the operator knows what a safe fallback is for its
domain (hold last command, emit neutral output, return immediately). Here the degraded
branch consumes the sample, records the status for the post-run summary, and returns
immediately instead of doing nominal-path work.

### Why `kEmitWitness` exists separately from `kSafeDefault`

The invocation shape is identical; the difference is intent and evidence. `kEmitWitness`
records violation evidence runtime-side and signals that the operator may emit a witness
of the violation. The consumer here reads `ctx.activation()` and reports the violation
reason (`kMaxStartDelayExceeded`) — the "witness" line in the output — instead of silently
degrading. Use it when downstream systems must be able to *see* that compute started late.

### Why `kFailRun` never calls `compute()` and how the evidence surfaces

`kFailRun` is the terminal policy: a late candidate is consumed without invoking user
code and the run fails closed — `session.wait()` returns with
`RunTermination::kFailed` and `session.diagnostics()` carries exactly one bounded
warning `MAX_START_DELAY_MISS` record for the violating candidate plus one error
`RUNTIME_MAX_START_DELAY_FAIL_RUN` record for the terminal policy decision. Nonfatal
policies retain only the warning miss record, while a checked start-delay evaluation
failure has the distinct
`RUNTIME_MAX_START_DELAY_EVALUATION_FAILED` token. Because failure is asynchronous, `main()` polls
`session.is_running()` with a 10 s cap instead of blocking unconditionally, then
prints the termination and the diagnostics dump. All
result printing in both variants happens in `main()` after `wait()` — operator-side
`std::cout` from concurrent `compute()` calls is not line-atomic.

## HSDK 4.x comparison

HSDK 4.x had no first-class contract for bounding compute start. The closest
4.x practice was hand-rolled:

| Concern | HSDK 4.x | HSDK 5 |
|---|---|---|
| Detecting a late start | Manual `std::chrono` stamps inside `compute()` | `max_start_delay(budget)` in the contract, checked by the Runtime immediately before compute entry |
| Reacting to lateness | Ad-hoc `if` + logging, semantics per author | Closed `OnViolation` set: `kSafeDefault` / `kEmitWitness` / `kFailRun` |
| Terminal misbehavior | Process keeps running unless the author aborts | `kFailRun`: run fails closed with `RunTermination::kFailed` + diagnostics |
| Evidence | Log grep | `ctx.activation()` in-band; `session.diagnostics()` post-run |

The rationale: timing obligations are part of an operator's *contract* with the graph,
so they are declared next to triggers and selectors where the compiler and Runtime can
enforce them uniformly, rather than re-implemented differently inside every operator.

## Migrating early-access contracts

Use the guide's [complete migration map](https://docs.nvidia.com/holoscan/sdk-user-guide/building-operators/temporal-contracts#migrating-early-access-contracts)
for the renamed C++ and Python APIs, inspection types, diagnostics, and serialized
artifacts. Rebuild C++ callers and regenerate old artifacts; no compatibility
aliases or legacy-format reader are provided. Timing semantics are unchanged.

## What's next

- Next in the contract family: [`temporal_contracts/on_predicate`](../on_predicate/README.md)
  (E12) — data-dependent activation gating (preview tier).
- Guide deep-dive: building-operators/temporal-contracts, maximum start delay and violation-policy
  section (see the guide's Performance section for measurement topics).

## The contract family

This example is E11 in the family chain:
[E04 on_any_input](../on_any_input/README.md) → [E05 on_each](../on_each/README.md) →
[E06 on_clock](../on_clock/README.md) → [E07 on_aligned](../on_aligned/README.md) →
[E08 on_window](../on_window/README.md) → [E09 selectors](../selectors/README.md) →
[E10 missing_data](../missing_data/README.md) → **E11 max_start_delay** →
[E12 on_predicate](../on_predicate/README.md).

Contract vocabulary matrix — this example demonstrates the **bolded** cells:

| Vocabulary group | Options |
|---|---|
| Trigger | OnAnyInput, **OnEach**, OnClock, OnAligned, OnWindow, OnPredicate |
| Selector | **Exact** (implicit on the trigger port), LatestWithin, DrainAll, BracketedWindow |
| Missing-data policy | Skip, SafeDefault, ReuseLast |
| Maximum start delay | **max_start_delay(budget) + on_violation(kSafeDefault / kEmitWitness / kFailRun)** |

**When to choose this pattern:** add a maximum start delay whenever an operator's output loses
value (or becomes unsafe) if compute starts late — control loops, actuation, safety
interlocks. Pick `kSafeDefault` when a domain fallback exists, `kEmitWitness` when
downstream must observe the slip, and `kFailRun` when running late is worse than not
running at all.

## Concepts at a glance

| API | Role |
|---|---|
| `Contract::max_start_delay(std::chrono::nanoseconds)` | Declares the positive latest-permitted-compute-start budget from frozen source coordinate |
| `Contract::on_violation(OnViolation)` | Declares the sole action for a late final invocation gate; mandatory partner of `max_start_delay()` |
| `holoscan::OnViolation` | Closed policy set: `kSafeDefault`, `kEmitWitness`, `kFailRun` |
| `ExecutionContext::activation()` | Returns the frozen `ActivationStatus` for this callback; side-effect free |
| `holoscan::ActivationStatus` | `.reason` (`kNominal` / `kMaxStartDelayExceeded` / `kInvariantViolated`) + `.policy` (compiled action) |
| `RunSession::termination()` | `kStopped` after cooperative stop; `kFailed` after a `kFailRun` violation |
| `RunSession::diagnostics()` | Bounded post-run evidence; `kFailRun` carries warning `MAX_START_DELAY_MISS` plus error `RUNTIME_MAX_START_DELAY_FAIL_RUN` |
| `holoscan::OnClock{.period}` | Clock-driven source trigger paced by the graph's default `RealtimeClock` |
