# Transport abstraction

holomq separates GPU/buffer lifecycle from the messaging backend. Applications use
`Publisher<Payload>`, `OutboundMessage<Payload>`, `InboundMessage<Payload>`, and
`Subscriber<Payload>` without including iceoryx2 headers.

**Shipped backend:** HoloMQ currently builds and links **iceoryx2 only**
(`HOLOMQ_TRANSPORT=iox2`, compile definition `HOLOMQ_TRANSPORT_IOX2=1`). The internal
`detail/transport/` layer keeps a backend contract so pub/sub code stays backend-agnostic,
but no alternate messaging backend ships in-tree today.

## CMake configuration

Configure with the default transport (SDK builds map `HOLOSCAN_MQ_BACKEND=iceoryx2` /
`HOLOMQ_DEFAULT_BACKEND=iceoryx2` to this value when `HOLOMQ_TRANSPORT` is unset):

```bash
cmake -B build -DHOLOMQ_TRANSPORT=iox2
```

`detail/transport/backend.hpp` includes `transport/iox2/backend.hpp` when
`HOLOMQ_TRANSPORT_IOX2` is defined.

There is **no** runtime backend switch and no `ContextBuilder::with_transport`.

## Performance model

The pub/sub hot path (`loan`, `send`, `try_take`) uses **concrete template ports**, not
virtual interfaces:

- `Publisher<Payload>` holds transport state in a `shared_ptr` to `detail::PublisherImpl`
  (port + sample-release registry).
- `OutboundMessage<Payload>` holds a `transport::OutboundLoan<Payload>` member.
- `Subscriber<Payload>` holds the transport port in a `shared_ptr` so `WaitSet` attachments
  survive subscriber moves.

Port methods compile to direct inline forwards into iceoryx2 APIs—no vtable on port calls.

## Backend contract

The iceoryx2 implementation under `transport/iox2/` exports these aliases via
`transport/iox2/backend.hpp`:

| Type | Role |
|------|------|
| `Runtime` | Node/process setup; `create_publisher<Payload>(name, …)` |
| `PublisherPort<Payload>` | Outbound port: `loan()`, `send()`, `update_connections()` |
| `OutboundLoan<Payload>` | Move-only loan; `payload()`, `release_key()` |
| `SubscriberPort<Payload>` | Inbound port: internal `receive()`, `has_samples()` |
| `InboundMessage<Payload>` | Move-only received sample until destroyed |

Shared entry point: `include/holomq/detail/transport/backend.hpp`.

## iceoryx2 QoS and limits

Implementation lives under `src/transport/iox2/`. Default QoS values live in
`include/holomq/limits.hpp`:

- `k_subscriber_queue_depth = 8`
- `k_subscriber_max_borrowed_samples = 0` (derive from configured queue depth)
- `k_publisher_history_depth = 0`
- `k_lossless_flow = true`
- `k_max_in_flight` mirrors iceoryx2 `max_loaned_samples` for the current lossless setting

`holomq::k_max_in_flight` is the default publisher-loan limit, not a complete
external `BufferPool` capacity recommendation. A conservative fixed pool also
covers distinct publisher-history samples and every subscriber's effective
unread queue plus borrowed-sample allowance:

```text
external_buffer_slots >=
    max_loaned_samples
  + publisher_history_depth
  + sum(effective_queue_depth[i] + max_borrowed_samples[i])
```

For `kDropOldest`, `effective_queue_depth[i]` is the configured positive queue
depth; depth one provides a freshest-only mailbox. Sharing one publication
across subscribers can reduce the exact number of distinct retained buffers;
the sum is the fail-safe bound when an embedding layer does not prove that
overlap.

Typed pub/sub builders separate static service configuration from endpoint-local
configuration. Both publisher and subscriber builders accept the same
{cpp:class}`holomq::PubSubServiceOptions`; either role can create the service
first without inventing different queue, fanout, borrowed-sample, history, or
event-wake defaults. When a service already exists, HoloMQ compares queue
policy, queue and borrow maxima, history, and each explicit non-zero subscriber
capacity, rejecting a mismatch with `kIncompatible` before creating the
endpoint. A zero subscriber capacity remains a backend-default wildcard for
compatibility. `enable_event_wake` and exact `queue_policy` are recorded as
required sample-service attributes because the backend's numeric static
configuration does not preserve those distinctions. The configured
`max_subscriber_queue_depth` is instead verified directly against the backend's
numeric subscriber maximum. Planned endpoints therefore get the same checked
static contract in either order.

Publishers use {cpp:class}`holomq::PublisherPubSubOptions` only for publisher
loan limits, release/replacement callbacks, and endpoint-local pressure response. A zero
`max_loaned_samples` keeps
the current compatibility default by deriving the value from the service's
maximum subscriber queue depth and publisher history. Subscribers use
{cpp:class}`holomq::SubscriberPubSubOptions` to request their own queue depth
and borrowed-sample allowance, bounded by the shared service maximums.
The default borrowed-sample allowance is `0`, meaning "derive from the configured
queue depth." Publisher-side service setup derives from the configured maximum
subscriber queue depth, while subscriber-side endpoint setup derives from that
subscriber's configured queue depth. This preserves the expected behavior that a
subscriber may receive and hold queued samples unless the application explicitly
chooses a smaller concurrent-borrow limit.
Safe-overflow behavior is a service-level transport property and must match for
every endpoint attached to that service. {cpp:enumerator}`holomq::QueuePolicy::kBlocking`
maps to lossless delivery. {cpp:enumerator}`holomq::QueuePolicy::kDropOldest`
maps to iceoryx2 safe overflow and honors the configured queue depth, retaining
the newest N unread samples. Depth one preserves mailbox-style newest-sample
behavior. Overflow replacement is per subscriber when the backend supports
per-recipient attribution. Blocking and safe-overflow subscribers require
separate services.

{cpp:enum}`holomq::PublisherPressurePolicy` changes only how one publisher responds when a
blocking recipient is full. `kFollowService` retains the standalone wait-until-delivered default.
`kRejectWhenFull` installs an iceoryx2 unable-to-deliver handler that returns before submitting to
the full recipient. A fanout send may still have reached another recipient before the aggregate
send reports `kBackpressured`; callers needing a retryable zero-delivery proof must therefore also
possess an exact single-recipient contract. One immutable stateless process-lifetime handler is
cold-bound only for rejecting publishers; the ordinary publisher object layout remains unchanged,
and the handler performs no work on an accepted publication.
`kBoundedRetry` installs a publisher-owned unable-to-deliver handler with the exact non-zero
`bounded_retry_budget_ns` requested during setup. The handler returns `Retry` while iceoryx2's
per-delivery elapsed time is below the budget and `DiscardDataAndFail` afterward. iceoryx2 borrows
the handler address, so the transport port retires its backend publisher before releasing the
handler and preserves that ordering across moves. Arbitrary endpoint-local budgets cannot share
the stateless rejecting handler; the bounded handler is allocated once during cold publisher
creation and is not inspected by HoloMQ on pressure-free accepted sends. An unable-to-deliver
callback records the offending receiver port identity in thread-local failure evidence. Only a
corresponding failed `send()` consumes and formats that evidence; the pressure-free accepted path
performs no identity recording, lookup, formatting, or allocation.

iceoryx2 physical sample storage is larger than the publisher loan limit alone:

```text
number_of_samples =
    max_subscribers * (subscriber_max_buffer_size + subscriber_max_borrowed_samples)
  + publisher_history_size
  + max_loaned_samples
```

Holoscan Runtime should eventually compute these values from graph topology,
queue policy, and buffer lifetime behavior. Application authors should not have
to size them manually for normal Channel use.

Subscriber-first and publisher-first service creation use the same
`publisher_history_depth` from {cpp:class}`holomq::PubSubServiceOptions`.
This removes endpoint-order dependence: a non-zero history contract is valid
even when a subscriber creates the service first, and a later endpoint with a
different history contract is rejected.

Typed pub/sub capabilities are reported through {cpp:class}`holomq::PubSubCapabilities`
({cpp:func}`holomq::Context::pubsub_capabilities` after context creation, or
{cpp:func}`holomq::default_pubsub_capabilities` before a context exists). This is setup-time
metadata for Holoscan Runtime graph finalization, not a steady-state dispatch
object. The important split is:

- HoloMQ typed pub/sub can carry a small POD payload containing inline data,
  {cpp:class}`holomq::LocalBufferDescriptor`, or a HoloMQ
  {cpp:class}`holomq::BufferHandle`.
- HoloMQ resolves {cpp:class}`holomq::BufferHandle` through the existing
  cross-process buffer plane.
- HoloMQ does **not** resolve {cpp:class}`holomq::LocalBufferDescriptor` values into Tensor/buffer
  lifetimes; that remains a Holoscan Runtime sidecar-store responsibility so
  same-process channels avoid CUDA VMM, fd-pass, and SHM ref-count import/export.
- The iox2 patch series exposes per-recipient delivery attribution for
  safe-overflow replacement. Descriptor-backed drop-oldest channels can therefore
  release the superseded recipient's runtime sidecar claim without pessimistic
  rejection or an always-on mirror for that backend. HoloMQ installs this
  replacement callback only for
  {cpp:enumerator}`holomq::QueuePolicy::kDropOldest` publishers that set
  {cpp:member}`holomq::PublisherPubSubOptions::replacement_attribution_required`; calling
  {cpp:func}`holomq::PublisherBuilder::on_sample_replaced` requests that option on the builder.
  HoloMQ does not gate on current subscriber count; Holoscan Runtime owns the
  static graph contract and can leave the option disabled when it can infer the
  affected subscriber locally. Replacement notification is separate from sample
  release notification: drop-oldest replacement identifies the recipient whose
  queue entry was overwritten, while a later sample-release event can still report
  that the physical transport sample bucket is now free. Backends without
  {cpp:enumerator}`holomq::DeliveryAttribution::kPerRecipient` still need a
  runtime fallback policy.

{cpp:class}`holomq::LocalBufferDescriptor` is intentionally only a 40-byte, standard-layout,
trivially-copyable control token (`abi_version`, runtime flags, sidecar
`store_id`, `record_index`, `generation`, and runtime `token`). It contains no
raw process-local pointers, CUDA VMM handles, file descriptors, or Tensor schema
metadata. Holoscan Runtime is expected to embed it inside a higher-level Channel
payload or sample envelope and resolve it against the runtime-owned sidecar store
only when both endpoints are in the same process.

Sample-release callbacks for GPU buffer pools are wired when creating the publisher port.

## Public API types

| Type | Role |
|------|------|
| `Publisher<Payload>` | Outbound port; `send`, `loan` via `OutboundMessage` |
| `OutboundMessage<Payload>` | Move-only loaned sample for fill + send |
| `Subscriber<Payload>` | Inbound port; `try_take`, `wait_and_take` |
| `InboundMessage<Payload>` | Move-only received sample |

Builders expose `service_options` on both roles, endpoint-local
`publisher_options` / `subscriber_options`, and focused convenience setters such
as `queue_policy`, `max_subscriber_queue_depth`, `queue_depth`, and
`max_loaned_samples`. Backend-specific QoS outside those options remains
internal.

Public `Subscriber::try_take()` / `wait_and_take()` return `expected<InboundMessage<Payload>, Error>`.
Internal `SubscriberPort::receive()` is poll-based (empty when no sample).

## Adding another backend (contributor note)

To land a new in-tree backend in the future:

1. Implement `transport/<name>/` with the aliases above (templates preferred; `.cpp` only
   for runtime setup).
2. Add a CMake branch in `holomq/CMakeLists.txt` for `HOLOMQ_TRANSPORT=<name>`.
3. Extend `detail/transport/backend.hpp` with `#elif HOLOMQ_TRANSPORT_<NAME>`.
4. Provide a small pub/sub test over your payload type.

## Out of scope

- Runtime backend plugins
- Backend-specific QoS configuration beyond
  {cpp:class}`holomq::PubSubServiceOptions`,
  {cpp:class}`holomq::PublisherPubSubOptions`, and
  {cpp:class}`holomq::SubscriberPubSubOptions`
- Virtual type-erased ports
