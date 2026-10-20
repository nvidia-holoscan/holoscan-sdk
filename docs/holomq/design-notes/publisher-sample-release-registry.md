# Publisher sample release registry

**Related code:**

- `include/holomq/detail/publisher_sample_release_registry.hpp`, `src/publisher_sample_release_registry.cpp`
- `include/holomq/publisher.hpp` — `send(message, on_released)`
- `include/holomq/publisher_builder.hpp` — wires `iox2::SampleReleasedHandler` at create time
- `tests/unit/test_publisher_sample_release_registry.cpp`
- `examples/holomq_main/main.cpp`

**See also:** [`ref-count-model.md`](ref-count-model.md)

---

## Role

When a publisher sends a sample that references a holomq `Buffer` (via `BufferHandle` in
the payload), the owner `Buffer` must stay alive until iceoryx2 releases that sample's SHM
bucket. Otherwise the pool slot may be reused while a subscriber still holds a handle from
the payload.

`Publisher::send(message, on_released)` registers an optional per-send `void()` callback.
The application moves `Buffer`s into the lambda capture; holomq holds the callable until
iceoryx2 invokes `SampleReleasedHandler`, then runs and destroys the callback.

Publishers carrying only self-contained payloads can set
`PublisherPubSubOptions::enable_sample_release_callbacks` to `false`. HoloMQ then
omits both the registry and the iceoryx2 handler, avoiding an empty registry dispatch
for every released sample. Plain `send(message)` remains valid; the callback-taking
overload returns `kNotImplemented`.

---

## End-to-end flow

```text
  App                          holomq::Publisher              PublisherSampleReleaseRegistry        iceoryx2
  │                                    │                                    │                          │
  │ send(sample, [buf=move(buf)](){})  │                                    │                          │
  ├───────────────────────────────────►│ register_pending(payload_ptr, fn)  │                          │
  │                                    ├───────────────────────────────────►│ occupy slot(key, fn)     │
  │                                    │ iox2::send(sample)                 │                          │
  │                                    ├───────────────────────────────────────────────────────────────►│
  │                                    │                                    │                          │ refs held by
  │                                    │                                    │                          │ subscribers / history
  │                                    │                                    │◄───────────────────────────┤ SampleReleasedHandler
  │                                    │                                    │ dispatch(payload_ptr)    │
  │                                    │                                    │ invoke fn; ~captures     │
```

When `send()` reports a failure, the callback stays registered until the transport releases the
sample. A fanout send may fail after earlier recipients have retained the sample, so the error
result alone cannot prove that releasing the callback's captured owner leases is safe.

---

## Components

| Component | Responsibility |
|-----------|----------------|
| `PublisherBuilder::create()` | When release callbacks are enabled, creates one `PublisherSampleReleaseRegistry` per publisher and installs an iceoryx2 `SampleReleasedHandler` that forwards to `registry->dispatch(info.payload_ptr())`; otherwise creates neither |
| `Publisher::send(OutboundMessage&&, Fn&&)` | Register callback keyed by payload address and retain it until transport release, including after reported send failure; returns `kNotImplemented` when callback support is disabled or `kInvalidArgument` if a callback is already pending for that address |
| `Publisher::send(OutboundMessage&&)` | No registry interaction |

### Payload correlation key

iceoryx2 uses `&message.payload()` as the registry key. Release notifications arrive via
`SampleReleasedInfo::payload_ptr()`. The key is also exposed via `OutboundLoan::release_key()`
(`&payload()`).

### Registry storage

```cpp
std::mutex mutex_;
std::deque<Slot> slots_;
std::atomic<std::size_t> pending_count_;
```

- **Slot key:** `const void*` (payload base in publisher SHM).
- **Slot entry:** reusable move-only type-erased callback (`Entry` / `EntryImpl<Fn>`).
- **Slot state:** available, pending, or dispatching.
- **Thread safety:** mutex on all operations (handler may run on transport threads).

| Operation | Behavior |
|-----------|----------|
| `register_pending(key, fn)` | Lock; scan for a duplicate key and an available slot; prepare or reuse its callback entry; mark pending; returns `false` if `key` is already pending |
| `dispatch(payload_ptr)` | Lock; find the pending slot; mark dispatching; unlock; invoke; lock; return the slot to available |

`on_released` callables must not throw.

For {cpp:enumerator}`holomq::QueuePolicy::kDropOldest`, a setup-time
{cpp:func}`holomq::PublisherBuilder::on_sample_replaced` callback can fire
before the physical transport sample bucket is released. That replacement event
describes a recipient queue overwrite and carries the replaced payload for
runtime sidecar cleanup. It is intentionally distinct from the
{cpp:func}`holomq::Publisher::send` `on_released` event, which runs only when the transport releases the sample slot; both events
can refer to the same payload address at different points in its lifetime.

---

## iceoryx2: capacity vs `max_loaned_samples`

| Limit | Meaning |
|-------|---------|
| **`max_loaned_samples`** | Max parallel publisher `SampleMut` loans |
| **`number_of_samples`** | Physical SHM buckets in the publisher data segment |

iceoryx2 sizes the segment as:

```text
number_of_samples =
    max_subscribers × (subscriber_max_buffer_size + subscriber_max_borrowed_samples)
  + publisher_history_size
  + max_loaned_samples
```

After `send()`, the publisher loan returns, but buckets may stay alive via subscriber
queues, borrowed samples, and publisher history. The maximum concurrent pending release
callbacks is bounded by **`number_of_samples`**, not **`max_loaned_samples` alone**.

Example (1 subscriber, queue 8, borrowed 1, history 0, max_loaned 10):

```text
number_of_samples = 1 × (8 + 1) + 0 + 10 = 19
```

You may loan 10 samples at once, but up to 19 buckets can be in flight over time.

---

## Registry characteristics

The registry uses a reusable **`deque` of slots** keyed by payload address. Publisher setup sizes
that deque from the complete `number_of_samples` formula above and prepares callback entries on
the first registration for a callback type. The steady-state path scans the bounded slots but does
not allocate while the accepted service and publisher limits are respected. A zero
`max_subscribers` request uses iceoryx2's default capacity of eight; it is not interpreted as a
single-subscriber service for registry sizing.

Callbacks capture move-only `Buffer` values via a virtual `Entry` type-erasure wrapper
(`std::function` is unsuitable for move-only captures).

---

## Buffer pool sizing

Application pool depth should cover in-flight owner `Buffer`s tied to unreleased iceoryx2
samples — the same window as pending release callbacks, not `max_loaned` alone.

For `holomq_main`, pool sizing uses `max(subscriber_queue_depth, publisher_history_depth)`.
The `number_of_samples` formula above is the tighter match to iceoryx2 preallocation.
