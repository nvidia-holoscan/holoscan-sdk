# WaitSet architecture

Multiplexed blocking wait over multiple subscriber ports on the iceoryx2 backend.

**Related code:**

- `include/holomq/wait_set.hpp`, `wait_set_handle.hpp`, `src/wait_set.cpp`
- `include/holomq/subscriber.hpp` — `try_take()`, `wait_and_take()`
- `src/transport/iox2/wait_set_backend.cpp` — event service + kernel listener
- `tests/unit/test_wait_set.cpp`

**See also:** [`PUBLIC_API.md`](../PUBLIC_API.md) (WaitSet section), [`transport-abstraction.md`](transport-abstraction.md)

---

## Role

One thread often serves **multiple subscribers** (camera, control, metrics). Polling each
port with `try_take()` wastes CPU. `WaitSet` blocks once, then drains ready attachments with
`try_take()`.

For a **single** port, `Subscriber::wait_and_take()` is enough only when the wait needs no
deadline and no cancellation. It takes a `WaitStrategy` and nothing else, so once it enters a
`KernelWait` only a publication or the strategy timeout can end it. A wait that must honor a
caller deadline, be cancellable, or cover more than one lane needs a `WaitSet` even for one
subscriber, because cancellation is delivered by a `WaitTrigger` and a trigger is a wait-set
attachment. That is why the Channel layer gives every waiting subscriber its own wait set.

---

## Wait modes

All modes spin in userspace for `WaitStrategy::spin_duration` first. Post-spin behavior:

| `WaitMode` | Mechanism | Event wake |
|------------|-----------|------------|
| `Return` | Return ready/timeout | Not used |
| `TimedWait` | `sleep(timeout)` | Not used |
| `KernelWait` | Kernel multiplex wait | **Required** on at least one attachment |

`WaitStrategy::timeout` is ignored for `Return` (warning logged when set). For `TimedWait`, unset
`timeout` defaults to a 1 ms sleep. For `KernelWait`, unset `timeout` blocks indefinitely; set
`timeout` for a bounded kernel wait.

---

## IOX2 backend

iceoryx2 pub/sub does not integrate wake into `send()` alone. HoloMQ adds a **parallel
event service** per subscriber when `SubscriberServiceBuilder::enable_event_wake(true)`:

1. Publisher `send()` notifies the subscriber's event channel after a sample is queued.
2. The WaitSet registers a kernel **Listener** on that event fd (for `KernelWait` only).
3. `WaitSet::wait()` blocks until at least one attachment may have work.

**Kernel wait:** `WaitMode::KernelWait` requires at least one attached subscriber created
with `enable_event_wake(true)`. Otherwise the wait fails with an error log.

Publisher and subscriber on the same service should both enable event wake when using
`KernelWait` on a WaitSet or `Subscriber::wait_and_take`.

`WaitSet::attach` shares ownership of subscriber transport ports (via `shared_ptr`); an
attached `Subscriber` may be moved. At most one WaitSet per subscriber. `WaitSet` and
`WaitSetAttachment` may outlive each other; attachments keep the backend alive until
destroyed.

---

## Application loop

```cpp
auto waitset_or = ctx.create_waitset();
if (!waitset_or.has_value()) { /* handle error */ }
holomq::WaitSet& waitset = waitset_or.value();

auto cam_att_or = waitset.attach(cam_sub);
auto ctrl_att_or = waitset.attach(ctrl_sub);
if (!cam_att_or.has_value() || !ctrl_att_or.has_value()) { /* handle error */ }

while (running) {
    auto waited = waitset.wait();
    if (!waited.has_value()) {
        // Handle or log the wait error before retrying.
        continue;
    }
    if (waited.value() != WaitResult::ready) { continue; }

    for (WaitSetHandle h : waitset.ready_handles()) {
        if (h == cam_att_or->handle()) {
            if (auto msg_or = cam_sub.try_take(); msg_or.has_value()) { /* ... */ }
        } else if (h == ctrl_att_or->handle()) {
            if (auto msg_or = ctrl_sub.try_take(); msg_or.has_value()) { /* ... */ }
        }
    }
}
```

After IOX2 kernel wake, `ready_handles()` may be **empty** (spurious wake). Loop and call
`try_take()` — do not assume every ready handle has a sample.

---

## Limits

- iceoryx2 `Listener` / `Notifier` types are not exposed in the public API
- Only the iceoryx2 WaitSet backend is implemented
- Single waiter per `WaitSet` instance (do not call `wait()` concurrently from multiple threads)
- A `WaitTrigger` latch is monotonic and single-use; a caller that must wait again after a
  cancellation attaches a replacement trigger rather than resetting the fired one
- A trigger occupies a kernel-notification slot but is not a subscriber, so it is excluded from
  `attachment_count` and cannot on its own satisfy the event-wake capability `KernelWait` requires
