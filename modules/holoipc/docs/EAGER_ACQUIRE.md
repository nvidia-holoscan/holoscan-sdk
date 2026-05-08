# Eager acquire

This document describes the subscriber-side **`acquire_pointer_eager`** API: the IPC handle is opened **before** the publisher’s **ACK** arrives, trading a documented safety window for lower latency. On-the-wire types are unchanged; only **local** mapping timing differs from the strict path.

The **`acquire_pointer`** API remains: it waits for **ACK** (or **NACK**) before calling `open_handle`, and returns a `Future` that completes with the pointer or empty on failure.

**Protocol spec:** [PROTOCOL_SPEC.md](PROTOCOL_SPEC.md) (section 5.1, steps 4–5, and section 5.2) describe strict vs eager subscriber behavior; this file is the library-focused design note.

---

## Motivation

In some deployments the application already knows the allocation is still valid (e.g. same pipeline phase, ordering guarantees, or a publisher that cannot free until a later barrier). Waiting for the publisher’s **ACK** adds a full control round-trip. The eager path returns a usable pointer without that wait while still participating in the control protocol.

---

## API

- **`Context::acquire_pointer_eager`** forwards to **`IpcCore::acquire_pointer_eager`**. Validation uses the same **`validate_pointer_descriptor`** checks as **`acquire_pointer`** (e.g. non-empty `reply_to_topic_name`, supported `handle_type`, valid handle bytes for that backend).

This is **not** the default on the strict acquire APIs.

**Return type:** `std::shared_ptr<void>`. Same RAII semantics: destroying the object sends **RELEASED**. No **`Future`** is returned.

---

## Implementation behavior (Holoscan IPC)

1. **`validate_pointer_descriptor()`** (same rules as **`acquire_pointer`**).
2. **`open_handle()`** via the backend (returns a `shared_ptr` whose deleter only tears down the **local** mapping for that `handle_type`—backend-specific; no **RELEASED** on the control channel yet).
3. **Synchronously** (under `IpcCore`’s mutex) **`push_reply_promise(inbox, key, promise, eager_acquire = true)`**, then send **ACQUIRED** (promise is registered first so a failed enqueue cannot follow a successful send; if the send fails or throws, **`abort_last_reply_promise()`** fulfills that promise with **NACK**). If this step fails after step 2, the local `shared_ptr` from **`open_handle`** is destroyed and the backend closes the mapping; **RELEASED** is not sent (no successful acquire path was completed).
4. **`attach_release_deleter()`** so dropping the returned pointer sends **RELEASED** on the control channel.

The remaining risk is use **before** the publisher has processed the subscriber’s **ACQUIRED** and **before** **ACK** (see below).

---

## Risk window (user accepts)

The subscriber may use memory **before** the publisher has finished processing the subscriber’s **ACQUIRED** and **before** **ACK**. Callers must ensure publisher teardown policy is compatible (e.g. does not free in a way that races with this window).

---

## NACK after eager open

If the publisher sends **NACK**, the subscriber has already mapped memory the publisher rejected. **Behavior:** **`std::terminate()`** (similar in spirit to destroying a joinable **`std::thread`**).

Strict **`acquire_pointer`** path is unchanged: **NACK** yields an empty pointer, no **`std::terminate()`**. Only queue entries marked **eager** trigger termination on **NACK**.

---

## Summary

| Aspect | `acquire_pointer` | `acquire_pointer_eager` |
|--------|------------------------------------------|-----------------------------------------------------|
| When mapping is returned | After **ACK** | After successful **open** + **ACQUIRED** send, **before** **ACK** |
| **ACQUIRED** sent | Yes | Yes |
| **RELEASED** on drop | Yes | Yes |
| **NACK** | Empty pointer | **`std::terminate()`** |
| Protocol wire format | Per PROTOCOL_SPEC | Same |

---

## Related code / examples

- **`examples/fastdds_benchmark`**: subscriber **`--eager`** uses **`acquire_pointer_eager`** on the AccelBuffer path (see **`ListenerSubscriberApp.cpp`** and **`USAGE.md`**).
