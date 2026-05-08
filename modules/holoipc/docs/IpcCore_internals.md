# IpcCore data maps – add/remove usage

This document lists each map (and map-like) container in `IpcCore` and where/when entries are **added** and **removed** during the object's lifetime.

---

## 1. `backends_`

**Type:** `BackendsMap` = `std::unordered_map<HandleType, std::shared_ptr<IpcBackendInterface>>`
**Purpose:** Backends keyed by `holoscan::ipc::HandleType`; used by `share_pointer`, `acquire_pointer`, and `acquire_pointer_eager`.

| When | Where |
|------|--------|
| **Add** | Constructor initializer list via `make_backends()` (e.g. CUDA_IPC backend). No dynamic insertions. |
| **Remove** | Never. Map is fixed after construction. |

---

## 2. `control_writers_`

**Type:** `ControlWritersMap` = `std::unordered_map<std::string, std::shared_ptr<TransportInterface::ControlWriter>>`
**Key:** Topic name (e.g. subscriber or publisher inbox topic name).
**Purpose:** Caches control writer instances per topic so we can send ACK/NACK/RELEASED/ACQUIRED to remote inboxes.

| When | Where |
|------|--------|
| **Add** | **`get_control_writer(topic_name)`** (or **`get_control_writer_locked(topic_name)`** when **`IpcCore::mutex_`** is already held) — If the topic is missing, **`transport_->create_control_writer(topic_name)`** and **`control_writers_.emplace(topic_name, std::move(writer))`**. Called whenever we need to send to a topic (ACK, NACK, RELEASED, or ACQUIRED from subscriber). |
| **Remove** | **1)** **`control_reader_disconnections_handler()`** — For each disconnected endpoint (publication handle no longer matched), after `subscribers_remove(inbox_topic)` and `publishers_remove(inbox_topic)`, `control_writers_.erase(inbox_topic)`.<br>**2)** **`~IpcCore()`** — `control_writers_.clear()` in destructor. |

---

## 3. `pointer_descriptors_`

**Type:** `PointerDescriptorMap` = `std::map<Key, PointerDescriptorEntry>`
**Key:** `Key` (handle type + device pointer id).
**Purpose:** Tracks shared pointer descriptors and their ref counts (one ref per ACQUIRED from a subscriber). When ref count goes to 0, `descriptor_` is reset but the map entry is **not** erased—so as long as the user holds the descriptor (from `share_pointer`), new subscribers can still acquire via the same key (`add_refs` restores `descriptor_` from `weak_ref_.lock()`). The entry is removed only when the last `shared_ptr<PointerDescriptor>` is destroyed (custom deleter).

| When | Where |
|------|--------|
| **Add** | **`share_pointer()`** — After exporting the handle and creating the `PointerDescriptor`, `pointer_descriptors_.emplace(key, PointerDescriptorEntry(descriptor_shared))` adds the entry (ref count 0). |
| **Remove** | **`share_pointer()` custom deleter** — When the last `shared_ptr<PointerDescriptor>` is destroyed (user dropped descriptor and no refs remain), the deleter runs under lock and calls `pointer_descriptors_.erase(key)`. **`pointer_descriptors_remove_refs(key, n)`** only decrements refs and resets `descriptor_` when ref count reaches 0; it does **not** erase the map entry (used from **`handle_released_message()`** and **`subscribers_remove()`**). |

---

## 4. `publishers_` and `PublisherEntry`

**Types:**

- **`publishers_`:** `PublishersMap` = `std::map<std::string, PublisherEntry>` — key = publisher inbox topic name (when this process is the subscriber).
- **`PublisherEntry`:** Contains **`reply_promises_`** = `std::map<Key, std::deque<PendingAcquireReply>>` — per-key FIFO of pending acquire replies for that publisher (deque allows rolling back the last enqueue if **ACQUIRED** send fails after **`push_reply_promise`**).

**Purpose:** When this process is the **subscriber**, track per-publisher pending ACK/NACK promises (from **`acquire_pointer`** and **`acquire_pointer_eager`**). Each queued **`PendingAcquireReply`** holds a `Promise<ControlMessageType>` and an **`eager_acquire`** flag. **`handle_ack_nack_message()`** uses **`msg.reply_to_topic_name`** as the publisher inbox to find the entry, pops the FIFO head, then either **`set_value`** with **ACK**/**NACK** or, on **NACK** with **`eager_acquire`**, logs and **`std::terminate()`** without fulfilling the promise (see [EAGER_ACQUIRE.md](EAGER_ACQUIRE.md)).

| When | Where |
|------|--------|
| **Add (publisher entry or promise)** | **`validate_pointer_descriptor()`** checks one descriptor for acquire; callers call **`push_reply_promise(inbox, key, promise, eager_acquire)`** *before* sending **ACQUIRED** ( **`acquire_pointer`** posted task, **`acquire_pointer_eager`** on the caller thread after **`open_handle`** ). If **ACQUIRED** send fails or throws after enqueue, **`abort_last_reply_promise()`** pops the tail and fulfills with **NACK**. |
| **Remove (from `reply_promises_` only)** | **`handle_ack_nack_message()`** — Pops the front **`PendingAcquireReply`**; on **NACK** with **`eager_acquire`**, logs an error and **`std::terminate()`** (promise is **not** fulfilled). Otherwise **`set_value`** on the promise. If that key's deque is empty, erase the key from `reply_promises_`. **`abort_last_reply_promise()`** pops the **back** after a failed send. The publisher entry itself is **not** removed here. |
| **Remove (publisher entry from `publishers_`)** | **1)** **`publishers_remove(publisher_inbox_topic_name)`** — Erases `publishers_[publisher_inbox_topic_name]`. Called from **`control_reader_disconnections_handler()`** for each disconnected (handle, inbox_topic).<br>**2)** **`~IpcCore()`** — `publishers_.clear()` in destructor so **`acquire_pointer`** futures still waiting on ACK/NACK get broken promises. |

---

## 5. `publication_handle_to_inbox_topic_`

**Type:** `PublicationHandleToInboxTopicMap` = `std::map<PublicationHandle, std::string>` — `PublicationHandle` is the transport-opaque writer id (`std::vector<uint8_t>` in **`control_message.hpp`**), not a DDS type inside **`IpcCore`**. Maps each remote writer to that endpoint’s inbox topic name.

**Purpose:** Correlate each remote control writer (**`PublicationHandle`**) with the inbox topic name from **`msg.reply_to_topic_name`** on incoming control messages, so **`control_reader_disconnections_handler()`** can run **`subscribers_remove` / `publishers_remove`** and erase the matching **`control_writers_`** entry when that writer disconnects. Updated in the **dispatcher** on every accepted message; removed only in the disconnection handler.

| When | Where |
|------|--------|
| **Add** | **`control_reader_message_dispatcher()`** — For every control message, if `!msg.reply_to_topic_name.empty()`, sets `publication_handle_to_inbox_topic_[publication_handle] = msg.reply_to_topic_name`. |
| **Remove** | **`control_reader_disconnections_handler()`** only — For each disconnected (handle, inbox_topic), after `subscribers_remove(inbox_topic)` and `publishers_remove(inbox_topic)`, `publication_handle_to_inbox_topic_.erase(publication_handle)`. |

---

## 6. `subscribers_`

**Type:** `SubscribersMap` = `std::map<std::string, SubscriberEntry>` — key = subscriber inbox topic name, value = ref counts per key.

**Purpose:** Track remote "subscribers" (writers to our control reader): which keys each has acquired. Used to send ACK/NACK and to decrement refs on RELEASED. Updated only when **ACQUIRED** or **RELEASED** messages are received (not on ACK/NACK).

| When | Where |
|------|--------|
| **Add** | **`subscribers_add(subscriber_inbox_topic_name, key)`** — `subscribers_[subscriber_inbox_topic_name].add_ref(key)`. Called from **`handle_acquired_message()`** (ACQUIRED only). |
| **Remove** | **1)** **`handle_released_message()`** — Uses `msg.reply_to_topic_name` as inbox; finds subscriber by inbox, calls `remove_ref(key)`; if that subscriber's `ref_counts_` becomes empty, erases the subscriber entry only.<br>**2)** **`subscribers_remove(subscriber_inbox_topic_name)`** — For each key in that entry calls `pointer_descriptors_remove_refs`, then erases the subscriber entry. Called from **`control_reader_disconnections_handler()`** with the inbox_topic for each disconnected endpoint. |

**Inner map `SubscriberEntry::ref_counts_`:**

- **Add / increment:** **`SubscriberEntry::add_ref(key)`** in **`subscribers_add()`**.
- **Decrement / remove key:** **`SubscriberEntry::remove_ref(key)`** in **`handle_released_message()`** (erases key when count reaches 0).

---

## Summary table

| Map | Add | Remove |
|-----|-----|--------|
| `backends_` | Constructor via `make_backends()` | Never |
| `control_writers_` | `get_control_writer()` | `control_reader_disconnections_handler()` (per disconnected endpoint); `~IpcCore()` (clear) |
| `pointer_descriptors_` | `share_pointer()` | Deleter of descriptor only (when last `shared_ptr` to descriptor is destroyed) |
| `publishers_` | `push_reply_promise()` (from `acquire_pointer` or `acquire_pointer_eager`) | Keys/queues in `reply_promises_`: `handle_ack_nack_message()`. Publisher entry: `publishers_remove()` (from `control_reader_disconnections_handler`); `~IpcCore()` (clear) |
| `publication_handle_to_inbox_topic_` | `control_reader_message_dispatcher()` (on every message with non-empty reply_to_topic_name) | `control_reader_disconnections_handler()` only |
| `subscribers_` | `subscribers_add()` (from `handle_acquired_message`) | `handle_released_message()` (erase subscriber entry when ref_counts_ empty); `subscribers_remove()` (from `control_reader_disconnections_handler`) |

---

## Disconnection and destructor checklist

Use this to verify no resource is left behind when an endpoint disconnects or when `~IpcCore()` runs.

**`control_reader_disconnections_handler()`** — For each disconnected (publication_handle, inbox_topic), in order:

1. `subscribers_remove(inbox_topic)` — releases pointer_descriptor refs, erases subscriber entry
2. `publishers_remove(inbox_topic)` — erases publisher entry (and any pending promises)
3. `publication_handle_to_inbox_topic_.erase(publication_handle)`
4. `control_writers_.erase(inbox_topic)` — destroys the ControlWriter for that topic

**`~IpcCore()`** — In order:

1. Under lock: `publishers_.clear()` — so **`acquire_pointer`** futures waiting on ACK/NACK get broken promises
2. `io_context_->stop()` then `worker_thread_.join()` — no further posted tasks run
3. Under lock: `control_writers_.clear()` — destroy writers after the worker thread has joined (see `ipc_core.cpp` destructor comments).
