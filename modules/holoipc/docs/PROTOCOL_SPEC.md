# Holoscan IPC (holoscan::ipc) — Protocol Specification (DRAFT)

## Document status

The *DRAFT* label applies to **this specification**—the normative description of wire format, fields, and lifecycle rules—not to the maturity of the Holoscan IPC (`holoscan::ipc`) **implementation**. The `holoipc` module is a supported Holoscan SDK component; for each SDK release, the library implements the protocol as defined by this document at that revision. If this specification changes in a future release, the implementation is expected to be updated to match; until the document is explicitly promoted (for example to *STABLE*), wire- and behavior-level compatibility across Holoscan SDK versions is not guaranteed. Integrators are advised to target a specific Holoscan SDK release and to consult release notes when upgrading.

---

This document specifies the **wire format and semantics** of the Holoscan IPC protocol for zero-copy GPU (or other device) memory sharing over DDS. Implementations that conform to this specification can interoperate: publisher and subscriber may use different DDS stacks or different libraries, as long as they speak this protocol.

**Scope:** The protocol defines (1) the structure of the pointer descriptor carried on the data channel, (2) the structure and use of control messages on the control channel, and (3) the lifecycle rules (acquisition and release) that ensure memory is not freed while in use. Transport, discovery, and topic naming use standard DDS; this spec does not define DDS configuration. **When** a subscriber maps handle bytes to a local pointer relative to **ACK** may be **implementation-defined** for optional library APIs (e.g. eager acquire; see [EAGER_ACQUIRE.md](EAGER_ACQUIRE.md)), while **control-message ordering and semantics** on the wire remain as in §5. API shape and other implementation details (threading, internal data structures) are otherwise out of scope.

---

## 1. Overview

The protocol uses two logical channels over standard DDS:

| Channel | Owner | Content |
|--------|--------|--------|
| **Data topic** | Application | Carries a **pointer descriptor** (and any application-defined metadata). The descriptor identifies the shared memory and tells the subscriber where to send control messages. |
| **Control channel** | Protocol | **Control messages** (ACQUIRED, RELEASED, ACK, NACK) over **inbox topics** (one per participant). Used so the publisher can track how many subscribers hold a reference and only deallocate when all have released. |

**Goals:**

- The process that allocated the memory (the **publisher**) must not free it while any other process (a **subscriber**) still holds a reference. The protocol coordinates acquisition and release so the publisher can enforce this.
- Pointer data is not copied over the wire; only a **handle** (opaque bytes) is sent. The subscriber opens the handle in its process to obtain a local pointer (e.g. GPU pointer via CUDA IPC).
- The protocol is **backend-agnostic**: handle format and meaning are determined by a **handle type** (e.g. CUDA_IPC). New backends can be added without changing the wire format.

---

## 2. Conventions

- **MUST** / **MUST NOT**: Required for conformance; implementations that violate these are non-conformant.
- **SHOULD** / **SHOULD NOT**: Recommended; implementations should follow unless there is a strong reason not to.
- **MAY**: Optional; implementations are free to do or not do.

---

## 3. Wire format

All protocol messages use the IDL definitions below. Implementations that use a different type system (e.g. non-IDL) MUST produce and consume payloads that are equivalent to the serialization of these structures (e.g. same CDR layout when using DDS/RTPS).

### 3.1 Pointer descriptor (data channel)

The **PointerDescriptor** is carried on the application’s data topic. It may be embedded in a larger application-defined type (e.g. a struct that also carries size or layout). The descriptor is the protocol’s public wire format for the data channel.

The canonical IDL lives in `public/modules/holoipc/idl/pointer_descriptor.idl`; the excerpt below MUST match that file.

```idl
module holoscan {
  module ipc {

    enum HandleType
    {
      CUDA_IPC   // CUDA IPC memory handle (cudaIpcMemHandle_t); handle size 64 bytes (value 0)
    };

    struct PointerDescriptor
    {
      string version;                // Protocol version (for forward compatibility if the protocol changes)
      sequence<octet> key;
      HandleType handle_type;        // How to interpret handle bytes (enables multiple backends)
      sequence<octet> handle;        // Opaque handle bytes (size/format depend on handle_type)
      string reply_to_topic_name;    // Topic where publisher receives ACQUIRED/RELEASED (subscriber sends reply here)
    };

  }; // module ipc
}; // module holoscan
```

- **key**: Opaque bytes that uniquely identify the pointer for the lifetime of the share. The publisher MUST use a value that is unique among pointers it currently shares. The publisher MAY reuse a key after all subscribers have released the associated pointer, but MUST ensure that straggler control messages (e.g. delayed **RELEASED** from reordering or retransmission) for the old pointer cannot affect the publisher’s ref count for a new pointer that reuses the same key octets. Implementations SHOULD incorporate a generation counter, timestamp, or equivalent distinguisher in the key, or delay reuse until a quiescence period has elapsed, to satisfy that requirement. The same key MUST be used in all control messages (ACQUIRED, RELEASED, ACK, NACK) referring to this pointer. **How** a publisher encodes correlation into those octets is **implementation-defined**; other participants MUST treat `key` as an opaque blob and echo it verbatim in control messages. (The Holoscan IPC reference library uses a tight packed encoding documented in `key.hpp`.)
- **handle_type**: Declares how the handle bytes were produced and how they must be interpreted. Receivers that do not support the given handle_type MUST treat acquisition as failed (e.g. by not sending ACQUIRED, or by having the local implementation fail the acquire call before sending ACQUIRED, or by sending NACK where applicable so no reference is added on the publisher).
- **handle**: Opaque bytes. Format and length are defined by the handle_type (e.g. CUDA IPC handle format). The publisher produces these when sharing; the subscriber consumes them when acquiring.
- **reply_to_topic_name**: The DDS topic name where the **publisher** receives control messages (ACQUIRED, RELEASED). The subscriber MUST send ACQUIRED and RELEASED to this topic so the publisher can associate them with the correct participant (see §5).

### 3.2 Control message (control channel)

The **ControlMessage** is used only on the control channel (inbox topics). It is not part of the application data topic type; the application typically does not see this type. It is specified so that any implementation can produce and consume the same wire format.

The canonical IDL lives in `public/modules/holoipc/idl/control_message.idl`; the excerpt below MUST match that file.

```idl
module holoscan {
  module ipc {

    enum ControlMessageType
    {
      ACQUIRED,   // Subscriber acquired the pointer (increment ref count)
      RELEASED,   // Subscriber released the pointer (decrement ref count)
      ACK,        // Publisher acknowledges ref added; subscriber may use buffer
      NACK        // Publisher did not have buffer (e.g. timeout); subscriber must fail acquire
    };

    struct ControlMessage
    {
      string version;                   // Protocol version (see protocol spec)
      sequence<octet> key;
      ControlMessageType message_type;  // Type of control message
      string reply_to_topic_name;       // Sender inbox (where they receive control messages)
    };

  }; // module ipc
}; // module holoscan
```

- **key**: MUST match the **key** of the PointerDescriptor for the pointer this control message refers to.
- **message_type**: One of ACQUIRED, RELEASED, ACK, NACK. Semantics in §5.
- **reply_to_topic_name**: The inbox topic name of the **sender** of this message. The receiver uses it to send responses (e.g. publisher sends ACK/NACK to the subscriber’s inbox).

---

## 4. Data channel

- The **data topic** is defined and owned by the application. Its payload MUST include (or wholly consist of) a **PointerDescriptor** so that subscribers can identify the pointer and know where to send control messages.
- The publisher produces a PointerDescriptor (e.g. from a local pointer and handle_type), fills **version**, **key**, **handle_type**, **handle**, and **reply_to_topic_name** (the publisher’s inbox topic name), and publishes it on the data topic.
- The subscriber receives PointerDescriptor samples on the data topic. To acquire the pointer, the subscriber uses the descriptor to send ACQUIRED to the publisher’s inbox (see §5.1). No acquisition is implied by receipt alone; acquisition is triggered only when the subscriber initiates it (e.g. by calling an acquire API).
- Implementations **MAY** offer an API that opens the handle **before** **ACK** arrives (**eager** acquire, e.g. `acquire_pointer_eager`). That affects **only local** mapping timing, not which control messages are sent. See [EAGER_ACQUIRE.md](EAGER_ACQUIRE.md).
- Other metadata (e.g. size, layout) may be carried in the same or another topic; the protocol does not define it.

---

## 5. Control channel and lifecycle

Each participant (publisher and each subscriber) has an **inbox topic** on which it receives control messages. Topic names are chosen by the implementation or application; the protocol only requires that the publisher’s inbox topic name appears in **PointerDescriptor.reply_to_topic_name** and that each sender’s inbox topic name appears in **ControlMessage.reply_to_topic_name** when that sender sends a control message.

Every **ControlMessage** on the wire MUST include all fields in §3.2—**version** (see §6) as well as **key**, **message_type**, and **reply_to_topic_name**. The numbered steps below call out lifecycle semantics; they do not relax that requirement.

### 5.1 Acquisition

1. Subscriber has a PointerDescriptor (from the data topic).
2. Subscriber sends a **ControlMessage** with `message_type = ACQUIRED`, `key = descriptor.key`, and `reply_to_topic_name =` subscriber’s inbox topic name, to the topic named in **descriptor.reply_to_topic_name** (publisher’s inbox).
3. Publisher receives ACQUIRED. If it will honor the acquisition, it MUST increment the ref count for that (key, subscriber) by **one** for this message (counted semantics; see the paragraph after step 5). It then sends:
   - **ACK** to the subscriber’s inbox (topic named in the received message’s `reply_to_topic_name`) if the pointer is still valid and the ref was recorded; or
   - **NACK** to the same inbox if the pointer is no longer valid or the ref cannot be honored.
4. Subscriber receives ACK or NACK. On **ACK**, the subscriber may open the handle (using **handle_type** and **handle**) and use the pointer. On **NACK**, the subscriber MUST NOT use the pointer and MUST NOT send RELEASED for this key (no ref was added).
5. **Optional eager subscriber behavior:** An implementation **MAY** map the handle locally **before** **ACK** is received, only if **ACQUIRED** for that key has been sent (per step 2). In that mode, **NACK** after a local open indicates a **contract violation** (the subscriber holds a mapping the publisher did not grant); this specification does **not** define recovery. Implementations **MAY** end the process (e.g. **`std::terminate()`**). See [EAGER_ACQUIRE.md](EAGER_ACQUIRE.md).

Refs are per (key, subscriber). The same subscriber MAY acquire the same key multiple times (e.g. multiple library acquire calls). For interoperability, publishers MUST use **counted** ref semantics: each **ACQUIRED** that the publisher accepts (leading to **ACK**) increases the ref count for that (key, subscriber) by exactly one; each **RELEASED** decreases it by one. Implementations MUST **not** treat multiple **ACQUIRED** messages from the same subscriber for the same key as **idempotent** (collapsing them into a single ref). The pointer MUST NOT be dropped or reused until the publisher’s ref count for that key reaches zero (every accepted **ACQUIRED** matched by a **RELEASED**, modulo best-effort and liveliness as in §5.2 and §7).

### 5.2 Release

1. Subscriber is done with the pointer (e.g. drops its local reference).
2. Subscriber sends a **ControlMessage** with `message_type = RELEASED`, `key =` that pointer’s key, and `reply_to_topic_name =` subscriber’s inbox topic name, to the **publisher’s** inbox topic.
3. Publisher receives RELEASED. It MUST decrement the ref count for that (key, subscriber) by one when the count is greater than zero. If there is no ref to remove (e.g. duplicate **RELEASED** or reordering), the publisher MAY ignore the message for ref-count purposes. When the ref count for that key reaches zero (all subscribers that had acquired have sent **RELEASED**), the publisher MAY drop its own reference to the pointer and deallocate or reuse the memory. If it later publishes another **PointerDescriptor** reusing the same **key** octets, it MUST satisfy the **key** reuse and straggler-safety rules in §3.1.

The subscriber SHOULD send RELEASED exactly once per successful acquisition. For strict acquire (open only after **ACK**), that is typically one RELEASED per **ACK** received for that key. For eager acquire (local open before **ACK**), the subscriber SHOULD still send **RELEASED** once when dropping the mapping if it sent **ACQUIRED** for that acquisition—so publisher ref counts stay consistent once **ACQUIRED** is processed—even though **ACK** was not waited on. **NACK** means no ref was added, so **RELEASED** is **not** appropriate for a strict acquire that never opened; for eager acquire, **NACK** is undefined / fatal as in §5.1 step 5. Best-effort delivery is acceptable (e.g. on process exit); the publisher may use timeouts or liveliness to treat missing RELEASED as release.

### 5.3 Summary of control message flow

| Sender    | Message  | Destination        | Meaning |
|-----------|----------|--------------------|--------|
| Subscriber| ACQUIRED | Publisher’s inbox | “I am acquiring this pointer; add a ref.” |
| Subscriber| RELEASED | Publisher’s inbox  | “I am releasing this pointer; remove a ref.” |
| Publisher | ACK      | Subscriber’s inbox  | “Ref added; you may use the pointer.” |
| Publisher | NACK     | Subscriber’s inbox  | “Ref not added; do not use the pointer.” |

**Note:** **ACK** implies the publisher recorded a ref—strict subscribers should open only after **ACK**. **NACK** implies no ref—strict subscribers must not open. Eager subscribers that opened before **ACK** are covered by §5.1 step 5.

---

## 6. Version and handle type

- **version** (in PointerDescriptor and ControlMessage): Protocol version string. Implementations SHOULD set it to a value that identifies the protocol version they implement. Receivers MAY use it to detect mismatches and adapt or reject. This allows the protocol to evolve (e.g. new message types or fields) with forward compatibility. The holoscan IPC reference implementation uses the string **`"1.0"`** (`kIpcProtocolVersion` in `holoscan/ipc/detail/control_message.hpp`) for both descriptor and control messages.
- **handle_type**: Identifies how handle bytes are created and interpreted. The protocol does not define the byte layout per handle_type; that is defined by the backend (e.g. CUDA IPC). New handle types can be added without changing this specification’s wire format; implementations that do not support a given handle_type MUST treat acquisition as failed for that descriptor, consistent with the **handle_type** rules in §3.1.

---

## 7. Edge cases and responsibilities

- **Subscriber crash or disconnect:** The publisher is not required to receive RELEASED. It MAY use DDS liveliness (or equivalent) to infer that a subscriber has left and treat that subscriber as having released all refs, so it can eventually deallocate when all remaining subscribers have released.
- **Publisher crash or disconnect:** Detecting publisher loss and releasing or avoiding use of pointers that may have been freed is the **application’s responsibility**. The data topic and use of DDS liveliness are under application control; the protocol does not define a library-level helper for this.
- **Duplicate or reordered messages:** Implementations MUST associate **ACQUIRED**/**RELEASED** with (key, subscriber) using the **counted** rules in §5.1 and §5.2: each accepted **ACQUIRED** increments by one; each **RELEASED** decrements by one when possible. **ACQUIRED** MUST **not** be handled idempotently (multiple **ACQUIRED** → multiple refs). Extra **RELEASED** when the per-(key, subscriber) count is already zero MAY be ignored so counts do not go negative.
- **NACK after eager local open:** Not a valid outcome for a strict subscriber; for eager implementations, treat as non-recoverable (§5.1 step 5, [EAGER_ACQUIRE.md](EAGER_ACQUIRE.md)).

---

## 8. Implementation freedom

The following are **not** specified by this protocol and are left to implementations:

- Topic naming schemes (data topic name, inbox topic names).
- Threading and concurrency model.
- How refs are stored (e.g. key + subscriber identity).
- API shape (e.g. `share_pointer`, `acquire_pointer`, `acquire_pointer_eager`, futures, callbacks).
- Optional **eager** acquire (local open before **ACK**); **implementation-defined**; see [EAGER_ACQUIRE.md](EAGER_ACQUIRE.md).
- How handle bytes are produced or consumed for a given handle_type (e.g. CUDA IPC API usage).
- Discovery and transport configuration (standard DDS).

**Related documents:** [EAGER_ACQUIRE.md](EAGER_ACQUIRE.md) (eager subscriber acquire semantics), [IDL_TYPES.md](IDL_TYPES.md) (Fast DDS IDL layout and regeneration for the wire types above).

Conformance to this specification is determined by: (1) using the specified wire format for PointerDescriptor and ControlMessage, (2) sending and receiving control messages as described in §5, and (3) maintaining the lifecycle semantics on the wire (no deallocation while a ref is held; **RELEASED** paired with acquisitions as in §5.2). **Wire** conformance is independent of whether a subscriber library opens handles strictly after **ACK** or uses a documented eager mode. For interoperability with unknown peers, subscribers **SHOULD** follow §5.1 step 4 (use only after **ACK**) unless both ends agree on stronger guarantees outside this spec.

---

*End of protocol specification.*
