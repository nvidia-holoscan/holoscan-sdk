# `VmmPoolImportCache` entry lifetime

How the subscriber-side GPU import cache retains VMM pool mappings and when entries are
released.

**Related code:** `src/vmm_pool_import_cache.hpp`, `vmm_pool_import_cache.cpp` (owned by `Context`).

**See also:** [`multi-publisher-shm-and-gpu-pool-accounting.md`](multi-publisher-shm-and-gpu-pool-accounting.md),
[`fd-pass-one-server-per-context-vs-per-pool.md`](fd-pass-one-server-per-context-vs-per-pool.md)

---

## Cache structure

`VmmPoolImportCache` imports a publisher CUDA VMM region over fd-pass once per
`(pool_cookie, fd_pass_socket_path)` and stores the result in:

```cpp
std::unordered_map<Key, VmmPoolImportEntry, KeyHash> entries_;
// Key = { pool_cookie, fd_pass_socket_path }
```

`import_pool_base(VmmPoolImportDescriptor)` looks up that compound key, creates an entry on
first use, and returns a `VmmPoolImportBase` summary for slot offset math. The socket path is
part of the key because cookies are opaque non-secrets scoped to one publisher FD-pass server;
cookie-only indexing would allow a hostile peer to poison a victim cookie via a different
socket.

---

## Retention policy

- **One map entry per distinct `(pool_cookie, fd_pass_socket_path)`** (typically one device or
  split-gen generation pool per publisher process / FD-pass endpoint).
- **Destroying a `ConstBuffer` does not remove the cache entry.** Short-lived imports reuse
  the warm mapping and avoid repeated fd-pass on every sample.
- **Entries are released when `VmmPoolImportCache` is destroyed** (usually with `Context`).

A new entry appears when `import_pool_base` sees a new compound key (second publisher,
publisher restart with a new VMM reservation, etc.). Slot indices on the same key share
one entry.

If the same `pool_cookie` is reused with a different `pool_total_bytes` or `slot_stride_bytes`,
`import_pool_base` fails at the implementation layer (`std::nullopt`) and logs a mismatch error.
That path is not a public API; subscriber import surfaces **`ErrorCode::kNotFound`** or
**`ErrorCode::kFailure`** via `Context::import_buffer`.

---

## Trade-off

Warm retention favors steady single-publisher streams (lower fd-pass latency). Long-lived
subscribers that observe many distinct `pool_cookie` values can accumulate CUDA imports
until `Context` teardown. There is no public API to prune individual entries while the
cache remains alive.

---

## Other retention policies

holomq retains entries until `Context` teardown. Other policies trade warm reuse against
memory growth when many distinct `pool_cookie` values appear:

| Strategy | Behavior | Trade-off |
|----------|----------|-----------|
| Retain until cache destroy | holomq default | Best reuse; growth with many cookies |
| Evict when last `ConstBuffer` drops | Remove entry when no live import views | Automatic cleanup; re-import every sample if views do not overlap |
| Explicit `prune` API | Application drops unreferenced entries on a schedule | Warm default + manual memory discipline |

`import_pool_base` is not thread-safe; any eviction policy must preserve that contract or
add locking.

Host SHM sample lifetime (iceoryx2) is separate from import-cache lifetime.
