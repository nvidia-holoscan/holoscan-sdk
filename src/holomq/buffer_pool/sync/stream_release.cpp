/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "buffer_pool/sync/stream_release.hpp"

#include <array>
#include <atomic>
#include <charconv>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <new>
#include <string>
#include <thread>
#include <utility>

#include <pthread.h>
#include <sched.h>
#include <unistd.h>

#include <cuda.h>
#include <holomq/logger.hpp>

namespace holomq {

struct BufferStreamRelease::Group {
  ShmRefCountMappingPtr mapping;
  std::uint32_t slot{};
  Owners owners;
  std::unique_ptr<Entry> arming;
  std::atomic_size_t remaining{};
  void resolve() noexcept {
    if (remaining.fetch_sub(1U, std::memory_order_acq_rel) == 1U) {
      mapping->table().release_ref(slot);
    }
  }
};

struct BufferStreamRelease::Entry : cuda::StreamLifetime::DeferredRelease {
  std::shared_ptr<const cuda::StreamLifetime> lifetime;
  std::shared_ptr<Group> group;  // destroy GPU mappings before the lifetime's native context owner
  CUstream stream{};
  CUcontext context{};
  CUdevice device{};
  Entry* queue_next{};
  enum class Kind { kConsumer, kCompleted, kEndpoint, kEndpointCompleted, kContextRelease };
  Kind kind{Kind::kConsumer};
  bool ready_outside_capture{};
  Owners backing;
  std::shared_ptr<cuda::StreamLifetime> owned_lifetime;
};

namespace {
using Entry = BufferStreamRelease::Entry;
using Lifetime = cuda::StreamLifetime;

// Preserve the process CPU set instead of inheriting the first importing lane's pin.
const cpu_set_t kInitialCPUAffinity = [] {
  cpu_set_t allowed{};
  if (sched_getaffinity(getpid(), sizeof(allowed), &allowed) != 0) {
    CPU_ZERO(&allowed);
  }
  return allowed;
}();

// Remove the kernel-isolated CPUs (isolcpus) listed in sysfs, such as "2-9,20-27", from `mask`.
// The Python Tensor release worker in imported_tensor.cpp keeps the same rule; change both.
void remove_isolated_cpus(cpu_set_t& mask) noexcept {
  std::string list;
  try {
    std::ifstream file{"/sys/devices/system/cpu/isolated"};
    std::getline(file, list);
  } catch (...) {
    return;
  }
  const char* position = list.data();
  const char* const end = list.data() + list.size();
  while (position < end) {
    int first = 0;
    auto parsed = std::from_chars(position, end, first);
    if (parsed.ec != std::errc{}) {
      return;
    }
    int last = first;
    position = parsed.ptr;
    if (position < end && *position == '-') {
      parsed = std::from_chars(position + 1, end, last);
      if (parsed.ec != std::errc{}) {
        return;
      }
      position = parsed.ptr;
    }
    for (int cpu = first; cpu <= last && cpu < CPU_SETSIZE; ++cpu) {
      if (cpu >= 0) {
        CPU_CLR(cpu, &mask);
      }
    }
    if (position < end && *position == ',') {
      ++position;
    } else {
      break;
    }
  }
}

// The kernel never balances a thread onto or off an isolated CPU, so a worker allowed there can
// stay behind a busy pinned thread, such as a Channel pump, and delay every retirement. Keep it on
// the process's other CPUs; use the whole process set only when nothing else remains.
const cpu_set_t kWorkerCPUAffinity = [] {
  cpu_set_t allowed = kInitialCPUAffinity;
  remove_isolated_cpus(allowed);
  return CPU_COUNT(&allowed) != 0 ? allowed : kInitialCPUAffinity;
}();

// Keep bootstrap placement and final affinity restoration together; CPU-set macros inflate
// the score even though the control flow is a single bounded mask subtraction.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
void restore_worker_affinity(pthread_t thread, bool creator = false) noexcept {
  if (creator) {
    cpu_set_t creator_mask{};
    cpu_set_t bootstrap = kWorkerCPUAffinity;
    if (sched_getaffinity(0, sizeof(creator_mask), &creator_mask) == 0) {
      for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
        if (CPU_ISSET(cpu, &creator_mask)) {
          CPU_CLR(cpu, &bootstrap);
        }
      }
      // Expanding affinity alone need not move a child off an isolated real-time CPU.
      // Force a first placement outside the creator's pin, then restore the full mask.
      if (CPU_COUNT(&bootstrap) != 0) {
        static_cast<void>(pthread_setaffinity_np(thread, sizeof(bootstrap), &bootstrap));
      }
    }
  }
  if (CPU_COUNT(&kWorkerCPUAffinity) == 0 ||
      pthread_setaffinity_np(thread, sizeof(kWorkerCPUAffinity), &kWorkerCPUAffinity) != 0) {
    log_literal(LogLevel::Error, "buffer release worker could not restore process CPU affinity");
  }
}

// Endpoint streams must not borrow a callback's Green Context or a caller's temporary
// context. Aliasing lifetime pointers retain this primary context independently of the
// native stream, including mapping cleanup queued after successful retirement.
struct PrimaryContextOwner {
  CUcontext context{};
  CUdevice device{};
  std::unique_ptr<Entry> cleanup{std::make_unique<Entry>()};
  ~PrimaryContextOwner();
};
struct EndpointLifetime {
  std::shared_ptr<PrimaryContextOwner> context;
  Lifetime lifetime;
  EndpointLifetime(std::shared_ptr<PrimaryContextOwner> owner, CUstream stream)
      : context(std::move(owner)), lifetime(stream, context->device) {}
};

void enqueue_completed(Entry* entry) noexcept;
void CUDAAPI completed(void* data) {
  auto* entry = static_cast<Entry*>(data);
  entry->group->resolve();
  entry->kind = Entry::Kind::kCompleted;
  // Unmapping payloads or unregistering pinned memory can call CUDA. Cleanup runs on the
  // worker; this callback only drops the SHM reference and publishes its preallocated node.
  enqueue_completed(entry);
}

// Every failure deliberately retains the preallocated entry, including its mapping and
// group reference. Neither a failed enqueue nor a closed admission proves memory safe.
void retain_failed_release() noexcept {
  log_literal(LogLevel::Critical, "buffer release completion unproven; retaining slot and mapping");
}

// Called only on the worker, never from an outcome callback or CUDA host function.
void report_retirement_failure(const char* operation, CUdevice device, CUresult result) noexcept {
  const char* error_name{};
  if (cuGetErrorName(result, &error_name) != CUDA_SUCCESS || error_name == nullptr) {
    error_name = "unknown";
  }
  std::array<char, 256> message{};
  std::snprintf(message.data(),
                message.size(),
                "%s failed (device=%d, CUDA result=%d [%s]); retaining retirement resources",
                operation,
                static_cast<int>(device),
                static_cast<int>(result),
                error_name);
  log_literal(LogLevel::Critical, message.data());
}

void defer_closing(Entry* entry, const std::shared_ptr<const Lifetime>& lifetime) noexcept {
  entry->on_outcome = [](Lifetime::DeferredRelease& node, Lifetime::State outcome) noexcept {
    if (outcome == Lifetime::State::kRetired) {
      completed(static_cast<Entry*>(&node));
    }
    // A quarantined owner already reports its failed barrier. Retain this node silently:
    // user log sinks can lock or wait, so outcome delivery must not invoke them.
  };
  if (!lifetime->defer(*entry)) {
    entry->on_outcome(*entry, lifetime->state());
  }
}

void retire_owner(Entry* entry) noexcept {
  const auto lifetime = entry->owned_lifetime;
  if (entry->kind == Entry::Kind::kEndpointCompleted) {
    const auto selected_result = cuCtxPushCurrent(entry->context);
    const bool selected = selected_result == CUDA_SUCCESS;
    const auto destroyed = selected ? cuStreamDestroy(entry->stream) : selected_result;
    if (selected) {
      CUcontext previous{};
      static_cast<void>(cuCtxPopCurrent(&previous));
    }
    if (destroyed == CUDA_SUCCESS) {
      delete entry;
    } else {
      report_retirement_failure(
          selected ? "endpoint stream destruction" : "endpoint cleanup context selection",
          entry->device,
          destroyed);
    }
    return;
  }
  lifetime->close_admission();
  lifetime->wait_for_pins();
  const auto selected_result = cuCtxPushCurrent(entry->context);
  const bool selected = selected_result == CUDA_SUCCESS;
  // A foreign generation wait may never finish. Only enqueue its tail proof here:
  // neither unrelated leases nor process exit may wait for this endpoint on the shared worker.
  const auto queued = selected ? cuLaunchHostFunc(
                                     entry->stream,
                                     [](void* data) {
                                       auto* owner = static_cast<Entry*>(data);
                                       if (owner->owned_lifetime->publish_retired()) {
                                         owner->kind = Entry::Kind::kEndpointCompleted;
                                         enqueue_completed(owner);
                                       }
                                     },
                                     entry)
                               : selected_result;
  if (selected) {
    CUcontext previous{};
    static_cast<void>(cuCtxPopCurrent(&previous));
  }
  if (queued != CUDA_SUCCESS) {
    static_cast<void>(lifetime->publish_quarantined());
    report_retirement_failure(
        selected ? "endpoint retirement tail enqueue" : "endpoint retirement context selection",
        entry->device,
        queued);
  }
}

// Keep lifetime-state dispatch beside the pin and CUDA context boundary so every outcome
// either proves completion or retains the entry without destroying live backing.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
void prepare(Entry* entry) noexcept {
  const auto lifetime = entry->lifetime;  // keep the pin target alive if a callback deletes entry
  if (lifetime != nullptr) {
    const auto state = lifetime->state();
    if (state == Lifetime::State::kRetired) {
      completed(entry);
      return;
    }
    if (state == Lifetime::State::kClosing) {
      defer_closing(entry, lifetime);
      return;
    }
    if (state == Lifetime::State::kQuarantined) {
      retain_failed_release();
      return;
    }
  }
  auto admission = lifetime != nullptr ? lifetime->try_pin() : Lifetime::PinResult{};
  if (lifetime == nullptr || admission.pin) {
    // Admission also protects the context that owns an SDK stream. Only thread-context
    // switches, nonblocking queries, and the nonblocking enqueue occur here; never synchronize
    // while pinned.
    const bool selected = cuCtxPushCurrent(entry->context) == CUDA_SUCCESS;
    auto* const stream = lifetime != nullptr ? admission.pin.stream() : entry->stream;
    // Every reader enqueued its work on this stream before the last owner released the buffer,
    // so a stream with nothing pending already proves completion. Returning the slot now skips
    // CUDA's host-function delivery, which can take far longer than the work it waits for.
    // Querying a stream that is being captured invalidates its capture, so query only a stream
    // known not to be capturing. Any other stream keeps the host function, which a capture
    // records, so the slot returns when the captured graph runs.
    CUstreamCaptureStatus capture = CU_STREAM_CAPTURE_STATUS_ACTIVE;
    const bool idle = selected && cuStreamIsCapturing(stream, &capture) == CUDA_SUCCESS &&
                      capture == CU_STREAM_CAPTURE_STATUS_NONE &&
                      cuStreamQuery(stream) == CUDA_SUCCESS;
    const auto result = idle       ? CUDA_SUCCESS
                        : selected ? cuLaunchHostFunc(stream, completed, entry)
                                   : CUDA_ERROR_INVALID_CONTEXT;
    const bool fallback_completed = lifetime == nullptr && selected && result != CUDA_SUCCESS &&
                                    cuStreamSynchronize(stream) == CUDA_SUCCESS;
    if (selected) {
      CUcontext previous{};
      static_cast<void>(cuCtxPopCurrent(&previous));
    }
    admission.pin.reset();
    if (idle) {
      completed(entry);
    } else if (result != CUDA_SUCCESS) {
      // Caller-owned streams retain the existing synchronous fallback contract.
      if (fallback_completed) {
        completed(entry);
      } else {
        retain_failed_release();
      }
    }
  } else if (admission.state == Lifetime::State::kClosing) {
    defer_closing(entry, lifetime);
  } else if (admission.state == Lifetime::State::kRetired) {
    completed(entry);
  } else {
    retain_failed_release();
  }
}

// Worker only. Retaining a primary context that nothing else holds creates it, which takes hundreds
// of milliseconds, and the matching release destroys it again. A process that works only in its
// own contexts would pay both on every cleanup. Keep the first retain for each device until
// process exit, so later retains and releases only move the reference count. True means the
// caller's retain is now held and must not be released.
[[nodiscard]] bool hold_primary_context(const CUdevice device) noexcept {
  static std::array<CUdevice, 64> held{};
  static std::size_t held_count{};
  for (std::size_t index = 0U; index < held_count; ++index) {
    if (held[index] == device) {
      return false;
    }
  }
  if (held_count == held.size()) {
    return false;
  }
  held[held_count++] = device;
  return true;
}

// Process one node independently of the queue bookkeeping and wake protocol.
void process_release(Entry* entry) noexcept {
  if (entry->kind == Entry::Kind::kContextRelease) {
    const auto released = cuDevicePrimaryCtxRelease(entry->device);
    if (released == CUDA_SUCCESS) {
      delete entry;
    } else {
      report_retirement_failure("endpoint primary context release", entry->device, released);
    }
  } else if (entry->kind == Entry::Kind::kEndpoint ||
             entry->kind == Entry::Kind::kEndpointCompleted) {
    retire_owner(entry);
  } else if (entry->kind == Entry::Kind::kCompleted) {
    const auto lifetime = entry->lifetime;  // retain endpoint context through the pop
    CUcontext context{};
    const CUdevice device = entry->device;
    const bool retained = cuDevicePrimaryCtxRetain(&context, device) == CUDA_SUCCESS;
    const bool selected = retained && cuCtxPushCurrent(context) == CUDA_SUCCESS;
    if (selected) {
      delete entry;
      CUcontext previous{};
      static_cast<void>(cuCtxPopCurrent(&previous));
    } else {
      log_literal(LogLevel::Critical,
                  "buffer cleanup context unavailable; retaining mapped storage");
    }
    if (retained && !hold_primary_context(device)) {
      static_cast<void>(cuDevicePrimaryCtxRelease(device));
    }
  } else {
    prepare(entry);
  }
}

// SDK buffers can be dropped inside a CUDA host function. Its thread must not call CUDA,
// wait, or take a mutex. Metadata and the worker are reserved when the stream is admitted.
class ReleaseWorker {
 public:
  ReleaseWorker() : thread_([this] { run(); }) {
    // Restore from the creator as well: the inherited real-time CPU may never schedule us.
    restore_worker_affinity(thread_.native_handle(), true);
  }
  void enqueue(Entry* entry) noexcept {
    if (stopped_.load(std::memory_order_acquire)) {
      return;  // late releases retain the entry; the inert worker remains allocated
    }
    pending_.fetch_add(1U, std::memory_order_relaxed);
    auto* head = head_.load(std::memory_order_relaxed);
    do {
      entry->queue_next = head;
    } while (!head_.compare_exchange_weak(
        head, entry, std::memory_order_release, std::memory_order_relaxed));
    wake();
  }
  void drain() noexcept {
    for (;;) {
      const auto sequence = sequence_.load(std::memory_order_acquire);
      if (pending_.load(std::memory_order_acquire) == 0U ||
          stopped_.load(std::memory_order_acquire)) {
        return;
      }
      sequence_.wait(sequence, std::memory_order_acquire);
    }
  }
  void shutdown() noexcept {
    stopped_.store(true, std::memory_order_release);
    wake();
    if (thread_.joinable()) {
      thread_.join();
    }
  }

 private:
  void wake() noexcept {
    sequence_.fetch_add(1U, std::memory_order_release);
    sequence_.notify_all();
  }
  void run() noexcept {
    restore_worker_affinity(pthread_self());
    static_cast<void>(pthread_setname_np(pthread_self(), "holomq-retire"));
    for (;;) {
      const auto sequence = sequence_.load(std::memory_order_acquire);
      auto* batch = head_.exchange(nullptr, std::memory_order_acquire);
      if (batch == nullptr) {
        if (stopped_.load(std::memory_order_acquire)) {
          return;
        }
        sequence_.wait(sequence, std::memory_order_acquire);
        continue;
      }
      while (batch != nullptr) {
        auto* entry = batch;
        batch = entry->queue_next;
        process_release(entry);
        if (pending_.fetch_sub(1U, std::memory_order_acq_rel) == 1U) {
          wake();
        }
      }
    }
  }
  std::atomic<Entry*> head_{};
  std::atomic_uint64_t sequence_{};
  std::atomic_bool stopped_{};
  std::atomic_size_t pending_{};
  std::thread thread_;
};

ReleaseWorker* created_worker{};
ReleaseWorker& worker() {
  static auto* instance = [] {
    auto* result = new ReleaseWorker;
    created_worker = result;
    if (std::atexit([] { created_worker->shutdown(); }) != 0) {
      // Keeping the worker alive is safe; process exit does not destroy its thread object.
      log_literal(LogLevel::Error, "could not register buffer release worker shutdown");
    }
    return result;
  }();
  return *instance;
}
void enqueue_completed(Entry* entry) noexcept {
  created_worker->enqueue(entry);
}

PrimaryContextOwner::~PrimaryContextOwner() {
  if (context != nullptr) {
    cleanup->device = device;
    cleanup->kind = Entry::Kind::kContextRelease;
    enqueue_completed(cleanup.release());
  }
}
}  // namespace

BufferStreamRelease::BufferStreamRelease() noexcept = default;
BufferStreamRelease::~BufferStreamRelease() = default;
bool BufferStreamRelease::defer_cuda_owners(Owners& owners, const int device) noexcept {
  try {
    static_cast<void>(worker());
    auto cleanup = std::make_unique<Entry>();
    cleanup->device = device;
    cleanup->kind = Entry::Kind::kCompleted;
    cleanup->backing = owners;
    // The shared_ptr constructor invokes this deleter even when its control allocation fails.
    // Transfer the node first; leave the caller's owners intact until wrapping succeeds.
    auto owner = std::shared_ptr<void>(cleanup.release(), [](void* data) noexcept {
      enqueue_completed(static_cast<Entry*>(data));
    });
    owners = {std::move(owner), {}};
    return true;
  } catch (...) {
    return false;
  }
}
BufferStreamRelease::BufferStreamRelease(BufferStreamRelease&& other) noexcept
    : group_(std::move(other.group_)), entries_(std::move(other.entries_)) {}
BufferStreamRelease& BufferStreamRelease::operator=(BufferStreamRelease&& other) noexcept {
  group_ = std::move(other.group_);
  entries_ = std::move(other.entries_);
  return *this;
}
bool BufferStreamRelease::empty() const noexcept {
  // Last release/reuse owns the lease exclusively; a CUDA host function must not take a mutex.
  return entries_.empty();
}
bool BufferStreamRelease::contains(CUstream_st* stream) const noexcept {
  const std::lock_guard lock{mutex_};
  for (const auto& entry : entries_) {
    if (entry->stream == stream && entry->ready_outside_capture &&
        (entry->lifetime == nullptr || entry->lifetime->state() == Lifetime::State::kOpen)) {
      // Ordinary stream ordering does not become a dependency of a newly captured graph.
      CUstreamCaptureStatus capture{};
      return cuStreamIsCapturing(stream, &capture) == CUDA_SUCCESS &&
             capture == CU_STREAM_CAPTURE_STATUS_NONE;
    }
  }
  return false;
}

void BufferStreamRelease::mark_ready(CUstream_st* stream) noexcept {
  // A captured wait belongs to that graph, not to later execution or another capture.
  CUstreamCaptureStatus capture{};
  if (cuStreamIsCapturing(stream, &capture) != CUDA_SUCCESS ||
      capture != CU_STREAM_CAPTURE_STATUS_NONE) {
    return;
  }
  const std::lock_guard lock{mutex_};
  for (auto& entry : entries_) {
    if (entry->stream == stream &&
        (entry->lifetime == nullptr || entry->lifetime->state() == Lifetime::State::kOpen)) {
      entry->ready_outside_capture = true;
    }
  }
}

// Keep fallible reservation and pinning before the single publication point so rejected
// registrations cannot leave a partially captured slot or stream owner.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
bool BufferStreamRelease::add(const ShmRefCountMappingPtr& mapping, std::uint32_t slot,
                              CUstream_st* stream, const std::shared_ptr<const Lifetime>& lifetime,
                              const Owners& owners) noexcept {
  const std::lock_guard lock{mutex_};
  if (stream == nullptr || mapping == nullptr ||
      (lifetime != nullptr &&
       (lifetime->handle() != stream || lifetime->state() != Lifetime::State::kOpen))) {
    return false;
  }
  try {
    static_cast<void>(worker());
    Entry* existing{};
    for (const auto& entry : entries_) {
      if (entry->stream == stream &&
          (entry->lifetime == nullptr || entry->lifetime->state() == Lifetime::State::kOpen) &&
          (entry->lifetime == lifetime || entry->lifetime == nullptr || lifetime == nullptr)) {
        existing = entry.get();
        break;
      }
    }
    if (existing != nullptr && (existing->lifetime != nullptr || lifetime == nullptr)) {
      return true;
    }
    auto group = group_;
    if (group == nullptr) {
      group = std::make_shared<Group>();
      group->arming = std::make_unique<Entry>();
      group->mapping = mapping;
      group->slot = slot;
      group->owners = owners;
    }
    auto candidate = existing == nullptr ? std::make_unique<Entry>() : nullptr;
    auto* entry = existing != nullptr ? existing : candidate.get();
    // Reserve container growth before pinning. SDK admission is noexcept and fails on OOM.
    if (candidate != nullptr) {
      entries_.reserve(entries_.size() + 1U);
    }
    auto admission = lifetime != nullptr ? lifetime->try_pin() : Lifetime::PinResult{};
    if (lifetime != nullptr && !admission.pin) {
      return false;
    }
    CUcontext context{};
    if (cuStreamGetCtx(stream, &context) != CUDA_SUCCESS) {
      return false;
    }
    CUdevice device{};
    if (lifetime != nullptr) {
      device = lifetime->device();
    } else {
      if (cuCtxPushCurrent(context) != CUDA_SUCCESS) {
        return false;
      }
      const auto queried = cuCtxGetDevice(&device);
      CUcontext previous{};
      static_cast<void>(cuCtxPopCurrent(&previous));
      if (queried != CUDA_SUCCESS) {
        return false;
      }
    }
    // Publish the group and entry together, only after all fallible admission steps.
    // A rejected first registration must not leave the old slot captured across rebind.
    group_ = std::move(group);
    entry->group = group_;
    entry->stream = stream;
    entry->context = context;
    entry->device = device;
    entry->lifetime = lifetime;
    if (candidate != nullptr) {
      entries_.push_back(std::move(candidate));
    }
    return true;
  } catch (...) {
    return false;
  }
}

void BufferStreamRelease::release() noexcept {
  if (entries_.empty()) {
    group_.reset();
    return;
  }
  group_->remaining.store(entries_.size() + 1U, std::memory_order_relaxed);
  auto* arming = group_->arming.release();
  arming->context = entries_.front()->context;
  arming->device = entries_.front()->device;
  arming->lifetime = entries_.front()->lifetime;
  arming->group = std::move(group_);
  for (auto& owner : entries_) {
    auto* entry = owner.release();
    if (entry->lifetime != nullptr) {
      // The worker was created by add(), before any entry could be accepted.
      if (entry->lifetime->state() == Lifetime::State::kRetired) {
        completed(entry);
      } else {
        created_worker->enqueue(entry);
      }
    } else {
      prepare(entry);
    }
  }
  entries_.clear();
  // The worker can delete every stream node before handoff returns. Transfer the group's
  // arming owner as well, so this thread never destroys GPU backing in a CUDA host function.
  completed(arming);
}

void BufferStreamRelease::prepare_for_reuse() noexcept {
  const std::lock_guard lock{mutex_};
  // The pool proves reuse before calling this. Keep SDK registrations warm; caller-owned
  // streams only need to survive the current lease, rather than the owner's reusable control.
  std::erase_if(entries_, [](const auto& entry) {
    return entry->lifetime == nullptr || entry->lifetime->state() == Lifetime::State::kRetired;
  });
  if (entries_.empty()) {
    group_.reset();
  }
}

EndpointStream::EndpointStream(std::unique_ptr<BufferStreamRelease::Entry> entry) noexcept
    : entry_(std::move(entry)) {}
EndpointStream::~EndpointStream() {
  if (entry_ != nullptr) {
    created_worker->enqueue(entry_.release());
  }
}
void EndpointStream::drain_retirements() noexcept {
  worker().drain();
}
CUstream_st* EndpointStream::get() const noexcept {
  return entry_->stream;
}
const std::shared_ptr<const cuda::StreamLifetime>& EndpointStream::lifetime() const noexcept {
  return entry_->lifetime;
}
std::unique_ptr<EndpointStream> EndpointStream::create() noexcept {
  try {
    static_cast<void>(worker());
    auto entry = std::make_unique<BufferStreamRelease::Entry>();
    auto context = std::make_shared<PrimaryContextOwner>();
    CUcontext current{};
    if (cuCtxGetCurrent(&current) != CUDA_SUCCESS ||
        (current != nullptr ? cuCtxGetDevice(&context->device)
                            : cuDeviceGet(&context->device, 0)) != CUDA_SUCCESS ||
        cuDevicePrimaryCtxRetain(&context->context, context->device) != CUDA_SUCCESS) {
      return nullptr;
    }
    entry->context = context->context;
    entry->device = context->device;
    entry->kind = Entry::Kind::kEndpoint;
    if (cuCtxPushCurrent(context->context) != CUDA_SUCCESS) {
      return nullptr;
    }
    struct RestoreContext {
      ~RestoreContext() {
        CUcontext previous{};
        static_cast<void>(cuCtxPopCurrent(&previous));
      }
    };
    const RestoreContext restore;
    auto result = std::unique_ptr<EndpointStream>(new EndpointStream(std::move(entry)));
    if (cuStreamCreate(&result->entry_->stream, CU_STREAM_NON_BLOCKING) != CUDA_SUCCESS) {
      // Do not enqueue an unmaterialized owner.
      result->entry_.reset();
      return nullptr;
    }
    try {
      auto control = std::make_shared<EndpointLifetime>(context, result->entry_->stream);
      result->entry_->owned_lifetime = std::shared_ptr<Lifetime>(control, &control->lifetime);
      result->entry_->lifetime = result->entry_->owned_lifetime;
    } catch (...) {
      static_cast<void>(cuStreamDestroy(result->entry_->stream));
      result->entry_.reset();
      return nullptr;
    }
    return result;
  } catch (...) {
    return nullptr;
  }
}

}  // namespace holomq
