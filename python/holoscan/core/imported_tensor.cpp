/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "imported_tensor.hpp"

#include <cuda.h>
#include <nanobind/nanobind.h>

#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#include <unistd.h>
#endif

#include <algorithm>
#include <array>
#include <atomic>
#include <charconv>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <holoscan/core/error_helpers.hpp>
#include <holoscan/logger/logger.hpp>

#include "core/domain/tensor_access.hpp"

namespace holoscan {
namespace {
namespace nb = nanobind;

#if defined(__linux__)
// Capture the process leader's allowed CPUs before an execution lane can start this worker.
const cpu_set_t kInitialCPUAffinity = [] {
  cpu_set_t allowed{};
  if (sched_getaffinity(getpid(), sizeof(allowed), &allowed) != 0) {
    CPU_ZERO(&allowed);
  }
  return allowed;
}();

// Remove the kernel-isolated CPUs (isolcpus) listed in sysfs, such as "2-9,20-27", from `mask`.
// The HoloMQ release worker in stream_release.cpp keeps the same rule; change both.
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
// stay behind a busy pinned thread. Keep it on the process's other CPUs; use the whole process set
// only when nothing else remains.
const cpu_set_t kWorkerCPUAffinity = [] {
  cpu_set_t allowed = kInitialCPUAffinity;
  remove_isolated_cpus(allowed);
  return CPU_COUNT(&allowed) != 0 ? allowed : kInitialCPUAffinity;
}();

// Keep bootstrap placement and final restoration together; CPU-set macro expansion inflates
// the score for this bounded mask subtraction and best-effort error reporting.
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
    try {
      HOLOSCAN_LOG_ERROR("Python Tensor release worker could not restore process CPU affinity");
    } catch (...) {  // NOLINT(bugprone-empty-catch)
      // Logging must not throw from the noexcept retirement path.
    }
  }
}
#endif

class ScopedDevice {
 public:
  explicit ScopedDevice(int device) noexcept {
    valid_ = cudaGetDevice(&previous_) == cudaSuccess &&
             (previous_ == device || cudaSetDevice(device) == cudaSuccess);
    changed_ = valid_ && previous_ != device;
  }
  ~ScopedDevice() {
    if (changed_) {
      (void)cudaSetDevice(previous_);
    }
  }
  [[nodiscard]] bool valid() const noexcept { return valid_; }

 private:
  int previous_{};
  bool valid_{};
  bool changed_{};
};

// Completed readiness and consumer events, kept for the next device Tensor import. Every import
// creates a readiness event and every consumer stream another, and creating and destroying them
// cost more than the rest of a CuPy import. An event returns here only from a retired record whose
// waits all succeeded, so recording it again cannot replace a pending state. Pools are keyed by the
// driver's process-unique context ID: an event is reused only in the context that created it, and
// the events of a destroyed or reset context are never handed out again; they are reclaimed with
// their context. Pools are kept for up to eight contexts. A further context takes over an entry
// only once that entry's pool is empty, so no cached event is ever dropped; until then, that
// context's completed events are destroyed.
class ImportEventPool {
 public:
  // Take a completed event of the current context, or create one and report its context.
  [[nodiscard]] bool acquire(cudaEvent_t& event, std::uint64_t& context_id) noexcept {
    context_id = current_context_id();
    if (context_id != 0) {
      const std::lock_guard lock{mutex_};
      for (auto& pool : pools_) {
        if (pool.context_id == context_id && !pool.events.empty()) {
          event = pool.events.back();
          pool.events.pop_back();
          return true;
        }
      }
    }
    if (cudaEventCreateWithFlags(&event, cudaEventDisableTiming | cudaEventBlockingSync) !=
        cudaSuccess) {
      return false;
    }
    if (context_id == 0) {
      // Creating the event made the device's primary context current.
      context_id = current_context_id();
    }
    return true;
  }

  // Keep a completed event for its context, or destroy it when that context has no pool with room.
  void release(std::uint64_t context_id, int device, cudaEvent_t event) noexcept {
    if (context_id != 0) {
      const std::lock_guard lock{mutex_};
      if (auto* events = context_events(context_id);
          events != nullptr && events->size() < events->capacity()) {
        events->push_back(event);  // within reserved capacity, so this cannot allocate
        return;
      }
    }
    const ScopedDevice scoped{device};
    if (scoped.valid()) {
      (void)cudaEventDestroy(event);
    }
  }

 private:
  struct ContextEvents {
    std::uint64_t context_id{};
    std::vector<cudaEvent_t> events;
  };

  [[nodiscard]] static std::uint64_t current_context_id() noexcept {
    CUcontext context{};
    unsigned long long id{};  // NOLINT(runtime/int) - cuCtxGetId's parameter type
    if (cuCtxGetCurrent(&context) != CUDA_SUCCESS || context == nullptr ||
        cuCtxGetId(context, &id) != CUDA_SUCCESS) {
      return 0;
    }
    return static_cast<std::uint64_t>(id);
  }

  // Find or add the context's pool, or take over an entry whose pool is empty. Returns null when
  // every entry still holds events: dropping their handles would leak them while their context
  // lives, and destroying them could reach a context that no longer exists.
  std::vector<cudaEvent_t>* context_events(std::uint64_t context_id) noexcept {
    ContextEvents* empty = nullptr;
    for (auto& pool : pools_) {
      if (pool.context_id == context_id) {
        return &pool.events;
      }
      if (empty == nullptr && pool.events.empty()) {
        empty = &pool;
      }
    }
    if (pools_.size() < kContexts) {
      try {
        auto& added = pools_.emplace_back();
        added.events.reserve(kEventsPerContext);
        empty = &added;
      } catch (...) {
        return nullptr;
      }
    }
    if (empty == nullptr) {
      return nullptr;
    }
    empty->context_id = context_id;
    return &empty->events;
  }

  static constexpr std::size_t kContexts = 8;
  static constexpr std::size_t kEventsPerContext = 256;
  std::mutex mutex_;
  std::vector<ContextEvents> pools_;
};

ImportEventPool& import_event_pool() {
  // Never destroyed: records retire on the worker and in late Python finalizers, which may run
  // after static destruction starts, and pooled events need no CUDA call at exit.
  static auto* const pool = std::make_unique<ImportEventPool>().release();
  return *pool;
}

class ReleaseWorker;
struct ReleaseRecord;
using Lifetime = holomq::cuda::StreamLifetime;

struct StreamUse : Lifetime::DeferredRelease {
  cudaStream_t stream{};
  cudaEvent_t event{};
  std::uint64_t event_context_id{};
  int device{};
  bool active{};
  std::shared_ptr<const Lifetime> lifetime;
  ReleaseRecord* release_record{};
};

struct ReleaseRecord {
  std::shared_ptr<void> owner;
  StreamUse readiness;
  std::vector<StreamUse> consumers;
  std::vector<int> untracked_devices;
  bool failed{};
  std::unique_ptr<ReleaseRecord> next;
  ReleaseRecord* outcome_next{};
  ReleaseWorker* worker{};
  std::atomic_size_t unresolved{1U};  // arming reference until every stream is registered
  std::atomic_bool outcome_failed{};
  bool outcome_route{};
  bool needs_sdk_fences{};
  bool events_complete{};  // every event's recorded work finished; it may be reused

  void resolve_outcome() noexcept;
  [[nodiscard]] bool prepare_sdk_fences() noexcept;

  [[nodiscard]] bool wait(bool include_untracked = true) noexcept {
    const auto wait_event = [](const StreamUse& use) {
      if (use.active) {
        return false;
      }
      if (use.event == nullptr) {
        return true;
      }
      const ScopedDevice device{use.device};
      return device.valid() && cudaEventSynchronize(use.event) == cudaSuccess;
    };
    if (failed || outcome_failed.load(std::memory_order_acquire) || !wait_event(readiness)) {
      return false;
    }
    for (const auto& consumer : consumers) {
      if (!wait_event(consumer)) {
        return false;
      }
    }
    events_complete = true;
    if (!include_untracked) {
      return true;
    }
    for (const auto ordinal : untracked_devices) {
      const ScopedDevice device{ordinal};
      if (!device.valid() || cudaDeviceSynchronize() != cudaSuccess) {
        return false;
      }
    }
    return true;
  }

  ~ReleaseRecord() {
    const auto retire = [this](const StreamUse& use) {
      if (use.event == nullptr) {
        return;
      }
      if (events_complete) {
        import_event_pool().release(use.event_context_id, use.device, use.event);
        return;
      }
      const ScopedDevice device{use.device};
      if (device.valid()) {
        (void)cudaEventDestroy(use.event);
      }
    };
    retire(readiness);
    for (const auto& consumer : consumers) {
      retire(consumer);
    }
  }
};

// Keep the owner alive when CUDA completion or interpreter access cannot prove deletion safe.
// There is no later teardown path that can safely reclaim this record.
void retain_release_record(std::unique_ptr<ReleaseRecord> record) noexcept {
  static_cast<void>(record.release());  // NOLINT(bugprone-unused-return-value)
}

void complete_release(std::unique_ptr<ReleaseRecord> record) noexcept {
  if (!record->wait()) {
    // A CUDA failure cannot establish safe storage reuse. Retain ownership rather than invoke
    // an allocator deleter on possibly in-flight storage. No CUDA API runs in a host callback.
    retain_release_record(std::move(record));
    try {
      HOLOSCAN_LOG_ERROR("Python Tensor CUDA retirement failed; retaining its allocation owner");
    } catch (...) {  // NOLINT(bugprone-empty-catch)
      // Logging must not throw from the noexcept retirement path.
    }
  }
}

struct ReleaseBatchCounts {
  std::size_t batch_count{};
  std::size_t outcome_count{};
};

// Prove precise stream completion and coalesce untracked exports into one fence per device.
[[nodiscard]] ReleaseBatchCounts wait_for_batch(ReleaseRecord* batch) noexcept {
  std::array<std::pair<int, bool>, 32> devices{};
  std::size_t device_count{};
  ReleaseBatchCounts counts;
  for (auto* record = batch; record != nullptr; record = record->next.get()) {
    ++counts.batch_count;
    counts.outcome_count += record->outcome_route ? 1U : 0U;
    record->failed = !record->wait(false);
    for (const auto ordinal : record->untracked_devices) {
      const auto end = devices.begin() + device_count;
      auto found = std::find_if(
          devices.begin(), end, [&](const auto& entry) { return entry.first == ordinal; });
      bool completed{};
      if (found != end) {
        completed = found->second;
      } else {
        const ScopedDevice device{ordinal};
        completed = device.valid() && cudaDeviceSynchronize() == cudaSuccess;
        if (device_count < devices.size()) {
          devices[device_count++] = {ordinal, completed};
        }
      }
      record->failed |= !completed;
    }
  }
  return counts;
}

// Route SDK fences that still depend on stream retirement before waiting on the ready batch.
[[nodiscard]] std::size_t prepare_release_batch(std::unique_ptr<ReleaseRecord>& batch) noexcept {
  std::size_t deferred_count{};
  std::unique_ptr<ReleaseRecord> ready;
  while (batch != nullptr) {
    auto record = std::move(batch);
    batch = std::move(record->next);
    if (record->needs_sdk_fences) {
      auto* raw = record.release();
      if (raw->prepare_sdk_fences()) {
        ++deferred_count;
        continue;
      }
      record.reset(raw);
    }
    record->next = std::move(ready);
    ready = std::move(record);
  }
  batch = std::move(ready);
  return deferred_count;
}

// Drop proven-ready owners under one GIL handoff, then handle failed records without the GIL.
void release_batch_owners(std::unique_ptr<ReleaseRecord> batch) noexcept {
  {
    // Run all ready Python deleters in one GIL handoff; do not hold it during CUDA waits.
    const nb::gil_scoped_acquire gil;
    for (auto* record = batch.get(); record != nullptr; record = record->next.get()) {
      if (!record->failed && gil.is_valid()) {
        record->owner.reset();
      } else {
        record->failed = true;
      }
    }
  }
  while (batch != nullptr) {
    auto record = std::move(batch);
    batch = std::move(record->next);
    if (record->failed) {
      complete_release(std::move(record));
    }
  }
}

class ReleaseWorker {
 public:
  ReleaseWorker() : worker_([this] { run(); }) {
#if defined(__linux__)
    // A pinned real-time creator may never yield enough CPU for the child to reset itself.
    restore_worker_affinity(worker_.native_handle(), true);
#endif
  }

  void enqueue(std::unique_ptr<ReleaseRecord> record) noexcept {
    bool queued = false;
    {
      const std::lock_guard lock{mutex_};
      if (!stopped_ && pending_ < kCapacity) {
        auto* tail = record.get();
        if (tail_ != nullptr) {
          tail_->next = std::move(record);
        } else {
          head_ = std::move(record);
        }
        tail_ = tail;
        ++pending_;
        queued = true;
      }
    }
    if (queued) {
      wake();
      return;
    }
    // Apply bounded backpressure only when the worker is saturated or already shut down.
    // Avoid acquiring the GIL for known native callback threads.
    if (!nb::is_alive()) {
      retain_release_record(std::move(record));
      return;
    }
    if (detail::TensorConsumerScope::active() && !PyImportedTensorScope::active()) {
      // A native callback does not own the Python GIL.
      complete_release(std::move(record));
    } else {
      // CPython's limited API cannot check GIL ownership, and the pinned nanobind abi3 backend
      // reports gil_check() as always true. Keep the conservative guard for unknown callers.
      const nb::gil_scoped_acquire gil;
      if (!gil.is_valid()) {
        retain_release_record(std::move(record));
        return;
      }
      const nb::gil_scoped_release release;
      complete_release(std::move(record));
    }
  }

  // Publication callbacks cannot wait for queue space, take a mutex, use CUDA, or take the GIL.
  // The worker and each record are preallocated before deferral; late publication is retained.
  void enqueue_outcome(ReleaseRecord* record) noexcept {
    if (stopped_.load(std::memory_order_acquire)) {
      return;
    }
    record->outcome_route = true;
    outcome_pending_.fetch_add(1U, std::memory_order_relaxed);
    auto* head = outcomes_.load(std::memory_order_relaxed);
    do {
      record->outcome_next = head;
    } while (!outcomes_.compare_exchange_weak(
        head, record, std::memory_order_release, std::memory_order_relaxed));
    wake();
  }

  bool drain(double timeout) {
    const nb::gil_scoped_release release;
    std::unique_lock lock{mutex_};
    const auto ready = [this] { return pending_ == 0 && outcome_pending_.load() == 0; };
    if (timeout < 0) {
      changed_.wait(lock, ready);
      return true;
    }
    return changed_.wait_for(lock, std::chrono::duration<double>{timeout}, ready);
  }

  void shutdown() {
    const nb::gil_scoped_release release;
    {
      const std::lock_guard lock{mutex_};
      stopped_ = true;
    }
    wake();
    if (worker_.joinable()) {
      worker_.join();
    }
  }

 private:
  void wake() noexcept {
    // Coalesce publications until the single consumer acknowledges them. A 32-bit word
    // also lets supported Linux standard libraries wait directly on a futex.
    if (wake_pending_.exchange(1U, std::memory_order_release) == 0U) {
      wake_pending_.notify_one();
    }
  }

  void run() noexcept {
#if defined(__linux__)
    restore_worker_affinity(pthread_self());
    (void)pthread_setname_np(pthread_self(), "hsdk-py-retire");
#endif
    for (;;) {
      // Only this thread clears the latch, always before checking both queues. A publication
      // before this exchange is found by the queue check; one after it leaves the latch set
      // until the next iteration, so wait(0) cannot miss it. Outcome producers never take mutex_.
      (void)wake_pending_.exchange(0U, std::memory_order_acquire);
      std::unique_ptr<ReleaseRecord> batch;
      bool stopped{};
      {
        const std::lock_guard lock{mutex_};
        batch = std::move(head_);
        tail_ = nullptr;
        stopped = stopped_.load(std::memory_order_acquire);
      }
      auto* outcomes = outcomes_.exchange(nullptr, std::memory_order_acquire);
      while (outcomes != nullptr) {
        auto* record = outcomes;
        outcomes = record->outcome_next;
        record->next = std::move(batch);
        batch.reset(record);
      }
      if (batch == nullptr) {
        if (stopped) {
          return;
        }
        wake_pending_.wait(0U, std::memory_order_acquire);
        continue;
      }
      const auto deferred_count = prepare_release_batch(batch);
      if (batch == nullptr) {
        {
          const std::lock_guard lock{mutex_};
          outcome_pending_.fetch_sub(deferred_count, std::memory_order_release);
        }
        changed_.notify_all();
        continue;
      }
      // Detached records have no live array aliases. One device fence can retire all their
      // untracked exports on that device. Precise stream events use blocking host waits.
      const auto counts = wait_for_batch(batch.get());
      release_batch_owners(std::move(batch));
      {
        const std::lock_guard lock{mutex_};
        pending_ -= counts.batch_count - counts.outcome_count;
        outcome_pending_.fetch_sub(counts.outcome_count + deferred_count,
                                   std::memory_order_release);
      }
      changed_.notify_all();
    }
  }

  static constexpr std::size_t kCapacity = 2048;
  std::mutex mutex_;
  std::condition_variable changed_;
  std::unique_ptr<ReleaseRecord> head_;
  ReleaseRecord* tail_{};
  std::size_t pending_{};
  std::atomic_bool stopped_{};
  std::atomic<ReleaseRecord*> outcomes_{};
  std::atomic_size_t outcome_pending_{};
  std::atomic_uint32_t wake_pending_{};
  std::thread worker_;
};

void ReleaseRecord::resolve_outcome() noexcept {
  if (unresolved.fetch_sub(1U, std::memory_order_acq_rel) == 1U) {
    worker->enqueue_outcome(this);
  }
}

bool ReleaseRecord::prepare_sdk_fences() noexcept {
  needs_sdk_fences = false;
  bool deferred{};
  for (auto& use : consumers) {
    if (use.lifetime == nullptr) {
      continue;
    }
    const ScopedDevice device{use.device};
    auto admission = use.lifetime->try_pin();
    if (admission.pin) {
      if (!device.valid() || cudaEventRecord(use.event, admission.pin.stream()) != cudaSuccess) {
        failed = true;
      } else {
        use.active = false;
      }
    } else if (admission.state == Lifetime::State::kClosing) {
      deferred = true;
      use.release_record = this;
      use.on_outcome = [](Lifetime::DeferredRelease& base, Lifetime::State outcome) noexcept {
        auto& entry = static_cast<StreamUse&>(base);
        auto* owner = entry.release_record;
        if (outcome == Lifetime::State::kRetired) {
          entry.active = false;
        } else {
          owner->outcome_failed.store(true, std::memory_order_relaxed);
        }
        owner->resolve_outcome();
      };
      unresolved.fetch_add(1U, std::memory_order_relaxed);
      if (!use.lifetime->defer(use)) {
        use.on_outcome(use, use.lifetime->state());
      }
    } else if (admission.state == Lifetime::State::kRetired) {
      use.active = false;
    } else {
      failed = true;
    }
  }
  if (deferred) {
    resolve_outcome();
  }
  return deferred;
}

std::atomic<ReleaseWorker*>& created_worker() noexcept {
  static std::atomic<ReleaseWorker*> worker{};
  return worker;
}

ReleaseWorker& release_worker() {
  // Explicit interpreter shutdown joins the worker. Keep the inert service available for late
  // Python finalizers, without depending on C++ static destruction order or acquiring a dead GIL.
  static auto* const worker = [] {
    auto* service = new ReleaseWorker;
    created_worker().store(service, std::memory_order_release);
    return service;
  }();
  return *worker;
}

PyImportedTensorScope*& active_scope() noexcept {
  static thread_local PyImportedTensorScope* scope{};
  return scope;
}

}  // namespace

class PyImportedMemoryReference final : public TensorMemoryReference,
                                        public detail::TensorConsumerCompletion,
                                        public detail::TensorStreamLifetimeConsumer {
 public:
  PyImportedMemoryReference(std::shared_ptr<void> owner, DLDevice device)
      : record_(std::make_unique<ReleaseRecord>()), device_(device) {
    record_->owner = std::move(owner);
  }

  ~PyImportedMemoryReference() override {
    if (detail::TensorConsumerScope::capturing() && device_.device_type != kDLCPU) {
      // Captured graph nodes can retain raw pointers beyond capture. Never inject retirement
      // records into capture or pretend completion of capture proves completion of replay.
      record_->failed = true;
      release_worker().enqueue(std::move(record_));
      return;
    }
    // A retained view may be dropped in a later Python or native callback, after the earlier
    // callback snapshot. Fence that callback's live stream again at the final release.
    if (const auto stream = detail::TensorConsumerScope::stream();
        stream && device_.device_type != kDLCPU && !readiness_failed_) {
      try {
        if (!set_deallocation_stream(*stream)) {
          track_untracked_export();
        }
      } catch (...) {
        record_->failed = true;
      }
    } else if (detail::TensorConsumerScope::active() && device_.device_type != kDLCPU &&
               !readiness_failed_) {
      // start()/stop() have no assigned compute stream. Their library stream use is untracked.
      try {
        track_untracked_export();
      } catch (...) {
        record_->failed = true;
      }
    }
    if (record_->readiness.event == nullptr && !record_->readiness.active &&
        record_->consumers.empty() && record_->untracked_devices.empty() && !record_->failed) {
      return;
    }
    if (std::any_of(record_->consumers.begin(), record_->consumers.end(), [](const auto& use) {
          return use.lifetime != nullptr;
        })) {
      for (auto& use : record_->consumers) {
        if (use.lifetime == nullptr) {
          finish(use);  // external streams still need their final record before release returns
        }
      }
      record_->worker = &release_worker();
      record_->needs_sdk_fences = true;
      // SDK stream work belongs on the worker. A native CUDA host function can drop the last
      // Tensor reference: its destructor must enqueue without calling CUDA or taking a mutex.
      release_worker().enqueue_outcome(record_.release());
      return;
    }
    for (auto& use : record_->consumers) {
      finish(use);
    }
    release_worker().enqueue(std::move(record_));
  }

  bool set_readiness_stream(cudaStream_t stream) override {
    const std::lock_guard lock{mutex_};
    auto& ready = record_->readiness;
    readiness_failed_ = true;
    ready.device = device_.device_id;
    ready.active = true;
    int readiness_device = device_.device_id;
#if CUDART_VERSION >= 12080
    if (stream != nullptr && stream != cudaStreamLegacy && stream != cudaStreamPerThread &&
        cudaStreamGetDevice(stream, &readiness_device) != cudaSuccess) {
      return false;
    }
#endif
    const ScopedDevice device{readiness_device};
    ready.device = readiness_device;
    if (!device.valid() ||
        (ready.event == nullptr &&
         !import_event_pool().acquire(ready.event, ready.event_context_id)) ||
        cudaEventRecord(ready.event, stream) != cudaSuccess) {
      if (device.valid() && cudaStreamSynchronize(stream) == cudaSuccess) {
        ready.active = false;
        if (ready.event != nullptr) {
          (void)cudaEventDestroy(ready.event);
          ready.event = nullptr;
        }
      }
      return false;
    }
    ready.active = false;
    readiness_failed_ = false;
    return true;
  }

  [[nodiscard]] bool has_readiness_stream_contract() const noexcept override {
    const std::lock_guard lock{mutex_};
    return record_->readiness.event != nullptr && !record_->readiness.active;
  }
  [[nodiscard]] bool tracks_deallocation_stream() const noexcept override { return true; }

  bool set_deallocation_stream(cudaStream_t stream) override {
    return register_consumer_stream(stream, detail::TensorConsumerScope::lifetime(stream));
  }

  bool register_consumer_stream(cudaStream_t stream,
                                const std::shared_ptr<const Lifetime>& lifetime) noexcept override {
    const std::lock_guard lock{mutex_};
    int ordinal{};
    if (lifetime != nullptr) {
      if (lifetime->handle() != stream || lifetime->state() != Lifetime::State::kOpen) {
        return false;
      }
      ordinal = lifetime->device();
    } else if (cudaGetDevice(&ordinal) != cudaSuccess) {
      return false;
    }
#if CUDART_VERSION >= 12080
    if (lifetime == nullptr && stream != nullptr && stream != cudaStreamLegacy &&
        stream != cudaStreamPerThread && cudaStreamGetDevice(stream, &ordinal) != cudaSuccess) {
      return false;
    }
#endif
    for (auto& use : record_->consumers) {
      if (use.stream == stream && use.device == ordinal && use.lifetime == lifetime) {
        use.active = true;
        return true;
      }
    }
    // Python imports are not allocation-free runtime pools. Grow instead of imposing a
    // lifetime-wide fan-out limit that counts the source stream against its consumers.
    try {
      record_->consumers.emplace_back();
    } catch (...) {
      // Admission is noexcept; metadata exhaustion must remain a recoverable failure.
      return false;
    }
    auto& use = record_->consumers.back();
    use.device = ordinal;
    const ScopedDevice device{ordinal};
    if (!device.valid() || !import_event_pool().acquire(use.event, use.event_context_id)) {
      record_->consumers.pop_back();
      return false;
    }
    use.stream = stream;
    use.lifetime = lifetime;
    use.active = true;
    return true;
  }

  expected<void, Error> synchronize_for_consumer(cudaStream_t stream) override {
    const std::lock_guard lock{mutex_};
    if (record_->readiness.event == nullptr) {
      return {};
    }
    int device{};
    if (cudaGetDevice(&device) != cudaSuccess) {
      return make_unexpected(Error{ErrorCode::kFailure});
    }
#if CUDART_VERSION >= 12080
    if (cudaStreamGetDevice(stream, &device) != cudaSuccess) {
      return make_unexpected(Error{ErrorCode::kFailure});
    }
#endif
    const ScopedDevice guard{device};
    if (!guard.valid()) {
      return make_unexpected(Error{ErrorCode::kFailure});
    }
    const auto& lifetime = detail::TensorConsumerScope::lifetime(stream);
    auto admission = lifetime != nullptr ? lifetime->try_pin() : Lifetime::PinResult{};
    if (lifetime != nullptr && !admission.pin) {
      return make_unexpected(Error{ErrorCode::kClosed});
    }
    return error_detail::make_cuda_error(cudaStreamWaitEvent(stream, record_->readiness.event, 0),
                                         "Python Tensor readiness wait failed");
  }

  expected<void, Error> synchronize_for_host() override {
    const std::lock_guard lock{mutex_};
    if (record_->readiness.event == nullptr) {
      return {};
    }
    const ScopedDevice device{record_->readiness.device};
    return error_detail::make_cuda_error(cudaEventSynchronize(record_->readiness.event),
                                         "Python Tensor host readiness wait failed");
  }

  void track_untracked_export() {
    const std::lock_guard lock{mutex_};
    int current{};
    if (cudaGetDevice(&current) != cudaSuccess) {
      throw std::runtime_error("Failed to query CUDA device for imported Tensor export");
    }
    for (const auto device : {current, device_.device_id, record_->readiness.device}) {
      if (std::find(record_->untracked_devices.begin(), record_->untracked_devices.end(), device) ==
          record_->untracked_devices.end()) {
        record_->untracked_devices.push_back(device);
      }
    }
  }

  std::shared_ptr<const Lifetime> stream_lifetime(cudaStream_t stream) const {
    const std::lock_guard lock{mutex_};
    for (const auto& use : record_->consumers) {
      if (use.stream == stream && use.lifetime != nullptr) {
        return use.lifetime;
      }
    }
    return {};
  }

  void record_consumer_completion(cudaStream_t stream) noexcept override {
    const std::lock_guard lock{mutex_};
    if (std::none_of(
            record_->consumers.begin(), record_->consumers.end(), [stream](const auto& use) {
              return use.stream == stream && use.lifetime == nullptr;
            })) {
      return;
    }
    int device{};
    if (cudaGetDevice(&device) != cudaSuccess) {
      record_->failed = true;
      return;
    }
#if CUDART_VERSION >= 12080
    if (stream != nullptr && stream != cudaStreamLegacy && stream != cudaStreamPerThread &&
        cudaStreamGetDevice(stream, &device) != cudaSuccess) {
      record_->failed = true;
      return;
    }
#endif
    for (auto& use : record_->consumers) {
      if (use.stream == stream && use.device == device && use.lifetime == nullptr) {
        // Explicit completion covers later work even after an earlier snapshot.
        finish(use, true);
      }
    }
  }

 private:
  static void finish(StreamUse& use, bool snapshot_live_stream = false) noexcept {
    if (!use.active && !snapshot_live_stream) {
      return;
    }
    const ScopedDevice device{use.device};
    if (!device.valid() || cudaEventRecord(use.event, use.stream) != cudaSuccess) {
      // Failed recording cannot justify release. Leave an unsatisfied retirement event by
      // retaining this owner below; callers must keep external streams alive through release.
      // Normal SDK callbacks snapshot before stream destruction, including exception cleanup.
      use.active = true;
      return;
    }
    use.active = false;
  }

  std::unique_ptr<ReleaseRecord> record_;
  DLDevice device_;
  bool readiness_failed_{};
  mutable std::mutex mutex_;
};

PyImportedTensorScope::PyImportedTensorScope(std::optional<cudaStream_t> stream) noexcept
    : previous_(std::exchange(active_scope(), this)),
      stream_(detail::TensorConsumerScope::capturing() ? std::nullopt : stream) {}

PyImportedTensorScope::~PyImportedTensorScope() {
  for (const auto& [reference, stream] : references_) {
    reference->record_consumer_completion(stream);
  }
  // Keep the Python/GIL marker active while releasing the scope's retained references.
  references_.clear();
  active_scope() = previous_;
}

void PyImportedTensorScope::track(std::shared_ptr<PyImportedMemoryReference> reference,
                                  cudaStream_t stream) {
  if (std::none_of(references_.begin(), references_.end(), [&](const auto& entry) {
        return entry.first == reference && entry.second == stream;
      })) {
    references_.emplace_back(std::move(reference), stream);
  }
}

std::optional<cudaStream_t> PyImportedTensorScope::stream() noexcept {
  return active_scope() != nullptr ? active_scope()->stream_ : std::nullopt;
}
bool PyImportedTensorScope::active() noexcept {
  return active_scope() != nullptr;
}

std::shared_ptr<TensorMemoryReference> make_python_tensor_memory_reference(
    std::shared_ptr<void> owner, DLDevice device) {
  if (device.device_type != kDLCPU) {
    (void)release_worker();
  }
  return std::make_shared<PyImportedMemoryReference>(std::move(owner), device);
}

void track_python_tensor(const Tensor& tensor, std::optional<cudaStream_t> consumer_stream) {
  const auto callback_stream = PyImportedTensorScope::stream();
  if (tensor.device().device_type == kDLCPU || (!callback_stream && !consumer_stream)) {
    return;
  }
  const auto& context = detail::TensorAccess::context(tensor);
  if (context == nullptr) {
    return;
  }
  // The Tensor argument owns this reference throughout registration. Only callback snapshots
  // need an additional owner; SDK deferral retains its own release record and stream lifetime.
  auto* reference = dynamic_cast<PyImportedMemoryReference*>(context->memory_reference.get());
  if (reference != nullptr) {
    const auto stream = consumer_stream ? *consumer_stream : *callback_stream;
    if (stream == cudaStreamPerThread) {
      reference->track_untracked_export();
      return;
    }
    const auto& lifetime = detail::TensorConsumerScope::lifetime(stream);
    if (!reference->register_consumer_stream(stream, lifetime)) {
      throw std::runtime_error("Failed to register Python Tensor consumer stream");
    }
    if (active_scope() != nullptr && lifetime == nullptr) {
      active_scope()->track(
          std::static_pointer_cast<PyImportedMemoryReference>(context->memory_reference), stream);
    }
  }
}

std::shared_ptr<const Lifetime> python_tensor_stream_lifetime(const Tensor& tensor,
                                                              cudaStream_t stream) {
  const auto& context = detail::TensorAccess::context(tensor);
  if (context == nullptr) {
    return {};
  }
  const auto reference =
      std::dynamic_pointer_cast<PyImportedMemoryReference>(context->memory_reference);
  return reference != nullptr ? reference->stream_lifetime(stream) : nullptr;
}

void track_python_tensor_untracked_export(const Tensor& tensor) {
  if (tensor.device().device_type == kDLCPU) {
    return;
  }
  const auto& context = detail::TensorAccess::context(tensor);
  if (context != nullptr) {
    if (auto* reference =
            dynamic_cast<PyImportedMemoryReference*>(context->memory_reference.get())) {
      reference->track_untracked_export();
    }
  }
}

void drain_python_tensor_releases(double timeout) {
  // Leave room for the wait's clock conversion and deadline addition. Validate even
  // before the worker exists, so invalid test deadlines never reach wait_for().
  const auto remaining =
      std::chrono::duration<double>{
          std::chrono::steady_clock::time_point::max().time_since_epoch()} -
      std::chrono::duration<double>{std::chrono::steady_clock::now().time_since_epoch()};
  if (!std::isfinite(timeout) || timeout > remaining.count() / 2) {
    throw std::invalid_argument(
        "timeout must be finite and within the steady-clock deadline range");
  }
  if (auto* worker = created_worker().load(std::memory_order_acquire)) {
    if (!worker->drain(timeout)) {
      throw std::runtime_error("Timed out waiting for Python Tensor retirement worker to drain");
    }
  }
}
void shutdown_python_tensor_releases() {
  if (auto* worker = created_worker().load(std::memory_order_acquire)) {
    worker->shutdown();
  }
}

}  // namespace holoscan
