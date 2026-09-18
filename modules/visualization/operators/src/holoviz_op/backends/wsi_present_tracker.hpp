// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include <holoviz/common.hpp>
#include "holoviz_op/backends/present_backend.hpp"
#include "holoviz_op/types.hpp"

namespace holoscan::visualization::backends {

/// Per-present-id outcome bookkeeping for WsiPresentBackend: which recent ids were queued
/// to the display and which are known drops, so wait_for_present() answers exactly instead
/// of defaulting old or never-issued ids to "displayed". Bounded state — a ring of the
/// last (image_count + 2) tagged presents plus watermarks; ids that fell off the ring (or
/// predate an image-count-changing recreation, which resets it) report kEvicted and the
/// caller maps that to kNotFound. Pure logic, no device calls; header-inline so it is
/// unit-testable without linking module-private symbols. Single-threaded, like the backend.
class WsiPresentTracker {
 public:
  enum class Outcome : uint8_t {
    kQueued,   ///< handed to the presentation engine
    kDropped,  ///< known drop: the frame never reached the display
  };

  enum class Lookup : uint8_t {
    kQueuedLive,     ///< queued this swapchain generation; wait on feedback / render-done
    kQueuedRetired,  ///< queued before the last recreation -> outcome unresolvable (kNotFound)
    kDropped,        ///< known drop -> WaitResult::kDropped
    kNeverIssued,    ///< a gap in the strictly increasing id sequence -> kNotFound
    kEvicted,        ///< outcome no longer tracked (older than the ring) -> kNotFound
  };

  struct LookupResult {
    Lookup state = Lookup::kNeverIssued;
    holoviz::SubmitSyncToken render_done{};  ///< valid for kQueuedLive only
  };

  /// (Re)size for a (re)created swapchain image count. Forgets all recorded outcomes: the
  /// eviction watermark moves past last_id(), so waits across an image-count-changing
  /// recreation report kEvicted (cross-recreation waits are documented as unsupported).
  void reset(uint32_t image_count) {
    ring_.assign(static_cast<size_t>(image_count) + 2, Record{});
    next_ = 0;
    evicted_below_ = last_id_ + 1;
  }

  /// Full reset for a brand-new swapchain (create_swapchain()): ids restart at 0.
  void reset_ids() {
    for (Record& record : ring_) {
      record = Record{};
    }
    next_ = 0;
    last_id_ = 0;
    retired_id_ = 0;
    evicted_below_ = 1;
  }

  /// Nonzero ids must be strictly increasing; zero (untagged) always passes.
  [[nodiscard]] bool is_monotonic(PresentId id) const { return id == 0 || id > last_id_; }

  /// Record the outcome of a tagged present and advance the id watermark. Untagged
  /// (id == 0) presents are not tracked.
  void record(PresentId id, Outcome outcome, holoviz::SubmitSyncToken render_done) {
    if (id == 0) {
      return;
    }
    last_id_ = id;
    if (ring_.empty()) {  // never sized (no reset() yet): track the watermark only
      evicted_below_ = id + 1;
      return;
    }
    Record& slot = ring_[next_];
    if (slot.id != 0) {
      // Ids are strictly increasing, so the overwritten record is always the oldest.
      evicted_below_ = slot.id + 1;
    }
    slot = Record{id, outcome, render_done};
    next_ = (next_ + 1) % ring_.size();
  }

  /// A recreation happened: every id queued so far is retired — its display outcome can no
  /// longer be resolved (the recreation drained rendering, which proves nothing about
  /// display), so a wait on it reports kNotFound. Known drops stay drops.
  void retire_all() { retired_id_ = last_id_; }

  [[nodiscard]] PresentId last_id() const { return last_id_; }

  /// Precondition: 0 < id <= last_id() (the caller maps id > last_id() to kNotFound).
  [[nodiscard]] LookupResult lookup(PresentId id) const {
    for (const Record& record : ring_) {
      if (record.id == id) {
        if (record.outcome == Outcome::kDropped) {
          return {Lookup::kDropped, {}};
        }
        if (id <= retired_id_) {
          return {Lookup::kQueuedRetired, {}};
        }
        return {Lookup::kQueuedLive, record.render_done};
      }
    }
    return {id < evicted_below_ ? Lookup::kEvicted : Lookup::kNeverIssued, {}};
  }

 private:
  struct Record {
    PresentId id = 0;
    Outcome outcome = Outcome::kQueued;
    holoviz::SubmitSyncToken render_done{};
  };

  std::vector<Record> ring_;  ///< image_count + 2 slots, overwritten oldest-first
  size_t next_ = 0;
  PresentId last_id_ = 0;
  PresentId retired_id_ = 0;
  /// Ids below this had their records evicted (or the ring was reset); their outcome is
  /// forgotten and lookup() reports kEvicted, never a defaulted "displayed".
  PresentId evicted_below_ = 1;
};

}  // namespace holoscan::visualization::backends
