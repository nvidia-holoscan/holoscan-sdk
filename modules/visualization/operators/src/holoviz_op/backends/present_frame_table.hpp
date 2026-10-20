// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// The pending-frame table shared by the module's PresentBackend implementations: one slot
// per output image, holding the AcquirePayload between acquire() and present(). Pure
// bookkeeping — no device calls — so the acquire/present protocol violations it detects
// (double-acquire, present-without-acquire, stale-generation tokens) are unit-testable
// without a display.

#pragma once

#include <algorithm>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include <holoviz/errors.hpp>
#include <holoviz/expected.hpp>
#include "holoviz_op/types.hpp"

#include "holoviz_op/backends/backend_tokens.hpp"

namespace holoscan::visualization::backends {

class PresentFrameTable {
 public:
  /// Size the table for a (re)created swapchain, dropping any pending payloads.
  void reset(uint32_t image_count) {
    pending_.clear();
    pending_.resize(image_count);
  }

  void clear() { pending_.assign(pending_.size(), nullptr); }

  uint32_t size() const { return static_cast<uint32_t>(pending_.size()); }

  /// The issuances of every pending (acquired, not yet consumed) frame, sorted. Captured
  /// before a changed-count reset() so the retired acquires can keep answering
  /// kFrameDropped (see RetiredAcquireLog) instead of decaying into the retryable
  /// "not acquired"/"out of range" answers of the resized table.
  [[nodiscard]] std::vector<uint64_t> pending_issuances() const {
    std::vector<uint64_t> issuances;
    for (const std::shared_ptr<AcquirePayload>& payload : pending_) {
      if (payload != nullptr) {
        issuances.push_back(payload->issuance);
      }
    }
    std::sort(issuances.begin(), issuances.end());
    return issuances;
  }

  /// Record an acquired frame. An occupied same-generation slot means the image was
  /// acquired twice without an intervening present — a protocol violation, not a
  /// recoverable state. An occupant from an OLDER generation died with its swapchain (a
  /// same-image-count recreation deliberately keeps such entries so presenting one reports
  /// kFrameDropped); a fresh acquire on the same index replaces it silently instead of
  /// wedging the frame loop.
  [[nodiscard]] expected<void> store(uint32_t image_index,
                                     std::shared_ptr<AcquirePayload> payload) {
    if (image_index >= pending_.size()) {
      return make_unexpected(
          Error{.code = ErrorCode::kInvalidArgument,
                .message = "[holoviz] PresentFrameTable: image index out of range."});
    }
    if (pending_[image_index] != nullptr &&
        pending_[image_index]->generation == payload->generation) {
      return make_unexpected(
          Error{.code = ErrorCode::kInvalidState,
                .message = "[holoviz] Swapchain image acquired twice without an intervening "
                           "present."});
    }
    pending_[image_index] = std::move(payload);
    return {};
  }

  /// Non-consuming lookup for present(): the entry stays in the table until the lower
  /// present reports the image consumed (erase()) — a failure that consumed nothing must
  /// leave the caller's only AcquireSyncToken in place for retry or release. The exact
  /// acquire identity is validated first: an occupant with a different issuance means the
  /// caller's acquire was superseded (a fresh acquire on a recycled index replaced it in
  /// store()), so the caller's frame is gone (kFrameDropped) and the occupant — someone
  /// else's live frame — must not be touched (see holds()). A matching payload minted
  /// under an older swapchain generation is likewise reported as kFrameDropped without
  /// touching its gates (they point into recreated-away storage) and WITHOUT clearing the
  /// slot; the caller records the drop, then erases explicitly.
  [[nodiscard]] expected<std::shared_ptr<AcquirePayload>> peek(uint32_t image_index,
                                                               uint64_t current_generation,
                                                               uint64_t acquire_issuance) {
    if (image_index >= pending_.size()) {
      return make_unexpected(
          Error{.code = ErrorCode::kInvalidArgument,
                .message = "[holoviz] PresentFrameTable: image index out of range."});
    }
    if (pending_[image_index] == nullptr) {
      return make_unexpected(
          Error{.code = ErrorCode::kInvalidState,
                .message = "[holoviz] Presenting a swapchain image that was not acquired."});
    }
    if (pending_[image_index]->issuance != acquire_issuance) {
      return make_unexpected(
          Error{.code = ErrorCode::kFrameDropped,
                .message = "[holoviz] The acquire behind this present was superseded by a "
                           "fresh acquire of the same image index; the frame is gone — "
                           "re-acquire and render again."});
    }
    if (pending_[image_index]->generation != current_generation) {
      return make_unexpected(
          Error{.code = ErrorCode::kFrameDropped,
                .message = "[holoviz] AcquireToken predates a swapchain recreation; the "
                           "frame is gone — re-acquire and render again."});
    }
    return pending_[image_index];
  }

  /// True when the slot's occupant is exactly this acquire. The erase-permission predicate
  /// for peek()'s kFrameDropped exits: a stale-generation occupant that IS the caller's
  /// acquire may be retired by the caller, while a superseded caller must leave the slot's
  /// fresh occupant alone.
  [[nodiscard]] bool holds(uint32_t image_index, uint64_t acquire_issuance) const {
    return image_index < pending_.size() && pending_[image_index] != nullptr &&
           pending_[image_index]->issuance == acquire_issuance;
  }

  /// True when the slot holds a pending (acquired, not yet consumed) frame of any generation.
  [[nodiscard]] bool occupied(uint32_t image_index) const {
    return image_index < pending_.size() && pending_[image_index] != nullptr;
  }

  /// Clear a slot after the frame was consumed (queued or dropped). Out-of-range or
  /// already-empty is a no-op.
  void erase(uint32_t image_index) {
    if (image_index < pending_.size()) {
      pending_[image_index] = nullptr;
    }
  }

 private:
  std::vector<std::shared_ptr<AcquirePayload>> pending_;
};

/// Issuance tombstones for pending acquires retired wholesale by a changed-count
/// swapchain recreation (the frame table is resized, dropping their slots). Presenting
/// such an acquire must report the consumed-frame contract answer (kFrameDropped, id
/// burned, wait answers kDropped) — never the retryable "not acquired"/"out of range"
/// the resized table would produce. Only PENDING issuances are remembered: a frame
/// already consumed before the recreation stays a "not acquired" protocol violation on
/// double-present, never a spurious drop.
class RetiredAcquireLog {
 public:
  /// Bounded memory: payload-pool backpressure caps live acquires at image_count + 2, so
  /// this covers many consecutive count-changing recreations under hoarded tokens; an
  /// evicted (oldest) issuance degrades to the table's "not acquired" answer.
  static constexpr size_t kCapacity = 64;

  /// Append one recreation's batch of pending issuances. Issuances are strictly
  /// increasing for the backend's lifetime and every remembered batch is retired before
  /// any later issuance is minted, so appending sorted batches keeps retired_ sorted —
  /// front eviction is oldest-first.
  void remember(std::vector<uint64_t> issuances) {
    retired_.insert(retired_.end(), issuances.begin(), issuances.end());
    if (retired_.size() > kCapacity) {
      retired_.erase(retired_.begin(),
                     retired_.begin() + static_cast<ptrdiff_t>(retired_.size() - kCapacity));
    }
  }

  /// Single-shot: true (and forgotten) when the issuance belongs to a retired acquire —
  /// the caller records the drop and burns the id exactly once. Unknown issuances answer
  /// false and are left for the frame table to classify.
  [[nodiscard]] bool consume(uint64_t issuance) {
    const auto it = std::lower_bound(retired_.begin(), retired_.end(), issuance);
    if (it == retired_.end() || *it != issuance) {
      return false;
    }
    retired_.erase(it);
    return true;
  }

  void clear() { retired_.clear(); }

 private:
  std::vector<uint64_t> retired_;
};

}  // namespace holoscan::visualization::backends
