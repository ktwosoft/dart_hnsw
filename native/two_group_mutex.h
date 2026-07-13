#pragma once

#include <condition_variable>
#include <cstddef>
#include <mutex>
#include <utility>

// Coordinates two internally-concurrent operation groups with a third,
// fully-exclusive administrative tier. Once both shared groups are waiting,
// admission proceeds in finite cohorts so neither group can starve the other.
class TwoGroupMutex {
 public:
  enum class Mode { kIdle, kQuery, kUpsert, kAdmin };

  class ScopedLock {
   public:
    ScopedLock(const ScopedLock&) = delete;
    ScopedLock& operator=(const ScopedLock&) = delete;

    ScopedLock(ScopedLock&& other) noexcept
        : owner_(std::exchange(other.owner_, nullptr)), mode_(other.mode_) {}

    ScopedLock& operator=(ScopedLock&& other) noexcept {
      if (this == &other) return *this;
      release();
      owner_ = std::exchange(other.owner_, nullptr);
      mode_ = other.mode_;
      return *this;
    }

    ~ScopedLock() { release(); }

   private:
    friend class TwoGroupMutex;

    ScopedLock(TwoGroupMutex* owner, Mode mode)
        : owner_(owner), mode_(mode) {}

    void release() noexcept {
      if (owner_ == nullptr) return;
      owner_->release(mode_);
      owner_ = nullptr;
    }

    TwoGroupMutex* owner_;
    Mode mode_;
  };

  TwoGroupMutex() = default;
  TwoGroupMutex(const TwoGroupMutex&) = delete;
  TwoGroupMutex& operator=(const TwoGroupMutex&) = delete;

  [[nodiscard]] ScopedLock acquire_query() {
    return acquire_group(Mode::kQuery);
  }

  [[nodiscard]] ScopedLock acquire_upsert() {
    return acquire_group(Mode::kUpsert);
  }

  [[nodiscard]] ScopedLock acquire_admin() {
    std::unique_lock<std::mutex> lock(mutex_);
    ++waiting_admin_;
    schedule_if_idle();
    condition_.wait(lock, [this] {
      return mode_ == Mode::kAdmin && active_count_ == 0;
    });
    --waiting_admin_;
    active_count_ = 1;
    return ScopedLock(this, Mode::kAdmin);
  }

 private:
  [[nodiscard]] ScopedLock acquire_group(Mode requested) {
    std::unique_lock<std::mutex> lock(mutex_);
    std::size_t& waiting = waiting_for(requested);
    ++waiting;
    schedule_if_idle();

    condition_.wait(lock, [this, requested] {
      if (mode_ != requested) return false;
      if (admission_grants_ > 0) return true;
      return waiting_admin_ == 0 && waiting_for(other_group(requested)) == 0;
    });

    --waiting;
    if (admission_grants_ > 0) --admission_grants_;
    ++active_count_;
    return ScopedLock(this, requested);
  }

  void release(Mode released) noexcept {
    std::unique_lock<std::mutex> lock(mutex_);
    if (mode_ != released || active_count_ == 0) return;

    --active_count_;
    if (active_count_ == 0 && admission_grants_ == 0) {
      mode_ = Mode::kIdle;
      schedule_if_idle();
      condition_.notify_all();
    }
  }

  void schedule_if_idle() {
    if (mode_ != Mode::kIdle) return;

    if (waiting_admin_ > 0) {
      mode_ = Mode::kAdmin;
      admission_grants_ = 0;
      condition_.notify_all();
      return;
    }

    const bool query_waiting = waiting_query_ > 0;
    const bool upsert_waiting = waiting_upsert_ > 0;
    if (!query_waiting && !upsert_waiting) return;

    Mode selected = Mode::kQuery;
    if (query_waiting && upsert_waiting) {
      selected = next_group_;
    } else if (upsert_waiting) {
      selected = Mode::kUpsert;
    }

    mode_ = selected;
    admission_grants_ = waiting_for(selected);
    next_group_ = other_group(selected);
    condition_.notify_all();
  }

  static Mode other_group(Mode group) {
    return group == Mode::kQuery ? Mode::kUpsert : Mode::kQuery;
  }

  std::size_t& waiting_for(Mode group) {
    return group == Mode::kQuery ? waiting_query_ : waiting_upsert_;
  }

  const std::size_t& waiting_for(Mode group) const {
    return group == Mode::kQuery ? waiting_query_ : waiting_upsert_;
  }

  std::mutex mutex_;
  std::condition_variable condition_;
  Mode mode_ = Mode::kIdle;
  Mode next_group_ = Mode::kQuery;
  std::size_t active_count_ = 0;
  std::size_t admission_grants_ = 0;
  std::size_t waiting_query_ = 0;
  std::size_t waiting_upsert_ = 0;
  std::size_t waiting_admin_ = 0;
};
