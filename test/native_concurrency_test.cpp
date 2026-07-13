#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "hnswlib.h"
#include "two_group_mutex.h"

namespace {

using namespace std::chrono_literals;

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

template <typename Predicate>
bool wait_until(Predicate predicate,
                std::chrono::milliseconds timeout = 2000ms) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (!predicate()) {
    if (std::chrono::steady_clock::now() >= deadline) return false;
    std::this_thread::yield();
  }
  return true;
}

void update_peak(std::atomic<int>& peak, int value) {
  int observed = peak.load();
  while (observed < value &&
         !peak.compare_exchange_weak(observed, value)) {
  }
}

template <typename Acquire>
void verify_shared_group_overlap(const std::string& name, Acquire acquire) {
  TwoGroupMutex mutex;
  std::atomic<int> active{0};
  std::atomic<int> peak{0};
  std::atomic<bool> release{false};

  auto worker = [&] {
    auto lock = acquire(mutex);
    const int now_active = active.fetch_add(1) + 1;
    update_peak(peak, now_active);
    while (!release.load()) std::this_thread::yield();
    active.fetch_sub(1);
  };

  std::thread first(worker);
  std::thread second(worker);
  const bool overlapped = wait_until([&] { return peak.load() == 2; });
  release = true;
  first.join();
  second.join();

  require(overlapped, name + " operations did not overlap");
}

void verify_query_upsert_exclusion_and_fair_handoff() {
  TwoGroupMutex mutex;
  std::atomic<bool> release_initial_query{false};
  std::atomic<bool> initial_query_entered{false};
  std::atomic<bool> upsert_waiting{false};
  std::atomic<bool> upsert_entered{false};
  std::atomic<bool> late_query_entered{false};
  std::atomic<int> sequence{0};
  std::atomic<int> upsert_order{0};
  std::atomic<int> late_query_order{0};

  std::thread initial_query([&] {
    auto lock = mutex.acquire_query();
    initial_query_entered = true;
    while (!release_initial_query.load()) std::this_thread::yield();
  });
  require(wait_until([&] { return initial_query_entered.load(); }),
          "initial query did not enter");

  std::thread upsert([&] {
    upsert_waiting = true;
    auto lock = mutex.acquire_upsert();
    upsert_order = sequence.fetch_add(1) + 1;
    upsert_entered = true;
  });
  require(wait_until([&] { return upsert_waiting.load(); }),
          "upsert did not start waiting");
  std::this_thread::sleep_for(20ms);

  std::thread late_query([&] {
    auto lock = mutex.acquire_query();
    late_query_order = sequence.fetch_add(1) + 1;
    late_query_entered = true;
  });
  std::this_thread::sleep_for(20ms);
  require(!upsert_entered.load(), "upsert overlapped an active query");
  require(!late_query_entered.load(),
          "late query bypassed a waiting upsert phase");

  release_initial_query = true;
  initial_query.join();
  upsert.join();
  late_query.join();

  require(upsert_order.load() == 1 && late_query_order.load() == 2,
          "query/upsert phases did not hand off fairly");
}

void verify_admin_priority_and_exclusion() {
  TwoGroupMutex mutex;
  std::atomic<bool> release_initial_query{false};
  std::atomic<bool> release_admin{false};
  std::atomic<bool> initial_query_entered{false};
  std::atomic<bool> admin_waiting{false};
  std::atomic<bool> admin_entered{false};
  std::atomic<bool> late_upsert_entered{false};

  std::thread initial_query([&] {
    auto lock = mutex.acquire_query();
    initial_query_entered = true;
    while (!release_initial_query.load()) std::this_thread::yield();
  });
  require(wait_until([&] { return initial_query_entered.load(); }),
          "initial query did not enter for admin test");

  std::thread admin([&] {
    admin_waiting = true;
    auto lock = mutex.acquire_admin();
    admin_entered = true;
    while (!release_admin.load()) std::this_thread::yield();
  });
  require(wait_until([&] { return admin_waiting.load(); }),
          "admin did not start waiting");
  std::this_thread::sleep_for(20ms);

  std::thread late_upsert([&] {
    auto lock = mutex.acquire_upsert();
    late_upsert_entered = true;
  });
  std::this_thread::sleep_for(20ms);
  require(!admin_entered.load(), "admin overlapped an active query");
  require(!late_upsert_entered.load(),
          "upsert bypassed an administrative waiter");

  release_initial_query = true;
  initial_query.join();
  require(wait_until([&] { return admin_entered.load(); }),
          "admin was not prioritized after the query drained");
  require(!late_upsert_entered.load(), "upsert overlapped an active admin");

  release_admin = true;
  admin.join();
  late_upsert.join();
  require(late_upsert_entered.load(), "upsert did not resume after admin");
}

void verify_hnsw_concurrent_insert_and_update() {
  constexpr std::size_t dimensions = 4;
  constexpr int thread_count = 8;
  constexpr int items_per_thread = 250;
  constexpr int item_count = thread_count * items_per_thread;
  constexpr int replacement_count = 200;

  hnswlib::L2Space space(dimensions);
  hnswlib::HierarchicalNSW<float> index(&space, 4000, 16, 100, 100, true);
  std::atomic<bool> failed{false};

  auto run_parallel = [&](auto operation) {
    std::vector<std::thread> workers;
    workers.reserve(thread_count);
    for (int worker = 0; worker < thread_count; ++worker) {
      workers.emplace_back([&, worker] {
        try {
          operation(worker);
        } catch (...) {
          failed = true;
        }
      });
    }
    for (auto& worker : workers) worker.join();
    require(!failed.exchange(false), "concurrent hnswlib operation failed");
  };

  run_parallel([&](int worker) {
    for (int offset = 0; offset < items_per_thread; ++offset) {
      const auto label = static_cast<hnswlib::labeltype>(
          worker * items_per_thread + offset);
      const float vector[dimensions] = {
          static_cast<float>(label), 1.0f, 2.0f, 3.0f};
      index.addPoint(vector, label, true);
    }
  });
  require(index.getCurrentElementCount() == item_count,
          "concurrent inserts lost elements");

  run_parallel([&](int worker) {
    for (int offset = 0; offset < items_per_thread; ++offset) {
      const auto label = static_cast<hnswlib::labeltype>(
          worker * items_per_thread + offset);
      const float vector[dimensions] = {
          static_cast<float>(label), 4.0f, 5.0f, 6.0f};
      index.addPoint(vector, label, true);
    }
  });

  for (int label = 0; label < replacement_count; ++label) {
    index.markDelete(static_cast<hnswlib::labeltype>(label));
  }

  run_parallel([&](int worker) {
    const int replacements_per_thread = replacement_count / thread_count;
    for (int offset = 0; offset < replacements_per_thread; ++offset) {
      const auto label = static_cast<hnswlib::labeltype>(
          item_count + worker * replacements_per_thread + offset);
      const float vector[dimensions] = {
          static_cast<float>(label), 7.0f, 8.0f, 9.0f};
      index.addPoint(vector, label, true);
    }
  });

  require(index.getCurrentElementCount() == item_count,
          "deleted-slot replacement unexpectedly grew the index");
  require(index.getDeletedCount() == 0,
          "deleted-slot replacement left deleted elements behind");
}

}  // namespace

int main() {
  try {
    verify_shared_group_overlap("query", [](TwoGroupMutex& mutex) {
      return mutex.acquire_query();
    });
    verify_shared_group_overlap("upsert", [](TwoGroupMutex& mutex) {
      return mutex.acquire_upsert();
    });
    verify_query_upsert_exclusion_and_fair_handoff();
    verify_admin_priority_and_exclusion();
    verify_hnsw_concurrent_insert_and_update();
    std::cout << "Native concurrency tests passed.\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "Native concurrency test failed: " << error.what() << '\n';
    return 1;
  }
}
