#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "hnswlib.h"
#include "two_group_mutex.h"

#ifdef _WIN32
#define HNSW_EXPORT __declspec(dllexport)
#else
#define HNSW_EXPORT
#endif

enum HnswErrorCode {
  HNSW_OK = 0,
  HNSW_ERR_NULL_ARG = -1,
  HNSW_ERR_INVALID_ARG = -2,
  HNSW_ERR_CONFIG_MISMATCH = -3,
  HNSW_ERR_GENERATION_CONFLICT = -4,
  HNSW_ERR_LOAD_FAILED = -5,
  HNSW_ERR_SAVE_FAILED = -6,
  HNSW_ERR_ZERO_VECTOR = -8,
  HNSW_ERR_NONFINITE_VECTOR = -9,
  HNSW_ERR_HNSWLIB = -10,
  HNSW_ERR_EXCLUSIVE_MUTATION_ACCESS = -11,
};

struct HnswStatsSnapshot {
  uint64_t element_count;
  uint64_t deleted_count;
  uint64_t max_elements;
  int32_t ef_search;
};

namespace {

using NativeSpace = hnswlib::SpaceInterface<float>;
using NativeIndex = hnswlib::HierarchicalNSW<float>;

constexpr int kMaxDimensions = 65536;
constexpr int kMaxElements = 10000000;
constexpr int kMinM = 2;
constexpr int kMaxM = 10000;
constexpr int kMaxEf = 1000000;
constexpr int kMaxBatchSize = 100000;
constexpr int kMaxSearchResults = 100000;
constexpr size_t kMaxBatchFloatValues = 16 * 1024 * 1024;
constexpr size_t kMaxCheckpointBytes = 512 * 1024 * 1024;
constexpr size_t kMaxIndexBaseBytes = 1024ULL * 1024 * 1024;

static bool is_valid_index_config(int dimensions,
                                  int max_elements,
                                  int m,
                                  int ef_construction,
                                  int ef_search,
                                  int metric) {
  if (dimensions <= 0 || dimensions > kMaxDimensions ||
      max_elements <= 0 || max_elements > kMaxElements || m < kMinM ||
      m > kMaxM || ef_construction <= 0 || ef_construction > kMaxEf ||
      ef_search <= 0 || ef_search > kMaxEf ||
      (metric != 0 && metric != 1 && metric != 2)) {
    return false;
  }

  const size_t data_size = static_cast<size_t>(dimensions) * sizeof(float);
  const size_t level0_links =
      static_cast<size_t>(m) * 2 * sizeof(hnswlib::tableint) +
      sizeof(hnswlib::linklistsizeint);
  if (level0_links > std::numeric_limits<size_t>::max() - data_size ||
      level0_links + data_size >
          std::numeric_limits<size_t>::max() - sizeof(hnswlib::labeltype)) {
    return false;
  }

  const size_t bytes_per_element =
      level0_links + data_size + sizeof(hnswlib::labeltype);
  return static_cast<size_t>(max_elements) <=
         kMaxIndexBaseBytes / bytes_per_element;
}

template <typename T>
static bool read_checkpoint_value(const uint8_t* data,
                                  size_t data_size,
                                  size_t* offset,
                                  T* value) {
  if (*offset > data_size || sizeof(T) > data_size - *offset) return false;
  std::memcpy(value, data + *offset, sizeof(T));
  *offset += sizeof(T);
  return true;
}

// hnswlib trusts graph sizing values stored in its binary header. Validate
// those values before loadIndex() uses them to allocate the target capacity.
static bool has_safe_checkpoint_layout(const uint8_t* data,
                                       size_t data_size,
                                       int dimensions,
                                       int requested_max_elements) {
  size_t offset = 0;
  size_t offset_level0 = 0;
  size_t stored_max_elements = 0;
  size_t current_elements = 0;
  size_t bytes_per_element = 0;
  size_t label_offset = 0;
  size_t offset_data = 0;
  int max_level = 0;
  hnswlib::tableint enterpoint = 0;
  size_t stored_max_m = 0;
  size_t stored_max_m0 = 0;
  size_t stored_m = 0;
  double mult = 0.0;
  size_t stored_ef_construction = 0;

  if (!read_checkpoint_value(data, data_size, &offset, &offset_level0) ||
      !read_checkpoint_value(data, data_size, &offset, &stored_max_elements) ||
      !read_checkpoint_value(data, data_size, &offset, &current_elements) ||
      !read_checkpoint_value(data, data_size, &offset, &bytes_per_element) ||
      !read_checkpoint_value(data, data_size, &offset, &label_offset) ||
      !read_checkpoint_value(data, data_size, &offset, &offset_data) ||
      !read_checkpoint_value(data, data_size, &offset, &max_level) ||
      !read_checkpoint_value(data, data_size, &offset, &enterpoint) ||
      !read_checkpoint_value(data, data_size, &offset, &stored_max_m) ||
      !read_checkpoint_value(data, data_size, &offset, &stored_max_m0) ||
      !read_checkpoint_value(data, data_size, &offset, &stored_m) ||
      !read_checkpoint_value(data, data_size, &offset, &mult) ||
      !read_checkpoint_value(data, data_size, &offset,
                             &stored_ef_construction)) {
    return false;
  }

  if (stored_max_elements == 0 || stored_max_elements > kMaxElements ||
      current_elements > stored_max_elements || stored_m < kMinM ||
      stored_m > kMaxM || stored_max_m != stored_m ||
      stored_max_m0 != stored_m * 2 ||
      stored_ef_construction < stored_m ||
      stored_ef_construction > kMaxEf || !std::isfinite(mult) ||
      mult <= 0.0 || offset_level0 != 0) {
    return false;
  }

  const size_t vector_bytes = static_cast<size_t>(dimensions) * sizeof(float);
  const size_t level0_links =
      stored_max_m0 * sizeof(hnswlib::tableint) +
      sizeof(hnswlib::linklistsizeint);
  if (level0_links > std::numeric_limits<size_t>::max() - vector_bytes ||
      level0_links + vector_bytes >
          std::numeric_limits<size_t>::max() - sizeof(hnswlib::labeltype)) {
    return false;
  }

  const size_t expected_bytes_per_element =
      level0_links + vector_bytes + sizeof(hnswlib::labeltype);
  if (bytes_per_element != expected_bytes_per_element ||
      offset_data != level0_links || label_offset != level0_links + vector_bytes) {
    return false;
  }

  const size_t load_capacity =
      static_cast<size_t>(requested_max_elements) < current_elements
          ? stored_max_elements
          : static_cast<size_t>(requested_max_elements);
  return load_capacity <= kMaxIndexBaseBytes / bytes_per_element;
}

static bool is_finite_vector(const float* values, int dim) {
  for (int i = 0; i < dim; i++) {
    if (!std::isfinite(values[i])) return false;
  }
  return true;
}

static double vector_norm(const float* values, int dim) {
  double sum = 0.0;
  for (int i = 0; i < dim; i++) {
    sum += static_cast<double>(values[i]) * values[i];
  }
  return std::sqrt(sum);
}

static void normalize_vector(float* values, int dim) {
  const auto norm = vector_norm(values, dim);
  if (norm <= 0.0f) return;

  const auto inv = 1.0 / norm;
  for (int i = 0; i < dim; i++) {
    values[i] = static_cast<float>(values[i] * inv);
  }
}

// ---------------------------------------------------------------------------
// Shared context — keyed by collection name (not file path)
// ---------------------------------------------------------------------------

struct SharedHnswContext {
  uint64_t id = 0;
  std::string collection_name;

  TwoGroupMutex operation_lock;

  NativeSpace* space = nullptr;
  NativeIndex* index = nullptr;

  int dimensions = 0;
  int max_elements = 0;
  int m = 16;
  int ef_construction = 200;
  int ef_search = 64;
  int metric = 0;
  bool allow_replace_deleted = true;
  uint64_t exclusive_mutation_owner_handle_id = 0;
  std::atomic<uint64_t> generation{0};

  ~SharedHnswContext() {
    delete index;
    delete space;
  }
};

struct ContextHandle {
  std::shared_ptr<SharedHnswContext> ctx;
  uint64_t id = 0;
};

static std::mutex registry_mutex;
static std::unordered_map<std::string, std::weak_ptr<SharedHnswContext>>
    registry;
static std::atomic<uint64_t> next_context_id{1};
static std::atomic<uint64_t> next_handle_id{1};

static bool context_config_matches(const SharedHnswContext& ctx,
                                   int dimensions,
                                   int metric,
                                   int allow_replace_deleted) {
  return ctx.dimensions == dimensions && ctx.metric == metric &&
         ctx.allow_replace_deleted == (allow_replace_deleted != 0);
}

static std::unique_ptr<NativeSpace> make_space(int dimensions, int metric) {
  if (metric == 1 || metric == 2) {
    return std::make_unique<hnswlib::InnerProductSpace>(dimensions);
  }
  return std::make_unique<hnswlib::L2Space>(dimensions);
}

static std::unique_ptr<NativeIndex> make_empty_index(
    NativeSpace* space,
    int max_elements,
    int m,
    int ef_construction,
    bool allow_replace_deleted) {
  return std::make_unique<NativeIndex>(space, max_elements, m,
                                       ef_construction, 100,
                                       allow_replace_deleted);
}

struct LoadedGraph {
  std::unique_ptr<NativeSpace> space;
  std::unique_ptr<NativeIndex> index;
};

static LoadedGraph build_loaded_graph(const uint8_t* data,
                                      int data_size,
                                      int dimensions,
                                      int max_elements,
                                      int m,
                                      int ef_construction,
                                      int ef_search,
                                      int metric,
                                      bool allow_replace_deleted) {
  if (!has_safe_checkpoint_layout(data, static_cast<size_t>(data_size),
                                  dimensions, max_elements)) {
    throw std::runtime_error("Unsafe checkpoint layout");
  }

  LoadedGraph graph;
  graph.space = make_space(dimensions, metric);
  graph.index = make_empty_index(graph.space.get(), 1, m, ef_construction,
                                 allow_replace_deleted);
  const std::string buffer(reinterpret_cast<const char*>(data),
                           static_cast<size_t>(data_size));
  std::istringstream input(buffer, std::ios::in | std::ios::binary);
  graph.index->loadIndex(input, graph.space.get(),
                         static_cast<size_t>(max_elements));
  graph.index->setEf(ef_search);
  return graph;
}

static ContextHandle* make_handle(std::shared_ptr<SharedHnswContext> ctx) {
  auto* handle = new ContextHandle();
  handle->ctx = std::move(ctx);
  handle->id = next_handle_id.fetch_add(1, std::memory_order_relaxed);
  return handle;
}

static int mutation_access_error(const SharedHnswContext& ctx,
                                 uint64_t handle_id) {
  return ctx.exclusive_mutation_owner_handle_id != 0 &&
                 ctx.exclusive_mutation_owner_handle_id != handle_id
             ? HNSW_ERR_EXCLUSIVE_MUTATION_ACCESS
             : HNSW_OK;
}

static int replace_loaded_graph(ContextHandle* handle,
                                LoadedGraph graph,
                                int64_t expected_generation) {
  auto ctx = handle->ctx;
  NativeIndex* old_index = nullptr;
  NativeSpace* old_space = nullptr;
  {
    auto lock = ctx->operation_lock.acquire_admin();
    const int access_error = mutation_access_error(*ctx, handle->id);
    if (access_error != HNSW_OK) return access_error;
    if (expected_generation < 0 ||
        ctx->generation.load() != static_cast<uint64_t>(expected_generation)) {
      return HNSW_ERR_GENERATION_CONFLICT;
    }

    old_index = ctx->index;
    old_space = ctx->space;
    ctx->index = graph.index.release();
    ctx->space = graph.space.release();
    ctx->generation.fetch_add(1);
  }
  delete old_index;
  delete old_space;
  return HNSW_OK;
}

static std::shared_ptr<SharedHnswContext> get_context(void* handle) {
  auto* typed = static_cast<ContextHandle*>(handle);
  return typed ? typed->ctx : std::shared_ptr<SharedHnswContext>();
}

static ContextHandle* get_handle(void* handle) {
  return static_cast<ContextHandle*>(handle);
}

}  // namespace

extern "C" {

HNSW_EXPORT int hnsw_abi_version() { return 3; }

// ---------------------------------------------------------------------------
// hnsw_create — create a new in-memory index (no file path)
// ---------------------------------------------------------------------------

HNSW_EXPORT void* hnsw_create(const char* collection_name,
                              int dimensions,
                              int max_elements,
                              int m,
                              int ef_construction,
                              int ef_search,
                              int metric,
                              int allow_replace_deleted,
                              int* error_code) {
  try {
    if (collection_name == nullptr) {
      if (error_code) *error_code = HNSW_ERR_NULL_ARG;
      return nullptr;
    }

    if (!is_valid_index_config(dimensions, max_elements, m, ef_construction,
                               ef_search, metric)) {
      if (error_code) *error_code = HNSW_ERR_INVALID_ARG;
      return nullptr;
    }

    const std::string name(collection_name);

    {
      std::lock_guard<std::mutex> lock(registry_mutex);
      auto it = registry.find(name);
      if (it != registry.end()) {
        auto ctx = it->second.lock();
        if (ctx) {
          if (!context_config_matches(*ctx, dimensions, metric,
                                      allow_replace_deleted)) {
            if (error_code) *error_code = HNSW_ERR_CONFIG_MISMATCH;
            return nullptr;
          }

          if (error_code) *error_code = HNSW_OK;
          return make_handle(std::move(ctx));
        }
        registry.erase(it);
      }
    }

    auto ctx = std::make_shared<SharedHnswContext>();
    ctx->id = next_context_id.fetch_add(1, std::memory_order_relaxed);
    ctx->collection_name = name;
    ctx->dimensions = dimensions;
    ctx->max_elements = max_elements;
    ctx->m = m;
    ctx->ef_construction = ef_construction;
    ctx->ef_search = ef_search;
    ctx->metric = metric;
    ctx->allow_replace_deleted = (allow_replace_deleted != 0);

    try {
      auto space = make_space(dimensions, metric);
      auto index = make_empty_index(space.get(), max_elements, m,
                                    ef_construction,
                                    ctx->allow_replace_deleted);
      index->setEf(ef_search);
      ctx->space = space.release();
      ctx->index = index.release();
    } catch (...) {
      if (error_code) *error_code = HNSW_ERR_HNSWLIB;
      return nullptr;
    }

    {
      std::lock_guard<std::mutex> lock(registry_mutex);
      // Another thread may have finished the same first open while this
      // context was being constructed. Publish only one context per name.
      auto it = registry.find(name);
      if (it != registry.end()) {
        auto existing = it->second.lock();
        if (existing) {
          if (!context_config_matches(*existing, dimensions, metric,
                                      allow_replace_deleted)) {
            if (error_code) *error_code = HNSW_ERR_CONFIG_MISMATCH;
            return nullptr;
          }

          if (error_code) *error_code = HNSW_OK;
          return make_handle(std::move(existing));
        }
      }
      registry[name] = ctx;
    }

    if (error_code) *error_code = HNSW_OK;
    return make_handle(std::move(ctx));
  } catch (...) {
    if (error_code) *error_code = HNSW_ERR_HNSWLIB;
    return nullptr;
  }
}

// ---------------------------------------------------------------------------
// hnsw_load — atomically replace an existing shared in-memory index.
// ---------------------------------------------------------------------------

HNSW_EXPORT int hnsw_load(
    void* handle,
    const uint8_t* data,
    int data_size,
    int64_t expected_generation) {
  try {
    auto* context_handle = get_handle(handle);
    if (!context_handle || data == nullptr) return HNSW_ERR_NULL_ARG;
    auto ctx = context_handle->ctx;
    if (data_size <= 0 ||
        static_cast<size_t>(data_size) > kMaxCheckpointBytes) {
      return HNSW_ERR_INVALID_ARG;
    }

    int dimensions = 0;
    int max_elements = 0;
    int m = 0;
    int ef_construction = 0;
    int ef_search = 0;
    int metric = 0;
    bool allow_replace_deleted = false;
    {
      auto lock = ctx->operation_lock.acquire_admin();
      if (!ctx->index) return HNSW_ERR_NULL_ARG;
      const int access_error = mutation_access_error(*ctx, context_handle->id);
      if (access_error != HNSW_OK) return access_error;
      dimensions = ctx->dimensions;
      max_elements = ctx->max_elements;
      m = ctx->m;
      ef_construction = ctx->ef_construction;
      ef_search = ctx->ef_search;
      metric = ctx->metric;
      allow_replace_deleted = ctx->allow_replace_deleted;
    }

    LoadedGraph graph;
    try {
      graph = build_loaded_graph(data, data_size, dimensions, max_elements, m,
                                 ef_construction, ef_search, metric,
                                 allow_replace_deleted);
    } catch (...) {
      return HNSW_ERR_LOAD_FAILED;
    }

    return replace_loaded_graph(context_handle, std::move(graph),
                                expected_generation);
  } catch (...) {
    return HNSW_ERR_HNSWLIB;
  }
}

HNSW_EXPORT int hnsw_close(void* handle) {
  if (handle == nullptr) return HNSW_ERR_NULL_ARG;
  delete static_cast<ContextHandle*>(handle);
  return HNSW_OK;
}

HNSW_EXPORT uint64_t hnsw_context_id(void* handle) {
  auto ctx = get_context(handle);
  return ctx ? ctx->id : 0;
}

HNSW_EXPORT uint64_t hnsw_generation(void* handle) {
  auto ctx = get_context(handle);
  if (!ctx) return 0;
  return ctx->generation.load();
}

HNSW_EXPORT int hnsw_acquire_exclusive_mutation_access(void* handle) {
  try {
    auto* context_handle = get_handle(handle);
    if (!context_handle) return HNSW_ERR_NULL_ARG;
    auto ctx = context_handle->ctx;

    auto lock = ctx->operation_lock.acquire_admin();
    if (!ctx->index) return HNSW_ERR_NULL_ARG;
    const auto owner_id = ctx->exclusive_mutation_owner_handle_id;
    if (owner_id != 0 && owner_id != context_handle->id) {
      return HNSW_ERR_EXCLUSIVE_MUTATION_ACCESS;
    }
    ctx->exclusive_mutation_owner_handle_id = context_handle->id;
    return HNSW_OK;
  } catch (...) {
    return HNSW_ERR_HNSWLIB;
  }
}

HNSW_EXPORT int hnsw_release_exclusive_mutation_access(void* handle) {
  try {
    auto* context_handle = get_handle(handle);
    if (!context_handle) return HNSW_ERR_NULL_ARG;
    auto ctx = context_handle->ctx;

    auto lock = ctx->operation_lock.acquire_admin();
    if (!ctx->index) return HNSW_ERR_NULL_ARG;
    if (ctx->exclusive_mutation_owner_handle_id != context_handle->id) {
      return HNSW_ERR_EXCLUSIVE_MUTATION_ACCESS;
    }
    ctx->exclusive_mutation_owner_handle_id = 0;
    return HNSW_OK;
  } catch (...) {
    return HNSW_ERR_HNSWLIB;
  }
}

HNSW_EXPORT uint64_t hnsw_registry_size() {
  std::lock_guard<std::mutex> lock(registry_mutex);
  uint64_t count = 0;
  for (auto it = registry.begin(); it != registry.end();) {
    if (it->second.expired()) {
      it = registry.erase(it);
    } else {
      count++;
      ++it;
    }
  }
  return count;
}

HNSW_EXPORT int hnsw_clear(void* handle, int64_t expected_generation) {
  try {
    auto* context_handle = get_handle(handle);
    if (!context_handle) return HNSW_ERR_NULL_ARG;
    auto ctx = context_handle->ctx;

    NativeIndex* old_index = nullptr;
    {
      auto lock = ctx->operation_lock.acquire_admin();
      if (!ctx->index) return HNSW_ERR_NULL_ARG;
      const int access_error = mutation_access_error(*ctx, context_handle->id);
      if (access_error != HNSW_OK) return access_error;
      if (expected_generation < 0 ||
          ctx->generation.load() !=
              static_cast<uint64_t>(expected_generation)) {
        return HNSW_ERR_GENERATION_CONFLICT;
      }

      auto replacement = make_empty_index(
          ctx->space, ctx->max_elements, ctx->m, ctx->ef_construction,
          ctx->allow_replace_deleted);
      replacement->setEf(ctx->ef_search);

      old_index = ctx->index;
      ctx->index = replacement.release();
      ctx->generation.fetch_add(1);
    }
    delete old_index;
    return HNSW_OK;
  } catch (...) {
    return HNSW_ERR_HNSWLIB;
  }
}

HNSW_EXPORT int hnsw_upsert_vector(void* handle,
                                   uint64_t label,
                                   const float* vector) {
  try {
    auto* context_handle = get_handle(handle);
    if (!context_handle || vector == nullptr) return HNSW_ERR_NULL_ARG;
    auto ctx = context_handle->ctx;

    auto lock = ctx->operation_lock.acquire_upsert();
    if (!ctx->index) return HNSW_ERR_NULL_ARG;
    const int access_error = mutation_access_error(*ctx, context_handle->id);
    if (access_error != HNSW_OK) return access_error;
    if (!is_finite_vector(vector, ctx->dimensions)) {
      return HNSW_ERR_NONFINITE_VECTOR;
    }

    std::vector<float> normalized;
    const float* vec_to_use = vector;

    if (ctx->metric == 2) {
      if (vector_norm(vector, ctx->dimensions) == 0.0) {
        return HNSW_ERR_ZERO_VECTOR;
      }
      normalized.assign(vector, vector + ctx->dimensions);
      normalize_vector(normalized.data(), ctx->dimensions);
      vec_to_use = normalized.data();
    }

    ctx->index->addPoint(vec_to_use, label, ctx->allow_replace_deleted);
    ctx->generation.fetch_add(1);
    return HNSW_OK;
  } catch (...) {
    return HNSW_ERR_HNSWLIB;
  }
}

HNSW_EXPORT int hnsw_upsert_batch(void* handle,
                                  const uint64_t* labels,
                                  const float* vectors,
                                  int n) {
  try {
    auto* context_handle = get_handle(handle);
    if (!context_handle || labels == nullptr || vectors == nullptr) {
      return HNSW_ERR_NULL_ARG;
    }
    auto ctx = context_handle->ctx;
    if (n <= 0 || n > kMaxBatchSize) {
      return HNSW_ERR_INVALID_ARG;
    }

    auto lock = ctx->operation_lock.acquire_upsert();
    if (!ctx->index) return HNSW_ERR_NULL_ARG;
    const int access_error = mutation_access_error(*ctx, context_handle->id);
    if (access_error != HNSW_OK) return access_error;
    if (static_cast<size_t>(n) >
        kMaxBatchFloatValues / static_cast<size_t>(ctx->dimensions)) {
      return HNSW_ERR_INVALID_ARG;
    }

    for (int i = 0; i < n; i++) {
      const auto offset =
          static_cast<size_t>(i) * static_cast<size_t>(ctx->dimensions);
      const float* vec = vectors + offset;
      if (!is_finite_vector(vec, ctx->dimensions)) {
        return HNSW_ERR_NONFINITE_VECTOR;
      }
      if (ctx->metric == 2 && vector_norm(vec, ctx->dimensions) == 0.0) {
        return HNSW_ERR_ZERO_VECTOR;
      }
    }

    for (int i = 0; i < n; i++) {
      const auto offset =
          static_cast<size_t>(i) * static_cast<size_t>(ctx->dimensions);
      const float* vec = vectors + offset;
      if (ctx->metric == 2) {
        std::vector<float> normalized(vec, vec + ctx->dimensions);
        normalize_vector(normalized.data(), ctx->dimensions);
        ctx->index->addPoint(normalized.data(), labels[i],
                             ctx->allow_replace_deleted);
      } else {
        ctx->index->addPoint(vec, labels[i], ctx->allow_replace_deleted);
      }
      ctx->generation.fetch_add(1);
    }

    return HNSW_OK;
  } catch (...) {
    return HNSW_ERR_HNSWLIB;
  }
}

HNSW_EXPORT int hnsw_mark_deleted(void* handle, uint64_t label) {
  try {
    auto* context_handle = get_handle(handle);
    if (!context_handle) return HNSW_ERR_NULL_ARG;
    auto ctx = context_handle->ctx;
    auto lock = ctx->operation_lock.acquire_admin();
    if (!ctx->index) return HNSW_ERR_NULL_ARG;
    const int access_error = mutation_access_error(*ctx, context_handle->id);
    if (access_error != HNSW_OK) return access_error;
    ctx->index->markDelete(label);
    ctx->generation.fetch_add(1);
    return HNSW_OK;
  } catch (...) {
    return HNSW_ERR_HNSWLIB;
  }
}

HNSW_EXPORT int hnsw_resize(void* handle, int new_max_elements) {
  try {
    auto* context_handle = get_handle(handle);
    if (!context_handle) return HNSW_ERR_NULL_ARG;
    auto ctx = context_handle->ctx;
    if (new_max_elements <= 0) {
      return HNSW_ERR_INVALID_ARG;
    }

    auto lock = ctx->operation_lock.acquire_admin();
    if (!ctx->index) return HNSW_ERR_NULL_ARG;
    const int access_error = mutation_access_error(*ctx, context_handle->id);
    if (access_error != HNSW_OK) return access_error;
    if (!is_valid_index_config(ctx->dimensions, new_max_elements, ctx->m,
                               ctx->ef_construction, ctx->ef_search,
                               ctx->metric)) {
      return HNSW_ERR_INVALID_ARG;
    }
    ctx->index->resizeIndex(new_max_elements);
    ctx->max_elements = new_max_elements;
    ctx->generation.fetch_add(1);
    return HNSW_OK;
  } catch (...) {
    return HNSW_ERR_HNSWLIB;
  }
}

HNSW_EXPORT int hnsw_set_ef_search(void* handle, int ef_search) {
  try {
    auto* context_handle = get_handle(handle);
    if (!context_handle) return HNSW_ERR_NULL_ARG;
    auto ctx = context_handle->ctx;
    if (ef_search <= 0 || ef_search > kMaxEf) return HNSW_ERR_INVALID_ARG;

    auto lock = ctx->operation_lock.acquire_admin();
    if (!ctx->index) return HNSW_ERR_NULL_ARG;
    const int access_error = mutation_access_error(*ctx, context_handle->id);
    if (access_error != HNSW_OK) return access_error;
    ctx->index->setEf(ef_search);
    ctx->ef_search = ef_search;
    ctx->generation.fetch_add(1);
    return HNSW_OK;
  } catch (...) {
    return HNSW_ERR_HNSWLIB;
  }
}

HNSW_EXPORT int hnsw_search_knn(void* handle,
                                const float* query,
                                int k,
                                uint64_t* out_labels,
                                float* out_distances,
                                int* out_count) {
  try {
    auto ctx = get_context(handle);
    if (!ctx || query == nullptr || out_labels == nullptr ||
        out_distances == nullptr || out_count == nullptr) {
      return HNSW_ERR_NULL_ARG;
    }
    if (k <= 0 || k > kMaxSearchResults) return HNSW_ERR_INVALID_ARG;

    auto lock = ctx->operation_lock.acquire_query();
    if (!ctx->index) return HNSW_ERR_NULL_ARG;

    std::vector<float> normalized;
    const float* query_to_use = query;
    if (!is_finite_vector(query, ctx->dimensions)) {
      return HNSW_ERR_NONFINITE_VECTOR;
    }
    if (ctx->metric == 2) {
      if (vector_norm(query, ctx->dimensions) == 0.0) {
        return HNSW_ERR_ZERO_VECTOR;
      }
      normalized.assign(query, query + ctx->dimensions);
      normalize_vector(normalized.data(), ctx->dimensions);
      query_to_use = normalized.data();
    }

    auto result = ctx->index->searchKnn(query_to_use, k);
    const int count = static_cast<int>(result.size());
    for (int i = count - 1; i >= 0; i--) {
      out_distances[i] = result.top().first;
      out_labels[i] = result.top().second;
      result.pop();
    }
    *out_count = count;
    return HNSW_OK;
  } catch (...) {
    return HNSW_ERR_HNSWLIB;
  }
}

HNSW_EXPORT int hnsw_stats(void* handle, HnswStatsSnapshot* out_stats) {
  if (out_stats == nullptr) return HNSW_ERR_NULL_ARG;
  auto ctx = get_context(handle);
  if (!ctx) return HNSW_ERR_NULL_ARG;
  auto lock = ctx->operation_lock.acquire_admin();
  if (!ctx->index) return HNSW_ERR_NULL_ARG;
  out_stats->element_count = ctx->index->getCurrentElementCount();
  out_stats->deleted_count = ctx->index->getDeletedCount();
  out_stats->max_elements = ctx->index->getMaxElements();
  out_stats->ef_search = ctx->ef_search;
  return HNSW_OK;
}

// ---------------------------------------------------------------------------
// hnsw_checkpoint — serialize the index to a byte buffer.
//
// Two-phase API:
//   1. Call hnsw_checkpoint(handle, nullptr, 0, &size, generation) to get
//      the size for a known generation.
//   2. Allocate a buffer of `size` bytes on the Dart side.
//   3. Call hnsw_checkpoint(handle, buffer, size, &written, generation) to
//      fill it. A changed generation fails instead of returning new bytes.
//
// Returns HNSW_OK on success. The caller owns the buffer.
// ---------------------------------------------------------------------------

HNSW_EXPORT int hnsw_checkpoint(void* handle,
                                uint8_t* out_buffer,
                                int buffer_size,
                                int* out_written,
                                int64_t expected_generation) {
  try {
    auto ctx = get_context(handle);
    if (!ctx) return HNSW_ERR_NULL_ARG;

    auto lock = ctx->operation_lock.acquire_admin();
    if (!ctx->index) return HNSW_ERR_NULL_ARG;
    if (expected_generation < 0 ||
        ctx->generation.load() != static_cast<uint64_t>(expected_generation)) {
      return HNSW_ERR_GENERATION_CONFLICT;
    }

    std::string serialized;
    try {
      const size_t estimated_size = ctx->index->indexFileSize();
      if (estimated_size > kMaxCheckpointBytes ||
          estimated_size >
              static_cast<size_t>(std::numeric_limits<int>::max())) {
        return HNSW_ERR_SAVE_FAILED;
      }
      std::ostringstream oss(std::ios::out | std::ios::binary);
      ctx->index->saveIndex(oss);
      serialized = oss.str();
    } catch (...) {
      return HNSW_ERR_SAVE_FAILED;
    }

    if (serialized.size() > kMaxCheckpointBytes ||
        serialized.size() >
            static_cast<size_t>(std::numeric_limits<int>::max())) {
      return HNSW_ERR_SAVE_FAILED;
    }
    const auto total = static_cast<int>(serialized.size());

    if (out_buffer == nullptr) {
      // Phase 1: caller wants the size.
      if (out_written) *out_written = total;
      return HNSW_OK;
    }

    // Phase 2: caller provided a buffer.
    if (buffer_size < total) {
      // Let a caller retry if another handle grew the index between the size
      // query and this fill operation.
      if (out_written) *out_written = total;
      return HNSW_ERR_INVALID_ARG;
    }

    std::memcpy(out_buffer, serialized.data(), serialized.size());
    if (out_written) *out_written = total;
    return HNSW_OK;
  } catch (...) {
    return HNSW_ERR_HNSWLIB;
  }
}

}  // extern "C"
