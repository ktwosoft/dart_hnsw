#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "hnswlib.h"

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
  HNSW_ERR_LOAD_FAILED = -5,
  HNSW_ERR_SAVE_FAILED = -6,
  HNSW_ERR_ZERO_VECTOR = -8,
  HNSW_ERR_HNSWLIB = -10,
};

namespace {

static float vector_norm(const float* values, int dim) {
  float sum = 0.0f;
  for (int i = 0; i < dim; i++) {
    sum += values[i] * values[i];
  }
  return std::sqrt(sum);
}

static void normalize_vector(float* values, int dim) {
  const auto norm = vector_norm(values, dim);
  if (norm <= 0.0f) return;

  const auto inv = 1.0f / norm;
  for (int i = 0; i < dim; i++) {
    values[i] *= inv;
  }
}

// ---------------------------------------------------------------------------
// Shared context — keyed by collection name (not file path)
// ---------------------------------------------------------------------------

struct SharedHnswContext {
  uint64_t id = 0;
  std::string collection_name;

  std::shared_mutex rw_lock;

  hnswlib::SpaceInterface<float>* space = nullptr;
  hnswlib::HierarchicalNSW<float>* index = nullptr;

  int dimensions = 0;
  int max_elements = 0;
  int m = 16;
  int ef_construction = 200;
  int ef_search = 64;
  int metric = 0;
  bool allow_replace_deleted = true;

  ~SharedHnswContext() {
    delete index;
    delete space;
  }
};

struct ContextHandle {
  std::shared_ptr<SharedHnswContext> ctx;
};

static std::mutex registry_mutex;
static std::unordered_map<std::string, std::weak_ptr<SharedHnswContext>>
    registry;
static std::atomic<uint64_t> next_id{1};

static std::shared_ptr<SharedHnswContext> get_context(void* handle) {
  if (handle == nullptr) return {};
  auto* typed = static_cast<ContextHandle*>(handle);
  return typed->ctx;
}

}  // namespace

extern "C" {

HNSW_EXPORT int hnsw_abi_version() { return 2; }

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

    if (dimensions <= 0 || max_elements <= 0 || m <= 0 ||
        ef_construction <= 0 || ef_search <= 0 ||
        (metric != 0 && metric != 1 && metric != 2)) {
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
          if (ctx->dimensions != dimensions || ctx->metric != metric ||
              ctx->allow_replace_deleted != (allow_replace_deleted != 0)) {
            if (error_code) *error_code = HNSW_ERR_CONFIG_MISMATCH;
            return nullptr;
          }

          auto* handle = new ContextHandle();
          handle->ctx = std::move(ctx);
          if (error_code) *error_code = HNSW_OK;
          return handle;
        }
        registry.erase(it);
      }
    }

    auto ctx = std::make_shared<SharedHnswContext>();
    ctx->id = next_id.fetch_add(1, std::memory_order_relaxed);
    ctx->collection_name = name;
    ctx->dimensions = dimensions;
    ctx->max_elements = max_elements;
    ctx->m = m;
    ctx->ef_construction = ef_construction;
    ctx->ef_search = ef_search;
    ctx->metric = metric;
    ctx->allow_replace_deleted = (allow_replace_deleted != 0);

    if (metric == 1) {
      ctx->space = new hnswlib::InnerProductSpace(dimensions);
    } else {
      ctx->space = new hnswlib::L2Space(dimensions);
    }

    try {
      ctx->index = new hnswlib::HierarchicalNSW<float>(
          ctx->space, max_elements, m, ef_construction, 100,
          ctx->allow_replace_deleted);
    } catch (...) {
      delete ctx->space;
      ctx->space = nullptr;
      if (error_code) *error_code = HNSW_ERR_HNSWLIB;
      return nullptr;
    }

    ctx->index->setEf(ef_search);

    {
      std::lock_guard<std::mutex> lock(registry_mutex);
      // Another thread may have finished the same first open while this
      // context was being constructed. Publish only one context per name.
      auto it = registry.find(name);
      if (it != registry.end()) {
        auto existing = it->second.lock();
        if (existing) {
          if (existing->dimensions != dimensions ||
              existing->metric != metric ||
              existing->allow_replace_deleted !=
                  (allow_replace_deleted != 0)) {
            if (error_code) *error_code = HNSW_ERR_CONFIG_MISMATCH;
            return nullptr;
          }

          auto* handle = new ContextHandle();
          handle->ctx = std::move(existing);
          if (error_code) *error_code = HNSW_OK;
          return handle;
        }
      }
      registry[name] = ctx;
    }

    auto* handle = new ContextHandle();
    handle->ctx = std::move(ctx);
    if (error_code) *error_code = HNSW_OK;
    return handle;
  } catch (...) {
    if (error_code) *error_code = HNSW_ERR_HNSWLIB;
    return nullptr;
  }
}

// ---------------------------------------------------------------------------
// hnsw_load — create an index from a byte buffer (same binary format as
// what hnsw_checkpoint returns)
// ---------------------------------------------------------------------------

HNSW_EXPORT void* hnsw_load(const char* collection_name,
                            const uint8_t* data,
                            int data_size,
                            int dimensions,
                            int max_elements,
                            int m,
                            int ef_construction,
                            int ef_search,
                            int metric,
                            int allow_replace_deleted,
                            int* error_code) {
  try {
    if (collection_name == nullptr || data == nullptr) {
      if (error_code) *error_code = HNSW_ERR_NULL_ARG;
      return nullptr;
    }

    if (dimensions <= 0 || max_elements <= 0 || m <= 0 ||
        ef_construction <= 0 || ef_search <= 0 ||
        (metric != 0 && metric != 1 && metric != 2) || data_size <= 0) {
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
          if (ctx->dimensions != dimensions || ctx->metric != metric ||
              ctx->allow_replace_deleted != (allow_replace_deleted != 0)) {
            if (error_code) *error_code = HNSW_ERR_CONFIG_MISMATCH;
            return nullptr;
          }

          auto* handle = new ContextHandle();
          handle->ctx = std::move(ctx);
          if (error_code) *error_code = HNSW_OK;
          return handle;
        }
        registry.erase(it);
      }
    }

    auto ctx = std::make_shared<SharedHnswContext>();
    ctx->id = next_id.fetch_add(1, std::memory_order_relaxed);
    ctx->collection_name = name;
    ctx->dimensions = dimensions;
    ctx->max_elements = max_elements;
    ctx->m = m;
    ctx->ef_construction = ef_construction;
    ctx->ef_search = ef_search;
    ctx->metric = metric;
    ctx->allow_replace_deleted = (allow_replace_deleted != 0);

    if (metric == 1) {
      ctx->space = new hnswlib::InnerProductSpace(dimensions);
    } else {
      ctx->space = new hnswlib::L2Space(dimensions);
    }

    try {
      ctx->index = new hnswlib::HierarchicalNSW<float>(
          ctx->space, ctx->allow_replace_deleted);
      {
        std::string buf(data, data + data_size);
        std::istringstream iss(std::move(buf), std::ios::in | std::ios::binary);
        ctx->index->loadIndex(iss, ctx->space,
                              static_cast<size_t>(max_elements));
      }
    } catch (...) {
      delete ctx->index;
      ctx->index = nullptr;
      delete ctx->space;
      ctx->space = nullptr;
      if (error_code) *error_code = HNSW_ERR_LOAD_FAILED;
      return nullptr;
    }

    ctx->index->setEf(ef_search);

    {
      std::lock_guard<std::mutex> lock(registry_mutex);
      // Another thread may have finished the same first open while this
      // context was being loaded. Publish only one context per name.
      auto it = registry.find(name);
      if (it != registry.end()) {
        auto existing = it->second.lock();
        if (existing) {
          if (existing->dimensions != dimensions ||
              existing->metric != metric ||
              existing->allow_replace_deleted !=
                  (allow_replace_deleted != 0)) {
            if (error_code) *error_code = HNSW_ERR_CONFIG_MISMATCH;
            return nullptr;
          }

          auto* handle = new ContextHandle();
          handle->ctx = std::move(existing);
          if (error_code) *error_code = HNSW_OK;
          return handle;
        }
      }
      registry[name] = ctx;
    }

    auto* handle = new ContextHandle();
    handle->ctx = std::move(ctx);
    if (error_code) *error_code = HNSW_OK;
    return handle;
  } catch (...) {
    if (error_code) *error_code = HNSW_ERR_HNSWLIB;
    return nullptr;
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

HNSW_EXPORT int hnsw_upsert_vector(void* handle,
                                   uint64_t label,
                                   const float* vector) {
  try {
    auto ctx = get_context(handle);
    if (!ctx || !ctx->index || vector == nullptr) return HNSW_ERR_NULL_ARG;

    std::unique_lock<std::shared_mutex> lock(ctx->rw_lock);

    std::vector<float> normalized;
    const float* vec_to_use = vector;

    if (ctx->metric == 2) {
      if (vector_norm(vector, ctx->dimensions) == 0.0f) {
        return HNSW_ERR_ZERO_VECTOR;
      }
      normalized.assign(vector, vector + ctx->dimensions);
      normalize_vector(normalized.data(), ctx->dimensions);
      vec_to_use = normalized.data();
    }

    ctx->index->addPoint(vec_to_use, label, ctx->allow_replace_deleted);
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
    auto ctx = get_context(handle);
    if (!ctx || !ctx->index || labels == nullptr || vectors == nullptr) {
      return HNSW_ERR_NULL_ARG;
    }
    if (n <= 0) return HNSW_ERR_INVALID_ARG;

    std::unique_lock<std::shared_mutex> lock(ctx->rw_lock);

    for (int i = 0; i < n; i++) {
      const float* vec = vectors + i * ctx->dimensions;
      if (ctx->metric == 2) {
        if (vector_norm(vec, ctx->dimensions) == 0.0f) {
          return HNSW_ERR_ZERO_VECTOR;
        }
        std::vector<float> normalized(vec, vec + ctx->dimensions);
        normalize_vector(normalized.data(), ctx->dimensions);
        ctx->index->addPoint(normalized.data(), labels[i],
                             ctx->allow_replace_deleted);
      } else {
        ctx->index->addPoint(vec, labels[i], ctx->allow_replace_deleted);
      }
    }

    return HNSW_OK;
  } catch (...) {
    return HNSW_ERR_HNSWLIB;
  }
}

HNSW_EXPORT int hnsw_mark_deleted(void* handle, uint64_t label) {
  try {
    auto ctx = get_context(handle);
    if (!ctx || !ctx->index) return HNSW_ERR_NULL_ARG;
    std::unique_lock<std::shared_mutex> lock(ctx->rw_lock);
    ctx->index->markDelete(label);
    return HNSW_OK;
  } catch (...) {
    return HNSW_ERR_HNSWLIB;
  }
}

HNSW_EXPORT int hnsw_resize(void* handle, int new_max_elements) {
  try {
    auto ctx = get_context(handle);
    if (!ctx || !ctx->index) return HNSW_ERR_NULL_ARG;
    if (new_max_elements <= 0) return HNSW_ERR_INVALID_ARG;

    std::unique_lock<std::shared_mutex> lock(ctx->rw_lock);
    ctx->index->resizeIndex(new_max_elements);
    ctx->max_elements = new_max_elements;
    return HNSW_OK;
  } catch (...) {
    return HNSW_ERR_HNSWLIB;
  }
}

HNSW_EXPORT int hnsw_set_ef_search(void* handle, int ef_search) {
  try {
    auto ctx = get_context(handle);
    if (!ctx || !ctx->index) return HNSW_ERR_NULL_ARG;
    if (ef_search <= 0) return HNSW_ERR_INVALID_ARG;

    std::unique_lock<std::shared_mutex> lock(ctx->rw_lock);
    ctx->index->setEf(ef_search);
    ctx->ef_search = ef_search;
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
    if (!ctx || !ctx->index || query == nullptr || out_labels == nullptr ||
        out_distances == nullptr || out_count == nullptr) {
      return HNSW_ERR_NULL_ARG;
    }
    if (k <= 0) return HNSW_ERR_INVALID_ARG;

    std::shared_lock<std::shared_mutex> lock(ctx->rw_lock);

    std::vector<float> normalized;
    const float* query_to_use = query;
    if (ctx->metric == 2) {
      if (vector_norm(query, ctx->dimensions) == 0.0f) {
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

HNSW_EXPORT uint64_t hnsw_element_count(void* handle) {
  auto ctx = get_context(handle);
  if (!ctx || !ctx->index) return 0;
  std::shared_lock<std::shared_mutex> lock(ctx->rw_lock);
  return ctx->index->getCurrentElementCount();
}

HNSW_EXPORT uint64_t hnsw_deleted_count(void* handle) {
  auto ctx = get_context(handle);
  if (!ctx || !ctx->index) return 0;
  std::shared_lock<std::shared_mutex> lock(ctx->rw_lock);
  return ctx->index->getDeletedCount();
}

HNSW_EXPORT uint64_t hnsw_max_elements(void* handle) {
  auto ctx = get_context(handle);
  if (!ctx || !ctx->index) return 0;
  std::shared_lock<std::shared_mutex> lock(ctx->rw_lock);
  return ctx->index->getMaxElements();
}

HNSW_EXPORT int hnsw_get_ef_search(void* handle) {
  auto ctx = get_context(handle);
  if (!ctx) return HNSW_ERR_NULL_ARG;
  std::shared_lock<std::shared_mutex> lock(ctx->rw_lock);
  return ctx->ef_search;
}

// ---------------------------------------------------------------------------
// hnsw_checkpoint — serialize the index to a byte buffer.
//
// Two-phase API:
//   1. Call hnsw_checkpoint(handle, nullptr, 0, &size) to get the size.
//   2. Allocate a buffer of `size` bytes on the Dart side.
//   3. Call hnsw_checkpoint(handle, buffer, size, &written) to fill it.
//
// Returns HNSW_OK on success. The caller owns the buffer.
// ---------------------------------------------------------------------------

HNSW_EXPORT int hnsw_checkpoint(void* handle,
                                uint8_t* out_buffer,
                                int buffer_size,
                                int* out_written) {
  try {
    auto ctx = get_context(handle);
    if (!ctx || !ctx->index) return HNSW_ERR_NULL_ARG;

    std::unique_lock<std::shared_mutex> lock(ctx->rw_lock);

    std::string serialized;
    try {
      std::ostringstream oss(std::ios::out | std::ios::binary);
      ctx->index->saveIndex(oss);
      serialized = oss.str();
    } catch (...) {
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
