#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#else
#include <dirent.h>
#include <fcntl.h>
#include <unistd.h>
#endif

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
  HNSW_ERR_LOCKED_PROCESS = -4,
  HNSW_ERR_LOAD_FAILED = -5,
  HNSW_ERR_SAVE_FAILED = -6,
  HNSW_ERR_RENAME_FAILED = -7,
  HNSW_ERR_ZERO_VECTOR = -8,
  HNSW_ERR_HNSWLIB = -10,
};

namespace {

#ifdef _WIN32
using LockHandle = HANDLE;
constexpr LockHandle kInvalidLockHandle = INVALID_HANDLE_VALUE;
#else
using LockHandle = int;
constexpr LockHandle kInvalidLockHandle = -1;
#endif

static bool is_valid_lock_handle(LockHandle handle) {
#ifdef _WIN32
  return handle != nullptr && handle != INVALID_HANDLE_VALUE;
#else
  return handle >= 0;
#endif
}

static void remove_file_if_present(const std::filesystem::path& path) {
  std::error_code ec;
  std::filesystem::remove(path, ec);
}

static LockHandle acquire_process_lock(const std::filesystem::path& lock_path) {
#ifdef _WIN32
  HANDLE handle = CreateFileW(
      lock_path.c_str(), GENERIC_READ | GENERIC_WRITE,
      FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr,
      OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
  if (handle == INVALID_HANDLE_VALUE) return kInvalidLockHandle;

  OVERLAPPED overlapped{};
  if (!LockFileEx(handle, LOCKFILE_EXCLUSIVE_LOCK | LOCKFILE_FAIL_IMMEDIATELY,
                  0, MAXDWORD, MAXDWORD, &overlapped)) {
    CloseHandle(handle);
    return kInvalidLockHandle;
  }
  return handle;
#else
  int fd = open(lock_path.c_str(), O_CREAT | O_RDWR, 0666);
  if (fd < 0) return kInvalidLockHandle;

  struct flock fl {};
  fl.l_type = F_WRLCK;
  fl.l_whence = SEEK_SET;
  fl.l_start = 0;
  fl.l_len = 0;

  if (fcntl(fd, F_SETLK, &fl) != 0) {
    close(fd);
    return kInvalidLockHandle;
  }
  return fd;
#endif
}

static void release_process_lock(LockHandle handle) {
  if (!is_valid_lock_handle(handle)) return;
#ifdef _WIN32
  OVERLAPPED overlapped{};
  UnlockFileEx(handle, 0, MAXDWORD, MAXDWORD, &overlapped);
  CloseHandle(handle);
#else
  struct flock fl {};
  fl.l_type = F_UNLCK;
  fl.l_whence = SEEK_SET;
  fl.l_start = 0;
  fl.l_len = 0;
  fcntl(handle, F_SETLK, &fl);
  close(handle);
#endif
}

static bool sync_file(const std::filesystem::path& path) {
#ifdef _WIN32
  HANDLE handle = CreateFileW(path.c_str(), GENERIC_WRITE,
                              FILE_SHARE_READ | FILE_SHARE_WRITE |
                                  FILE_SHARE_DELETE,
                              nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL,
                              nullptr);
  if (handle == INVALID_HANDLE_VALUE) return false;
  bool ok = FlushFileBuffers(handle) != 0;
  CloseHandle(handle);
  return ok;
#else
  int fd = open(path.c_str(), O_RDONLY);
  if (fd < 0) return false;
#ifdef __APPLE__
  bool ok = fcntl(fd, F_FULLFSYNC) == 0;
#else
  bool ok = fsync(fd) == 0;
#endif
  close(fd);
  return ok;
#endif
}

static bool sync_directory_best_effort(const std::filesystem::path& path) {
#ifdef _WIN32
  (void)path;
  return true;
#else
  DIR* dir = opendir(path.empty() ? "." : path.c_str());
  if (dir == nullptr) return false;
  const int dir_fd = dirfd(dir);
#ifdef __APPLE__
  const bool ok = dir_fd >= 0 && fcntl(dir_fd, F_FULLFSYNC) == 0;
#else
  const bool ok = dir_fd >= 0 && fsync(dir_fd) == 0;
#endif
  closedir(dir);
  return ok;
#endif
}

static bool replace_file(const std::filesystem::path& from,
                         const std::filesystem::path& to) {
#ifdef _WIN32
  if (std::filesystem::exists(to)) {
    if (ReplaceFileW(to.c_str(), from.c_str(), nullptr,
                     REPLACEFILE_IGNORE_MERGE_ERRORS, nullptr,
                     nullptr) != 0) {
      return true;
    }
  }

  return MoveFileExW(from.c_str(), to.c_str(),
                     MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH) != 0;
#else
  return std::rename(from.c_str(), to.c_str()) == 0;
#endif
}

static std::string canonicalize_path(const char* raw_path) {
  std::filesystem::path path(raw_path);
  std::error_code ec;

  auto canonical = std::filesystem::weakly_canonical(path, ec);
  if (!ec) {
    return canonical.string();
  }

  ec.clear();
  auto absolute = std::filesystem::absolute(path, ec);
  if (!ec) {
    return absolute.lexically_normal().string();
  }

  return path.lexically_normal().string();
}

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

struct SharedHnswContext {
  uint64_t id = 0;
  std::string canonical_path;
  std::string lock_file_path;

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

  LockHandle lock_handle = kInvalidLockHandle;

  ~SharedHnswContext() {
    delete index;
    delete space;
    release_process_lock(lock_handle);
    if (!lock_file_path.empty()) {
      remove_file_if_present(std::filesystem::path(lock_file_path));
    }
  }
};

struct ContextHandle {
  std::shared_ptr<SharedHnswContext> ctx;
};

static std::mutex registry_mutex;
static std::unordered_map<std::string, std::weak_ptr<SharedHnswContext>>
    registry;
static uint64_t next_id = 1;

static std::shared_ptr<SharedHnswContext> get_context(void* handle) {
  if (handle == nullptr) return {};
  auto* typed = static_cast<ContextHandle*>(handle);
  return typed->ctx;
}

}  // namespace

extern "C" {

HNSW_EXPORT int hnsw_abi_version() { return 1; }

HNSW_EXPORT void* hnsw_open(const char* path,
                            int dimensions,
                            int max_elements,
                            int m,
                            int ef_construction,
                            int ef_search,
                            int metric,
                            int allow_replace_deleted,
                            int reserved_open_flags,
                            int* error_code) {
  try {
    (void)reserved_open_flags;

    if (path == nullptr) {
      if (error_code) *error_code = HNSW_ERR_NULL_ARG;
      return nullptr;
    }

    if (dimensions <= 0 || max_elements <= 0 || m <= 0 ||
        ef_construction <= 0 || ef_search <= 0 ||
        (metric != 0 && metric != 1 && metric != 2)) {
      if (error_code) *error_code = HNSW_ERR_INVALID_ARG;
      return nullptr;
    }

    const std::string canonical = canonicalize_path(path);

    {
      std::lock_guard<std::mutex> lock(registry_mutex);
      auto it = registry.find(canonical);
      if (it != registry.end()) {
        auto ctx = it->second.lock();
        if (ctx) {
          if (ctx->dimensions != dimensions || ctx->metric != metric ||
              ctx->allow_replace_deleted !=
                  (allow_replace_deleted != 0)) {
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

    const auto canonical_path = std::filesystem::path(canonical);
    const auto lock_path =
        std::filesystem::path(canonical + ".lock");
    const auto lock_handle = acquire_process_lock(lock_path);
    if (!is_valid_lock_handle(lock_handle)) {
      if (error_code) *error_code = HNSW_ERR_LOCKED_PROCESS;
      return nullptr;
    }

    auto ctx = std::make_shared<SharedHnswContext>();
    ctx->id = next_id++;
    ctx->canonical_path = canonical;
    ctx->lock_file_path = lock_path.string();
    ctx->lock_handle = lock_handle;
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

    std::error_code ec;
    const bool file_exists = std::filesystem::exists(canonical_path, ec);

    try {
      if (file_exists) {
        ctx->index = new hnswlib::HierarchicalNSW<float>(
            ctx->space, canonical, false, max_elements,
            ctx->allow_replace_deleted);
      } else {
        ctx->index = new hnswlib::HierarchicalNSW<float>(
            ctx->space, max_elements, m, ef_construction, 100,
            ctx->allow_replace_deleted);
      }
    } catch (...) {
      if (error_code) *error_code = HNSW_ERR_LOAD_FAILED;
      return nullptr;
    }

    ctx->index->setEf(ef_search);

    {
      std::lock_guard<std::mutex> lock(registry_mutex);
      registry[canonical] = ctx;
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

HNSW_EXPORT int hnsw_checkpoint(void* handle) {
  try {
    auto ctx = get_context(handle);
    if (!ctx || !ctx->index) return HNSW_ERR_NULL_ARG;

    std::unique_lock<std::shared_mutex> lock(ctx->rw_lock);

    const auto index_path = std::filesystem::path(ctx->canonical_path);
    const auto tmp_path =
        std::filesystem::path(ctx->canonical_path + ".tmp");

    try {
      ctx->index->saveIndex(tmp_path.string());
    } catch (...) {
      return HNSW_ERR_SAVE_FAILED;
    }

    if (!sync_file(tmp_path)) {
      remove_file_if_present(tmp_path);
      return HNSW_ERR_SAVE_FAILED;
    }

    if (!replace_file(tmp_path, index_path)) {
      remove_file_if_present(tmp_path);
      return HNSW_ERR_RENAME_FAILED;
    }
    sync_directory_best_effort(index_path.parent_path());
    return HNSW_OK;
  } catch (...) {
    return HNSW_ERR_HNSWLIB;
  }
}

}  // extern "C"
