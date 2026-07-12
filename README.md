# dart_hnsw

Cross-platform Dart Native bindings for [hnswlib](https://github.com/nmslib/hnswlib) — fast approximate nearest neighbor search through native C++ code with shared in-process indexes, reader-writer locking, and multi-isolate concurrency.

---

## Features

- **Approximate nearest neighbor (ANN)** search through native `hnswlib`
- **Native C++ performance** — search and index operations run in native code via `dart:ffi`, not in the Dart VM. No pure-Dart ANN overhead.
- **Pure in-memory** — no filesystem dependency. The index lives entirely in native memory.
- **One shared native context** per collection name inside a Dart process
- **Reader-writer locking** so concurrent searches across isolates can proceed together while writes stay exclusive
- **Symmetric multi-isolate access** without duplicating the same index in each isolate
- **Cross-platform native assets**: macOS, Linux, Windows, iOS, Android

---

## Why This Package

Most local HNSW packages are optimized for single-isolate, pure-Dart usage. `dart_hnsw` exists for the case where multiple isolates in the same Dart process need to work against the same native index efficiently.

Native efficiency. Index construction, vector insertion, and nearest-neighbor search all run in C++ through `dart:ffi` — not in the Dart VM. This means:

- **No serialization overhead** — vectors are passed directly as native float arrays
- **No GC pressure** from large index structures — the entire graph lives off-heap in native memory
- **Shared memory** — one native context per collection name, shared across all isolates in the process
- **Zero-copy reads** — search traverses the native graph directly without marshalling the index into Dart

Core concurrency model:

- opening the same collection name from multiple isolates attaches them to one shared native context
- searches use a shared read lock, so concurrent readers can run at the same time
- upserts, deletes, resize, and `checkpoint()` use an exclusive write lock

This lets you build isolate-heavy CLI or server workloads without copying one large HNSW index into every isolate separately.

### No filesystem dependency

The index lives entirely in native memory. `checkpoint()` serializes the index to a `Uint8List` that you can store anywhere — a file, a database BLOB, Redis, S3, etc. `load()` restores from bytes. The native layer has zero filesystem code, making it truly cross-platform (iOS, Android, macOS, Linux, Windows).

---

## Getting Started

### 1. Add dependency

```yaml
dependencies:
  dart_hnsw: ^0.2.0
```

### 2. Import the package

```dart
import 'package:dart_hnsw/dart_hnsw.dart';
```

### 3. Use the index

```dart
final index = HnswIndex.create(
  collectionName: 'my_collection',
  dimensions: 3,
  metric: HnswMetric.cosine,
);

index.upsert(1, [1, 0, 0]);
index.upsert(2, [0, 1, 0]);

final results = index.search([1, 0, 0], k: 1);
print(results.first.id);

// checkpoint() returns raw bytes — store them wherever you want
final bytes = index.checkpoint();
await File('my_index.bin').writeAsBytes(bytes);

index.close();
```

### 4. Restore from bytes

```dart
final bytes = await File('my_index.bin').readAsBytes();
final index = HnswIndex.load(
  collectionName: 'my_collection',
  bytes: bytes,
  dimensions: 3,
  metric: HnswMetric.cosine,
);

final results = index.search([1, 0, 0], k: 1);
print(results.first.id);

index.close();
```

---

## How Should I Run It

### `dart run`

Use this for normal development.

```sh
dart run bin/your_app.dart
```

The native library is built and loaded automatically through Dart hooks/code assets. No manual `cmake` step is required.

This mode preserves the shared native context behavior for multiple isolates inside the same Dart process.

### `dart compile exe`

Use this when shipping a compiled CLI, but copy the shared library next to the executable.

```sh
dart compile exe bin/your_app.dart -o dist/your_app
```

Sidecar library names:

- macOS: `libdart_hnsw.dylib`
- Linux: `libdart_hnsw.so`
- Windows: `dart_hnsw.dll`

Typical packaging flow on macOS:

```sh
dart compile exe bin/your_app.dart -o dist/your_app
cp .dart_tool/lib/libdart_hnsw.dylib dist/
```

On Linux, copy `libdart_hnsw.so`. On Windows, copy `dart_hnsw.dll`.

At runtime, `dart_hnsw` resolves native code in this order:

1. Dart native-asset mapping, if available
2. `DART_HNSW_LIBRARY_PATH`, if set
3. the executable directory
4. the current working directory

Example override:

```sh
DART_HNSW_LIBRARY_PATH=/opt/dart_hnsw/libdart_hnsw.so ./your_app
```

---

## Persistence

- `close()` only releases the handle. It does not save.
- `checkpoint()` serializes the index to a `Uint8List` using the same binary format as hnswlib's native `saveIndex`.
- The caller is responsible for storing the bytes (filesystem, database, blob storage, etc.).
- `HnswIndex.load()` restores an index from bytes previously produced by `checkpoint()`.
- The binary payload does not include all wrapper configuration. Store and restore the same `dimensions`, `metric`, and `allowReplaceDeleted` values yourself; using different values is unsupported.
- Checkpoint bytes use hnswlib's native binary layout, including native `size_t` fields. Restore them on a compatible ABI, rather than treating them as a portable interchange format between 32-bit and 64-bit targets.

---

## Concurrency Model

- Same process, same collection name: one shared native index context
- Same process, multiple isolates: all isolates attach to that shared context
- Searches and getters: shared reader lock
- Upserts, deletes, resize, and checkpoint: exclusive writer lock

This is one of the package's core design goals: isolate-friendly shared native indexing, not just local ANN search.

---

## Supported Platforms

The hook and native source are configured for:

- macOS
- Linux
- Windows

Android and iOS are also declared in `pubspec.yaml` and use the same C++17 hook path. They require validation in the consuming Flutter app for the target ABI/device; this package does not provide a mobile application fixture.

The package hook has been smoke-tested here for Android arm64 (NDK API 21) and iOS arm64 simulator (iOS 13). Actual Flutter app/device runtime validation remains the responsibility of the consuming application.

The native layer has **zero filesystem dependencies** — all index state lives in memory, and checkpoint/load operate on byte buffers.

Web is unsupported (requires native FFI).

---

## API Overview

### `HnswIndex`

| Method / Property | Description |
|-------|-------------|
| `HnswIndex.create(...)` | Create a new in-memory index |
| `HnswIndex.load(...)` | Load an index from a byte buffer |
| `upsert(id, vector)` | Insert or replace one vector |
| `upsertBatch(entries)` | Insert or replace multiple vectors |
| `search(query, {k})` | Approximate nearest neighbor search |
| `markDeleted(id)` | Mark a label deleted |
| `resize(newMaxElements)` | Increase index capacity |
| `setEfSearch(ef)` | Adjust search breadth |
| `checkpoint()` | Serialize the index to `Uint8List` |
| `close()` | Release the native handle |
| `elementCount` | Total stored element count |
| `deletedCount` | Deleted entry count |
| `liveCount` | Convenience count of non-deleted entries |
| `maxElements` | Current capacity |
| `efSearch` | Current search breadth |
| `contextId` | Shared native context identifier for same-process/same-collection handles |

### `HnswMetric`

- `HnswMetric.l2`
- `HnswMetric.innerProduct`
- `HnswMetric.cosine`

---

## Running Tests

```sh
dart test
```
