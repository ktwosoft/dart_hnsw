import 'dart:ffi' as ffi;

import 'package:ffi/ffi.dart';

import 'native_bindings.dart';

final HnswNativeBindings _nativeBindings = HnswNativeBindings();

/// Error codes returned by the native hnswlib layer.
///
/// Each value maps to a specific failure mode from the C++ bindings.
enum HnswErrorCode {
  /// Operation completed successfully.
  ok(0),

  /// A null argument was passed to the native layer.
  nullArg(-1),

  /// An argument value was invalid.
  invalidArg(-2),

  /// Configuration mismatch when attaching to an existing shared context.
  configMismatch(-3),

  /// The index is locked by another process.
  lockedProcess(-4),

  /// Failed to load the index from disk.
  loadFailed(-5),

  /// Failed to save the index to disk.
  saveFailed(-6),

  /// Failed to rename checkpoint files atomically.
  renameFailed(-7),

  /// Zero vector provided where not allowed (e.g. cosine metric).
  zeroVector(-8),

  /// Error propagated from the underlying hnswlib library.
  hnswlib(-10);

  final int code;
  const HnswErrorCode(this.code);

  static HnswErrorCode fromCode(int code) => values.firstWhere(
    (value) => value.code == code,
    orElse: () => HnswErrorCode.hnswlib,
  );
}

/// Base exception for all dart_hnsw errors.
///
/// Specific subclasses distinguish open failures, dimension mismatches,
/// capacity limits, concurrency conflicts, and native errors.
class HnswException implements Exception {
  /// Human-readable description of the error.
  final String message;

  /// Creates an exception with the given [message].
  const HnswException(this.message);

  @override
  String toString() => 'HnswException: $message';
}

/// Thrown when opening or creating an index fails.
class HnswOpenException extends HnswException {
  const HnswOpenException(super.message);
}

/// Thrown when vector dimensions or metric settings do not match
/// the active shared index context.
class HnswDimensionMismatchException extends HnswException {
  const HnswDimensionMismatchException(super.message);
}

/// Thrown when the index has reached its maximum element capacity.
class HnswCapacityExceededException extends HnswException {
  const HnswCapacityExceededException(super.message);
}

/// Thrown when another process holds the exclusive file lock.
class HnswConcurrentProcessException extends HnswException {
  const HnswConcurrentProcessException(super.message);
}

/// Thrown when loading a persisted index from disk fails.
class HnswLoadException extends HnswException {
  const HnswLoadException(super.message);
}

/// Thrown when saving (checkpointing) the index to disk fails.
class HnswCheckpointException extends HnswException {
  const HnswCheckpointException(super.message);
}

/// Thrown when a zero vector is used with a metric that does not allow it
/// (e.g. cosine).
class HnswZeroVectorException extends HnswException {
  const HnswZeroVectorException(super.message);
}

/// Thrown when an unrecognized error code is returned from the native layer.
class HnswNativeException extends HnswException {
  const HnswNativeException(super.message);
}

/// Distance metric used for nearest neighbor search.
///
/// Maps directly to the metrics supported by hnswlib.
enum HnswMetric {
  /// Euclidean (L2) distance.
  l2,

  /// Inner product (dot product) similarity.
  innerProduct,

  /// Cosine similarity (1 - cosine of the angle between vectors).
  cosine,
}

/// A single nearest-neighbor search result.
///
/// Contains the [id] of the matched vector and its [distance] from the query.
class HnswResult {
  /// Label of the matched vector.
  final int id;

  /// Distance from the query vector (interpretation depends on the metric).
  final double distance;

  /// Creates a search result with the given [id] and [distance].
  const HnswResult(this.id, this.distance);
}

/// A vector entry used for batch upserts.
///
/// Pairs a label [id] with its corresponding [vector] data.
class HnswVectorEntry {
  /// Label for this vector entry.
  final int id;

  /// The vector data (must match the index dimensions).
  final List<double> vector;

  /// Creates a vector entry with the given [id] and [vector].
  const HnswVectorEntry(this.id, this.vector);
}

ffi.Pointer<ffi.Float> _listToFloatPtr(List<double> values) {
  final ptr = calloc<ffi.Float>(values.length);
  for (var i = 0; i < values.length; i++) {
    ptr[i] = values[i];
  }
  return ptr;
}

HnswException _errorForCode(int code, String context) {
  switch (code) {
    case -1:
      return HnswException('$context: null argument');
    case -2:
      return HnswException('$context: invalid argument');
    case -3:
      return HnswDimensionMismatchException(
        '$context: dimensions, metric, or delete-reuse setting mismatch with the active shared index context',
      );
    case -4:
      return HnswConcurrentProcessException(
        '$context: index is locked by another process',
      );
    case -5:
      return HnswLoadException('$context: failed to load index');
    case -6:
      return HnswCheckpointException('$context: failed to save index');
    case -7:
      return HnswCheckpointException(
        '$context: failed to replace checkpoint files',
      );
    case -8:
      return HnswZeroVectorException(
        '$context: zero vector not allowed for cosine metric',
      );
    default:
      return HnswNativeException('$context: native error $code');
  }
}

int _metricCode(HnswMetric metric) {
  switch (metric) {
    case HnswMetric.l2:
      return 0;
    case HnswMetric.innerProduct:
      return 1;
    case HnswMetric.cosine:
      return 2;
  }
}

/// A shared native HNSW index backed by hnswlib.
///
/// Opening the same canonical [path] from multiple isolates or locations
/// within the same process attaches them to one shared native context.
/// Searches use a shared reader lock; mutations use an exclusive writer lock.
///
/// {@category Index}
class HnswIndex {
  ffi.Pointer<ffi.Void>? _handle;
  final int _dimensions;

  HnswIndex._(this._handle, this._dimensions);

  /// Opens or creates an HNSW index at [path].
  ///
  /// If the same [path] is already open in this process, the existing shared
  /// native context is reused. The [dimensions], [metric], and
  /// [allowReplaceDeleted] values must match the existing context.
  ///
  /// [maxElements] sets the initial capacity. [m] and [efConstruction] control
  /// graph build quality. [efSearch] controls the default search breadth.
  static HnswIndex open({
    required String path,
    required int dimensions,
    HnswMetric metric = HnswMetric.cosine,
    int maxElements = 100000,
    int m = 16,
    int efConstruction = 200,
    int efSearch = 64,
    bool allowReplaceDeleted = true,
  }) {
    final errorCode = calloc<ffi.Int32>();
    final pathPtr = path.toNativeUtf8();

    try {
      final handle = _nativeBindings.hnswOpen(
        pathPtr,
        dimensions,
        maxElements,
        m,
        efConstruction,
        efSearch,
        _metricCode(metric),
        allowReplaceDeleted ? 1 : 0,
        0,
        errorCode,
      );

      if (errorCode.value != 0) {
        throw _errorForCode(errorCode.value, 'HnswIndex.open(\'$path\')');
      }

      return HnswIndex._(handle, dimensions);
    } finally {
      calloc.free(errorCode);
      calloc.free(pathPtr);
    }
  }

  /// Whether the native handle has been released via [close].
  bool get isClosed => _handle == null;

  ffi.Pointer<ffi.Void> _requireOpenHandle() {
    final handle = _handle;
    if (handle == null || handle == ffi.nullptr) {
      throw StateError('HnswIndex is closed');
    }
    return handle;
  }

  /// Inserts or replaces the vector for the given label [id].
  ///
  /// The [vector] length must match the index [dimensions] specified at open.
  /// If a vector with the same [id] already exists it is replaced.
  void upsert(int id, List<double> vector) {
    if (vector.length != _dimensions) {
      throw ArgumentError(
        'Expected $_dimensions dimensions, got ${vector.length}',
      );
    }

    final vecPtr = _listToFloatPtr(vector);
    try {
      final result = _nativeBindings.hnswUpsertVector(
        _requireOpenHandle(),
        id,
        vecPtr,
      );
      if (result != 0) {
        throw _errorForCode(result, 'upsert(label=$id)');
      }
    } finally {
      calloc.free(vecPtr);
    }
  }

  /// Inserts or replaces multiple vectors in a single native call.
  ///
  /// More efficient than repeated [upsert] calls when adding many vectors.
  /// All entries must match the index [dimensions].
  void upsertBatch(List<HnswVectorEntry> entries) {
    if (entries.isEmpty) return;

    final labels = calloc<ffi.Uint64>(entries.length);
    final vectors = calloc<ffi.Float>(entries.length * _dimensions);

    try {
      for (var i = 0; i < entries.length; i++) {
        final entry = entries[i];
        if (entry.vector.length != _dimensions) {
          throw ArgumentError(
            'Entry ${entry.id}: expected $_dimensions dimensions, got ${entry.vector.length}',
          );
        }

        labels[i] = entry.id;
        for (var j = 0; j < _dimensions; j++) {
          vectors[i * _dimensions + j] = entry.vector[j];
        }
      }

      final result = _nativeBindings.hnswUpsertBatch(
        _requireOpenHandle(),
        labels,
        vectors,
        entries.length,
      );
      if (result != 0) {
        throw _errorForCode(result, 'upsertBatch(${entries.length} entries)');
      }
    } finally {
      calloc.free(labels);
      calloc.free(vectors);
    }
  }

  /// Searches for the [k] nearest neighbors of [query].
  ///
  /// Returns up to [k] results sorted by distance (ascending). The distance
  /// interpretation depends on the index [metric].
  List<HnswResult> search(List<double> query, {int k = 10}) {
    if (query.length != _dimensions) {
      throw ArgumentError(
        'Expected $_dimensions dimensions, got ${query.length}',
      );
    }

    final queryPtr = _listToFloatPtr(query);
    final outLabels = calloc<ffi.Uint64>(k);
    final outDistances = calloc<ffi.Float>(k);
    final outCount = calloc<ffi.Int32>();

    try {
      final result = _nativeBindings.hnswSearchKnn(
        _requireOpenHandle(),
        queryPtr,
        k,
        outLabels,
        outDistances,
        outCount,
      );
      if (result != 0) {
        throw _errorForCode(result, 'search(k=$k)');
      }

      return List.generate(
        outCount.value,
        (index) => HnswResult(outLabels[index], outDistances[index].toDouble()),
      );
    } finally {
      calloc.free(queryPtr);
      calloc.free(outLabels);
      calloc.free(outDistances);
      calloc.free(outCount);
    }
  }

  /// Marks the vector with label [id] as deleted.
  ///
  /// Deleted entries are excluded from search results. Their storage is
  /// reused when [allowReplaceDeleted] was enabled at open.
  void markDeleted(int id) {
    final result = _nativeBindings.hnswMarkDeleted(_requireOpenHandle(), id);
    if (result != 0) {
      throw _errorForCode(result, 'markDeleted(label=$id)');
    }
  }

  /// Increases the index capacity to [newMaxElements].
  ///
  /// Cannot shrink the index below the current [elementCount].
  void resize(int newMaxElements) {
    final result = _nativeBindings.hnswResize(
      _requireOpenHandle(),
      newMaxElements,
    );
    if (result != 0) {
      throw _errorForCode(result, 'resize($newMaxElements)');
    }
  }

  /// Sets the search breadth parameter [ef].
  ///
  /// Higher values improve recall at the cost of search speed.
  void setEfSearch(int ef) {
    final result = _nativeBindings.hnswSetEfSearch(_requireOpenHandle(), ef);
    if (result != 0) {
      throw _errorForCode(result, 'setEfSearch($ef)');
    }
  }

  /// Total number of elements stored in the index (including deleted).
  int get elementCount =>
      _nativeBindings.hnswElementCount(_requireOpenHandle());

  /// Number of elements marked as deleted.
  int get deletedCount =>
      _nativeBindings.hnswDeletedCount(_requireOpenHandle());

  /// Convenience property: [elementCount] - [deletedCount].
  int get liveCount => elementCount - deletedCount;

  /// Current maximum capacity of the index.
  int get maxElements => _nativeBindings.hnswMaxElements(_requireOpenHandle());

  /// Current search breadth (ef) parameter.
  int get efSearch => _nativeBindings.hnswGetEfSearch(_requireOpenHandle());

  /// Opaque identifier for the shared native context.
  ///
  /// Handles attached to the same canonical path share the same context id.
  int get contextId => _nativeBindings.hnswContextId(_requireOpenHandle());

  /// Persists the current index state to disk.
  ///
  /// Writes the raw hnswlib index file. [close] does **not** save —
  /// call [checkpoint] explicitly before closing to persist changes.
  void checkpoint() {
    final result = _nativeBindings.hnswCheckpoint(_requireOpenHandle());
    if (result != 0) {
      throw _errorForCode(result, 'checkpoint');
    }
  }

  /// Releases the native handle.
  ///
  /// Does **not** save the index — call [checkpoint] first to persist changes.
  /// After calling [close] the instance must not be used further.
  void close() {
    final handle = _handle;
    if (handle == null || handle == ffi.nullptr) {
      return;
    }
    _nativeBindings.hnswClose(handle);
    _handle = null;
  }
}
