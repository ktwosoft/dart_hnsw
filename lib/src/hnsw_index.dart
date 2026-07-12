import 'dart:ffi' as ffi;
import 'dart:typed_data';

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

  /// Failed to load the index from a byte buffer.
  loadFailed(-5),

  /// Failed to save (checkpoint) the index to a byte buffer.
  saveFailed(-6),

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

/// Thrown when creating or loading an index fails.
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

/// Thrown when loading a persisted index from bytes fails.
class HnswLoadException extends HnswException {
  const HnswLoadException(super.message);
}

/// Thrown when saving (checkpointing) the index to bytes fails.
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
    case -5:
      return HnswLoadException('$context: failed to load index');
    case -6:
      return HnswCheckpointException('$context: failed to save index');
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
/// Opening the same [collectionName] from multiple isolates within the same
/// process attaches them to one shared native context. Searches use a shared
/// reader lock; mutations use an exclusive writer lock.
///
/// The index lives entirely in native memory. Use [checkpoint] to serialize
/// the index to a [Uint8List] that you can store anywhere (filesystem,
/// database, blob storage, etc.). Use [HnswIndex.load] to restore from bytes.
///
/// {@category Index}
class HnswIndex {
  ffi.Pointer<ffi.Void>? _handle;
  final int _dimensions;

  HnswIndex._(this._handle, this._dimensions);

  /// Creates a new in-memory HNSW index.
  ///
  /// If an index with the same [collectionName] is already open in this
  /// process, the existing shared native context is reused. The [dimensions],
  /// [metric], and [allowReplaceDeleted] values must match the existing
  /// context.
  ///
  /// [maxElements] sets the initial capacity. [m] and [efConstruction] control
  /// graph build quality. [efSearch] controls the default search breadth.
  static HnswIndex create({
    required String collectionName,
    required int dimensions,
    HnswMetric metric = HnswMetric.cosine,
    int maxElements = 100000,
    int m = 16,
    int efConstruction = 200,
    int efSearch = 64,
    bool allowReplaceDeleted = true,
  }) {
    final errorCode = calloc<ffi.Int32>();
    final namePtr = collectionName.toNativeUtf8();

    try {
      final handle = _nativeBindings.hnswCreate(
        namePtr,
        dimensions,
        maxElements,
        m,
        efConstruction,
        efSearch,
        _metricCode(metric),
        allowReplaceDeleted ? 1 : 0,
        errorCode,
      );

      if (errorCode.value != 0) {
        throw _errorForCode(
          errorCode.value,
          'HnswIndex.create(\'$collectionName\')',
        );
      }

      return HnswIndex._(handle, dimensions);
    } finally {
      calloc.free(errorCode);
      calloc.free(namePtr);
    }
  }

  /// Loads an HNSW index from a byte buffer previously produced by
  /// [checkpoint].
  ///
  /// The [dimensions], [metric], and [allowReplaceDeleted] must match the
  /// values used when the index was originally created. The binary payload
  /// does not store all wrapper configuration, so the caller is responsible
  /// for preserving and supplying the original settings. [maxElements] can
  /// be set larger than the original capacity to allow future growth.
  ///
  /// If an index with the same [collectionName] is already open in this
  /// process, the existing shared native context is reused (the byte buffer
  /// is ignored).
  static HnswIndex load({
    required String collectionName,
    required Uint8List bytes,
    required int dimensions,
    HnswMetric metric = HnswMetric.cosine,
    int maxElements = 100000,
    int m = 16,
    int efConstruction = 200,
    int efSearch = 64,
    bool allowReplaceDeleted = true,
  }) {
    final errorCode = calloc<ffi.Int32>();
    final namePtr = collectionName.toNativeUtf8();
    final dataPtr = calloc<ffi.Uint8>(bytes.length);
    for (var i = 0; i < bytes.length; i++) {
      dataPtr[i] = bytes[i];
    }

    try {
      final handle = _nativeBindings.hnswLoad(
        namePtr,
        dataPtr,
        bytes.length,
        dimensions,
        maxElements,
        m,
        efConstruction,
        efSearch,
        _metricCode(metric),
        allowReplaceDeleted ? 1 : 0,
        errorCode,
      );

      if (errorCode.value != 0) {
        throw _errorForCode(
          errorCode.value,
          'HnswIndex.load(\'$collectionName\')',
        );
      }

      return HnswIndex._(handle, dimensions);
    } finally {
      calloc.free(errorCode);
      calloc.free(namePtr);
      calloc.free(dataPtr);
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
  /// The [vector] length must match the index [dimensions] specified at
  /// creation. If a vector with the same [id] already exists it is replaced.
  void upsert(int id, List<double> vector) {
    final handle = _requireOpenHandle();
    if (vector.length != _dimensions) {
      throw ArgumentError(
        'Expected $_dimensions dimensions, got ${vector.length}',
      );
    }

    final vecPtr = _listToFloatPtr(vector);
    try {
      final result = _nativeBindings.hnswUpsertVector(
        handle,
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
    final handle = _requireOpenHandle();
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
        handle,
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
    final handle = _requireOpenHandle();
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
        handle,
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
  /// reused when [allowReplaceDeleted] was enabled at creation.
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
  /// Handles attached to the same collection name share the same context id.
  int get contextId => _nativeBindings.hnswContextId(_requireOpenHandle());

  /// Serializes the current index state to a [Uint8List].
  ///
  /// The returned bytes use the same binary format as hnswlib's native
  /// `saveIndex`. The caller is responsible for storing them (filesystem,
  /// database, blob storage, etc.). [close] does **not** save — call
  /// [checkpoint] explicitly before closing to persist changes.
  ///
  /// To restore, pass the bytes to [HnswIndex.load].
  Uint8List checkpoint() {
    final handle = _requireOpenHandle();
    final sizeOut = calloc<ffi.Int32>();

    try {
      // Phase 1: get the size.
      final phase1 = _nativeBindings.hnswCheckpoint(
        handle,
        ffi.Pointer<ffi.Uint8>.fromAddress(0),
        0,
        sizeOut,
      );
      if (phase1 != 0) {
        throw _errorForCode(phase1, 'checkpoint (size query)');
      }

      var size = sizeOut.value;
      if (size <= 0) {
        throw HnswCheckpointException('checkpoint: invalid size $size');
      }

      // The index may grow after the size query because another isolate can
      // mutate the shared context. Retry with the required size reported by
      // the fill call until a complete snapshot fits.
      for (var attempt = 0; attempt < 16; attempt++) {
        final buffer = calloc<ffi.Uint8>(size);
        try {
          sizeOut.value = 0;
          final phase2 = _nativeBindings.hnswCheckpoint(
            handle,
            buffer,
            size,
            sizeOut,
          );
          final written = sizeOut.value;
          if (phase2 == 0) {
            if (written <= 0 || written > size) {
              throw HnswCheckpointException(
                'checkpoint: native returned invalid size $written',
              );
            }
            return Uint8List.fromList(buffer.asTypedList(written));
          }

          if (phase2 == HnswErrorCode.invalidArg.code && written > size) {
            size = written;
            continue;
          }
          throw _errorForCode(phase2, 'checkpoint');
        } finally {
          calloc.free(buffer);
        }
      }

      throw HnswCheckpointException(
        'checkpoint: index changed continuously while serializing',
      );
    } finally {
      calloc.free(sizeOut);
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
