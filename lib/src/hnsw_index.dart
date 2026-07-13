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

  /// A destructive operation used a stale shared-index generation.
  generationConflict(-4),

  /// Failed to load the index from a byte buffer.
  loadFailed(-5),

  /// Failed to save (checkpoint) the index to a byte buffer.
  saveFailed(-6),

  /// Zero vector provided where not allowed (e.g. cosine metric).
  zeroVector(-8),

  /// A vector contains NaN or infinity.
  nonFiniteVector(-9),

  /// Error propagated from the underlying hnswlib library.
  hnswlib(-10),

  /// A different handle owns exclusive mutation access to the collection.
  exclusiveMutationAccess(-11);

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

/// Thrown when another isolate changed the shared index first.
class HnswGenerationConflictException extends HnswException {
  const HnswGenerationConflictException(super.message);
}

/// Thrown when another handle owns exclusive mutation access to a collection.
class HnswExclusiveMutationAccessException extends HnswException {
  const HnswExclusiveMutationAccessException(super.message);
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

/// Thrown when a vector contains NaN or infinity.
class HnswNonFiniteVectorException extends HnswException {
  const HnswNonFiniteVectorException(super.message);
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

/// One consistent snapshot of index metadata.
///
/// Reading [HnswIndex.stats] briefly takes the fully exclusive native
/// administrative lock. Read it infrequently and reuse the returned snapshot
/// instead of requesting metadata fields independently.
final class HnswStats {
  const HnswStats({
    required this.elementCount,
    required this.deletedCount,
    required this.maxElements,
    required this.efSearch,
  });

  /// Total number of occupied elements, including deleted entries.
  final int elementCount;

  /// Number of entries currently marked as deleted.
  final int deletedCount;

  /// Current maximum index capacity.
  final int maxElements;

  /// Current search breadth parameter.
  final int efSearch;

  /// Number of non-deleted entries in this snapshot.
  int get liveCount => elementCount - deletedCount;

  @override
  bool operator ==(Object other) =>
      other is HnswStats &&
      other.elementCount == elementCount &&
      other.deletedCount == deletedCount &&
      other.maxElements == maxElements &&
      other.efSearch == efSearch;

  @override
  int get hashCode =>
      Object.hash(elementCount, deletedCount, maxElements, efSearch);

  @override
  String toString() =>
      'HnswStats(elementCount: $elementCount, '
      'deletedCount: $deletedCount, maxElements: $maxElements, '
      'efSearch: $efSearch)';
}

/// Safety limits enforced by both the Dart and native layers.
abstract final class HnswLimits {
  /// Maximum vector dimensions.
  static const maxDimensions = 65536;

  /// Maximum index capacity.
  static const maxElements = 10000000;

  /// Maximum graph connectivity parameter.
  static const maxM = 10000;

  /// Minimum graph connectivity parameter.
  static const minM = 2;

  /// Maximum search or construction breadth.
  static const maxEf = 1000000;

  /// Maximum results requested from one search.
  static const maxSearchResults = 100000;

  /// Maximum entries accepted by one batch upsert.
  static const maxBatchSize = 100000;

  /// Maximum float values accepted by one batch upsert.
  static const maxBatchFloatValues = 16 * 1024 * 1024;

  /// Maximum checkpoint payload size.
  static const maxCheckpointBytes = 512 * 1024 * 1024;
}

const _maxSignedInt64 = 9223372036854775807;
const _maxLabel = _maxSignedInt64;

ffi.Pointer<ffi.Float> _listToFloatPtr(List<double> values) {
  final ptr = calloc<ffi.Float>(values.length);
  for (var i = 0; i < values.length; i++) {
    ptr[i] = values[i];
  }
  return ptr;
}

void _validateIndexOptions({
  required String collectionName,
  required int dimensions,
  required int maxElements,
  required int m,
  required int efConstruction,
  required int efSearch,
}) {
  if (collectionName.isEmpty) {
    throw ArgumentError.value(
      collectionName,
      'collectionName',
      'must not be empty',
    );
  }
  if (dimensions <= 0 || dimensions > HnswLimits.maxDimensions) {
    throw ArgumentError.value(
      dimensions,
      'dimensions',
      'must be between 1 and ${HnswLimits.maxDimensions}',
    );
  }
  if (maxElements <= 0 || maxElements > HnswLimits.maxElements) {
    throw ArgumentError.value(
      maxElements,
      'maxElements',
      'must be between 1 and ${HnswLimits.maxElements}',
    );
  }
  if (m < HnswLimits.minM || m > HnswLimits.maxM) {
    throw ArgumentError.value(
      m,
      'm',
      'must be between ${HnswLimits.minM} and ${HnswLimits.maxM}',
    );
  }
  if (efConstruction <= 0 || efConstruction > HnswLimits.maxEf) {
    throw ArgumentError.value(
      efConstruction,
      'efConstruction',
      'must be between 1 and ${HnswLimits.maxEf}',
    );
  }
  if (efSearch <= 0 || efSearch > HnswLimits.maxEf) {
    throw ArgumentError.value(
      efSearch,
      'efSearch',
      'must be between 1 and ${HnswLimits.maxEf}',
    );
  }
}

int _generationValue(int expectedGeneration) {
  if (expectedGeneration < 0 || expectedGeneration > _maxSignedInt64) {
    throw ArgumentError.value(
      expectedGeneration,
      'expectedGeneration',
      'must be a valid generation',
    );
  }
  return expectedGeneration;
}

void _validateLabel(int id, String name) {
  if (id < 0 || id > _maxLabel) {
    throw ArgumentError.value(
      id,
      name,
      'must fit in the supported signed 64-bit label range',
    );
  }
}

void _validateFiniteVector(List<double> values, String name) {
  for (final value in values) {
    if (!value.isFinite) {
      throw HnswNonFiniteVectorException(
        '$name must contain only finite values',
      );
    }
  }
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
      return HnswGenerationConflictException(
        '$context: shared index changed since the expected generation',
      );
    case -5:
      return HnswLoadException('$context: failed to load index');
    case -6:
      return HnswCheckpointException('$context: failed to save index');
    case -8:
      return HnswZeroVectorException(
        '$context: zero vector not allowed for cosine metric',
      );
    case -9:
      return HnswNonFiniteVectorException(
        '$context: vector contains NaN or infinity',
      );
    case -11:
      return HnswExclusiveMutationAccessException(
        '$context: another handle owns exclusive mutation access',
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
/// process attaches them to one shared native context. Searches can run
/// concurrently with other searches, and upserts can run concurrently with
/// other upserts. Search and upsert groups do not overlap. Administrative
/// operations such as delete, resize, load, clear, and checkpoint are fully
/// exclusive.
///
/// The index lives entirely in native memory. Use [checkpoint] to serialize
/// the index to a [Uint8List] that you can store anywhere (filesystem,
/// database, blob storage, etc.). Use [load] to restore from bytes.
///
/// {@category Index}
class HnswIndex {
  ffi.Pointer<ffi.Void>? _handle;
  final int _dimensions;

  HnswIndex._(this._handle, this._dimensions);

  /// Opens a shared in-memory HNSW index, creating an empty one if needed.
  ///
  /// If an index with the same [collectionName] is already open in this
  /// process, the existing shared native context is reused. The [dimensions],
  /// [metric], and [allowReplaceDeleted] values must match the existing
  /// context.
  ///
  /// [maxElements] sets the initial capacity. [m] and [efConstruction] control
  /// graph build quality. [efSearch] controls the default search breadth.
  static HnswIndex open({
    required String collectionName,
    required int dimensions,
    HnswMetric metric = HnswMetric.cosine,
    int maxElements = 100000,
    int m = 16,
    int efConstruction = 200,
    int efSearch = 64,
    bool allowReplaceDeleted = true,
  }) {
    _validateIndexOptions(
      collectionName: collectionName,
      dimensions: dimensions,
      maxElements: maxElements,
      m: m,
      efConstruction: efConstruction,
      efSearch: efSearch,
    );
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
          'HnswIndex.open(\'$collectionName\')',
        );
      }

      return HnswIndex._(handle, dimensions);
    } finally {
      calloc.free(errorCode);
      calloc.free(namePtr);
    }
  }

  /// Replaces this shared index with bytes produced by [checkpoint].
  ///
  /// Existing handles for the same collection remain valid and see either the
  /// old complete graph or the new complete graph. [expectedGeneration] is
  /// required to prevent a stale checkpoint from overwriting newer content.
  void load(Uint8List bytes, {required int expectedGeneration}) {
    final handle = _requireOpenHandle();
    if (bytes.isEmpty || bytes.length > HnswLimits.maxCheckpointBytes) {
      throw ArgumentError.value(
        bytes.length,
        'bytes.length',
        'must be between 1 and ${HnswLimits.maxCheckpointBytes}',
      );
    }
    final expected = _generationValue(expectedGeneration);
    final dataPtr = calloc<ffi.Uint8>(bytes.length);
    for (var i = 0; i < bytes.length; i++) {
      dataPtr[i] = bytes[i];
    }

    try {
      final result = _nativeBindings.hnswLoad(
        handle,
        dataPtr,
        bytes.length,
        expected,
      );

      if (result != 0) throw _errorForCode(result, 'HnswIndex.load');
    } finally {
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
  /// open time. If a vector with the same [id] already exists it is replaced.
  /// Upserts from multiple isolates may execute concurrently; searches wait
  /// until the active upsert phase finishes.
  void upsert(int id, List<double> vector) {
    final handle = _requireOpenHandle();
    _validateLabel(id, 'id');
    if (vector.length != _dimensions) {
      throw ArgumentError(
        'Expected $_dimensions dimensions, got ${vector.length}',
      );
    }
    _validateFiniteVector(vector, 'vector');

    final vecPtr = _listToFloatPtr(vector);
    try {
      final result = _nativeBindings.hnswUpsertVector(handle, id, vecPtr);
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
  /// All entries must match the index [dimensions]. One batch is one member of
  /// the concurrent upsert group and may overlap other upsert calls.
  void upsertBatch(List<HnswVectorEntry> entries) {
    final handle = _requireOpenHandle();
    if (entries.isEmpty) return;
    if (entries.length > HnswLimits.maxBatchSize ||
        entries.length > HnswLimits.maxBatchFloatValues ~/ _dimensions) {
      throw ArgumentError.value(
        entries.length,
        'entries.length',
        'exceeds the batch safety limit',
      );
    }

    for (final entry in entries) {
      _validateLabel(entry.id, 'entry.id');
      if (entry.vector.length != _dimensions) {
        throw ArgumentError(
          'Entry ${entry.id}: expected $_dimensions dimensions, got ${entry.vector.length}',
        );
      }
      _validateFiniteVector(entry.vector, 'Entry ${entry.id}.vector');
    }

    final labels = calloc<ffi.Uint64>(entries.length);
    final vectors = calloc<ffi.Float>(entries.length * _dimensions);

    try {
      for (var i = 0; i < entries.length; i++) {
        final entry = entries[i];
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
    if (k <= 0 || k > HnswLimits.maxSearchResults) {
      throw ArgumentError.value(
        k,
        'k',
        'must be between 1 and ${HnswLimits.maxSearchResults}',
      );
    }
    if (query.length != _dimensions) {
      throw ArgumentError(
        'Expected $_dimensions dimensions, got ${query.length}',
      );
    }
    _validateFiniteVector(query, 'query');

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
    final handle = _requireOpenHandle();
    _validateLabel(id, 'id');
    final result = _nativeBindings.hnswMarkDeleted(handle, id);
    if (result != 0) {
      throw _errorForCode(result, 'markDeleted(label=$id)');
    }
  }

  /// Increases the index capacity to [newMaxElements].
  ///
  /// Cannot shrink the index below the current element count.
  void resize(int newMaxElements) {
    final handle = _requireOpenHandle();
    if (newMaxElements <= 0 || newMaxElements > HnswLimits.maxElements) {
      throw ArgumentError.value(
        newMaxElements,
        'newMaxElements',
        'must be between 1 and ${HnswLimits.maxElements}',
      );
    }
    final result = _nativeBindings.hnswResize(handle, newMaxElements);
    if (result != 0) {
      throw _errorForCode(result, 'resize($newMaxElements)');
    }
  }

  /// Sets the search breadth parameter [ef].
  ///
  /// Higher values improve recall at the cost of search speed.
  void setEfSearch(int ef) {
    final handle = _requireOpenHandle();
    if (ef <= 0 || ef > HnswLimits.maxEf) {
      throw ArgumentError.value(
        ef,
        'ef',
        'must be between 1 and ${HnswLimits.maxEf}',
      );
    }
    final result = _nativeBindings.hnswSetEfSearch(handle, ef);
    if (result != 0) {
      throw _errorForCode(result, 'setEfSearch($ef)');
    }
  }

  /// Reads all index metadata under one native administrative lock.
  HnswStats get stats {
    final handle = _requireOpenHandle();
    final nativeStats = calloc<NativeHnswStats>();
    try {
      final result = _nativeBindings.hnswStats(handle, nativeStats);
      if (result != 0) {
        throw _errorForCode(result, 'stats');
      }
      final value = nativeStats.ref;
      return HnswStats(
        elementCount: value.elementCount,
        deletedCount: value.deletedCount,
        maxElements: value.maxElements,
        efSearch: value.efSearch,
      );
    } finally {
      calloc.free(nativeStats);
    }
  }

  /// Opaque identifier for the shared native context.
  ///
  /// Handles attached to the same collection name share the same context id.
  int get contextId => _nativeBindings.hnswContextId(_requireOpenHandle());

  /// Monotonically increasing shared-index state version.
  ///
  /// Mutations, [load], and [clear] advance this value. Use it with
  /// `expectedGeneration` to prevent stale destructive operations.
  int get generation => _nativeBindings.hnswGeneration(_requireOpenHandle());

  /// Acquires exclusive mutation access for this handle's shared collection.
  ///
  /// This waits for the active native operation phase to drain before assigning
  /// ownership. While held, only this handle may load, clear, upsert, delete,
  /// resize, or change the search breadth. Other handles may continue issuing
  /// searches, subject to the normal query/upsert phase boundary. The owner
  /// must call [releaseExclusiveMutationAccess] if normal mutations should
  /// resume.
  void acquireExclusiveMutationAccess() {
    final result = _nativeBindings.hnswAcquireExclusiveMutationAccess(
      _requireOpenHandle(),
    );
    if (result != 0) {
      throw _errorForCode(result, 'acquireExclusiveMutationAccess');
    }
  }

  /// Releases this handle's exclusive mutation access.
  ///
  /// Only the owning handle may release. Closing an owning handle without
  /// release intentionally leaves the remaining shared context mutation-
  /// blocked until every handle has closed.
  void releaseExclusiveMutationAccess() {
    final result = _nativeBindings.hnswReleaseExclusiveMutationAccess(
      _requireOpenHandle(),
    );
    if (result != 0) {
      throw _errorForCode(result, 'releaseExclusiveMutationAccess');
    }
  }

  /// Serializes the current index state to a [Uint8List].
  ///
  /// The returned bytes use the same binary format as hnswlib's native
  /// `saveIndex`. The caller is responsible for storing them (filesystem,
  /// database, blob storage, etc.). [close] does **not** save — call
  /// [checkpoint] explicitly before closing to persist changes.
  ///
  /// [expectedGeneration] is required so a checkpoint cannot silently capture
  /// a newer shared state than the caller intended.
  Uint8List checkpoint({required int expectedGeneration}) {
    final handle = _requireOpenHandle();
    final sizeOut = calloc<ffi.Int32>();
    final expected = _generationValue(expectedGeneration);

    try {
      final phase1 = _nativeBindings.hnswCheckpoint(
        handle,
        ffi.Pointer<ffi.Uint8>.fromAddress(0),
        0,
        sizeOut,
        expected,
      );
      if (phase1 != 0) {
        throw _errorForCode(phase1, 'checkpoint (size query)');
      }

      final size = sizeOut.value;
      if (size <= 0 || size > HnswLimits.maxCheckpointBytes) {
        throw HnswCheckpointException('checkpoint: invalid size $size');
      }

      final buffer = calloc<ffi.Uint8>(size);
      try {
        final phase2 = _nativeBindings.hnswCheckpoint(
          handle,
          buffer,
          size,
          sizeOut,
          expected,
        );
        if (phase2 != 0) throw _errorForCode(phase2, 'checkpoint');

        final written = sizeOut.value;
        if (written <= 0 || written > size) {
          throw HnswCheckpointException(
            'checkpoint: native returned invalid size $written',
          );
        }
        return Uint8List.fromList(buffer.asTypedList(written));
      } finally {
        calloc.free(buffer);
      }
    } finally {
      calloc.free(sizeOut);
    }
  }

  /// Clears all vectors while preserving the shared context configuration.
  ///
  /// Existing handles remain valid and immediately observe the empty index.
  /// Clearing fails if another isolate changed the context first.
  void clear({required int expectedGeneration}) {
    final result = _nativeBindings.hnswClear(
      _requireOpenHandle(),
      _generationValue(expectedGeneration),
    );
    if (result != 0) {
      throw _errorForCode(result, 'clear');
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
