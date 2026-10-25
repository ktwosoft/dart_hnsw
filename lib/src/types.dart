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
/// Reading `HnswIndex.stats` briefly takes the fully exclusive native
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
