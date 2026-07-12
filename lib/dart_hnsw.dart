/// Cross-platform HNSWlib native bindings for Dart.
///
/// Provides high-performance approximate nearest neighbor search through
/// native [hnswlib](https://github.com/nmslib/hnswlib) with:
/// - Shared in-process indexes — one native context per canonical path
/// - Reader-writer locking for safe multi-isolate concurrency
/// - Cross-process file locking for exclusive mutation
/// - Explicit `checkpoint()` persistence
///
/// Supports `l2`, `innerProduct`, and `cosine` distance metrics.
// ignore_for_file: dangling_library_doc_comments

export 'src/hnsw_index.dart'
    show
        HnswIndex,
        HnswMetric,
        HnswResult,
        HnswVectorEntry,
        HnswErrorCode,
        HnswException,
        HnswOpenException,
        HnswDimensionMismatchException,
        HnswCapacityExceededException,
        HnswConcurrentProcessException,
        HnswLoadException,
        HnswCheckpointException,
        HnswZeroVectorException,
        HnswNativeException;
