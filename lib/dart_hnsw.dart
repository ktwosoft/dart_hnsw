/// Cross-platform HNSWlib native bindings for Dart.
///
/// Provides high-performance approximate nearest neighbor search through
/// native [hnswlib](https://github.com/nmslib/hnswlib) with:
/// - Pure in-memory indexes — no filesystem dependency
/// - Shared in-process contexts — one native context per collection name
/// - Reader-writer locking for safe multi-isolate concurrency
/// - `checkpoint()` returns raw bytes so the caller controls storage
/// - `load()` restores from bytes — store them anywhere (FS, DB, blob, etc.)
///
/// Supports `l2`, `innerProduct`, and `cosine` distance metrics.
/// Native assets are configured for macOS, Linux, Windows, iOS, and Android.
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
        HnswLoadException,
        HnswCheckpointException,
        HnswZeroVectorException,
        HnswNativeException;
