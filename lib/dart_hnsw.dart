/// Native and browser HNSWlib bindings for Dart.
///
/// Provides high-performance approximate nearest neighbor search through
/// native [hnswlib](https://github.com/nmslib/hnswlib) with:
/// - Pure in-memory indexes — no filesystem dependency
/// - Shared in-process contexts — one native context per collection name
/// - Concurrent query and upsert groups with exclusive administration
/// - Handle-owned exclusive mutation access for stable checkpoint shutdowns
/// - `checkpoint()` returns raw bytes so the caller controls storage
/// - `load()` atomically restores or replaces content from bytes
///
/// Supports `l2`, `innerProduct`, and `cosine` distance metrics.
/// Native assets are configured for macOS, Linux, Windows, iOS, and Android.
/// On web, await HnswIndex.initialize before using the synchronous API.
/// Browser C++ Wasm assets are compiled separately with tool/build_web.sh.
// ignore_for_file: dangling_library_doc_comments

export 'src/hnsw_index.dart'
    show
        HnswIndex,
        HnswMetric,
        HnswResult,
        HnswStats,
        HnswVectorEntry,
        HnswLimits,
        HnswErrorCode,
        HnswException,
        HnswOpenException,
        HnswDimensionMismatchException,
        HnswGenerationConflictException,
        HnswExclusiveMutationAccessException,
        HnswCapacityExceededException,
        HnswLoadException,
        HnswCheckpointException,
        HnswZeroVectorException,
        HnswNonFiniteVectorException,
        HnswNativeException;
