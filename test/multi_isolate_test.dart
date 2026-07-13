import 'dart:ffi';
import 'dart:typed_data';

import 'package:ffi/ffi.dart';
import 'package:test/test.dart';

import 'package:dart_hnsw/src/hnsw_index.dart';
import 'package:dart_hnsw/src/native_bindings.dart';
import '_fixtures.dart';

// ---------------------------------------------------------------------------
// T9–T12, T19, T26, T31: Shared-context / multi-isolate tests
// ---------------------------------------------------------------------------

void main() {
  late HnswNativeBindings native;

  setUp(() {
    native = HnswNativeBindings();
  });

  tearDown(() {
    expectCleanRegistry(native);
  });

  // -----------------------------------------------------------------------
  // T9: open same collection A and B → same contextId; upsert via A, search via B
  // -----------------------------------------------------------------------
  test(
    'T9: open same collection A and B → contextId(A) == contextId(B); upsert via A, search via B sees it',
    () {
      final name = testCollectionName('t9_shared_context');
      final errorCode = calloc<Int32>();
      final namePtr = name.toNativeUtf8();

      try {
        final handleA = native.hnswCreate(
          namePtr,
          4,
          1000,
          16,
          200,
          64,
          0,
          1,
          errorCode,
        );
        expect(errorCode.value, equals(0));
        expect(isNullPointer(handleA), isFalse);

        final handleB = native.hnswCreate(
          namePtr,
          4,
          1000,
          16,
          200,
          64,
          0,
          1,
          errorCode,
        );
        expect(errorCode.value, equals(0));
        expect(isNullPointer(handleB), isFalse);

        // Same context id
        expect(
          native.hnswContextId(handleB),
          equals(native.hnswContextId(handleA)),
        );
        expect(native.hnswRegistrySize(), equals(1));

        // Upsert via A
        final vector = calloc<Float>(4);
        vector[0] = 42.0;
        vector[1] = 0.0;
        vector[2] = 0.0;
        vector[3] = 0.0;
        expect(native.hnswUpsertVector(handleA, 42, vector), equals(0));
        calloc.free(vector);

        // Search via B sees it
        final query = calloc<Float>(4);
        query[0] = 42.0;
        query[1] = 0.0;
        query[2] = 0.0;
        query[3] = 0.0;
        final outLabels = calloc<Uint64>(1);
        final outDistances = calloc<Float>(1);
        final outCount = calloc<Int32>();

        expect(
          native.hnswSearchKnn(
            handleB,
            query,
            1,
            outLabels,
            outDistances,
            outCount,
          ),
          equals(0),
        );
        expect(outCount.value, equals(1));
        expect(outLabels[0], equals(42));

        calloc.free(query);
        calloc.free(outLabels);
        calloc.free(outDistances);
        calloc.free(outCount);

        expect(native.hnswClose(handleA), equals(0));
        expect(native.hnswClose(handleB), equals(0));
        expect(native.hnswRegistrySize(), equals(0));
      } finally {
        calloc.free(errorCode);
        calloc.free(namePtr);
      }
    },
  );

  // -----------------------------------------------------------------------
  // T10: 10 isolates upsert 100 unique labels each → elementCount == 1000
  // -----------------------------------------------------------------------
  test(
    'T10: 10 isolates upsert 100 unique labels each → elementCount == 1000',
    () async {
      const isolateCount = 10;
      const iterationsPerIsolate = 100;

      final name = testCollectionName('t10_many_isolates');
      final errorCode = calloc<Int32>();
      final namePtr = name.toNativeUtf8();

      try {
        final mainHandle = native.hnswCreate(
          namePtr,
          4,
          10000,
          16,
          200,
          64,
          0,
          1,
          errorCode,
        );
        expect(errorCode.value, equals(0));
        final mainId = native.hnswContextId(mainHandle);

        final results = await spawnWorkers(
          collectionName: name,
          count: isolateCount,
          iterations: iterationsPerIsolate,
          sleepMs: 0,
          workerFn: upsertWorker,
        );

        expectWorkersSucceeded(results);

        for (final result in results) {
          expect(result.contextId, equals(mainId));
        }

        expect(
          readNativeStats(native, mainHandle).elementCount,
          equals(isolateCount * iterationsPerIsolate),
        );
        expect(
          native.hnswGeneration(mainHandle),
          equals(isolateCount * iterationsPerIsolate),
        );

        expect(native.hnswClose(mainHandle), equals(0));
        expect(native.hnswRegistrySize(), equals(0));
      } finally {
        calloc.free(errorCode);
        calloc.free(namePtr);
      }
    },
  );

  // -----------------------------------------------------------------------
  // T11: 20 isolates search concurrently — each verifies expected nearest label
  // -----------------------------------------------------------------------
  test(
    'T11: 20 isolates search concurrently — each verifies expected nearest label',
    () async {
      const isolateCount = 20;
      const iterationsPerIsolate = 10;

      final name = testCollectionName('t11_concurrent_search');
      final errorCode = calloc<Int32>();
      final namePtr = name.toNativeUtf8();

      try {
        final mainHandle = native.hnswCreate(
          namePtr,
          4,
          1000,
          16,
          200,
          64,
          0,
          1,
          errorCode,
        );
        expect(errorCode.value, equals(0));

        // Insert a known vector
        final vector = calloc<Float>(4);
        vector[0] = 1.0;
        vector[1] = 0.0;
        vector[2] = 0.0;
        vector[3] = 0.0;
        expect(native.hnswUpsertVector(mainHandle, 1, vector), equals(0));
        calloc.free(vector);

        final results = await spawnWorkers(
          collectionName: name,
          count: isolateCount,
          iterations: iterationsPerIsolate,
          sleepMs: 0,
          workerFn: searchWorker,
        );

        expectWorkersSucceeded(results);

        expect(native.hnswClose(mainHandle), equals(0));
        expect(native.hnswRegistrySize(), equals(0));
      } finally {
        calloc.free(errorCode);
        calloc.free(namePtr);
      }
    },
  );

  // -----------------------------------------------------------------------
  // T12: concurrent upsert + search stress — no crash
  // -----------------------------------------------------------------------
  test('T12: concurrent upsert + search stress — no crash', () async {
    const isolateCount = 20;
    const iterationsPerIsolate = 50;

    final name = testCollectionName('t12_mixed_stress');
    final errorCode = calloc<Int32>();
    final namePtr = name.toNativeUtf8();

    try {
      final mainHandle = native.hnswCreate(
        namePtr,
        4,
        10000,
        16,
        200,
        64,
        0,
        1,
        errorCode,
      );
      expect(errorCode.value, equals(0));
      final mainId = native.hnswContextId(mainHandle);

      final results = await spawnWorkers(
        collectionName: name,
        count: isolateCount,
        iterations: iterationsPerIsolate,
        sleepMs: 0,
        workerFn: mixedUpsertSearchWorker,
      );

      expectWorkersSucceeded(results);

      for (final result in results) {
        expect(result.contextId, equals(mainId));
      }

      final expectedWrites = results.fold<int>(
        0,
        (total, result) => total + result.writeCount,
      );
      expect(
        readNativeStats(native, mainHandle).elementCount,
        equals(expectedWrites),
      );
      expect(native.hnswGeneration(mainHandle), equals(expectedWrites));

      expect(native.hnswClose(mainHandle), equals(0));
      expect(native.hnswRegistrySize(), equals(0));
    } finally {
      calloc.free(errorCode);
      calloc.free(namePtr);
    }
  });

  // -----------------------------------------------------------------------
  // T19: 20 isolates open + upsert/search + close → registry_size == 0
  // -----------------------------------------------------------------------
  test(
    'T19: 20 isolates open + upsert/search + close → registry_size == 0',
    () async {
      const isolateCount = 20;
      const iterationsPerIsolate = 10;

      final name = testCollectionName('t19_lifecycle_cleanup');
      final errorCode = calloc<Int32>();
      final namePtr = name.toNativeUtf8();

      try {
        final mainHandle = native.hnswCreate(
          namePtr,
          4,
          10000,
          16,
          200,
          64,
          0,
          1,
          errorCode,
        );
        expect(errorCode.value, equals(0));

        final results = await spawnWorkers(
          collectionName: name,
          count: isolateCount,
          iterations: iterationsPerIsolate,
          sleepMs: 0,
          workerFn: upsertWorker,
        );

        expectWorkersSucceeded(results);

        expect(native.hnswClose(mainHandle), equals(0));
        expect(native.hnswRegistrySize(), equals(0));
      } finally {
        calloc.free(errorCode);
        calloc.free(namePtr);
      }
    },
  );

  // -----------------------------------------------------------------------
  // T26: close() on one handle does not affect another live handle
  // -----------------------------------------------------------------------
  test('T26: close() on one handle does not affect another live handle', () {
    final name = testCollectionName('t26_close_one_handle');
    final errorCode = calloc<Int32>();
    final namePtr = name.toNativeUtf8();

    try {
      final handleA = native.hnswCreate(
        namePtr,
        4,
        1000,
        16,
        200,
        64,
        0,
        1,
        errorCode,
      );
      expect(errorCode.value, equals(0));

      final handleB = native.hnswCreate(
        namePtr,
        4,
        1000,
        16,
        200,
        64,
        0,
        1,
        errorCode,
      );
      expect(errorCode.value, equals(0));

      // Close A
      expect(native.hnswClose(handleA), equals(0));
      expect(native.hnswRegistrySize(), equals(1)); // B still alive

      // B still works
      final vector = calloc<Float>(4);
      vector[0] = 7.0;
      expect(native.hnswUpsertVector(handleB, 7, vector), equals(0));
      calloc.free(vector);

      expect(readNativeStats(native, handleB).elementCount, equals(1));

      expect(native.hnswClose(handleB), equals(0));
      expect(native.hnswRegistrySize(), equals(0));
    } finally {
      calloc.free(errorCode);
      calloc.free(namePtr);
    }
  });

  test('T31: concurrent first opens share one context', () async {
    const isolateCount = 100;
    final name = testCollectionName('t31_concurrent_first_open');

    final results = await spawnWorkers(
      collectionName: name,
      count: isolateCount,
      iterations: 0,
      sleepMs: 0,
      workerFn: createOnlyWorker,
    );

    expectWorkersSucceeded(results);
    expect(results.map((result) => result.contextId).toSet(), hasLength(1));
  });

  test(
    'T38: read operations remain valid while another isolate reloads',
    () async {
      Uint8List makeCheckpoint(int label) {
        final source = HnswIndex.open(
          collectionName: testCollectionName('t38_source_$label'),
          dimensions: 4,
          metric: HnswMetric.l2,
          maxElements: 100,
        );
        source.upsert(label, [label.toDouble(), 0, 0, 0]);
        final bytes = source.checkpoint(expectedGeneration: source.generation);
        source.close();
        return bytes;
      }

      final firstCheckpoint = makeCheckpoint(1);
      final secondCheckpoint = makeCheckpoint(2);
      final name = testCollectionName('t38_replace_while_searching');
      final index = HnswIndex.open(
        collectionName: name,
        dimensions: 4,
        metric: HnswMetric.l2,
        maxElements: 100,
      );
      final contextId = index.contextId;

      final workerFuture = spawnWorkers(
        collectionName: name,
        count: 1,
        iterations: 5000,
        sleepMs: 0,
        workerFn: searchWorker,
      );

      for (var i = 0; i < 50; i++) {
        final bytes = i.isEven ? firstCheckpoint : secondCheckpoint;
        index.load(bytes, expectedGeneration: index.generation);
        expect(index.contextId, equals(contextId));
      }

      final results = await workerFuture;
      expectWorkersSucceeded(results);
      index.close();
    },
  );

  test(
    'T44: administrative calls complete during mixed query/upsert load',
    () async {
      const isolateCount = 10;
      const iterationsPerIsolate = 300;
      const adminOperations = 100;
      final name = testCollectionName('t39_admin_priority');
      final errorCode = calloc<Int32>();
      final namePtr = name.toNativeUtf8();

      try {
        final mainHandle = native.hnswCreate(
          namePtr,
          4,
          10000,
          16,
          200,
          64,
          0,
          1,
          errorCode,
        );
        expect(errorCode.value, equals(0));

        final seed = calloc<Float>(4);
        seed[0] = 1.0;
        expect(native.hnswUpsertVector(mainHandle, 1, seed), equals(0));
        calloc.free(seed);

        final workerFuture = spawnWorkers(
          collectionName: name,
          count: isolateCount,
          iterations: iterationsPerIsolate,
          sleepMs: 0,
          workerFn: mixedUpsertSearchWorker,
        );

        for (var i = 0; i < adminOperations; i++) {
          final ef = 32 + (i % 2);
          expect(native.hnswSetEfSearch(mainHandle, ef), equals(0));
          expect(readNativeStats(native, mainHandle).efSearch, equals(ef));
        }

        final results = await workerFuture.timeout(const Duration(seconds: 15));
        expectWorkersSucceeded(results);
        final expectedWrites = results.fold<int>(
          0,
          (total, result) => total + result.writeCount,
        );
        expect(
          readNativeStats(native, mainHandle).elementCount,
          equals(1 + expectedWrites),
        );
        expect(
          native.hnswGeneration(mainHandle),
          equals(1 + expectedWrites + adminOperations),
        );

        expect(native.hnswClose(mainHandle), equals(0));
      } finally {
        calloc.free(errorCode);
        calloc.free(namePtr);
      }
    },
  );
}
