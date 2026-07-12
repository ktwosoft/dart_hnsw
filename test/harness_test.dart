import 'dart:ffi';

import 'package:ffi/ffi.dart';
import 'package:test/test.dart';

import 'package:dart_hnsw/src/native_bindings.dart';
import '_fixtures.dart';

// ---------------------------------------------------------------------------
// Legacy harness regression tests — adapted to use HnswNativeBindings
// ---------------------------------------------------------------------------

void main() {
  late HnswNativeBindings native;

  setUp(() {
    native = HnswNativeBindings();
  });

  tearDown(() {
    expectCleanRegistry(native);
  });

  test('same path returns distinct handles to the same shared context', () {
    final path = testIndexPath('harness_same_path.index');
    cleanTestIndex(path);

    final errorCode = calloc<Int32>();
    final pathPtr = path.toNativeUtf8();

    try {
      final handle1 = native.hnswOpen(
        pathPtr,
        4,
        100,
        16,
        200,
        64,
        0,
        1,
        0,
        errorCode,
      );
      expect(errorCode.value, equals(0));
      expect(isNullPointer(handle1), isFalse);

      final handle2 = native.hnswOpen(
        pathPtr,
        4,
        100,
        16,
        200,
        64,
        0,
        1,
        0,
        errorCode,
      );
      expect(errorCode.value, equals(0));
      expect(isNullPointer(handle2), isFalse);

      expect(handle2.address, isNot(equals(handle1.address)));
      expect(
        native.hnswContextId(handle2),
        equals(native.hnswContextId(handle1)),
      );
      expect(native.hnswRegistrySize(), equals(1));

      expect(native.hnswClose(handle1), equals(0));
      expect(native.hnswRegistrySize(), equals(1));

      expect(native.hnswClose(handle2), equals(0));
      expect(native.hnswRegistrySize(), equals(0));
    } finally {
      calloc.free(errorCode);
      calloc.free(pathPtr);
      cleanTestIndex(path);
    }
  });

  test('different paths create different shared contexts', () {
    final pathA = testIndexPath('harness_different_a.index');
    final pathB = testIndexPath('harness_different_b.index');
    cleanTestIndex(pathA);
    cleanTestIndex(pathB);

    final errorCode = calloc<Int32>();
    final pathPtrA = pathA.toNativeUtf8();
    final pathPtrB = pathB.toNativeUtf8();

    try {
      final handleA = native.hnswOpen(
        pathPtrA,
        4,
        100,
        16,
        200,
        64,
        0,
        1,
        0,
        errorCode,
      );
      expect(errorCode.value, equals(0));
      expect(isNullPointer(handleA), isFalse);

      final handleB = native.hnswOpen(
        pathPtrB,
        4,
        100,
        16,
        200,
        64,
        0,
        1,
        0,
        errorCode,
      );
      expect(errorCode.value, equals(0));
      expect(isNullPointer(handleB), isFalse);

      expect(handleA.address, isNot(equals(handleB.address)));
      expect(
        native.hnswContextId(handleA),
        isNot(equals(native.hnswContextId(handleB))),
      );
      expect(native.hnswRegistrySize(), equals(2));

      expect(native.hnswClose(handleA), equals(0));
      expect(native.hnswClose(handleB), equals(0));
      expect(native.hnswRegistrySize(), equals(0));
    } finally {
      calloc.free(errorCode);
      calloc.free(pathPtrA);
      calloc.free(pathPtrB);
      cleanTestIndex(pathA);
      cleanTestIndex(pathB);
    }
  });

  test('upserts through one handle are visible through another handle', () {
    final path = testIndexPath('harness_cross_handle_visibility.index');
    cleanTestIndex(path);

    final errorCode = calloc<Int32>();
    final pathPtr = path.toNativeUtf8();

    try {
      final writerHandle = native.hnswOpen(
        pathPtr,
        4,
        100,
        16,
        200,
        64,
        0,
        1,
        0,
        errorCode,
      );
      expect(errorCode.value, equals(0));

      final readerHandle = native.hnswOpen(
        pathPtr,
        4,
        100,
        16,
        200,
        64,
        0,
        1,
        0,
        errorCode,
      );
      expect(errorCode.value, equals(0));

      // Upsert via writer
      final vector = calloc<Float>(4);
      vector[0] = 1.0;
      expect(native.hnswUpsertVector(writerHandle, 1, vector), equals(0));
      calloc.free(vector);

      // Search via reader
      final query = calloc<Float>(4);
      query[0] = 1.0;
      final outLabels = calloc<Uint64>(1);
      final outDistances = calloc<Float>(1);
      final outCount = calloc<Int32>();

      expect(
        native.hnswSearchKnn(
          readerHandle,
          query,
          1,
          outLabels,
          outDistances,
          outCount,
        ),
        equals(0),
      );
      expect(outCount.value, equals(1));
      expect(outLabels[0], equals(1));

      calloc.free(query);
      calloc.free(outLabels);
      calloc.free(outDistances);
      calloc.free(outCount);

      expect(native.hnswClose(writerHandle), equals(0));
      expect(native.hnswClose(readerHandle), equals(0));
      expect(native.hnswRegistrySize(), equals(0));
    } finally {
      calloc.free(errorCode);
      calloc.free(pathPtr);
      cleanTestIndex(path);
    }
  });

  test('many isolates opening the same path share one context', () async {
    const isolateCount = 20;
    final path = testIndexPath('harness_many_isolates_same_context.index');
    cleanTestIndex(path);

    final errorCode = calloc<Int32>();
    final pathPtr = path.toNativeUtf8();

    try {
      final mainHandle = native.hnswOpen(
        pathPtr,
        4,
        10000,
        16,
        200,
        64,
        0,
        1,
        0,
        errorCode,
      );
      expect(errorCode.value, equals(0));
      final mainId = native.hnswContextId(mainHandle);

      final results = await spawnWorkers(
        path: path,
        count: isolateCount,
        iterations: 0,
        sleepMs: 0,
        workerFn: upsertWorker,
      );

      expectWorkersSucceeded(results);

      for (final result in results) {
        expect(result.contextId, equals(mainId));
      }

      expect(native.hnswClose(mainHandle), equals(0));
      expect(native.hnswRegistrySize(), equals(0));
    } finally {
      calloc.free(errorCode);
      calloc.free(pathPtr);
      cleanTestIndex(path);
    }
  });

  test('reopen after full close creates a new context id', () {
    final path = testIndexPath('harness_reopen_after_close.index');
    cleanTestIndex(path);

    final errorCode = calloc<Int32>();
    final pathPtr = path.toNativeUtf8();

    try {
      final handle1 = native.hnswOpen(
        pathPtr,
        4,
        100,
        16,
        200,
        64,
        0,
        1,
        0,
        errorCode,
      );
      expect(errorCode.value, equals(0));
      final id1 = native.hnswContextId(handle1);

      expect(native.hnswClose(handle1), equals(0));
      expect(native.hnswRegistrySize(), equals(0));

      final handle2 = native.hnswOpen(
        pathPtr,
        4,
        100,
        16,
        200,
        64,
        0,
        1,
        0,
        errorCode,
      );
      expect(errorCode.value, equals(0));
      final id2 = native.hnswContextId(handle2);

      expect(id2, isNot(equals(id1)));

      expect(native.hnswClose(handle2), equals(0));
      expect(native.hnswRegistrySize(), equals(0));
    } finally {
      calloc.free(errorCode);
      calloc.free(pathPtr);
      cleanTestIndex(path);
    }
  });
}
