import 'dart:ffi';
import 'dart:io';

import 'package:ffi/ffi.dart';
import 'package:test/test.dart';

import 'package:dart_hnsw/src/hnsw_index.dart';
import 'package:dart_hnsw/src/native_bindings.dart';
import '_fixtures.dart';

// ---------------------------------------------------------------------------
// T1–T8: Single-isolate correctness
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
  // T1: create → upsert → search → close
  // -----------------------------------------------------------------------
  test(
    'T1: create → upsert(1,[1,0,0,0]) → search([1,0,0,0],k=1) → expect id 1 → close',
    () {
      final name = testCollectionName('t1_basic');
      final errorCode = calloc<Int32>();
      final namePtr = name.toNativeUtf8();

      try {
        final handle = native.hnswCreate(
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
        expect(isNullPointer(handle), isFalse);

        // Upsert
        final vector = calloc<Float>(4);
        vector[0] = 1.0;
        vector[1] = 0.0;
        vector[2] = 0.0;
        vector[3] = 0.0;
        expect(native.hnswUpsertVector(handle, 1, vector), equals(0));
        calloc.free(vector);

        // Search
        final query = calloc<Float>(4);
        query[0] = 1.0;
        query[1] = 0.0;
        query[2] = 0.0;
        query[3] = 0.0;
        final outLabels = calloc<Uint64>(1);
        final outDistances = calloc<Float>(1);
        final outCount = calloc<Int32>();

        expect(
          native.hnswSearchKnn(
            handle,
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

        expect(native.hnswClose(handle), equals(0));
      } finally {
        calloc.free(errorCode);
        calloc.free(namePtr);
      }
    },
  );

  // -----------------------------------------------------------------------
  // T2: dimension mismatch → ArgumentError (Dart guard)
  // -----------------------------------------------------------------------
  test(
    'T2: create with dim=4 → upsert vector of length 5 → expect ArgumentError',
    () {
      final name = testCollectionName('t2_dim_mismatch');
      final index = HnswIndex.create(
        collectionName: name,
        dimensions: 4,
        maxElements: 100,
      );

      expect(
        () => index.upsert(1, [1, 2, 3, 4, 5]),
        throwsA(isA<ArgumentError>()),
      );

      index.close();
    },
  );

  test('T2b: close() is idempotent and methods throw after close', () {
    final name = testCollectionName('t2b_closed_handle');
    final index = HnswIndex.create(
      collectionName: name,
      dimensions: 4,
      maxElements: 100,
    );

    index.upsert(1, [1, 0, 0, 0]);
    index.close();
    index.close();

    expect(() => index.search([1, 0, 0, 0]), throwsA(isA<StateError>()));
    expect(() => index.upsert(2, [0, 1, 0, 0]), throwsA(isA<StateError>()));
    expect(() => index.checkpoint(), throwsA(isA<StateError>()));
    expect(() => index.elementCount, throwsA(isA<StateError>()));
  });

  test('T2d: closed methods reject before validating their arguments', () {
    final index = HnswIndex.create(
      collectionName: testCollectionName('t2d_closed_argument_order'),
      dimensions: 4,
      maxElements: 100,
    );
    index.close();

    expect(() => index.upsert(1, [1, 2, 3]), throwsA(isA<StateError>()));
    expect(() => index.upsertBatch(const []), throwsA(isA<StateError>()));
    expect(() => index.search([1, 2, 3]), throwsA(isA<StateError>()));
  });

  test('T2c: sidecar dynamic library bindings work when forced', () {
    final sidecarPath = builtSidecarLibraryPath();
    expect(
      File(sidecarPath).existsSync(),
      isTrue,
      reason: 'Expected built sidecar library at $sidecarPath',
    );

    final dynamicNative = HnswNativeBindings.withOptions(
      libraryPath: sidecarPath,
      preferDynamicLibrary: true,
    );
    final name = testCollectionName('t2c_dynamic');
    final errorCode = calloc<Int32>();
    final namePtr = name.toNativeUtf8();

    try {
      final handle = dynamicNative.hnswCreate(
        namePtr,
        4,
        100,
        16,
        200,
        64,
        0,
        1,
        errorCode,
      );
      expect(errorCode.value, equals(0));
      expect(isNullPointer(handle), isFalse);

      final vector = calloc<Float>(4);
      vector[0] = 1.0;
      expect(dynamicNative.hnswUpsertVector(handle, 1, vector), equals(0));
      calloc.free(vector);

      expect(dynamicNative.hnswElementCount(handle), equals(1));
      expect(dynamicNative.hnswClose(handle), equals(0));
      expect(dynamicNative.hnswRegistrySize(), equals(0));
    } finally {
      calloc.free(errorCode);
      calloc.free(namePtr);
    }
  });

  // -----------------------------------------------------------------------
  // T3: upsert duplicate label → search reflects updated vector
  // -----------------------------------------------------------------------
  test('T3: upsert duplicate label → search reflects updated vector', () {
    final name = testCollectionName('t3_upsert_duplicate');
    final errorCode = calloc<Int32>();
    final namePtr = name.toNativeUtf8();

    try {
      final handle = native.hnswCreate(
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

      // Insert label 1 with [1,0,0,0]
      var vector = calloc<Float>(4);
      vector[0] = 1.0;
      vector[1] = 0.0;
      vector[2] = 0.0;
      vector[3] = 0.0;
      expect(native.hnswUpsertVector(handle, 1, vector), equals(0));
      calloc.free(vector);

      // Upsert same label with [0,1,0,0]
      vector = calloc<Float>(4);
      vector[0] = 0.0;
      vector[1] = 1.0;
      vector[2] = 0.0;
      vector[3] = 0.0;
      expect(native.hnswUpsertVector(handle, 1, vector), equals(0));
      calloc.free(vector);

      // Search for [0,1,0,0] → nearest should be label 1
      final query = calloc<Float>(4);
      query[0] = 0.0;
      query[1] = 1.0;
      query[2] = 0.0;
      query[3] = 0.0;
      final outLabels = calloc<Uint64>(1);
      final outDistances = calloc<Float>(1);
      final outCount = calloc<Int32>();

      expect(
        native.hnswSearchKnn(
          handle,
          query,
          1,
          outLabels,
          outDistances,
          outCount,
        ),
        equals(0),
      );
      expect(outCount.value, equals(1));
      expect(outLabels[0], equals(1), reason: 'upsert should update features');

      calloc.free(query);
      calloc.free(outLabels);
      calloc.free(outDistances);
      calloc.free(outCount);

      expect(native.hnswClose(handle), equals(0));
    } finally {
      calloc.free(errorCode);
      calloc.free(namePtr);
    }
  });

  // -----------------------------------------------------------------------
  // T4: markDeleted(id) → search no longer returns id
  // -----------------------------------------------------------------------
  test('T4: markDeleted(id) → search no longer returns id', () {
    final name = testCollectionName('t4_mark_deleted');
    final errorCode = calloc<Int32>();
    final namePtr = name.toNativeUtf8();

    try {
      final handle = native.hnswCreate(
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

      // Insert two vectors
      var vector = calloc<Float>(4);
      vector[0] = 1.0;
      vector[1] = 0.0;
      vector[2] = 0.0;
      vector[3] = 0.0;
      expect(native.hnswUpsertVector(handle, 1, vector), equals(0));
      calloc.free(vector);

      vector = calloc<Float>(4);
      vector[0] = 0.0;
      vector[1] = 1.0;
      vector[2] = 0.0;
      vector[3] = 0.0;
      expect(native.hnswUpsertVector(handle, 2, vector), equals(0));
      calloc.free(vector);

      // Mark label 1 as deleted
      expect(native.hnswMarkDeleted(handle, 1), equals(0));

      // Search for [1,0,0,0] — should return label 2 (not 1)
      final query = calloc<Float>(4);
      query[0] = 1.0;
      query[1] = 0.0;
      query[2] = 0.0;
      query[3] = 0.0;
      final outLabels = calloc<Uint64>(1);
      final outDistances = calloc<Float>(1);
      final outCount = calloc<Int32>();

      expect(
        native.hnswSearchKnn(
          handle,
          query,
          1,
          outLabels,
          outDistances,
          outCount,
        ),
        equals(0),
      );
      expect(outCount.value, equals(1));
      expect(
        outLabels[0],
        equals(2),
        reason: 'deleted label 1 should not appear',
      );

      calloc.free(query);
      calloc.free(outLabels);
      calloc.free(outDistances);
      calloc.free(outCount);

      expect(native.hnswClose(handle), equals(0));
    } finally {
      calloc.free(errorCode);
      calloc.free(namePtr);
    }
  });

  // -----------------------------------------------------------------------
  // T6: resize → maxElements reflects new capacity → upsert beyond old cap
  // -----------------------------------------------------------------------
  test(
    'T6: resize(newMax) → maxElements reflects new capacity → upsert beyond old cap works',
    () {
      final name = testCollectionName('t6_resize');
      final errorCode = calloc<Int32>();
      final namePtr = name.toNativeUtf8();

      try {
        final handle = native.hnswCreate(
          namePtr,
          4,
          10,
          16,
          200,
          64,
          0,
          1,
          errorCode,
        );
        expect(errorCode.value, equals(0));

        expect(native.hnswMaxElements(handle), equals(10));

        // Fill to capacity
        for (var i = 0; i < 10; i++) {
          final vector = calloc<Float>(4);
          vector[0] = i.toDouble();
          expect(native.hnswUpsertVector(handle, i, vector), equals(0));
          calloc.free(vector);
        }

        expect(native.hnswElementCount(handle), equals(10));

        // Resize
        expect(native.hnswResize(handle, 20), equals(0));
        expect(native.hnswMaxElements(handle), equals(20));

        // Add beyond old capacity
        final vector = calloc<Float>(4);
        vector[0] = 99.0;
        expect(native.hnswUpsertVector(handle, 99, vector), equals(0));
        calloc.free(vector);

        expect(native.hnswElementCount(handle), equals(11));

        expect(native.hnswClose(handle), equals(0));
      } finally {
        calloc.free(errorCode);
        calloc.free(namePtr);
      }
    },
  );

  // -----------------------------------------------------------------------
  // T7: setEfSearch(32) → efSearch getter returns 32
  // -----------------------------------------------------------------------
  test('T7: setEfSearch(32) → efSearch getter returns 32', () {
    final name = testCollectionName('t7_ef_search');
    final errorCode = calloc<Int32>();
    final namePtr = name.toNativeUtf8();

    try {
      final handle = native.hnswCreate(
        namePtr,
        4,
        100,
        16,
        200,
        64,
        0,
        1,
        errorCode,
      );
      expect(errorCode.value, equals(0));

      expect(native.hnswGetEfSearch(handle), equals(64));

      expect(native.hnswSetEfSearch(handle, 32), equals(0));
      expect(native.hnswGetEfSearch(handle), equals(32));

      expect(native.hnswClose(handle), equals(0));
    } finally {
      calloc.free(errorCode);
      calloc.free(namePtr);
    }
  });

  // -----------------------------------------------------------------------
  // T8: elementCount and maxElements return correct values
  // -----------------------------------------------------------------------
  test('T8: elementCount and maxElements return correct values', () {
    final name = testCollectionName('t8_counts');
    final errorCode = calloc<Int32>();
    final namePtr = name.toNativeUtf8();

    try {
      final handle = native.hnswCreate(
        namePtr,
        4,
        500,
        16,
        200,
        64,
        0,
        1,
        errorCode,
      );
      expect(errorCode.value, equals(0));

      expect(native.hnswElementCount(handle), equals(0));
      expect(native.hnswMaxElements(handle), equals(500));

      for (var i = 0; i < 10; i++) {
        final vector = calloc<Float>(4);
        vector[0] = i.toDouble();
        expect(native.hnswUpsertVector(handle, i, vector), equals(0));
        calloc.free(vector);
      }

      expect(native.hnswElementCount(handle), equals(10));
      expect(native.hnswMaxElements(handle), equals(500));

      expect(native.hnswClose(handle), equals(0));
    } finally {
      calloc.free(errorCode);
      calloc.free(namePtr);
    }
  });
}
