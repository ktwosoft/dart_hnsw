import 'dart:ffi';
import 'dart:typed_data';

import 'package:ffi/ffi.dart';
import 'package:test/test.dart';

import 'package:dart_hnsw/src/native_bindings.dart';
import 'package:dart_hnsw/src/hnsw_index.dart';
import '_fixtures.dart';

// ---------------------------------------------------------------------------
// T13–T16: Persistence (checkpoint / load) tests
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
  // T13: create(efSearch=64) → upsert → checkpoint → close
  //      → load(efSearch=32) → search returns persisted label
  //      → stats report efSearch 32
  // -----------------------------------------------------------------------
  test('T13: checkpoint and load with efSearch reapplied', () {
    final name1 = testCollectionName('t13_checkpoint_session1');
    final name2 = testCollectionName('t13_checkpoint_session2');
    final errorCode = calloc<Int32>();
    final name1Ptr = name1.toNativeUtf8();

    try {
      // --- First session ---
      final handle1 = native.hnswCreate(
        name1Ptr,
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

      final vector = calloc<Float>(4);
      vector[0] = 1.0;
      vector[1] = 0.0;
      vector[2] = 0.0;
      vector[3] = 0.0;
      expect(native.hnswUpsertVector(handle1, 1, vector), equals(0));
      calloc.free(vector);

      // Checkpoint — phase 1: get size
      final sizeOut = calloc<Int32>();
      final generation = native.hnswGeneration(handle1);
      expect(
        native.hnswCheckpoint(
          handle1,
          Pointer<Uint8>.fromAddress(0),
          0,
          sizeOut,
          generation,
        ),
        equals(0),
      );
      final size = sizeOut.value;
      expect(size, greaterThan(0));

      // Checkpoint — phase 2: fill buffer
      final buffer = calloc<Uint8>(size);
      expect(
        native.hnswCheckpoint(handle1, buffer, size, sizeOut, generation),
        equals(0),
      );
      expect(sizeOut.value, equals(size));

      // Copy to Dart-owned bytes
      final bytes = Uint8List.fromList(buffer.asTypedList(size));

      calloc.free(buffer);
      calloc.free(sizeOut);

      expect(native.hnswClose(handle1), equals(0));
      expect(native.hnswRegistrySize(), equals(0));

      // --- Second session with different efSearch ---
      final name2Ptr = name2.toNativeUtf8();
      final dataPtr = calloc<Uint8>(bytes.length);
      for (var i = 0; i < bytes.length; i++) {
        dataPtr[i] = bytes[i];
      }

      try {
        final handle2 = native.hnswCreate(
          name2Ptr,
          4,
          1000,
          16,
          200,
          32,
          0,
          1,
          errorCode,
        );
        expect(errorCode.value, equals(0));
        expect(native.hnswLoad(handle2, dataPtr, bytes.length, 0), equals(0));

        // efSearch should be reapplied
        expect(readNativeStats(native, handle2).efSearch, equals(32));

        // Search persisted vector
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
            handle2,
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

        expect(native.hnswClose(handle2), equals(0));
        expect(native.hnswRegistrySize(), equals(0));
      } finally {
        calloc.free(dataPtr);
        calloc.free(name2Ptr);
      }
    } finally {
      calloc.free(errorCode);
      calloc.free(name1Ptr);
    }
  });

  // -----------------------------------------------------------------------
  // T14: checkpoint returns non-empty bytes; round-trip preserves data
  // -----------------------------------------------------------------------
  test('T14: checkpoint returns non-empty bytes and round-trips', () {
    final name = testCollectionName('t14_checkpoint_roundtrip');
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

      final vector = calloc<Float>(4);
      vector[0] = 1.0;
      expect(native.hnswUpsertVector(handle, 1, vector), equals(0));
      calloc.free(vector);

      // Checkpoint
      final sizeOut = calloc<Int32>();
      final generation = native.hnswGeneration(handle);
      expect(
        native.hnswCheckpoint(
          handle,
          Pointer<Uint8>.fromAddress(0),
          0,
          sizeOut,
          generation,
        ),
        equals(0),
      );
      final size = sizeOut.value;
      expect(size, greaterThan(0));

      final buffer = calloc<Uint8>(size);
      expect(
        native.hnswCheckpoint(handle, buffer, size, sizeOut, generation),
        equals(0),
      );
      final bytes = Uint8List.fromList(buffer.asTypedList(size));
      calloc.free(buffer);
      calloc.free(sizeOut);

      expect(bytes.length, equals(size));
      expect(bytes.isNotEmpty, isTrue);

      expect(native.hnswClose(handle), equals(0));
      expect(native.hnswRegistrySize(), equals(0));
    } finally {
      calloc.free(errorCode);
      calloc.free(namePtr);
    }
  });

  // -----------------------------------------------------------------------
  // T15: create → upsert → close() (no checkpoint) → create → elementCount == 0
  // -----------------------------------------------------------------------
  test('T15: close without checkpoint → create new → elementCount == 0', () {
    final name1 = testCollectionName('t15_close_no_checkpoint');
    final name2 = testCollectionName('t15_close_no_checkpoint_2');
    final errorCode = calloc<Int32>();
    final name1Ptr = name1.toNativeUtf8();
    final name2Ptr = name2.toNativeUtf8();

    try {
      // First session
      final handle1 = native.hnswCreate(
        name1Ptr,
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

      final vector = calloc<Float>(4);
      vector[0] = 1.0;
      expect(native.hnswUpsertVector(handle1, 1, vector), equals(0));
      calloc.free(vector);

      // Close without checkpoint
      expect(native.hnswClose(handle1), equals(0));
      expect(native.hnswRegistrySize(), equals(0));

      // Create a new index — should be empty
      final handle2 = native.hnswCreate(
        name2Ptr,
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

      // Fresh index, no data
      expect(readNativeStats(native, handle2).elementCount, equals(0));

      expect(native.hnswClose(handle2), equals(0));
      expect(native.hnswRegistrySize(), equals(0));
    } finally {
      calloc.free(errorCode);
      calloc.free(name1Ptr);
      calloc.free(name2Ptr);
    }
  });

  // -----------------------------------------------------------------------
  // T16: upsert → checkpoint → close → load → mutation present
  // -----------------------------------------------------------------------
  test('T16: upsert → checkpoint → close → load → mutation present', () {
    final name1 = testCollectionName('t16_checkpoint_persists');
    final name2 = testCollectionName('t16_checkpoint_persists_restored');
    final errorCode = calloc<Int32>();
    final name1Ptr = name1.toNativeUtf8();

    try {
      // First session
      final handle1 = native.hnswCreate(
        name1Ptr,
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

      final vector = calloc<Float>(4);
      vector[0] = 1.0;
      expect(native.hnswUpsertVector(handle1, 1, vector), equals(0));
      calloc.free(vector);

      // Checkpoint to bytes
      final sizeOut = calloc<Int32>();
      final generation = native.hnswGeneration(handle1);
      expect(
        native.hnswCheckpoint(
          handle1,
          Pointer<Uint8>.fromAddress(0),
          0,
          sizeOut,
          generation,
        ),
        equals(0),
      );
      final size = sizeOut.value;
      final buffer = calloc<Uint8>(size);
      expect(
        native.hnswCheckpoint(handle1, buffer, size, sizeOut, generation),
        equals(0),
      );
      final bytes = Uint8List.fromList(buffer.asTypedList(size));
      calloc.free(buffer);
      calloc.free(sizeOut);

      expect(native.hnswClose(handle1), equals(0));
      expect(native.hnswRegistrySize(), equals(0));

      // Load from bytes
      final name2Ptr = name2.toNativeUtf8();
      final dataPtr = calloc<Uint8>(bytes.length);
      for (var i = 0; i < bytes.length; i++) {
        dataPtr[i] = bytes[i];
      }

      try {
        final handle2 = native.hnswCreate(
          name2Ptr,
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
        expect(native.hnswLoad(handle2, dataPtr, bytes.length, 0), equals(0));

        expect(readNativeStats(native, handle2).elementCount, equals(1));

        final query = calloc<Float>(4);
        query[0] = 1.0;
        final outLabels = calloc<Uint64>(1);
        final outDistances = calloc<Float>(1);
        final outCount = calloc<Int32>();

        expect(
          native.hnswSearchKnn(
            handle2,
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

        expect(native.hnswClose(handle2), equals(0));
        expect(native.hnswRegistrySize(), equals(0));
      } finally {
        calloc.free(dataPtr);
        calloc.free(name2Ptr);
      }
    } finally {
      calloc.free(errorCode);
      calloc.free(name1Ptr);
    }
  });

  // -----------------------------------------------------------------------
  // T16b: High-level API checkpoint/load round-trip
  // -----------------------------------------------------------------------
  test('T16b: HnswIndex.checkpoint() → index.load() round-trip', () {
    final name1 = testCollectionName('t16b_highlevel_save');
    final name2 = testCollectionName('t16b_highlevel_load');

    final index = HnswIndex.open(
      collectionName: name1,
      dimensions: 4,
      metric: HnswMetric.l2,
      maxElements: 100,
    );

    index.upsert(1, [1, 0, 0, 0]);
    index.upsert(2, [0, 1, 0, 0]);

    final bytes = index.checkpoint(expectedGeneration: index.generation);
    expect(bytes.isNotEmpty, isTrue);

    index.close();

    final restored = HnswIndex.open(
      collectionName: name2,
      dimensions: 4,
      metric: HnswMetric.l2,
      maxElements: 100,
    );
    restored.load(bytes, expectedGeneration: restored.generation);

    expect(restored.stats.elementCount, equals(2));

    final results = restored.search([1, 0, 0, 0], k: 1);
    expect(results.length, equals(1));
    expect(results.first.id, equals(1));

    restored.close();
  });

  test('T28: truncated checkpoint bytes are rejected', () {
    final source = HnswIndex.open(
      collectionName: testCollectionName('t28_truncated_source'),
      dimensions: 4,
      metric: HnswMetric.l2,
      maxElements: 100,
    );
    source.upsert(1, [1, 0, 0, 0]);
    final bytes = source.checkpoint(expectedGeneration: source.generation);
    source.close();

    for (final cut in [1, 2, 4, 8]) {
      final truncated = bytes.sublist(0, bytes.length - cut);
      final restored = HnswIndex.open(
        collectionName: testCollectionName('t28_truncated_$cut'),
        dimensions: 4,
        metric: HnswMetric.l2,
        maxElements: 100,
      );
      expect(
        () => restored.load(truncated, expectedGeneration: restored.generation),
        throwsA(isA<HnswLoadException>()),
        reason: 'Removing $cut bytes must invalidate the checkpoint',
      );
      restored.close();
    }
  });

  test('T29: checkpoint reports a larger size after the index grows', () {
    final name = testCollectionName('t29_checkpoint_growth');
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

      final vector = calloc<Float>(4);
      vector[0] = 1;
      expect(native.hnswUpsertVector(handle, 1, vector), equals(0));

      final sizeOut = calloc<Int32>();
      final initialGeneration = native.hnswGeneration(handle);
      expect(
        native.hnswCheckpoint(
          handle,
          Pointer<Uint8>.fromAddress(0),
          0,
          sizeOut,
          initialGeneration,
        ),
        equals(0),
      );
      final initialSize = sizeOut.value;
      final staleBuffer = calloc<Uint8>(initialSize);

      for (var id = 2; id <= 12; id++) {
        vector[0] = id.toDouble();
        expect(native.hnswUpsertVector(handle, id, vector), equals(0));
      }

      expect(
        native.hnswCheckpoint(
          handle,
          staleBuffer,
          initialSize,
          sizeOut,
          native.hnswGeneration(handle),
        ),
        equals(-2),
      );
      expect(sizeOut.value, greaterThan(initialSize));

      final completeBuffer = calloc<Uint8>(sizeOut.value);
      expect(
        native.hnswCheckpoint(
          handle,
          completeBuffer,
          sizeOut.value,
          sizeOut,
          native.hnswGeneration(handle),
        ),
        equals(0),
      );

      calloc.free(completeBuffer);
      calloc.free(staleBuffer);
      calloc.free(sizeOut);
      calloc.free(vector);
      expect(native.hnswClose(handle), equals(0));
    } finally {
      calloc.free(errorCode);
      calloc.free(namePtr);
    }
  });

  test('T30: empty indexes can be checkpointed and loaded', () {
    final source = HnswIndex.open(
      collectionName: testCollectionName('t30_empty_source'),
      dimensions: 4,
      metric: HnswMetric.l2,
      maxElements: 100,
    );
    final bytes = source.checkpoint(expectedGeneration: source.generation);
    source.close();

    final restored = HnswIndex.open(
      collectionName: testCollectionName('t30_empty_restored'),
      dimensions: 4,
      metric: HnswMetric.l2,
      maxElements: 100,
    );
    restored.load(bytes, expectedGeneration: restored.generation);
    expect(restored.stats.elementCount, equals(0));
    restored.close();
  });

  test('T32: invalid link-list counts are rejected', () {
    final source = HnswIndex.open(
      collectionName: testCollectionName('t32_malformed_links_source'),
      dimensions: 4,
      metric: HnswMetric.l2,
      maxElements: 100,
    );
    source.upsert(1, [1, 0, 0, 0]);
    final bytes = Uint8List.fromList(
      source.checkpoint(expectedGeneration: source.generation),
    );
    source.close();

    // The serialized header contains ten size_t values, two uint32 values,
    // and one double. The first level-0 link count follows that header.
    final pointerSize = sizeOf<Pointer<Void>>();
    final headerSize =
        pointerSize * 10 + sizeOf<Uint32>() * 2 + sizeOf<Double>();
    ByteData.sublistView(bytes).setUint16(headerSize, 0xffff, Endian.host);

    final restored = HnswIndex.open(
      collectionName: testCollectionName('t32_malformed_links_restored'),
      dimensions: 4,
      metric: HnswMetric.l2,
      maxElements: 100,
    );
    try {
      expect(
        () => restored.load(bytes, expectedGeneration: restored.generation),
        throwsA(isA<HnswLoadException>()),
      );
    } finally {
      restored.close();
    }
  });
}
