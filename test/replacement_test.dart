import 'dart:ffi';
import 'dart:typed_data';

import 'package:test/test.dart';

import 'package:dart_hnsw/src/hnsw_index.dart';
import 'package:dart_hnsw/src/native_bindings.dart';
import '_fixtures.dart';

HnswIndex _openL2(String collectionName) {
  return HnswIndex.open(
    collectionName: collectionName,
    dimensions: 4,
    metric: HnswMetric.l2,
    maxElements: 100,
  );
}

Uint8List _checkpointWithLabel(int label) {
  final source = _openL2(testCollectionName('replacement_source_$label'));
  source.upsert(label, [label.toDouble(), 0, 0, 0]);
  final bytes = source.checkpoint(expectedGeneration: source.generation);
  source.close();
  return bytes;
}

Uint8List _allocationHostileCheckpoint() {
  const dimensions = 4;
  const maxM = HnswLimits.maxM;
  const maxM0 = maxM * 2;
  const sizeOfTableInt = 4;
  const sizeOfLinkListSize = 4;
  final sizeOfSizeT = sizeOf<IntPtr>();
  final sizeOfLabel = sizeOfSizeT;
  final headerSize = sizeOfSizeT * 10 + 16;

  final source = _openL2(testCollectionName('hostile_checkpoint_source'));
  source.upsert(1, [1, 0, 0, 0]);
  final original = source.checkpoint(expectedGeneration: source.generation);
  source.close();

  final level0Links = maxM0 * sizeOfTableInt + sizeOfLinkListSize;
  final offsetData = level0Links;
  final labelOffset = offsetData + dimensions * Float32List.bytesPerElement;
  final bytesPerElement = labelOffset + sizeOfLabel;
  final forged = Uint8List(headerSize + bytesPerElement + sizeOfLinkListSize);
  forged.setRange(0, headerSize, original);

  final header = ByteData.sublistView(forged);
  void writeSizeT(int offset, int value) {
    if (sizeOfSizeT == 8) {
      header.setUint64(offset, value, Endian.host);
    } else {
      header.setUint32(offset, value, Endian.host);
    }
  }

  writeSizeT(sizeOfSizeT * 3, bytesPerElement);
  writeSizeT(sizeOfSizeT * 4, labelOffset);
  writeSizeT(sizeOfSizeT * 5, offsetData);
  writeSizeT(sizeOfSizeT * 6 + 8, maxM);
  writeSizeT(sizeOfSizeT * 7 + 8, maxM0);
  writeSizeT(sizeOfSizeT * 8 + 8, maxM);
  return forged;
}

void main() {
  late HnswNativeBindings native;

  setUp(() {
    native = HnswNativeBindings();
  });

  tearDown(() {
    expectCleanRegistry(native);
  });

  test('T33: load atomically replaces content for all live handles', () {
    final replacement = _checkpointWithLabel(2);
    final name = testCollectionName('t33_atomic_replace');
    final first = _openL2(name);
    final peer = _openL2(name);
    first.upsert(1, [1, 0, 0, 0]);
    final generation = first.generation;

    first.load(replacement, expectedGeneration: generation);

    expect(peer.contextId, equals(first.contextId));
    expect(peer.generation, equals(generation + 1));
    expect(peer.stats.elementCount, equals(1));
    expect(peer.search([2, 0, 0, 0], k: 1).single.id, equals(2));

    peer.close();
    first.close();
  });

  test('T34: failed load leaves the active graph unchanged', () {
    final replacement = _checkpointWithLabel(2);
    final name = testCollectionName('t34_failed_replace');
    final index = _openL2(name);
    index.upsert(1, [1, 0, 0, 0]);
    final generation = index.generation;
    final malformed = Uint8List.fromList(
      replacement.sublist(0, replacement.length - 1),
    );

    expect(
      () => index.load(malformed, expectedGeneration: generation),
      throwsA(isA<HnswLoadException>()),
    );

    expect(index.generation, equals(generation));
    expect(index.stats.elementCount, equals(1));
    expect(index.search([1, 0, 0, 0], k: 1).single.id, equals(1));
    index.close();
  });

  test('T35: stale generations reject destructive operations', () {
    final replacement = _checkpointWithLabel(3);
    final name = testCollectionName('t35_stale_generation');
    final index = _openL2(name);
    index.upsert(1, [1, 0, 0, 0]);
    final staleGeneration = index.generation;
    index.upsert(2, [2, 0, 0, 0]);

    expect(
      () => index.load(replacement, expectedGeneration: staleGeneration),
      throwsA(isA<HnswGenerationConflictException>()),
    );
    expect(
      () => index.checkpoint(expectedGeneration: staleGeneration),
      throwsA(isA<HnswGenerationConflictException>()),
    );
    expect(
      () => index.clear(expectedGeneration: staleGeneration),
      throwsA(isA<HnswGenerationConflictException>()),
    );
    expect(index.stats.elementCount, equals(2));
    index.close();
  });

  test('T36: clear resets all live handles and advances generation', () {
    final name = testCollectionName('t36_clear');
    final first = _openL2(name);
    final peer = _openL2(name);
    first.upsert(1, [1, 0, 0, 0]);
    final generation = first.generation;

    first.clear(expectedGeneration: generation);

    expect(peer.stats.elementCount, equals(0));
    expect(peer.generation, equals(generation + 1));
    expect(peer.search([1, 0, 0, 0], k: 1), isEmpty);

    peer.close();
    first.close();
  });

  test('T37: public input limits reject non-finite and unsafe values', () {
    expect(
      () => HnswIndex.open(
        collectionName: testCollectionName('t37_large_dimension'),
        dimensions: HnswLimits.maxDimensions + 1,
      ),
      throwsA(isA<ArgumentError>()),
    );
    expect(
      () => HnswIndex.open(
        collectionName: testCollectionName('t37_large_allocation'),
        dimensions: HnswLimits.maxDimensions,
        maxElements: HnswLimits.maxElements,
      ),
      throwsA(isA<HnswException>()),
    );

    final index = _openL2(testCollectionName('t37_non_finite'));
    expect(
      () => index.upsert(1, [double.nan, 0, 0, 0]),
      throwsA(isA<HnswNonFiniteVectorException>()),
    );
    expect(
      () => index.upsertBatch([
        const HnswVectorEntry(1, [1, 0, 0, 0]),
        HnswVectorEntry(2, [double.infinity, 0, 0, 0]),
      ]),
      throwsA(isA<HnswNonFiniteVectorException>()),
    );
    expect(
      () => index.search([double.infinity, 0, 0, 0]),
      throwsA(isA<HnswNonFiniteVectorException>()),
    );
    expect(() => index.search([1, 0, 0, 0], k: 0), throwsArgumentError);
    index.close();
  });

  test(
    'T39: load rejects a checkpoint with an unsafe serialized allocation',
    () {
      final target = HnswIndex.open(
        collectionName: testCollectionName('t39_hostile_checkpoint'),
        dimensions: 4,
        metric: HnswMetric.l2,
        maxElements: 1000000,
      );
      try {
        target.load(
          _allocationHostileCheckpoint(),
          // This is a valid public configuration, but would request roughly
          // 80 GiB with the forged per-element layout.
          expectedGeneration: target.generation,
        );
        fail('The hostile checkpoint must be rejected before allocation');
      } on HnswLoadException {
        // Expected.
      } finally {
        target.close();
      }
    },
  );

  test('T40: m below 2 is rejected', () {
    HnswIndex? index;
    try {
      index = HnswIndex.open(
        collectionName: testCollectionName('t40_invalid_m'),
        dimensions: 4,
        metric: HnswMetric.l2,
        m: 1,
      );
      fail('m=1 must be rejected');
    } on ArgumentError {
      // Expected.
    } finally {
      index?.close();
    }
  });

  test('T41: m=2 checkpoint round-trips', () {
    final source = HnswIndex.open(
      collectionName: testCollectionName('t41_m2_source'),
      dimensions: 4,
      metric: HnswMetric.l2,
      maxElements: 100,
      m: 2,
    );
    source.upsert(1, [1, 0, 0, 0]);
    final bytes = source.checkpoint(expectedGeneration: source.generation);
    source.close();

    final restored = HnswIndex.open(
      collectionName: testCollectionName('t41_m2_restored'),
      dimensions: 4,
      metric: HnswMetric.l2,
      maxElements: 100,
      m: 2,
    );
    restored.load(bytes, expectedGeneration: restored.generation);
    expect(restored.search([1, 0, 0, 0], k: 1).single.id, equals(1));
    restored.close();
  });

  test(
    'T42: exclusive mutation access is handle-owned and persists on close',
    () {
      final replacement = _checkpointWithLabel(3);
      final name = testCollectionName('t42_mutation_gate');
      final shutdownWorker = _openL2(name);
      final peer = _openL2(name);
      shutdownWorker.upsert(1, [1, 0, 0, 0]);

      shutdownWorker.acquireExclusiveMutationAccess();
      shutdownWorker.upsert(2, [2, 0, 0, 0]);

      shutdownWorker.releaseExclusiveMutationAccess();
      peer.upsert(3, [3, 0, 0, 0]);

      shutdownWorker.acquireExclusiveMutationAccess();
      shutdownWorker.upsert(4, [4, 0, 0, 0]);
      final generation = shutdownWorker.generation;
      final bytes = shutdownWorker.checkpoint(expectedGeneration: generation);

      expect(peer.search([1, 0, 0, 0], k: 1).single.id, equals(1));
      expect(bytes, isNotEmpty);
      expect(
        () => peer.acquireExclusiveMutationAccess(),
        throwsA(isA<HnswExclusiveMutationAccessException>()),
      );
      expect(
        () => peer.upsert(3, [3, 0, 0, 0]),
        throwsA(isA<HnswExclusiveMutationAccessException>()),
      );
      expect(
        () => peer.upsertBatch([
          const HnswVectorEntry(2, [2, 0, 0, 0]),
        ]),
        throwsA(isA<HnswExclusiveMutationAccessException>()),
      );
      expect(
        () => peer.markDeleted(1),
        throwsA(isA<HnswExclusiveMutationAccessException>()),
      );
      expect(
        () => peer.load(replacement, expectedGeneration: generation),
        throwsA(isA<HnswExclusiveMutationAccessException>()),
      );
      expect(
        () => peer.clear(expectedGeneration: generation),
        throwsA(isA<HnswExclusiveMutationAccessException>()),
      );
      expect(
        () => peer.resize(200),
        throwsA(isA<HnswExclusiveMutationAccessException>()),
      );
      expect(
        () => peer.setEfSearch(32),
        throwsA(isA<HnswExclusiveMutationAccessException>()),
      );

      // The shutdown worker can close while the peer remains mutation-blocked.
      shutdownWorker.close();
      expect(
        () => peer.releaseExclusiveMutationAccess(),
        throwsA(isA<HnswExclusiveMutationAccessException>()),
      );
      expect(
        () => peer.upsert(3, [3, 0, 0, 0]),
        throwsA(isA<HnswExclusiveMutationAccessException>()),
      );
      peer.close();

      final reopened = _openL2(name);
      reopened.upsert(3, [3, 0, 0, 0]);
      expect(reopened.stats.elementCount, equals(1));
      reopened.close();
    },
  );

  test('T43: exclusive owner can perform every protected mutation', () {
    final replacement = _checkpointWithLabel(9);
    final name = testCollectionName('t43_exclusive_owner_mutations');
    final owner = _openL2(name);
    final peer = _openL2(name);

    owner.acquireExclusiveMutationAccess();
    owner.load(replacement, expectedGeneration: owner.generation);
    owner.resize(200);
    owner.setEfSearch(32);
    owner.upsertBatch([
      const HnswVectorEntry(10, [10, 0, 0, 0]),
      const HnswVectorEntry(11, [11, 0, 0, 0]),
    ]);
    owner.markDeleted(10);
    owner.clear(expectedGeneration: owner.generation);
    owner.upsert(12, [12, 0, 0, 0]);

    owner.releaseExclusiveMutationAccess();
    peer.upsert(13, [13, 0, 0, 0]);
    final stats = peer.stats;
    expect(stats.elementCount, equals(2));
    expect(stats.efSearch, equals(32));
    expect(stats.maxElements, equals(200));

    peer.close();
    owner.close();
  });
}
