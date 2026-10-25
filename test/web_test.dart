@TestOn('browser')
library;

import 'dart:convert';
import 'dart:math';
import 'dart:typed_data';
import 'package:dart_hnsw/dart_hnsw.dart';
import 'package:dart_hnsw/src/wasm_bindings.dart';
import 'package:test/test.dart';
import 'fixtures/native64_checkpoint.dart';

void main() {
  final handles = <HnswIndex>[];
  HnswIndex open({
    String name = 'web',
    HnswMetric metric = HnswMetric.cosine,
    int dimensions = 3,
    int capacity = 4,
    int ef = 100,
  }) {
    final index = HnswIndex.open(
      collectionName: name,
      dimensions: dimensions,
      metric: metric,
      maxElements: capacity,
      efSearch: ef,
    );
    handles.add(index);
    return index;
  }

  setUpAll(
    () => HnswIndex.initialize(
      loaderUrl: Uri.parse('../build/web_assets/hnsw.js'),
      wasmUrl: Uri.parse('../build/web_assets/hnsw.wasm'),
    ),
  );
  tearDown(() {
    for (final index in handles) {
      index.close();
    }
    handles.clear();
    expect(wasmInteger(wasmBindings.registrySize()), 0);
  });

  test(
    'same synchronous API: empty, sharing, update, delete, reuse, resize, reopen',
    () {
      final index = open(), other = open();
      expect(index.search([1, 0, 0]), isEmpty);
      expect(index.contextId, other.contextId);
      index.upsertBatch(const [
        HnswVectorEntry(4294967295, [1, 0, 0]),
        HnswVectorEntry(2, [0, 1, 0]),
        HnswVectorEntry(3, [0, 0, 1]),
        HnswVectorEntry(4, [-1, 0, 0]),
      ]);
      expect(other.search([1, 0, 0], k: 1).single.id, 4294967295);
      index.upsert(4294967295, [0.1, 1, 0]);
      expect(index.stats.elementCount, 4);
      expect(other.search([0.1, 1, 0], k: 1).single.id, 4294967295);
      index.markDeleted(2);
      expect(index.stats.deletedCount, 1);
      expect(() => index.markDeleted(2), throwsA(isA<HnswException>()));
      expect(() => index.markDeleted(999), throwsA(isA<HnswException>()));
      expect(index.search([0, 1, 0]).map((r) => r.id), isNot(contains(2)));
      index.upsert(5, [0, 1, 0]);
      expect(index.stats.deletedCount, 0);
      expect(index.stats.elementCount, 4);
      index.resize(8);
      index.upsert(6, [1, 1, 1]);
      index.setEfSearch(32);
      expect(index.stats.maxElements, 8);
      expect(index.stats.efSearch, 32);
      index.close();
      index.close();
      expect(other.stats.liveCount, 5);
      other.close();
      expect(open().stats.elementCount, 0);
    },
  );

  test('all deleted and resurrection', () {
    final index = open();
    index.upsert(7, [1, 0, 0]);
    index.markDeleted(7);
    expect(index.search([1, 0, 0]), isEmpty);
    index.upsert(7, [0, 1, 0]);
    expect(index.search([0, 1, 0]).single.id, 7);
    expect(index.stats.deletedCount, 0);
  });

  test('dimensions, finite/zero vectors, labels, options, closed handles', () {
    final index = open();
    expect(() => open(name: 'bad', dimensions: 0), throwsArgumentError);
    expect(
      () => open(dimensions: 2),
      throwsA(isA<HnswDimensionMismatchException>()),
    );
    expect(() => index.upsert(1, [1, 2]), throwsArgumentError);
    expect(() => index.search([1, 2]), throwsArgumentError);
    for (final value in [double.nan, double.infinity, 1e100]) {
      expect(
        () => index.upsert(1, [value, 0, 0]),
        throwsA(isA<HnswNonFiniteVectorException>()),
      );
      expect(
        () => index.search([value, 0, 0]),
        throwsA(isA<HnswNonFiniteVectorException>()),
      );
    }
    expect(
      () => index.upsert(1, [0, 0, 0]),
      throwsA(isA<HnswZeroVectorException>()),
    );
    expect(
      () => index.search([0, 0, 0]),
      throwsA(isA<HnswZeroVectorException>()),
    );
    for (final id in [-1, 4294967296]) {
      expect(() => index.upsert(id, [1, 0, 0]), throwsArgumentError);
      expect(() => index.markDeleted(id), throwsArgumentError);
    }
    expect(() => index.search([1, 0, 0], k: 0), throwsArgumentError);
    index.close();
    expect(() => index.upsert(1, [1, 0, 0]), throwsStateError);
    expect(() => index.upsertBatch([]), throwsStateError);
    expect(() => index.stats, throwsStateError);
    expect(() => index.checkpoint(expectedGeneration: 0), throwsStateError);
  });

  test('ranking and distances against brute force for all metrics', () {
    final random = Random(42);
    final vectors = List.generate(
      100,
      (_) => List.generate(3, (_) => random.nextDouble() + 0.01),
    );
    for (final metric in HnswMetric.values) {
      final graph = open(name: metric.name, metric: metric, capacity: 100);
      graph.upsertBatch([
        for (var i = 0; i < vectors.length; i++) HnswVectorEntry(i, vectors[i]),
      ]);
      for (final query in vectors.take(10)) {
        double distance(List<double> v) {
          var dot = 0.0, normQ = 0.0, normV = 0.0, l2 = 0.0;
          for (var j = 0; j < 3; j++) {
            dot += query[j] * v[j];
            normQ += query[j] * query[j];
            normV += v[j] * v[j];
            l2 += pow(query[j] - v[j], 2);
          }
          return switch (metric) {
            HnswMetric.cosine => 1 - dot / sqrt(normQ * normV),
            HnswMetric.l2 => l2,
            HnswMetric.innerProduct => 1 - dot,
          };
        }

        final expected = List.generate(
          100,
          (i) => i,
        )..sort((a, b) => distance(vectors[a]).compareTo(distance(vectors[b])));
        final actual = graph.search(query, k: 5);
        expect(actual.map((r) => r.id), expected.take(5));
        for (final r in actual) {
          expect(r.distance, closeTo(distance(vectors[r.id]), 1e-5));
        }
      }
    }
  });

  test(
    'raw checkpoints, atomic replacement, stale guard, native64 rejection',
    () {
      final index = open(), other = open();
      index.load(
        index.checkpoint(expectedGeneration: index.generation),
        expectedGeneration: index.generation,
      );
      index.upsert(1, [1, 0, 0]);
      final generation = index.generation;
      final bytes = index.checkpoint(expectedGeneration: generation);
      index.clear(expectedGeneration: generation);
      expect(other.stats.liveCount, 0);
      expect(
        () => index.load(bytes, expectedGeneration: generation),
        throwsA(isA<HnswGenerationConflictException>()),
      );
      index.load(bytes, expectedGeneration: index.generation);
      expect(other.search([1, 0, 0]).single.id, 1);
      expect(
        () => index.load(
          base64Decode(native64Checkpoint),
          expectedGeneration: index.generation,
        ),
        throwsA(isA<HnswLoadException>()),
      );
      expect(
        () => index.load(
          Uint8List.fromList(bytes.take(20).toList()),
          expectedGeneration: index.generation,
        ),
        throwsA(isA<HnswLoadException>()),
      );
      expect(
        () => index.clear(expectedGeneration: generation),
        throwsA(isA<HnswGenerationConflictException>()),
      );
      expect(
        () => index.checkpoint(expectedGeneration: generation),
        throwsA(isA<HnswGenerationConflictException>()),
      );
      expect(other.search([1, 0, 0]).single.id, 1);
      final replacement = open(name: 'replacement');
      replacement.upsert(2, [0, 1, 0]);
      index.load(
        replacement.checkpoint(expectedGeneration: replacement.generation),
        expectedGeneration: index.generation,
      );
      expect(other.search([0, 1, 0]).single.id, 2);
    },
  );

  test('exclusive mutation access matches native handle ownership', () {
    final index = open(), other = open();
    index.upsert(1, [1, 0, 0]);
    index.acquireExclusiveMutationAccess();
    expect(other.search([1, 0, 0]).single.id, 1);
    expect(
      () => other.upsert(2, [0, 1, 0]),
      throwsA(isA<HnswExclusiveMutationAccessException>()),
    );
    index.releaseExclusiveMutationAccess();
    other.upsert(2, [0, 1, 0]);
    index.acquireExclusiveMutationAccess();
    index.close();
    expect(
      () => other.resize(8),
      throwsA(isA<HnswExclusiveMutationAccessException>()),
    );
    other.close();
    expect(open().stats.liveCount, 0);
  });


  test('close and repeated failing calls free Wasm allocations', () {
    final before = wasmBindings.allocatedBytes();
    for (var i = 0; i < 20; i++) {
      final index = open();
      index.upsert(1, [1, 0, 0]);
      expect(
        () => index.upsert(2, [0, 0, 0]),
        throwsA(isA<HnswZeroVectorException>()),
      );
      index.checkpoint(expectedGeneration: index.generation);
      index.close();
    }
    expect(wasmInteger(wasmBindings.registrySize()), 0);
    expect(wasmBindings.allocatedBytes() - before, lessThan(32768));
  });

  test('256D retrieval smoke: incremental updates, replacement, repeated search', () {
    final random = Random(256);
    List<double> vector() =>
        List.generate(256, (_) => random.nextDouble() * 2 - 1);
    final entries = [
      for (var i = 0; i < 1000; i++) HnswVectorEntry(i, vector()),
    ];
    final index = open(dimensions: 256, capacity: 1100, ef: 64);
    final watch = Stopwatch()..start();
    index.upsertBatch(entries);
    final build = watch.elapsedMicroseconds / 1000;
    watch.reset();
    for (var i = 0; i < 100; i++) {
      entries[i] = HnswVectorEntry(i, vector());
    }
    index.upsertBatch(entries.sublist(0, 100));
    for (var i = 100; i < 120; i++) {
      index.markDeleted(i);
      entries[i] = HnswVectorEntry(i + 1000, vector());
    }
    index.upsertBatch(entries.sublist(100, 120));
    final incremental = watch.elapsedMicroseconds / 1000;
    watch.reset();
    final replacement = open(
      name: 'temporary',
      dimensions: 256,
      capacity: 1100,
      ef: 64,
    );
    replacement.upsertBatch(entries);
    index.load(
      replacement.checkpoint(expectedGeneration: replacement.generation),
      expectedGeneration: index.generation,
    );
    replacement.close();
    final replace = watch.elapsedMicroseconds / 1000;
    final timings = <double>[];
    for (var i = 0; i < 200; i++) {
      watch.reset();
      expect(index.search(entries[i].vector, k: 10), hasLength(10));
      timings.add(watch.elapsedMicroseconds / 1000);
    }
    timings.sort();
    final allocated = wasmBindings.allocatedBytes();
    index.close();
    print(
      'SMOKE build_ms=$build incremental_ms=$incremental replacement_ms=$replace '
      'p50_ms=${timings[100]} p95_ms=${timings[190]} heap_bytes=${wasmBindings.bytes.length} '
      'allocated_bytes=$allocated after_close_bytes=${wasmBindings.allocatedBytes()}',
    );
  });
  test('heap growth refreshes views during open and resize', () {
    final before = wasmBindings.bytes.length;
    final index = open(name: 'growth', dimensions: 256, capacity: 20000);
    final vector = List<double>.filled(256, 0)..[0] = 1;
    expect(wasmBindings.bytes.length, greaterThan(before));
    index.upsert(4294967295, vector);
    index.resize(40000);
    expect(index.search(vector, k: 1).single.id, 4294967295);
    final bytes = index.checkpoint(expectedGeneration: index.generation);
    index.load(bytes, expectedGeneration: index.generation);
    expect(index.search(vector, k: 1).single.id, 4294967295);
  });

}
