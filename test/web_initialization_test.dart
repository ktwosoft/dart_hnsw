@TestOn('browser')
library;

import 'package:dart_hnsw/dart_hnsw.dart';
import 'package:test/test.dart';

void main() {
  test(
    'explicit initialization, failed-load retry and concurrent initialization',
    () async {
      expect(
        () => HnswIndex.open(collectionName: 'early', dimensions: 3),
        throwsStateError,
      );
      final loader = Uri.parse('../build/web_assets/hnsw.js');
      final wasm = Uri.parse('../build/web_assets/hnsw.wasm');
      await expectLater(
        HnswIndex.initialize(loaderUrl: Uri.parse('missing.js'), wasmUrl: wasm),
        throwsA(anything),
      );
      await expectLater(
        HnswIndex.initialize(
          loaderUrl: loader,
          wasmUrl: Uri.parse('missing.wasm'),
        ),
        throwsA(anything),
      );
      await Future.wait([
        HnswIndex.initialize(loaderUrl: loader, wasmUrl: wasm),
        HnswIndex.initialize(loaderUrl: loader, wasmUrl: wasm),
      ]);
      await HnswIndex.initialize(loaderUrl: loader, wasmUrl: wasm);
      expect(
        () => HnswIndex.initialize(
          loaderUrl: Uri.parse('different.js'),
          wasmUrl: wasm,
        ),
        throwsStateError,
      );
      final index = HnswIndex.open(
        collectionName: 'ready',
        dimensions: 3,
        maxElements: 4,
      );
      try {
        index.upsert(1, [1, 0, 0]);
        expect(index.search([1, 0, 0]).single.id, 1);
      } finally {
        index.close();
      }
    },
  );
}
