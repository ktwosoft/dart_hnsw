import 'dart:io';

import 'package:dart_hnsw/dart_hnsw.dart';

void main() {
  // Create an in-memory index — no file path needed.
  final index = HnswIndex.create(
    collectionName: 'example',
    dimensions: 4,
    metric: HnswMetric.l2,
    maxElements: 1000,
  );

  index.upsert(1, [1, 0, 0, 0]);
  index.upsert(2, [0, 1, 0, 0]);

  final results = index.search([0.9, 0.1, 0, 0], k: 2);
  for (final r in results) {
    print('id=${r.id} distance=${r.distance}');
  }

  // checkpoint() returns raw bytes — you decide where to store them.
  final bytes = index.checkpoint();
  print('Checkpoint size: ${bytes.length} bytes');

  // Example: write to a file (but you could store in a DB, blob, etc.)
  File('example.hnsw').writeAsBytesSync(bytes);

  index.close();

  // Later — restore from bytes.
  final restored = HnswIndex.load(
    collectionName: 'example_restored',
    bytes: File('example.hnsw').readAsBytesSync(),
    dimensions: 4,
    metric: HnswMetric.l2,
    maxElements: 1000,
  );

  final restoredResults = restored.search([0.9, 0.1, 0, 0], k: 2);
  for (final r in restoredResults) {
    print('restored: id=${r.id} distance=${r.distance}');
  }

  restored.close();

  // Cleanup example file.
  File('example.hnsw').deleteSync();
}
