import 'package:dart_hnsw/dart_hnsw.dart';

void main() {
  // Open an in-memory index — no file path needed.
  final index = HnswIndex.open(
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

  // Capture the generation before checkpointing shared state.
  final generation = index.generation;
  final bytes = index.checkpoint(expectedGeneration: generation);
  print('Checkpoint size: ${bytes.length} bytes');

  // The application owns persistence and may pass these bytes directly to
  // another layer or retain them for a later load.
  // load() atomically replaces this shared context.
  index.load(bytes, expectedGeneration: generation);

  final restoredResults = index.search([0.9, 0.1, 0, 0], k: 2);
  for (final r in restoredResults) {
    print('restored: id=${r.id} distance=${r.distance}');
  }

  index.close();
}
