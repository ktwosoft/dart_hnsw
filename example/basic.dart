import 'package:dart_hnsw/dart_hnsw.dart';

void main() {
  final index = HnswIndex.open(
    path: 'example.hnsw',
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

  index.checkpoint(); // explicit save — close() does not save
  index.close();
}
