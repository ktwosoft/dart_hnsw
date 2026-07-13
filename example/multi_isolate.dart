import 'dart:isolate';
import 'dart:typed_data';

import 'package:dart_hnsw/dart_hnsw.dart';

const _collectionName = 'multi_isolate_example';

Future<void> main() async {
  // The application owns loading checkpoint bytes.
  final index = HnswIndex.open(
    collectionName: _collectionName,
    dimensions: 2,
    metric: HnswMetric.l2,
  );
  index.load(_initialCheckpoint(), expectedGeneration: index.generation);

  // Workers only open the same collection, write one vector, and close.
  await Future.wait([_writeFromIsolate(7), _writeFromIsolate(8)]);

  print(
    'Element count: ${index.stats.elementCount}',
  ); // 3, including loaded id 42.
  print('Nearest id: ${index.search([0, 7], k: 1).single.id}');
  index.close();
}

Future<void> _writeFromIsolate(int id) async {
  final reply = ReceivePort();
  await Isolate.spawn(_writeVector, [id, reply.sendPort]);
  await reply.first;
  reply.close();
}

void _writeVector(List<Object> message) {
  final id = message[0] as int;
  final reply = message[1] as SendPort;

  final index = HnswIndex.open(
    collectionName: _collectionName,
    dimensions: 2,
    metric: HnswMetric.l2,
  );
  index.upsert(id, [0, id.toDouble()]);
  index.close();
  reply.send(null);
}

Uint8List _initialCheckpoint() {
  final source = HnswIndex.open(
    collectionName: 'multi_isolate_source',
    dimensions: 2,
    metric: HnswMetric.l2,
  );
  source.upsert(42, [1, 0]);
  final bytes = source.checkpoint(expectedGeneration: source.generation);
  source.close();
  return bytes;
}
