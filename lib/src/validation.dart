import 'types.dart';

void validateIndexOptions({
  required String collectionName,
  required int dimensions,
  required int maxElements,
  required int m,
  required int efConstruction,
  required int efSearch,
}) {
  if (collectionName.isEmpty) {
    throw ArgumentError.value(
      collectionName,
      'collectionName',
      'must not be empty',
    );
  }
  if (dimensions <= 0 || dimensions > HnswLimits.maxDimensions) {
    throw ArgumentError.value(
      dimensions,
      'dimensions',
      'must be between 1 and ${HnswLimits.maxDimensions}',
    );
  }
  if (maxElements <= 0 || maxElements > HnswLimits.maxElements) {
    throw ArgumentError.value(
      maxElements,
      'maxElements',
      'must be between 1 and ${HnswLimits.maxElements}',
    );
  }
  if (m < HnswLimits.minM || m > HnswLimits.maxM) {
    throw ArgumentError.value(
      m,
      'm',
      'must be between ${HnswLimits.minM} and ${HnswLimits.maxM}',
    );
  }
  if (efConstruction <= 0 || efConstruction > HnswLimits.maxEf) {
    throw ArgumentError.value(
      efConstruction,
      'efConstruction',
      'must be between 1 and ${HnswLimits.maxEf}',
    );
  }
  if (efSearch <= 0 || efSearch > HnswLimits.maxEf) {
    throw ArgumentError.value(
      efSearch,
      'efSearch',
      'must be between 1 and ${HnswLimits.maxEf}',
    );
  }
}

void validateFiniteVector(List<double> values, String name) {
  for (final value in values) {
    if (!value.isFinite) {
      throw HnswNonFiniteVectorException(
        '$name must contain only finite values',
      );
    }
  }
}

HnswException errorForCode(int code, String context) {
  switch (code) {
    case -1:
      return HnswException('$context: null argument');
    case -2:
      return HnswException('$context: invalid argument');
    case -3:
      return HnswDimensionMismatchException(
        '$context: dimensions, metric, or delete-reuse setting mismatch with the active shared index context',
      );
    case -4:
      return HnswGenerationConflictException(
        '$context: shared index changed since the expected generation',
      );
    case -5:
      return HnswLoadException('$context: failed to load index');
    case -6:
      return HnswCheckpointException('$context: failed to save index');
    case -8:
      return HnswZeroVectorException(
        '$context: zero vector not allowed for cosine metric',
      );
    case -9:
      return HnswNonFiniteVectorException(
        '$context: vector contains NaN or infinity',
      );
    case -11:
      return HnswExclusiveMutationAccessException(
        '$context: another handle owns exclusive mutation access',
      );
    default:
      return HnswNativeException('$context: native error $code');
  }
}
