/// Detection and spoof details for one input photo.
class ImageMetadata {
  /// Creates metadata returned for one image.
  const ImageMetadata({
    required this.facesDetected,
    this.bbox,
    this.isLive,
    this.livenessScore,
  });

  /// Number of faces the detector found.
  final int facesDetected;

  /// Primary face box as `[x1, y1, x2, y2]`.
  final List<double>? bbox;

  /// Passive spoof verdict. True means a real face. Null when the check was off.
  final bool? isLive;

  /// Passive spoof confidence. Null when the check was off.
  final double? livenessScore;
}

/// Successful 1:1 comparison.
class CompareResult {
  /// Creates a parsed success response.
  const CompareResult({
    required this.channelId,
    required this.isMatch,
    required this.similarityScore,
    required this.similarityPercentage,
    required this.confidenceLevel,
    required this.image1,
    required this.image2,
    required this.executionTimeMs,
    required this.message,
    required this.traceId,
    required this.transactionId,
  });

  /// Channel id echoed by the API.
  final String channelId;

  /// `TRUE`, `Need Confirmation`, or `Different`.
  final String isMatch;

  /// Cosine similarity from 0 to 1.
  final double similarityScore;

  /// Similarity formatted by the API, for example `86.42%`.
  final String similarityPercentage;

  /// `HIGH`, `MEDIUM`, or `LOW`.
  final String confidenceLevel;

  /// Reference photo.
  final ImageMetadata image1;

  /// Photo being checked.
  final ImageMetadata image2;

  /// Server processing time in milliseconds.
  final double executionTimeMs;

  /// Summary from the API.
  final String message;

  /// Value of the `X-TRACE-ID` response header.
  final String traceId;

  /// Value of the `X-TRANSACTION-ID` response header.
  final String transactionId;
}

/// `GET /health` payload.
class HealthStatus {
  /// Creates a parsed health response.
  const HealthStatus({
    required this.status,
    required this.service,
    required this.version,
    required this.device,
    required this.environment,
  });

  /// `UP` when the process is serving.
  final String status;

  /// Service name from the API.
  final String service;

  /// Service version from the API.
  final String version;

  /// `GPU (CUDA)` or `CPU`.
  final String device;

  /// Deployment environment name.
  final String environment;
}
