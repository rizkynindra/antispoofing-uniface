import 'models.dart';

/// The API rejected the call, or the response could not be read.
///
/// Client-side problems, such as an empty image, use [httpStatus] null.
class FaceCompareException implements Exception {
  /// Creates an error with the API [errorCode] when the server sent one.
  FaceCompareException(
    this.message, {
    required this.errorCode,
    this.httpStatus,
    this.channelId = 'UNKNOWN',
    this.image1,
    this.image2,
    this.executionTimeMs,
    this.traceId = '',
    this.transactionId = '',
  });

  /// Human-readable explanation.
  final String message;

  /// API code such as `NO_FACE_IN_IMAGE_1`, or `TRANSPORT` when the host is unreachable.
  final String errorCode;

  /// HTTP status. Null when the failure happened before a response arrived.
  final int? httpStatus;

  /// Channel id from the error body.
  final String channelId;

  /// Metadata for the reference photo, when the API included it.
  final ImageMetadata? image1;

  /// Metadata for the photo being checked, when the API included it.
  final ImageMetadata? image2;

  /// Server processing time in milliseconds, when present.
  final double? executionTimeMs;

  /// Value of the `X-TRACE-ID` response header.
  final String traceId;

  /// Value of the `X-TRANSACTION-ID` response header.
  final String transactionId;

  @override
  String toString() => message;
}
