import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:http/http.dart' as http;

import 'errors.dart';
import 'image_ref.dart';
import 'models.dart';

/// Calls a running face comparison service from a Flutter or Dart app.
///
/// Face matching always runs. [FaceCompareClient.compare] can also run the
/// passive spoof check on either photo. That check is MiniFASNet: real face
/// versus a printed photo, a screen replay, or a mask. The API stores it in
/// `is_live` and `liveness_score`.
class FaceCompareClient {
  /// Points at the API. [baseUrl] has no trailing path.
  ///
  /// Pass [httpClient] in tests. The SDK closes a client it created itself.
  FaceCompareClient({
    this.baseUrl = 'http://127.0.0.1:2112',
    this.channelId = 'UNKNOWN',
    this.timeout = const Duration(seconds: 60),
    http.Client? httpClient,
  }) : _http = httpClient ?? http.Client(),
       _ownsClient = httpClient == null;

  /// API origin, for example `http://10.0.2.2:2112` from an Android emulator.
  final String baseUrl;

  /// Default `ChId` sent with every compare call.
  final String channelId;

  /// Limit for one HTTP call, including the upload.
  final Duration timeout;

  final http.Client _http;
  final bool _ownsClient;

  /// Closes the HTTP client when this SDK created it.
  void close() {
    if (_ownsClient) {
      _http.close();
    }
  }

  /// Service status, including whether that process is on GPU or CPU.
  Future<HealthStatus> health() async {
    final response = await _request('GET', '/health');
    final payload = response.payload;
    return HealthStatus(
      status: '${payload['status'] ?? ''}',
      service: '${payload['service'] ?? ''}',
      version: '${payload['version'] ?? ''}',
      device: '${payload['device'] ?? ''}',
      environment: '${payload['environment'] ?? ''}',
    );
  }

  /// Compare [image1] (reference) with [image2] (the photo being checked).
  ///
  /// Set [checkSpoof] to also run the passive spoof check on [image2].
  /// Set [checkSpoofOnReference] to run that same check on [image1].
  Future<CompareResult> compare({
    required ImageRef image1,
    required ImageRef image2,
    String? channelId,
    bool checkSpoof = false,
    bool checkSpoofOnReference = false,
    double? similarityThreshold,
    double? needConfirmationThreshold,
    String? traceId,
    String? prevTransactionId,
  }) async {
    final resolvedChannel = channelId ?? this.channelId;
    final body = <String, Object?>{
      'ChId': resolvedChannel,
      'image1': await encodeImage(image1),
      'image2': await encodeImage(image2),
    };
    final options = <String, Object?>{};
    if (checkSpoofOnReference) {
      options['check_liveness_image1'] = true;
    }
    if (checkSpoof) {
      options['check_liveness_image2'] = true;
    }
    if (similarityThreshold != null) {
      options['similarity_threshold'] = similarityThreshold;
    }
    if (needConfirmationThreshold != null) {
      options['need_confirmation_threshold'] = needConfirmationThreshold;
    }
    if (options.isNotEmpty) {
      body['options'] = options;
    }

    final headers = <String, String>{'X-ChId': resolvedChannel};
    if (traceId != null && traceId.isNotEmpty) {
      headers['X-TRACE-ID'] = traceId;
    }
    if (prevTransactionId != null && prevTransactionId.isNotEmpty) {
      headers['X-PREV-TRANS-ID'] = prevTransactionId;
    }

    final response = await _request(
      'POST',
      '/v1/compare',
      body: body,
      headers: headers,
    );
    return _parseCompareResult(response);
  }

  Future<_ApiResponse> _request(
    String method,
    String path, {
    Map<String, Object?>? body,
    Map<String, String>? headers,
  }) async {
    final request = http.Request(method, Uri.parse('$baseUrl$path'));
    request.headers['Accept'] = 'application/json';
    request.headers['User-Agent'] = 'face-compare-flutter/1.0.0';
    if (headers != null) {
      request.headers.addAll(headers);
    }
    if (body != null) {
      request.headers['Content-Type'] = 'application/json';
      request.body = jsonEncode(body);
    }

    final http.Response response;
    try {
      final streamed = await _http.send(request).timeout(timeout);
      response = await http.Response.fromStream(streamed);
    } on TimeoutException {
      throw FaceCompareException(
        'Could not reach $baseUrl$path: timed out.',
        errorCode: 'TRANSPORT',
      );
    } on http.ClientException catch (error) {
      throw FaceCompareException(
        'Could not reach $baseUrl$path: ${error.message}',
        errorCode: 'TRANSPORT',
      );
    } on SocketException catch (error) {
      throw FaceCompareException(
        'Could not reach $baseUrl$path: ${error.message}',
        errorCode: 'TRANSPORT',
      );
    }

    final traceId = response.headers['x-trace-id'] ?? '';
    final transactionId = response.headers['x-transaction-id'] ?? '';
    final payload = _decodeJson(
      response.bodyBytes,
      response.statusCode,
      traceId,
      transactionId,
    );
    if (response.statusCode >= 400 || payload['status'] == 'FAILED') {
      throw _errorFromPayload(
        payload,
        response.statusCode,
        traceId,
        transactionId,
      );
    }
    return _ApiResponse(payload, traceId, transactionId);
  }
}

class _ApiResponse {
  const _ApiResponse(this.payload, this.traceId, this.transactionId);

  final Map<String, dynamic> payload;
  final String traceId;
  final String transactionId;
}

/// Turns an [ImageRef] into the JSON value `POST /v1/compare` expects.
Future<Object> encodeImage(ImageRef source) async {
  if (source.bytes != null) {
    return _base64Payload(source.bytes!);
  }
  if (source.path != null) {
    final file = File(source.path!);
    if (!await file.exists()) {
      throw FaceCompareException(
        'Image file does not exist: ${source.path}',
        errorCode: 'IMAGE_NOT_FOUND',
      );
    }
    return _base64Payload(await file.readAsBytes());
  }

  final text = (source.value ?? '').trim();
  if (text.isEmpty) {
    throw FaceCompareException('Image data is empty.', errorCode: 'EMPTY_IMAGE');
  }
  return text;
}

Map<String, String> _base64Payload(Uint8List raw) {
  if (raw.isEmpty) {
    throw FaceCompareException('Image data is empty.', errorCode: 'EMPTY_IMAGE');
  }
  return {'type': 'base64', 'data': base64Encode(raw)};
}

Map<String, dynamic> _decodeJson(
  Uint8List raw,
  int status,
  String traceId,
  String transactionId,
) {
  final Object? payload;
  try {
    payload = jsonDecode(utf8.decode(raw));
  } on FormatException {
    throw FaceCompareException(
      'Response was not JSON (HTTP $status).',
      errorCode: 'INVALID_RESPONSE',
      httpStatus: status,
      traceId: traceId,
      transactionId: transactionId,
    );
  }
  if (payload is! Map) {
    throw FaceCompareException(
      'Response JSON was not an object (HTTP $status).',
      errorCode: 'INVALID_RESPONSE',
      httpStatus: status,
      traceId: traceId,
      transactionId: transactionId,
    );
  }
  return Map<String, dynamic>.from(payload);
}

FaceCompareException _errorFromPayload(
  Map<String, dynamic> payload,
  int status,
  String traceId,
  String transactionId,
) {
  final message = payload['message'];
  return FaceCompareException(
    message is String && message.isNotEmpty
        ? message
        : 'Request failed with HTTP $status.',
    errorCode: '${payload['error_code'] ?? 'HTTP_$status'}',
    httpStatus: status,
    channelId: '${payload['ChId'] ?? 'UNKNOWN'}',
    image1: _parseMetadata(payload['image1_metadata']),
    image2: _parseMetadata(payload['image2_metadata']),
    executionTimeMs: _optionalDouble(payload['execution_time_ms']),
    traceId: traceId,
    transactionId: transactionId,
  );
}

CompareResult _parseCompareResult(_ApiResponse response) {
  final payload = response.payload;
  final result = payload['result'];
  final image1 = _parseMetadata(payload['image1_metadata']);
  final image2 = _parseMetadata(payload['image2_metadata']);
  if (result is! Map || image1 == null || image2 == null) {
    throw FaceCompareException(
      'Comparison response is missing the verdict or image metadata.',
      errorCode: 'INVALID_RESPONSE',
      httpStatus: 200,
      channelId: '${payload['ChId'] ?? 'UNKNOWN'}',
      image1: image1,
      image2: image2,
      traceId: response.traceId,
      transactionId: response.transactionId,
    );
  }
  final verdict = Map<String, dynamic>.from(result);
  return CompareResult(
    channelId: '${payload['ChId'] ?? 'UNKNOWN'}',
    isMatch: '${verdict['is_match'] ?? ''}',
    similarityScore: (verdict['similarity_score'] as num).toDouble(),
    similarityPercentage: '${verdict['similarity_percentage'] ?? ''}',
    confidenceLevel: '${verdict['confidence_level'] ?? ''}',
    image1: image1,
    image2: image2,
    executionTimeMs: (payload['execution_time_ms'] as num?)?.toDouble() ?? 0,
    message: '${payload['message'] ?? ''}',
    traceId: response.traceId,
    transactionId: response.transactionId,
  );
}

ImageMetadata? _parseMetadata(Object? value) {
  if (value is! Map) {
    return null;
  }
  final data = Map<String, dynamic>.from(value);
  final bbox = data['bbox'];
  return ImageMetadata(
    facesDetected: (data['faces_detected'] as num?)?.toInt() ?? 0,
    bbox: bbox is List ? [for (final item in bbox) (item as num).toDouble()] : null,
    isLive: data['is_live'] as bool?,
    livenessScore: _optionalDouble(data['liveness_score']),
  );
}

double? _optionalDouble(Object? value) {
  if (value is num) {
    return value.toDouble();
  }
  return null;
}
