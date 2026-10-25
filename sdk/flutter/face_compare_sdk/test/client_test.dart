import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:face_compare_sdk/face_compare_sdk.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';
import 'package:test/test.dart';

void main() {
  test('encode keeps a url and encodes a local file', () async {
    final photo = File('${Directory.systemTemp.path}/face-compare-sdk-live.jpg');
    photo.writeAsBytesSync([1, 2, 3, 4]);
    addTearDown(() {
      if (photo.existsSync()) {
        photo.deleteSync();
      }
    });

    expect(await encodeImage(ImageRef.url('https://example.test/a.jpg')), 'https://example.test/a.jpg');
    expect(await encodeImage(ImageRef.raw('kyc/ref.jpg')), 'kyc/ref.jpg');
    expect(await encodeImage(ImageRef.file(photo.path)), {
      'type': 'base64',
      'data': base64Encode([1, 2, 3, 4]),
    });

    expect(
      () => encodeImage(ImageRef.file('${Directory.systemTemp.path}/missing-face-compare.jpg')),
      throwsA(
        isA<FaceCompareException>().having((error) => error.errorCode, 'errorCode', 'IMAGE_NOT_FOUND'),
      ),
    );
  });

  test('health reads the device', () async {
    final httpClient = MockClient((request) async {
      expect(request.method, 'GET');
      expect(request.url.path, '/health');
      return http.Response(
        jsonEncode({
          'status': 'UP',
          'service': 'svc-face-compare',
          'version': '1.0.0',
          'device': 'GPU (CUDA)',
          'environment': 'production',
        }),
        200,
      );
    });
    final client = FaceCompareClient(httpClient: httpClient);

    final health = await client.health();

    expect(health.device, 'GPU (CUDA)');
    expect(health.status, 'UP');
  });

  test('compare posts the match and the optional spoof check', () async {
    late http.Request captured;
    final httpClient = MockClient((request) async {
      captured = request;
      return http.Response(
        jsonEncode({
          'status': 'SUCCESS',
          'ChId': 'PORTAL_KLAIM',
          'result': {
            'is_match': 'TRUE',
            'similarity_score': 0.8642,
            'similarity_percentage': '86.42%',
            'confidence_level': 'HIGH',
          },
          'image1_metadata': {
            'faces_detected': 1,
            'bbox': [1, 2, 3, 4],
            'is_live': null,
          },
          'image2_metadata': {
            'faces_detected': 1,
            'bbox': [5, 6, 7, 8],
            'is_live': true,
            'liveness_score': 0.99,
          },
          'execution_time_ms': 68.4,
          'message': 'done',
        }),
        200,
        headers: {'x-trace-id': 'TC-test', 'x-transaction-id': 'TR-test'},
      );
    });
    final client = FaceCompareClient(channelId: 'PORTAL_KLAIM', httpClient: httpClient);

    final verdict = await client.compare(
      image1: ImageRef.raw('kyc/ref.jpg'),
      image2: ImageRef.bytes(Uint8List.fromList([9, 8, 7])),
      checkSpoof: true,
      similarityThreshold: 0.8,
      traceId: 'caller-trace',
    );

    expect(verdict.isMatch, 'TRUE');
    expect(verdict.similarityScore, 0.8642);
    expect(verdict.image2.isLive, isTrue);
    expect(verdict.traceId, 'TC-test');
    expect(verdict.transactionId, 'TR-test');

    final body = jsonDecode(captured.body) as Map<String, dynamic>;
    expect(body['ChId'], 'PORTAL_KLAIM');
    expect(body['image1'], 'kyc/ref.jpg');
    expect(body['image2'], {'type': 'base64', 'data': base64Encode([9, 8, 7])});
    expect(body['options'], {'check_liveness_image2': true, 'similarity_threshold': 0.8});
    expect(captured.headers['x-trace-id'], 'caller-trace');
    expect(captured.headers['x-chid'], 'PORTAL_KLAIM');
    expect(captured.url.path, '/v1/compare');
  });

  test('a no-face response raises the API error', () async {
    final httpClient = MockClient((request) async {
      return http.Response(
        jsonEncode({
          'status': 'FAILED',
          'ChId': 'PORTAL_KLAIM',
          'error_code': 'NO_FACE_IN_IMAGE_1',
          'message': 'No face detected in Image 1 (Reference photo).',
          'image1_metadata': {'faces_detected': 0},
          'image2_metadata': {
            'faces_detected': 1,
            'bbox': [1, 2, 3, 4],
          },
          'execution_time_ms': 12.5,
        }),
        400,
        headers: {'x-trace-id': 'TC-test', 'x-transaction-id': 'TR-test'},
      );
    });
    final client = FaceCompareClient(httpClient: httpClient);

    expect(
      () => client.compare(
        image1: ImageRef.url('http://example.test/a.jpg'),
        image2: ImageRef.url('http://example.test/b.jpg'),
      ),
      throwsA(
        isA<FaceCompareException>()
            .having((error) => error.errorCode, 'errorCode', 'NO_FACE_IN_IMAGE_1')
            .having((error) => error.httpStatus, 'httpStatus', 400)
            .having((error) => error.image1?.facesDetected, 'faces', 0)
            .having((error) => error.traceId, 'traceId', 'TC-test'),
      ),
    );
  });

  test('a connection failure is TRANSPORT', () async {
    final httpClient = MockClient((request) async {
      throw http.ClientException('connection refused');
    });
    final client = FaceCompareClient(httpClient: httpClient);

    expect(
      () => client.health(),
      throwsA(isA<FaceCompareException>().having((error) => error.errorCode, 'errorCode', 'TRANSPORT')),
    );
  });
}
