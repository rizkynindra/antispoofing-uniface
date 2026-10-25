# Face comparison SDK for Flutter

One Dart package for a Flutter app on Android and iOS. The app sends two photos. Matching and the optional spoof check run on the face comparison API.

The package does not embed the models, and it has no Android or iOS plugin code. The same dependency works on both platforms.

## Install

In the Flutter app `pubspec.yaml`:

```yaml
dependencies:
  face_compare_sdk:
    path: ../antispoofing-uniface/sdk/flutter/face_compare_sdk
```

Then run `flutter pub get`.

## Compare two photos

`image1` is the reference photo. `image2` is the photo being checked. Matching always runs.

`checkSpoof: true` also runs the passive spoof check on `image2`. The API reports that on `isLive` and `livenessScore`. The check looks for a printed photo, a screen replay, or a mask. It does not ask the person to blink or turn their head. Leave the flag off for a match-only call. `checkSpoofOnReference` runs the same check on `image1`.

```dart
import 'dart:typed_data';

import 'package:face_compare_sdk/face_compare_sdk.dart';

Future<void> matchPhoto(Uint8List capturedJpeg) async {
  final client = FaceCompareClient(
    baseUrl: 'http://10.0.2.2:2112',
    channelId: 'PORTAL_KLAIM',
  );
  try {
    print((await client.health()).device);
    final verdict = await client.compare(
      image1: ImageRef.url('http://s3.bpjsketenagakerjaan.go.id/kyc/ref.jpg'),
      image2: ImageRef.bytes(capturedJpeg),
      checkSpoof: true,
    );
    print('${verdict.isMatch} ${verdict.similarityScore}');
    print(verdict.image2.isLive);
  } on FaceCompareException catch (error) {
    print('${error.errorCode} ${error.message}');
  } finally {
    client.close();
  }
}
```

`ImageRef.url` and `ImageRef.raw` are sent through, so the API downloads them. `ImageRef.bytes` and `ImageRef.file` are uploaded as base64. A gallery or camera plugin should hand the SDK the file path or the JPEG bytes. This package does not open the camera.

`verdict.isMatch` is `TRUE`, `Need Confirmation`, or `Different`.

`FaceCompareException.errorCode` is the API code (`NO_FACE_IN_IMAGE_1`, `INVALID_IMAGE_2`, `INTERNAL_SERVER_ERROR`) or `TRANSPORT` when the phone cannot reach the API. `traceId` and `transactionId` come from the response headers.

## Where the phone should send the request

| App is running on | `baseUrl` |
|---|---|
| Android emulator | `http://10.0.2.2:2112` |
| iOS simulator | `http://127.0.0.1:2112` |
| A physical phone | `http://<the API machine's LAN address>:2112` |

The API must already be running and listening on `0.0.0.0`. Android blocks plain `http://` unless the app allows cleartext traffic. iOS blocks it unless the app's ATS settings allow that host. A production app should use `https://`.
