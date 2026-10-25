import 'dart:typed_data';

/// A photo to send to `POST /v1/compare`.
///
/// A URL or S3 key is forwarded so the API downloads it. Bytes and local files
/// are uploaded as base64.
class ImageRef {
  const ImageRef._({this.value, this.bytes, this.path});

  /// An `http`, `https`, `s3`, or `data:` value, or a raw base64 string.
  const ImageRef.url(String url) : this._(value: url);

  /// An S3 key or a base64 string the API should interpret itself.
  const ImageRef.raw(String raw) : this._(value: raw);

  /// Encoded image bytes, usually a JPEG from the camera or gallery.
  const ImageRef.bytes(Uint8List data) : this._(bytes: data);

  /// A file path on the phone. The SDK reads it and uploads base64.
  const ImageRef.file(String filePath) : this._(path: filePath);

  /// URL, S3 key, or base64 text.
  final String? value;

  /// Raw image bytes.
  final Uint8List? bytes;

  /// Local file path.
  final String? path;
}
