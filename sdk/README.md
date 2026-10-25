# Face comparison SDK

Python client for the face comparison API. It does not install the models or the GPU runtime. The API process has to be running already.

```bash
pip install -e ./sdk
```

```python
from face_compare_sdk import FaceCompareClient, FaceCompareError

with FaceCompareClient("http://127.0.0.1:2112", channel_id="PORTAL_KLAIM") as client:
    print(client.health().device)
    try:
        verdict = client.compare(
            "http://s3.bpjsketenagakerjaan.go.id/kyc/ref.jpg",
            "live.jpg",
            check_liveness_image2=True,
        )
    except FaceCompareError as exc:
        print(exc.error_code, exc)
    else:
        print(verdict.is_match, verdict.similarity_score)
```

`compare` accepts a URL, an S3 key, a base64 string, a `data:` URL, raw image bytes, or a path to a local file. A local file is uploaded as base64. A URL is sent through so the API downloads it.

`health().device` is `GPU (CUDA)` when that API process is using the GPU, and `CPU` when it is not.

Failed comparisons raise `FaceCompareError`. `error_code` matches the API (`NO_FACE_IN_IMAGE_1`, `INVALID_IMAGE_2`, `INTERNAL_SERVER_ERROR`). A connection failure uses `TRANSPORT`. `trace_id` and `transaction_id` come from the response headers.
