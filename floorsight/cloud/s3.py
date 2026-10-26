"""S3 implementation of EvidenceStore. Same reference format as storage.LocalEvidenceStore.

Evidence images are already blurred on the edge. The bucket (see infra/terraform)
blocks public access, encrypts with SSE-S3 and expires objects after 1 day.
"""

from __future__ import annotations

import threading
import time

import cv2
import numpy as np

from floorsight.cloud._aws import make_client, safe_ref


class S3EvidenceStore:
    def __init__(
        self,
        bucket: str,
        prefix: str = "evidence/",
        region: str | None = None,
        client=None,
        jpeg_quality: int = 85,
        presign_expiry_s: int = 900,
    ):
        self.bucket = bucket
        self.prefix = prefix if prefix.endswith("/") or not prefix else prefix + "/"
        self.jpeg_quality = jpeg_quality
        self.presign_expiry_s = presign_expiry_s
        self._s3 = make_client("s3", region, client)
        self._lock = threading.Lock()
        self._n = 0

    def key(self, ref: str) -> str:
        return f"{self.prefix}{ref}.jpg"

    def put(self, image: np.ndarray, label: str) -> str:
        safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in label)[:40]
        with self._lock:
            self._n += 1
            ref = f"{int(time.time() * 1000)}-{self._n:05d}-{safe}"
        ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        if not ok:
            raise RuntimeError("JPEG encode failed")
        self._s3.put_object(
            Bucket=self.bucket,
            Key=self.key(ref),
            Body=buf.tobytes(),
            ContentType="image/jpeg",
            ServerSideEncryption="AES256",
            Metadata={"label": safe, "blurred": "true"},
        )
        return ref

    def put_bytes(self, ref: str, data: bytes) -> None:
        """Store already-encoded (blurred) JPEG bytes under an existing ref (mirroring a local store)."""
        if not safe_ref(ref):
            raise ValueError(f"unsafe evidence ref {ref!r}")
        self._s3.put_object(Bucket=self.bucket, Key=self.key(ref), Body=data, ContentType="image/jpeg",
                            ServerSideEncryption="AES256", Metadata={"blurred": "true"})

    def get(self, ref: str) -> bytes | None:
        if not safe_ref(ref):
            return None
        from botocore.exceptions import ClientError

        try:
            obj = self._s3.get_object(Bucket=self.bucket, Key=self.key(ref))
        except ClientError as e:
            if e.response.get("Error", {}).get("Code") in ("NoSuchKey", "404", "NotFound"):
                return None
            raise
        return obj["Body"].read()

    def presigned_url(self, ref: str, expiry_s: int | None = None) -> str | None:
        """Short-lived GET link so the dashboard can show evidence without proxying bytes."""
        if not safe_ref(ref):
            return None
        return self._s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": self.bucket, "Key": self.key(ref)},
            ExpiresIn=expiry_s or self.presign_expiry_s,
        )
