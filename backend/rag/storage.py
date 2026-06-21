"""Optional S3 storage for uploaded source documents.

Ephemeral by design: objects live under <prefix>/<session_id>/<filename> and are
removed when the session is cleared or evicted; a bucket lifecycle rule expires
any stragglers. Enabled by setting S3_BUCKET — otherwise every call is a no-op,
so the app runs identically without S3. Credentials come from the standard AWS
chain (EC2 instance role in prod).
"""

import logging
from functools import lru_cache

from .config import settings

logger = logging.getLogger("rag.storage")


def s3_enabled() -> bool:
    return bool(settings.s3_bucket.strip())


@lru_cache(maxsize=1)
def _client():
    import boto3

    return boto3.client("s3", region_name=settings.aws_region)


def _key(session_id: str, filename: str) -> str:
    return f"{settings.s3_prefix}/{session_id}/{filename}"


def upload_document(session_id: str, filename: str, data: bytes) -> None:
    if not s3_enabled():
        return
    try:
        _client().put_object(
            Bucket=settings.s3_bucket,
            Key=_key(session_id, filename),
            Body=data,
            ContentType="application/pdf",
        )
    except Exception as exc:  # never let storage break ingestion
        logger.warning("S3 upload failed for %s: %s", filename, exc)


def presigned_url(session_id: str, filename: str) -> str | None:
    if not s3_enabled():
        return None
    try:
        return _client().generate_presigned_url(
            "get_object",
            Params={"Bucket": settings.s3_bucket, "Key": _key(session_id, filename)},
            ExpiresIn=settings.s3_url_ttl_seconds,
        )
    except Exception as exc:
        logger.warning("S3 presign failed for %s: %s", filename, exc)
        return None


def delete_session(session_id: str) -> None:
    if not s3_enabled():
        return
    try:
        client = _client()
        prefix = f"{settings.s3_prefix}/{session_id}/"
        listing = client.list_objects_v2(Bucket=settings.s3_bucket, Prefix=prefix)
        objects = [{"Key": o["Key"]} for o in listing.get("Contents", [])]
        if objects:
            client.delete_objects(Bucket=settings.s3_bucket, Delete={"Objects": objects})
    except Exception as exc:
        logger.warning("S3 delete failed for session %s: %s", session_id, exc)
