"""S3 storage tests — no AWS calls when the bucket is unset (default)."""

from rag import storage


def test_disabled_by_default():
    # With S3_BUCKET unset, S3 is off and every call is a safe no-op.
    assert storage.s3_enabled() is False
    assert storage.presigned_url("sess", "doc.pdf") is None
    storage.upload_document("sess", "doc.pdf", b"%PDF-1.4")  # no-op, no AWS call
    storage.delete_session("sess")                            # no-op, no AWS call


def test_key_layout():
    assert storage._key("abc123", "report.pdf") == "uploads/abc123/report.pdf"
