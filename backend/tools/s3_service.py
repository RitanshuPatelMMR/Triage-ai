import boto3
import json
import uuid
import os
from datetime import datetime
from botocore.exceptions import ClientError

# ── S3 client (lazy loaded) ───────────────────────────────────────────────
_s3_client = None

def get_s3():
    global _s3_client
    if _s3_client is None:
        _s3_client = boto3.client(
            "s3",
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name=os.getenv("AWS_REGION", "us-east-1")
        )
    return _s3_client

BUCKET = os.getenv("S3_BUCKET_NAME", "triageai-storage")


def upload_file(file_bytes: bytes, filename: str, input_type: str) -> str:
    """
    Upload uploaded file to S3.
    Returns S3 key (path inside bucket).
    """
    try:
        date_prefix = datetime.utcnow().strftime("%Y/%m/%d")
        unique_id = str(uuid.uuid4())[:8]
        extension = filename.split(".")[-1].lower() if "." in filename else "bin"
        s3_key = f"uploads/{date_prefix}/{unique_id}_{filename}"

        get_s3().put_object(
            Bucket=BUCKET,
            Key=s3_key,
            Body=file_bytes,
            ContentType=_get_content_type(extension),
            Metadata={
                "input_type": input_type,
                "original_filename": filename,
                "uploaded_at": datetime.utcnow().isoformat()
            }
        )
        print(f"✅ S3 upload: {s3_key}")
        return s3_key

    except ClientError as e:
        print(f"❌ S3 upload failed: {e}")
        return ""


def save_report(report: dict, input_type: str) -> str:
    """
    Save final report JSON to S3.
    Returns S3 key.
    """
    try:
        date_prefix = datetime.utcnow().strftime("%Y/%m/%d")
        unique_id = str(uuid.uuid4())[:8]
        s3_key = f"reports/{date_prefix}/{unique_id}_report.json"

        get_s3().put_object(
            Bucket=BUCKET,
            Key=s3_key,
            Body=json.dumps(report, indent=2),
            ContentType="application/json",
            Metadata={
                "input_type": input_type,
                "created_at": datetime.utcnow().isoformat()
            }
        )
        print(f"✅ S3 report saved: {s3_key}")
        return s3_key

    except ClientError as e:
        print(f"❌ S3 report save failed: {e}")
        return ""


def get_report(s3_key: str) -> dict:
    """
    Retrieve a saved report from S3 by key.
    """
    try:
        response = get_s3().get_object(Bucket=BUCKET, Key=s3_key)
        content = response["Body"].read().decode("utf-8")
        return json.loads(content)
    except ClientError as e:
        print(f"❌ S3 get report failed: {e}")
        return {}


def is_available() -> bool:
    """Check if S3 is configured and reachable."""
    try:
        if not os.getenv("AWS_ACCESS_KEY_ID"):
            return False
        get_s3().head_bucket(Bucket=BUCKET)
        return True
    except Exception:
        return False


def _get_content_type(extension: str) -> str:
    types = {
        "pdf": "application/pdf",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "png": "image/png",
        "txt": "text/plain",
        "json": "application/json",
    }
    return types.get(extension, "application/octet-stream")