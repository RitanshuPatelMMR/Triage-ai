import boto3
import json
import os
from datetime import datetime
from botocore.exceptions import ClientError

# ── DynamoDB client (lazy loaded) ─────────────────────────────────────────
_dynamo_client = None
TABLE = os.getenv("DYNAMODB_TABLE", "triageai-reports")


def get_dynamo():
    global _dynamo_client
    if _dynamo_client is None:
        _dynamo_client = boto3.resource(
            "dynamodb",
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name=os.getenv("AWS_REGION", "us-east-1")
        )
    return _dynamo_client


def get_table():
    return get_dynamo().Table(TABLE)


# ── Save report ───────────────────────────────────────────────────────────
def save_report(session_id: str, request_id: str, report: dict,
                input_type: str, s3_key: str = "") -> bool:
    """
    Save a completed report to DynamoDB.
    Returns True on success, False on failure.
    """
    if not is_available():
        return False

    try:
        created_at = datetime.utcnow().isoformat()
        item = {
            "session_id": session_id,
            "created_at": created_at,
            "request_id": request_id,
            "input_type": input_type,
            "report": json.dumps(report),       # store as JSON string
            "human_verified": False,
            "s3_key": s3_key,
            "note_length": len(report.get("plain_english_summary", "")),
        }
        get_table().put_item(Item=item)
        print(f"✅ DynamoDB saved: {session_id} / {created_at}")
        return True

    except ClientError as e:
        print(f"❌ DynamoDB save failed: {e}")
        return False


# ── Get all reports for a session ─────────────────────────────────────────
def get_history(session_id: str) -> list:
    """
    Get all reports for a session, newest first.
    Returns list of report dicts.
    """
    if not is_available():
        return []

    try:
        response = get_table().query(
            KeyConditionExpression="session_id = :sid",
            ExpressionAttributeValues={":sid": session_id},
            ScanIndexForward=False,   # newest first
            Limit=50                  # max 50 reports per session
        )
        items = response.get("Items", [])

        # Parse report JSON string back to dict
        results = []
        for item in items:
            try:
                item["report"] = json.loads(item["report"])
                results.append(item)
            except Exception:
                continue

        return results

    except ClientError as e:
        print(f"❌ DynamoDB get failed: {e}")
        return []


# ── Mark report as verified ───────────────────────────────────────────────
def mark_verified(session_id: str, created_at: str) -> bool:
    """Mark a specific report as human verified."""
    if not is_available():
        return False

    try:
        get_table().update_item(
            Key={"session_id": session_id, "created_at": created_at},
            UpdateExpression="SET human_verified = :v",
            ExpressionAttributeValues={":v": True}
        )
        return True
    except ClientError as e:
        print(f"❌ DynamoDB update failed: {e}")
        return False


# ── Delete a report ───────────────────────────────────────────────────────
def delete_report(session_id: str, created_at: str) -> bool:
    """Delete a specific report."""
    if not is_available():
        return False

    try:
        get_table().delete_item(
            Key={"session_id": session_id, "created_at": created_at}
        )
        return True
    except ClientError as e:
        print(f"❌ DynamoDB delete failed: {e}")
        return False


# ── Check if DynamoDB is configured ──────────────────────────────────────
def is_available() -> bool:
    try:
        if not os.getenv("AWS_ACCESS_KEY_ID"):
            return False
        if not os.getenv("DYNAMODB_TABLE"):
            return False
        return True
    except Exception:
        return False