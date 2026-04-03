import boto3
import json
import os
import time
import traceback
from datetime import datetime
from botocore.exceptions import ClientError

# ── CloudWatch client (lazy loaded) ──────────────────────────────────────
_cw_client = None
_log_group = "/triageai/backend"
_log_stream = None

def get_cw():
    global _cw_client
    if _cw_client is None:
        _cw_client = boto3.client(
            "logs",
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name=os.getenv("AWS_REGION", "us-east-1")
        )
    return _cw_client


def _get_stream():
    """Get or create today's log stream."""
    global _log_stream
    stream_name = datetime.utcnow().strftime("%Y/%m/%d")

    if _log_stream == stream_name:
        return stream_name

    try:
        get_cw().create_log_stream(
            logGroupName=_log_group,
            logStreamName=stream_name
        )
    except ClientError as e:
        if e.response["Error"]["Code"] != "ResourceAlreadyExistsException":
            print(f"CloudWatch stream error: {e}")

    _log_stream = stream_name
    return stream_name


def _send(event_type: str, data: dict):
    """Send one structured log event to CloudWatch."""
    if not os.getenv("AWS_ACCESS_KEY_ID"):
        # CloudWatch not configured — just print locally
        print(f"[{event_type}] {json.dumps(data)}")
        return

    try:
        stream = _get_stream()
        message = json.dumps({
            "event": event_type,
            "timestamp": datetime.utcnow().isoformat(),
            **data
        })
        get_cw().put_log_events(
            logGroupName=_log_group,
            logStreamName=stream,
            logEvents=[{
                "timestamp": int(time.time() * 1000),
                "message": message
            }]
        )
    except Exception as e:
        # Never crash the app because of logging failure
        print(f"CloudWatch log failed: {e}")


# ── Public logging functions ──────────────────────────────────────────────

def log_request(input_type: str, note_length: int, request_id: str):
    """Log incoming analysis request."""
    _send("REQUEST_START", {
        "request_id": request_id,
        "input_type": input_type,
        "note_length": note_length,
    })


def log_node(node_name: str, duration_ms: float, errors: list, request_id: str):
    """Log completion of each LangGraph node."""
    _send("NODE_COMPLETE", {
        "request_id": request_id,
        "node": node_name,
        "duration_ms": round(duration_ms, 2),
        "has_errors": len(errors) > 0,
        "errors": errors
    })


def log_report(report: dict, input_type: str, total_ms: float, request_id: str):
    """Log final report metrics."""
    pc = report.get("patient_card", {})
    _send("REPORT_COMPLETE", {
        "request_id": request_id,
        "input_type": input_type,
        "total_duration_ms": round(total_ms, 2),
        "conditions_count": len(pc.get("conditions_with_codes", [])),
        "medications_count": len(pc.get("medications", [])),
        "drug_warnings_count": len(report.get("drug_warnings", [])),
        "has_vitals": bool(pc.get("vitals")),
        "confidence_flags_count": len(report.get("confidence_flags", [])),
        "pipeline_errors": report.get("errors", [])
    })


def log_error(error: str, context: str, request_id: str):
    """Log any error in the pipeline."""
    _send("ERROR", {
        "request_id": request_id,
        "context": context,
        "error": error,
        "traceback": traceback.format_exc()
    })


def log_file_upload(filename: str, file_size_kb: float, s3_key: str, request_id: str):
    """Log file upload event."""
    _send("FILE_UPLOAD", {
        "request_id": request_id,
        "filename": filename,
        "file_size_kb": round(file_size_kb, 2),
        "s3_key": s3_key
    })


def is_available() -> bool:
    """Check if CloudWatch is configured."""
    return bool(os.getenv("AWS_ACCESS_KEY_ID"))