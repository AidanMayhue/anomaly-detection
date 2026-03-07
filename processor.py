#!/usr/bin/env python3
import json
import io
import logging
import boto3
import pandas as pd
from datetime import datetime

from baseline import BaselineManager
from detector import AnomalyDetector

s3 = boto3.client("s3")
logger = logging.getLogger("processor")

NUMERIC_COLS = ["temperature", "humidity", "pressure", "wind_speed"]  # students configure this


def sync_log_to_s3(bucket: str, log_file: str = "anomaly_detection.log"):
    """Upload the current log file to S3 alongside the baseline."""
    try:
        log_key = "logs/anomaly_detection.log"
        with open(log_file, "rb") as f:
            s3.put_object(
                Bucket=bucket,
                Key=log_key,
                Body=f.read(),
                ContentType="text/plain"
            )
        logger.info("Log file synced to s3://%s/%s", bucket, log_key)
    except Exception as e:
        logger.error("Failed to sync log file to S3: %s", e)


def process_file(bucket: str, key: str):
    logger.info("Starting processing: s3://%s/%s", bucket, key)

    # 1. Download raw file
    try:
        response = s3.get_object(Bucket=bucket, Key=key)
        df = pd.read_csv(io.BytesIO(response["Body"].read()))
        logger.info("Loaded %d rows from %s | columns: %s", len(df), key, list(df.columns))
    except Exception as e:
        logger.error("Failed to download or parse file s3://%s/%s: %s", bucket, key, e)
        return

    # 2. Load current baseline
    try:
        baseline_mgr = BaselineManager(bucket=bucket)
        baseline = baseline_mgr.load()
    except Exception as e:
        logger.error("Failed to load baseline: %s", e)
        return

    # 3. Update baseline with values from this batch BEFORE scoring
    #    (use only non-null values for each channel)
    try:
        for col in NUMERIC_COLS:
            if col in df.columns:
                clean_values = df[col].dropna().tolist()
                if clean_values:
                    baseline = baseline_mgr.update(baseline, col, clean_values)
    except Exception as e:
        logger.error("Failed to update baseline for file %s: %s", key, e)
        return

    # 4. Run detection
    try:
        detector = AnomalyDetector(z_threshold=3.0, contamination=0.05)
        scored_df = detector.run(df, NUMERIC_COLS, baseline, method="both")
    except Exception as e:
        logger.error("Detection failed for file %s: %s", key, e)
        return

    # 5. Write scored file to processed/ prefix
    try:
        output_key = key.replace("raw/", "processed/")
        csv_buffer = io.StringIO()
        scored_df.to_csv(csv_buffer, index=False)
        s3.put_object(
            Bucket=bucket,
            Key=output_key,
            Body=csv_buffer.getvalue(),
            ContentType="text/csv"
        )
        logger.info("Scored file written to s3://%s/%s", bucket, output_key)
    except Exception as e:
        logger.error("Failed to write scored file for %s: %s", key, e)
        return

    # 6. Save updated baseline back to S3 and sync log
    try:
        baseline_mgr.save(baseline)
        sync_log_to_s3(bucket)
    except Exception as e:
        logger.error("Failed to save baseline or sync log for file %s: %s", key, e)

    # 7. Build and return a processing summary
    try:
        anomaly_count = int(scored_df["anomaly"].sum()) if "anomaly" in scored_df else 0
        summary = {
            "source_key": key,
            "output_key": output_key,
            "processed_at": datetime.utcnow().isoformat(),
            "total_rows": len(df),
            "anomaly_count": anomaly_count,
            "anomaly_rate": round(anomaly_count / len(df), 4) if len(df) > 0 else 0,
            "baseline_observation_counts": {
                col: baseline.get(col, {}).get("count", 0) for col in NUMERIC_COLS
            }
        }

        summary_key = output_key.replace(".csv", "_summary.json")
        s3.put_object(
            Bucket=bucket,
            Key=summary_key,
            Body=json.dumps(summary, indent=2),
            ContentType="application/json"
        )

        logger.info(
            "Processing complete -- %s | anomalies: %d/%d (%.1f%%)",
            key, anomaly_count, len(df), summary["anomaly_rate"] * 100
        )
        return summary

    except Exception as e:
        logger.error("Failed to write summary for file %s: %s", key, e)
