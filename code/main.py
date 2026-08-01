"""
Main entry point for WhatsApp Message Notification Router.
Loads datasets, extracts contextual features, executes decision fusion routing, validates contract, and writes output.csv.
"""

import logging
import sys
from pathlib import Path

# Add code directory to sys.path
CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import pandas as pd
import config
from data_loader import load_all_datasets
from feature_engineering import FeatureExtractor
from router import NotificationRouter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def validate_predictions(df_out: pd.DataFrame, expected_count: int) -> bool:
    """Validates predictions DataFrame against the project contract in AGENTS.md §6."""
    if list(df_out.columns) != config.REQUIRED_OUTPUT_COLUMNS:
        raise ValueError(
            f"Columns mismatch! Expected {config.REQUIRED_OUTPUT_COLUMNS}, got {list(df_out.columns)}"
        )

    if len(df_out) != expected_count:
        raise ValueError(
            f"Row count mismatch! Expected {expected_count} rows, got {len(df_out)}"
        )

    invalid_actions = set(df_out["action"]) - config.ALLOWED_ACTIONS
    if invalid_actions:
        raise ValueError(f"Invalid action values found: {invalid_actions}")

    invalid_types = set(df_out["message_type"]) - config.ALLOWED_MESSAGE_TYPES
    if invalid_types:
        raise ValueError(f"Invalid message_type values found: {invalid_types}")

    if not df_out["confidence"].between(0.0, 1.0).all():
        raise ValueError("Confidence values must be numeric between 0.0 and 1.0")

    if df_out["evidence_message_ids"].isnull().any():
        raise ValueError("evidence_message_ids contains null values. Use 'none' instead.")

    return True


def run_pipeline() -> pd.DataFrame:
    """Runs end-to-end hybrid notification routing pipeline."""
    logger.info("Starting WhatsApp Notification Router Pipeline...")
    dataset = load_all_datasets()

    if dataset.messages.empty:
        raise FileNotFoundError(f"Messages file not found or empty at {config.MESSAGES_FILE}")

    logger.info(f"Loaded {len(dataset.messages)} messages to route.")

    feature_extractor = FeatureExtractor(dataset)
    router = NotificationRouter()

    predictions = []
    decision_counts = {"rule_engine": 0, "gemini_fusion": 0, "rule_engine_fallback": 0}

    for idx, row in dataset.messages.iterrows():
        features = feature_extractor.extract_features(row)
        pred = router.route_message(features)

        d_type = pred.pop("decision_type", "rule_engine")
        decision_counts[d_type] = decision_counts.get(d_type, 0) + 1
        predictions.append(pred)

    df_out = pd.DataFrame(predictions)[config.REQUIRED_OUTPUT_COLUMNS]

    # Validate output schema & values
    validate_predictions(df_out, len(dataset.messages))
    logger.info("Output schema and domain contract validation PASSED!")

    # Save predictions
    df_out.to_csv(config.OUTPUT_PATH, index=False)
    df_out.to_csv(config.DATASET_OUTPUT_PATH, index=False)

    logger.info(f"Decision Fusion Summary: {decision_counts}")
    logger.info(f"Successfully written {len(df_out)} predictions to:")
    logger.info(f" - {config.OUTPUT_PATH}")
    logger.info(f" - {config.DATASET_OUTPUT_PATH}")

    return df_out


if __name__ == "__main__":
    run_pipeline()
