"""
Configuration file for HackerRank Orchestrate Message Notification Router.
Defines dataset paths, allowed prediction categories, and project constants.
"""

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset"
MEDIA_DIR = DATASET_DIR / "media"
IMAGES_DIR = MEDIA_DIR / "images"
AUDIO_DIR = MEDIA_DIR / "audio"

# Output paths
OUTPUT_PATH = BASE_DIR / "output.csv"
DATASET_OUTPUT_PATH = DATASET_DIR / "output.csv"

# Dataset Files
MESSAGES_FILE = DATASET_DIR / "messages.csv"
SAMPLE_MESSAGES_FILE = DATASET_DIR / "sample_messages.csv"
USERS_FILE = DATASET_DIR / "users.csv"
GROUPS_FILE = DATASET_DIR / "groups.csv"
GROUP_MEMBERS_FILE = DATASET_DIR / "group_members.csv"
BUSINESS_ACCOUNTS_FILE = DATASET_DIR / "business_accounts.csv"
USER_BUSINESS_HISTORY_FILE = DATASET_DIR / "user_business_history.csv"
MESSAGE_HISTORY_FILE = DATASET_DIR / "message_history.csv"
MESSAGE_EVENTS_FILE = DATASET_DIR / "message_events.csv"
IMAGES_FILE = DATASET_DIR / "images.csv"
VOICE_NOTES_FILE = DATASET_DIR / "voice_notes.csv"
DAILY_NOTIFICATION_SUMMARY_FILE = DATASET_DIR / "daily_notification_summary.csv"

# Allowed schema values as defined by the problem statement contract
ALLOWED_ACTIONS = {"notify", "digest", "mute"}

ALLOWED_MESSAGE_TYPES = {
    "personal",
    "urgent",
    "event",
    "payment",
    "business_update",
    "promotion",
    "greeting",
    "forward",
    "spam",
    "scam",
    "unknown",
}

REQUIRED_OUTPUT_COLUMNS = [
    "message_id",
    "action",
    "message_type",
    "reason",
    "confidence",
    "evidence_message_ids",
]

# Gemini API & Decision Fusion Configuration
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemma-4-31b-it")
FUSION_RULE_CONFIDENCE_THRESHOLD = 0.90
MAX_API_RETRIES = 3
API_RETRY_DELAY = 1.0
