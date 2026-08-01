"""
Data loader module for WhatsApp Message Notification Router.
Loads and indexes all dataset CSV files into structured DataFrames and lookup maps.
"""

from typing import Any, Dict
import pandas as pd
import config


class DatasetContainer:
    """Container holding all loaded DataFrames and structured lookups."""

    def __init__(self, dataframes: Dict[str, pd.DataFrame]):
        self.messages = dataframes.get("messages", pd.DataFrame())
        self.sample_messages = dataframes.get("sample_messages", pd.DataFrame())
        self.users = dataframes.get("users", pd.DataFrame())
        self.groups = dataframes.get("groups", pd.DataFrame())
        self.group_members = dataframes.get("group_members", pd.DataFrame())
        self.business_accounts = dataframes.get("business_accounts", pd.DataFrame())
        self.user_business_history = dataframes.get("user_business_history", pd.DataFrame())
        self.message_history = dataframes.get("message_history", pd.DataFrame())
        self.message_events = dataframes.get("message_events", pd.DataFrame())
        self.images = dataframes.get("images", pd.DataFrame())
        self.voice_notes = dataframes.get("voice_notes", pd.DataFrame())
        self.daily_notification_summary = dataframes.get(
            "daily_notification_summary", pd.DataFrame()
        )

        # Build index lookups for fast retrieval
        self._build_lookups()

    def _build_lookups(self):
        # Users indexed by user_id
        if not self.users.empty and "user_id" in self.users.columns:
            self.user_map = self.users.set_index("user_id").to_dict(orient="index")
        else:
            self.user_map = {}

        # Groups indexed by group_id
        if not self.groups.empty and "group_id" in self.groups.columns:
            self.group_map = self.groups.set_index("group_id").to_dict(orient="index")
        else:
            self.group_map = {}

        # Group members indexed by (group_id, user_id) tuple
        self.group_member_map = {}
        if not self.group_members.empty:
            for _, row in self.group_members.iterrows():
                key = (row.get("group_id"), row.get("user_id"))
                self.group_member_map[key] = row.to_dict()

        # Business accounts indexed by business_id
        if not self.business_accounts.empty and "business_id" in self.business_accounts.columns:
            self.business_map = self.business_accounts.set_index("business_id").to_dict(
                orient="index"
            )
        else:
            self.business_map = {}

        # User-Business history indexed by (user_id, business_id) tuple
        self.user_business_map = {}
        if not self.user_business_history.empty:
            for _, row in self.user_business_history.iterrows():
                key = (row.get("user_id"), row.get("business_id"))
                self.user_business_map[key] = row.to_dict()

        # Images indexed by image_id
        if not self.images.empty and "image_id" in self.images.columns:
            self.image_map = self.images.set_index("image_id").to_dict(orient="index")
        else:
            self.image_map = {}

        # Voice notes indexed by voice_note_id
        if not self.voice_notes.empty and "voice_note_id" in self.voice_notes.columns:
            self.voice_note_map = self.voice_notes.set_index("voice_note_id").to_dict(
                orient="index"
            )
        else:
            self.voice_note_map = {}


def load_all_datasets() -> DatasetContainer:
    """Loads all CSV files in dataset directory into a DatasetContainer."""
    files_to_load = {
        "messages": config.MESSAGES_FILE,
        "sample_messages": config.SAMPLE_MESSAGES_FILE,
        "users": config.USERS_FILE,
        "groups": config.GROUPS_FILE,
        "group_members": config.GROUP_MEMBERS_FILE,
        "business_accounts": config.BUSINESS_ACCOUNTS_FILE,
        "user_business_history": config.USER_BUSINESS_HISTORY_FILE,
        "message_history": config.MESSAGE_HISTORY_FILE,
        "message_events": config.MESSAGE_EVENTS_FILE,
        "images": config.IMAGES_FILE,
        "voice_notes": config.VOICE_NOTES_FILE,
        "daily_notification_summary": config.DAILY_NOTIFICATION_SUMMARY_FILE,
    }

    dataframes = {}
    for name, filepath in files_to_load.items():
        if filepath.exists():
            dataframes[name] = pd.read_csv(filepath)
        else:
            dataframes[name] = pd.DataFrame()

    return DatasetContainer(dataframes)
