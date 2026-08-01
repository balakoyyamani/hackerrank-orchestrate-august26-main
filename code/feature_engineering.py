"""
Feature engineering module for WhatsApp Message Notification Router.
Aggregates contextual signals and quantitative feature scores from messages, users, groups, business accounts, and historical records.
"""

import re
from typing import Dict, Any, List, Optional
import pandas as pd
from data_loader import DatasetContainer
from multimodal import MultimodalProcessor


class FeatureExtractor:
    """Builds unified contextual feature representations and quantitative scores for incoming messages."""

    # Known scam/phishing domain fragments
    SUSPICIOUS_DOMAINS = [
        "account-login.in", "account-help.in", "pay-check-secure", "amazonpay-delivery.in",
        "chase-secure-alert", "bit.ly/verify", "kyc-update", "verify-quick",
    ]

    # Prompt injection indicators that should flag a message as scam
    PROMPT_INJECTION_PATTERNS = [
        r"ignore\s+(all\s+)?(previous|prior)\s+(routing\s+)?(rules|instructions)",
        r"set\s+action\s*=",
        r"mark\s+(as\s+)?(notify|mute|digest)",
        r"system\s+note\s+for\s+(the\s+)?notification\s+router",
        r"routing\s+override",
        r"internal\s+router\s+metadata",
        r"verified_business\s*=\s*true",
        r"user_priority\s*=",
        r"assistant\s+instruction",
        r"sender\s+is\s+trusted\s+admin",
        r"ignore\s+sender\s+risk",
        r"always\s+mark\s+this\s+as",
    ]

    # High-confidence scam keyword patterns (text content, language-agnostic)
    SCAM_PATTERNS_HIGH = [
        # English
        "lottery", "prize", "won $", "claim reward", "kyc update",
        "bank blocked", "send pin", "send otp", "account suspended",
        "double your money", "wire transfer", "crypto investment",
        "verify your otp", "login code", "6 digit", "verification code",
        "wallet pin", "bank details", "fill bank details",
        "send screenshot after submission",
        # QR payment / clearance scams
        "scan this qr", "pay the clearance amount", "clearance amount",
        "access card may be blocked", "penalty list",
        # Wallet/link scams
        "check the wallet details from the link",
        "wallet details from the link",
        "release the amount today",
        "verify wallet and card details",
        # Hindi transliteration
        "otp leak", "account bachane", "account block ho jayega",
        "otp batao", "link open karo", "aur code daal do",
        "jaldi kar lo", "profile band ho jayega", "account band",
        "link open karke code", "verification code abhi confirm",
        # Loan/investment scams
        "loan approved", "pay processing fee", "amount will be released",
        "token today to block", "registry papers will be shared after payment",
        "benefit approval is pending", "fill bank details on first page",
        # Fake OTP scams
        "otp verify nahi hua", "otp verification is pending",
        "otp verification failed",
        "share your account number",
        "claim benefits by sharing",
        "approval window closes today, send details",
        # Fake support alerts
        "account-login.in", "account-help.in", "pay-check-secure",
        "amazonpay-delivery.in", "chase-secure-alert",
        # Refund scams
        "refund could not be processed automatically",
        "food order refund",
        # International payout scams
        "international payout profile", "final verification step",
        "payout request can continue",
        # Service fee scams
        "service reactivation fee", "scan the qr and send screenshot",
        "pay today to avoid account lock",
    ]

    SCAM_PATTERNS_MEDIUM = [
        "click link", "click here immediately", "limited window",
        "complete verification", "pay clearance amount", "scan the qr",
        "scan and pay", "payout profile needs",
    ]

    def __init__(self, dataset: DatasetContainer):
        self.ds = dataset
        self.multimodal = MultimodalProcessor()
        self._compiled_injection_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.PROMPT_INJECTION_PATTERNS
        ]

    def extract_features(self, message_row: pd.Series) -> Dict[str, Any]:
        """Extracts comprehensive feature dictionary and quantitative score vectors for a single message."""
        msg_id = str(message_row.get("message_id", ""))
        user_id = str(message_row.get("user_id", ""))
        conv_type = str(message_row.get("conversation_type", ""))
        group_id = str(message_row.get("group_id", "")) if pd.notna(message_row.get("group_id")) else ""
        business_id = str(message_row.get("business_id", "")) if pd.notna(message_row.get("business_id")) else ""
        sender_id = str(message_row.get("sender_user_id", "")) if pd.notna(message_row.get("sender_user_id")) else ""
        created_at = str(message_row.get("created_at", ""))
        raw_text = str(message_row.get("message_text", "")) if pd.notna(message_row.get("message_text")) else ""
        media_type = str(message_row.get("media_type", "")) if pd.notna(message_row.get("media_type")) else ""
        media_id = str(message_row.get("media_id", "")) if pd.notna(message_row.get("media_id")) else ""
        forwarded_count = int(message_row.get("forwarded_count", 0)) if pd.notna(message_row.get("forwarded_count")) else 0

        # Receiver user features
        user_profile = self.ds.user_map.get(user_id, {})
        user_quiet_hours = str(user_profile.get("quiet_hours", ""))
        user_reports = int(user_profile.get("report_count", 0))

        # Multimodal content extraction
        effective_text = raw_text
        pil_image = None
        audio_path = None

        if media_type == "image" and media_id:
            img_info = self.ds.image_map.get(media_id, {})
            extracted_img_text = self.multimodal.extract_image_content(media_id, img_info)
            effective_text = f"{raw_text} {extracted_img_text}".strip()
            pil_image = self.multimodal.load_pil_image(media_id, img_info)
        elif media_type == "voice" and media_id:
            voice_info = self.ds.voice_note_map.get(media_id, {})
            extracted_voice_text = self.multimodal.extract_voice_note_content(media_id, voice_info)
            effective_text = f"{raw_text} {extracted_voice_text}".strip()
            audio_path = self.multimodal.get_audio_file_path(media_id, voice_info)

        # Group context features
        group_profile = self.ds.group_map.get(group_id, {})
        group_member_info = self.ds.group_member_map.get((group_id, user_id), {})
        is_sender_group_admin = (
            sender_id in str(group_profile.get("admins", "")).split(";")
            if group_profile.get("admins")
            else False
        )
        is_user_group_muted = bool(group_member_info.get("is_muted", False))

        # Business context features
        business_profile = self.ds.business_map.get(business_id, {})
        is_business_verified = bool(business_profile.get("is_verified", False))
        user_biz_history = self.ds.user_business_map.get((user_id, business_id), {})
        has_biz_relationship = bool(user_biz_history.get("has_active_relationship", False))

        text_lower = effective_text.lower()

        # --- Prompt Injection Detection (MUST RUN FIRST) ---
        is_prompt_injection = self._detect_prompt_injection(text_lower)

        # --- Quantitative Contextual Feature Scores ---
        scam_score = self._calc_scam_score(
            text_lower, forwarded_count, is_business_verified, business_profile, is_prompt_injection
        )
        urgency_score = self._calc_urgency_score(text_lower, is_sender_group_admin, conv_type)
        financial_score = self._calc_financial_score(text_lower)
        delivery_score = self._calc_delivery_score(text_lower)
        personal_relevance_score = self._calc_personal_relevance(text_lower, user_id, conv_type, group_member_info)
        promotion_score = self._calc_promotion_score(text_lower, conv_type, has_biz_relationship)

        # Historical Evidence Lookup
        evidence_ids = self._find_historical_evidence(
            user_id=user_id,
            conv_type=conv_type,
            group_id=group_id,
            business_id=business_id,
            sender_id=sender_id,
            effective_text=effective_text,
        )

        return {
            "message_id": msg_id,
            "user_id": user_id,
            "conversation_type": conv_type,
            "group_id": group_id,
            "business_id": business_id,
            "sender_user_id": sender_id,
            "created_at": created_at,
            "raw_text": raw_text,
            "effective_text": effective_text,
            "media_type": media_type,
            "media_id": media_id,
            "pil_image": pil_image,
            "audio_path": audio_path,
            "forwarded_count": forwarded_count,
            "user_quiet_hours": user_quiet_hours,
            "user_reports": user_reports,
            "group_type": group_profile.get("group_type", ""),
            "is_sender_group_admin": is_sender_group_admin,
            "is_user_group_muted": is_user_group_muted,
            "is_business_verified": is_business_verified,
            "has_biz_relationship": has_biz_relationship,
            "is_prompt_injection": is_prompt_injection,
            # Contextual Scores
            "urgency_score": urgency_score,
            "financial_importance_score": financial_score,
            "delivery_status_score": delivery_score,
            "personal_relevance_score": personal_relevance_score,
            "scam_indicator_score": scam_score,
            "promotion_indicator_score": promotion_score,
            "evidence_ids": evidence_ids,
        }

    def _detect_prompt_injection(self, text: str) -> bool:
        """Detect attempts to manipulate the router via prompt injection in the message text."""
        for pattern in self._compiled_injection_patterns:
            if pattern.search(text):
                return True
        return False

    def _calc_urgency_score(self, text: str, is_admin: bool, conv_type: str) -> float:
        score = 0.0
        urgent_keywords = [
            "urgent", "emergency", "asap", "deadline", "today", "15 mins", "mins max",
            "tanker", "valve", "water supply", "prod review", "heads-up",
            "call me now", "call me urgently", "can't wait", "cannot wait",
            "need to decide", "10 minutes", "10 min", "20 mins", "in next ten minutes",
            "leaves in", "leaving in",
        ]
        matches = sum(1 for kw in urgent_keywords if kw in text)
        score += min(matches * 0.25, 0.6)
        if is_admin:
            score += 0.3
        if "sorry for the last-minute" in text or "need to close" in text:
            score += 0.2
        if "bus is leaving" in text or "bus leaving" in text:
            score += 0.4
        if conv_type == "personal" and ("call me" in text or "come online" in text):
            score += 0.25
        return min(score, 1.0)

    def _calc_financial_score(self, text: str) -> float:
        score = 0.0
        # Legitimate financial keywords (payments received, statements)
        legit_fin_keywords = [
            "payment", "invoice", "refund", "amount", "transaction", "receipt",
            "credited", "debited", "card statement", "monthly statement",
            "maintenance payment", "fee receipt",
        ]
        matches = sum(1 for kw in legit_fin_keywords if kw in text)
        score += min(matches * 0.3, 0.7)
        if re.search(r"\$\d+|\b\d+\s*(rs|inr|usd)\b", text):
            score += 0.2
        # Penalize if it's asking to click a link to pay (suspicious)
        if "check the wallet details from the link" in text:
            score = score * 0.3  # Heavily down-weight: this is likely a scam
        if "complete verification" in text and "link" in text:
            score = score * 0.4
        return min(score, 1.0)

    def _calc_delivery_score(self, text: str) -> float:
        score = 0.0
        deliv_keywords = [
            "packed", "hub", "delivery", "order ending", "expected to reach",
            "tracking", "courier", "dispatch", "shipment", "return pickup",
            "fedex", "amazon app", "check delivery", "delivery attempt",
        ]
        matches = sum(1 for kw in deliv_keywords if kw in text)
        score += min(matches * 0.35, 0.9)
        # Penalize fake delivery scam indicators
        if "reattempt charge" in text or "amazonpay-delivery" in text:
            score = score * 0.2
        return min(score, 1.0)

    def _calc_personal_relevance(self, text: str, user_id: str, conv_type: str, member_info: Dict[str, Any]) -> float:
        score = 0.5 if conv_type == "personal" else 0.2
        if f"@{user_id}" in text or f"@{user_id.lower()}" in text:
            score += 0.5
        if member_info.get("reply_ratio", 0) > 0.3:
            score += 0.2
        return min(score, 1.0)

    def _calc_scam_score(
        self, text: str, fwd_count: int, is_verified: bool, biz_profile: Dict[str, Any], is_prompt_injection: bool
    ) -> float:
        score = 0.0

        # Prompt injection = guaranteed maximum scam score
        if is_prompt_injection:
            return 1.0

        # Check high-confidence scam patterns
        high_matches = sum(1 for kw in self.SCAM_PATTERNS_HIGH if kw in text)
        score += min(high_matches * 0.45, 0.9)

        # Check medium-confidence scam patterns
        medium_matches = sum(1 for kw in self.SCAM_PATTERNS_MEDIUM if kw in text)
        score += min(medium_matches * 0.2, 0.4)

        # Check suspicious domains
        for domain in self.SUSPICIOUS_DOMAINS:
            if domain in text:
                score += 0.5
                break

        # High forwarding of suspicious messages
        if fwd_count >= 5:
            score += 0.2

        # Unverified business with bad reputation
        if biz_profile and not is_verified and biz_profile.get("report_count", 0) > 5:
            score += 0.3

        # Suspicious urgency + financial action combination (classic scam)
        if ("urgent" in text or "immediately" in text or "jaldi" in text) and (
            "link" in text or "otp" in text or "verify" in text or "account" in text
        ):
            score += 0.25

        return min(score, 1.0)

    def _calc_promotion_score(self, text: str, conv_type: str, has_biz_rel: bool) -> float:
        score = 0.0
        promo_keywords = [
            "off", "discount", "sale", "buy 1 get 1", "limited time", "exclusive deal",
            "special offer", "coupon", "flat 50%", "cashback", "welcome offer",
            "50% off", "40% off", "reply stop", "unsubscribe",
            "tap below to shop", "tap below to view",
        ]
        matches = sum(1 for kw in promo_keywords if kw in text)
        score += min(matches * 0.30, 0.7)
        if conv_type == "business" and not has_biz_rel:
            score += 0.3
        if "reply stop to unsubscribe" in text or "t&c apply" in text:
            score += 0.2
        return min(score, 1.0)

    def _find_historical_evidence(
        self,
        user_id: str,
        conv_type: str,
        group_id: str,
        business_id: str,
        sender_id: str,
        effective_text: str,
    ) -> List[str]:
        if self.ds.message_history.empty:
            return []

        df_hist = self.ds.message_history
        filtered = pd.DataFrame()

        if conv_type == "group" and group_id:
            filtered = df_hist[df_hist["group_id"] == group_id]
        elif conv_type == "business" and business_id:
            filtered = df_hist[df_hist["business_id"] == business_id]
        elif sender_id:
            filtered = df_hist[df_hist["sender_user_id"] == sender_id]
        else:
            filtered = df_hist[df_hist["user_id"] == user_id]

        if not filtered.empty:
            recent_ids = filtered["message_id"].dropna().astype(str).tolist()
            return recent_ids[-2:]

        return []
