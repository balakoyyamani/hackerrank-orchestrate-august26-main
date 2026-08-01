"""
Notification router module for WhatsApp Message Notification Router.
Combines a weighted scoring rule engine, dynamic confidence calculation, Gemini 2.5 Flash API fallback, and decision fusion.
"""

import logging
import re
from typing import Dict, Any, Tuple, List, Optional
import config
from gemini_client import GeminiRouterClient

logger = logging.getLogger(__name__)


class NotificationRouter:
    """Hybrid notification router executing rule engine scoring, Gemini 2.5 Flash API calls, and decision fusion."""

    def __init__(self, gemini_client: Optional[GeminiRouterClient] = None):
        self.gemini_client = gemini_client or GeminiRouterClient()
        self.fusion_threshold = config.FUSION_RULE_CONFIDENCE_THRESHOLD

    def route_message(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Executes decision fusion routing for an incoming message."""
        # 1. Rule Engine Prediction & Dynamic Confidence
        rule_pred = self._evaluate_rule_engine(features)
        rule_confidence = rule_pred["confidence"]

        # 2. Decision Fusion: If rule confidence >= threshold, use rule prediction directly
        if rule_confidence >= self.fusion_threshold:
            rule_pred["decision_type"] = "rule_engine"
            return rule_pred

        # 3. Low confidence: Attempt Gemini 2.5 Flash Multimodal Reasoning
        logger.info(
            f"Rule confidence ({rule_confidence:.2f}) < {self.fusion_threshold}. Requesting Gemini API reasoning..."
        )
        gemini_pred = self.gemini_client.route_message_multimodal(features)

        if gemini_pred:
            gemini_pred["decision_type"] = "gemini_fusion"
            return gemini_pred

        # 4. Fallback to Rule Prediction if Gemini API call unavailable or fails
        rule_pred["decision_type"] = "rule_engine_fallback"
        return rule_pred

    def _evaluate_rule_engine(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Calculates weighted category scores and dynamic confidence using a priority-ordered rule cascade."""
        text = features.get("effective_text", "").lower()
        conv_type = features.get("conversation_type", "")
        forwarded_count = features.get("forwarded_count", 0)
        is_admin = features.get("is_sender_group_admin", False)
        is_user_muted = features.get("is_user_group_muted", False)
        is_biz_verified = features.get("is_business_verified", False)
        has_biz_rel = features.get("has_biz_relationship", False)
        is_prompt_injection = features.get("is_prompt_injection", False)
        evidence_list = features.get("evidence_ids", [])
        evidence_str = ";".join(evidence_list) if evidence_list else "none"

        # Feature scores from feature extractor
        scam_score = features.get("scam_indicator_score", 0.0)
        urgency_score = features.get("urgency_score", 0.0)
        financial_score = features.get("financial_importance_score", 0.0)
        delivery_score = features.get("delivery_status_score", 0.0)
        promotion_score = features.get("promotion_indicator_score", 0.0)
        personal_relevance = features.get("personal_relevance_score", 0.0)

        # ─────────────────────────────────────────────
        # RULE 0: Prompt Injection — ALWAYS MUTE
        # ─────────────────────────────────────────────
        if is_prompt_injection:
            return {
                "message_id": features["message_id"],
                "action": "mute",
                "message_type": "scam",
                "reason": "Prompt injection attack detected: message attempts to manipulate the notification router.",
                "confidence": 0.98,
                "evidence_message_ids": evidence_str,
            }

        # ─────────────────────────────────────────────
        # RULE 1: High-Confidence Scam / Phishing
        # ─────────────────────────────────────────────
        if scam_score >= 0.65:
            dyn_conf = min(0.97, 0.82 + scam_score * 0.18)
            # Determine scam type: generic spam vs targeted scam
            is_targeted_scam = any(
                kw in text for kw in [
                    "otp", "pin", "login code", "verification code", "bank details",
                    "account number", "wallet pin", "kyc", "6 digit", "account block",
                    "account suspended", "otp leak", "account bachane", "otp batao",
                    "profile band", "account band", "link open karo", "verify now",
                ]
            )
            m_type = "scam" if is_targeted_scam else "spam"
            return {
                "message_id": features["message_id"],
                "action": "mute",
                "message_type": m_type,
                "reason": "High scam indicator score: phishing, credential theft, or social engineering patterns detected.",
                "confidence": round(dyn_conf, 2),
                "evidence_message_ids": evidence_str,
            }

        # ─────────────────────────────────────────────
        # RULE 2: Medium Scam (0.35–0.65) with suspicious link context
        # ─────────────────────────────────────────────
        if scam_score >= 0.35 and (
            "link" in text or "click" in text or "open" in text or "verify" in text
        ):
            dyn_conf = min(0.90, 0.72 + scam_score * 0.20)
            is_financial_scam = any(
                kw in text for kw in [
                    "loan approved", "pay processing fee", "payment link", "scan the qr",
                    "token today", "scan and pay", "pay clearance", "service reactivation",
                    "reattempt charge", "payout profile", "verify wallet",
                ]
            )
            m_type = "scam" if is_financial_scam else "spam"
            return {
                "message_id": features["message_id"],
                "action": "mute",
                "message_type": m_type,
                "reason": "Message requests sensitive action (click/verify) with suspicious scam indicators.",
                "confidence": round(dyn_conf, 2),
                "evidence_message_ids": evidence_str,
            }

        # ─────────────────────────────────────────────
        # RULE 3: Spam / Chain Forwarding
        # ─────────────────────────────────────────────
        if forwarded_count >= 7:
            # Very high forwarding = chain message / viral spam
            dyn_conf = min(0.94, 0.82 + min(forwarded_count / 15.0, 0.12))
            # Is it a religious/blessing chain message?
            is_blessing = any(kw in text for kw in [
                "blessings", "bhagwan", "positive energy", "smile today", "stay blessed",
                "share this", "forward to", "share in family", "share with everyone",
                "good vibes", "good luck",
            ])
            m_type = "greeting" if is_blessing else "forward"
            return {
                "message_id": features["message_id"],
                "action": "mute",
                "message_type": m_type,
                "reason": "Highly forwarded chain message or viral spam with no personal relevance.",
                "confidence": round(dyn_conf, 2),
                "evidence_message_ids": evidence_str,
            }

        if forwarded_count >= 4 and scam_score >= 0.2:
            # Moderately forwarded with some scam indicators = suspicious
            dyn_conf = min(0.90, 0.75 + (forwarded_count / 20.0) + scam_score * 0.10)
            return {
                "message_id": features["message_id"],
                "action": "mute",
                "message_type": "spam",
                "reason": "Forwarded message with suspicious content patterns detected.",
                "confidence": round(dyn_conf, 2),
                "evidence_message_ids": evidence_str,
            }

        # ─────────────────────────────────────────────
        # RULE 4: Trusted Admin Time-Sensitive Operational Notice
        # ─────────────────────────────────────────────
        if is_admin and urgency_score >= 0.3 and scam_score < 0.35:
            dyn_conf = min(0.94, 0.80 + urgency_score * 0.15)
            # Classify the admin message type
            if any(kw in text for kw in ["bus", "bus leaving", "bus is leaving", "stadium", "school"]):
                m_type = "event"
                reason = "Trusted admin sent a time-sensitive event update."
            elif any(kw in text for kw in ["tanker", "valve", "water", "motor room"]):
                m_type = "urgent"
                reason = "Trusted admin sent a critical infrastructure/utility alert."
            elif any(kw in text for kw in ["gate", "lift", "maintenance", "fire alarm", "repair"]):
                m_type = "urgent"
                reason = "Trusted admin sent a building/facility operational notice."
            elif any(kw in text for kw in ["payment", "fee", "receipt"]):
                m_type = "payment"
                reason = "Trusted admin sent a payment-related notice."
            else:
                m_type = "urgent"
                reason = "Trusted group admin sent a time-sensitive operational update."
            return {
                "message_id": features["message_id"],
                "action": "notify",
                "message_type": m_type,
                "reason": reason,
                "confidence": round(dyn_conf, 2),
                "evidence_message_ids": evidence_str,
            }

        # ─────────────────────────────────────────────
        # RULE 5: High Urgency (non-admin, personal or group)
        # ─────────────────────────────────────────────
        if urgency_score >= 0.55 and scam_score < 0.35:
            dyn_conf = min(0.92, 0.75 + urgency_score * 0.18)
            # Determine message type more precisely
            if any(kw in text for kw in ["bus", "stadium", "school"]):
                m_type = "event"
            elif any(kw in text for kw in ["build", "prod", "deployment", "rollback", "client", "server"]):
                m_type = "urgent"
            elif any(kw in text for kw in ["call me", "come online", "clinic", "doctor", "specialist"]):
                m_type = "personal"
                dyn_conf = min(0.91, dyn_conf)
            else:
                m_type = "urgent"
            return {
                "message_id": features["message_id"],
                "action": "notify",
                "message_type": m_type,
                "reason": "Time-sensitive message with high urgency indicators.",
                "confidence": round(dyn_conf, 2),
                "evidence_message_ids": evidence_str,
            }

        # ─────────────────────────────────────────────
        # RULE 6: Legitimate Financial / Payment Notifications
        # ─────────────────────────────────────────────
        if financial_score >= 0.45 and scam_score < 0.25:
            dyn_conf = min(0.91, 0.75 + financial_score * 0.18)
            action = "notify" if (has_biz_rel or is_biz_verified) else "digest"
            # Direct maintenance/school fee payments in personal context get notify
            if conv_type in ("group", "personal") and any(
                kw in text for kw in ["maintenance payment", "fee receipt", "school", "payment due"]
            ):
                action = "notify"
            return {
                "message_id": features["message_id"],
                "action": action,
                "message_type": "payment",
                "reason": "Legitimate financial transaction or payment notification.",
                "confidence": round(dyn_conf, 2),
                "evidence_message_ids": evidence_str,
            }

        # ─────────────────────────────────────────────
        # RULE 7: Verified Delivery Status Notifications
        # ─────────────────────────────────────────────
        if delivery_score >= 0.45 and scam_score < 0.25:
            dyn_conf = min(0.91, 0.75 + delivery_score * 0.18)
            # Very high delivery score = near-certain real delivery notification → notify
            # Moderate score without verification → digest
            action = "notify" if (has_biz_rel or is_biz_verified or delivery_score >= 0.70) else "digest"
            return {
                "message_id": features["message_id"],
                "action": action,
                "message_type": "business_update",
                "reason": "Legitimate order fulfillment or delivery status notification.",
                "confidence": round(dyn_conf, 2),
                "evidence_message_ids": evidence_str,
            }

        # ─────────────────────────────────────────────
        # RULE 8: Promotional Marketing (business channel)
        # ─────────────────────────────────────────────
        if promotion_score >= 0.45:
            dyn_conf = min(0.91, 0.70 + promotion_score * 0.22)
            # Digest if user has opted in / has relationship, mute otherwise
            action = "digest" if has_biz_rel else "mute"
            return {
                "message_id": features["message_id"],
                "action": action,
                "message_type": "promotion",
                "reason": "Commercial marketing offer from a business account.",
                "confidence": round(dyn_conf, 2),
                "evidence_message_ids": evidence_str,
            }

        # ─────────────────────────────────────────────
        # RULE 9: Low-Substance Greetings
        # ─────────────────────────────────────────────
        is_greeting = any(kw in text for kw in [
            "good morning", "good evening", "happy weekend", "have a nice day",
            "good morning beta", "have a good day", "stay positive", "good vibes",
        ])
        if is_greeting and len(text) < 120:
            # Forwarded greetings → mute; direct greetings → digest
            action = "mute" if forwarded_count >= 3 else "digest"
            dyn_conf = 0.85 if forwarded_count >= 3 else 0.82
            return {
                "message_id": features["message_id"],
                "action": action,
                "message_type": "greeting",
                "reason": "Routine social greeting message with no urgent content.",
                "confidence": dyn_conf,
                "evidence_message_ids": evidence_str,
            }

        # ─────────────────────────────────────────────
        # RULE 10: Personal Messages — Context Dependent
        # ─────────────────────────────────────────────
        # Direct @ mention in a group = higher priority
        user_id = features.get("user_id", "")
        is_mentioned = f"@{user_id}" in features.get("effective_text", "") or f"@{user_id}" in features.get("effective_text", "").lower()

        if is_mentioned or personal_relevance >= 0.7:
            # Directly addressed message → notify unless muted
            action = "digest" if is_user_muted else "notify"
            dyn_conf = 0.85
            return {
                "message_id": features["message_id"],
                "action": action,
                "message_type": "personal",
                "reason": "Message directly mentions or is addressed to the user.",
                "confidence": dyn_conf,
                "evidence_message_ids": evidence_str,
            }

        # 7. General messages — below fusion threshold → Gemini reasoning
        action = "digest" if is_user_muted else "notify"
        dyn_conf = 0.68  # Below 0.90 threshold, prompting Gemini API decision fusion

        # Additional signal: if conv_type is personal and scam_score is low → higher confidence notify
        if conv_type == "personal" and scam_score < 0.15:
            dyn_conf = 0.75

        return {
            "message_id": features["message_id"],
            "action": action,
            "message_type": "personal",
            "reason": "Standard message requiring contextual analysis.",
            "confidence": dyn_conf,
            "evidence_message_ids": evidence_str,
        }
