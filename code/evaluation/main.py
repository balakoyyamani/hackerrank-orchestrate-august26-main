"""
Evaluation module for WhatsApp Message Notification Router.
Calculates Accuracy, Precision, Recall, F1 Score (Macro/Micro), and Confusion Matrix for ground truth data.
"""

import sys
from pathlib import Path

# Add code directory to sys.path
CODE_DIR = Path(__file__).resolve().parent.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import pandas as pd
import numpy as np
import config
from data_loader import load_all_datasets
from feature_engineering import FeatureExtractor
from router import NotificationRouter


def compute_metrics(y_true: list, y_pred: list, labels: list) -> dict:
    """Computes Accuracy, Precision, Recall, F1 (Macro/Micro), and Confusion Matrix."""
    total = len(y_true)
    if total == 0:
        return {}

    # Convert to pandas for easy confusion matrix construction
    label_to_idx = {lbl: i for i, lbl in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=int)

    correct = 0
    for t, p in zip(y_true, y_pred):
        if t == p:
            correct += 1
        t_idx = label_to_idx.get(t)
        p_idx = label_to_idx.get(p)
        if t_idx is not None and p_idx is not None:
            cm[t_idx, p_idx] += 1

    accuracy = correct / total

    # Class-wise precision, recall, f1
    precisions = []
    recalls = []
    f1s = []
    support = []

    for i, lbl in enumerate(labels):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)
        support.append(cm[i, :].sum())

    macro_precision = np.mean(precisions) if precisions else 0.0
    macro_recall = np.mean(recalls) if recalls else 0.0
    macro_f1 = np.mean(f1s) if f1s else 0.0

    # Micro averages (for multi-class, micro-precision = micro-recall = accuracy)
    micro_f1 = accuracy

    return {
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "confusion_matrix": pd.DataFrame(cm, index=labels, columns=labels),
        "class_report": pd.DataFrame(
            {
                "Precision": precisions,
                "Recall": recalls,
                "F1-Score": f1s,
                "Support": support,
            },
            index=labels,
        ),
    }


def evaluate_on_sample():
    """Runs router predictions on sample_messages.csv and evaluates comprehensive classification metrics."""
    dataset = load_all_datasets()
    if dataset.sample_messages.empty:
        print(f"Sample messages file missing at {config.SAMPLE_MESSAGES_FILE}")
        return

    sample_df = dataset.sample_messages
    print(f"\nEvaluating Notification Router on {len(sample_df)} ground truth messages...")

    feature_extractor = FeatureExtractor(dataset)
    router = NotificationRouter()

    gt_actions = []
    pred_actions = []

    gt_types = []
    pred_types = []

    decision_counts = {"rule_engine": 0, "gemini_fusion": 0, "rule_engine_fallback": 0}

    for _, row in sample_df.iterrows():
        features = feature_extractor.extract_features(row)
        pred = router.route_message(features)

        gt_act = str(row.get("action", "")).strip().lower()
        gt_tp = str(row.get("message_type", "")).strip().lower()

        pred_act = pred["action"].strip().lower()
        pred_tp = pred["message_type"].strip().lower()

        gt_actions.append(gt_act)
        pred_actions.append(pred_act)

        gt_types.append(gt_tp)
        pred_types.append(pred_tp)

        d_type = pred.get("decision_type", "rule_engine")
        decision_counts[d_type] = decision_counts.get(d_type, 0) + 1

    action_labels = sorted(list(config.ALLOWED_ACTIONS))
    type_labels = sorted(list(config.ALLOWED_MESSAGE_TYPES))

    action_metrics = compute_metrics(gt_actions, pred_actions, action_labels)
    type_metrics = compute_metrics(gt_types, pred_types, type_labels)

    print("\n=======================================================")
    print("           NOTIFICATION ROUTER EVALUATION REPORT       ")
    print("=======================================================\n")

    print(f"Decision Fusion Summary: {decision_counts}\n")

    print("--- 1. ACTION ROUTING METRICS ---")
    print(f"Accuracy:        {action_metrics['accuracy']*100:.2f}%")
    print(f"Macro Precision: {action_metrics['macro_precision']*100:.2f}%")
    print(f"Macro Recall:    {action_metrics['macro_recall']*100:.2f}%")
    print(f"Macro F1 Score:  {action_metrics['macro_f1']*100:.2f}%")
    print(f"Micro F1 Score:  {action_metrics['micro_f1']*100:.2f}%")
    print("\nClass-wise Action Report:")
    print(action_metrics["class_report"].to_string())
    print("\nAction Confusion Matrix (Rows=True, Cols=Pred):")
    print(action_metrics["confusion_matrix"].to_string())

    print("\n--- 2. MESSAGE TYPE METRICS ---")
    print(f"Accuracy:        {type_metrics['accuracy']*100:.2f}%")
    print(f"Macro Precision: {type_metrics['macro_precision']*100:.2f}%")
    print(f"Macro Recall:    {type_metrics['macro_recall']*100:.2f}%")
    print(f"Macro F1 Score:  {type_metrics['macro_f1']*100:.2f}%")
    print(f"Micro F1 Score:  {type_metrics['micro_f1']*100:.2f}%")
    print("\nClass-wise Message Type Report:")
    print(type_metrics["class_report"].to_string())
    print("\nMessage Type Confusion Matrix (Rows=True, Cols=Pred):")
    print(type_metrics["confusion_matrix"].to_string())
    print("=======================================================\n")


if __name__ == "__main__":
    evaluate_on_sample()
