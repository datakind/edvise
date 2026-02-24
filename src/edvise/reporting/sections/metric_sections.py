def register_metric_sections(card, registry):
    """
    Registers metric sections for a model, specifically the training metric and sample weights.
    If sample weights are not used, then that particular section is skipped.
    """

    @registry.register("primary_metric_section")
    def primary_metric():
        """
        Returns a markdown string describing the primary metric used for training the model. This section
        is meant to be more verbose in the model card, explaining what the metric is for a non-technical audience.
        """

        def normalize_metric(name: str | None) -> str | None:
            if not name:
                return None
            # Common punctuation normalization
            aliases = {
                "logloss": "log_loss",
                "log_loss": "log_loss",
                "auc": "roc_auc",
                "roc_auc": "roc_auc",
                "area_under_roc": "roc_auc",
                "areaunderroc": "roc_auc",
                "binary_auc": "roc_auc",
                "auprc": "pr_auc",
                "pr_auc": "pr_auc",
                "area_under_pr": "pr_auc",
                "average_precision": "pr_auc",
                "f1": "f1",
                "f1_score": "f1",
                "precision": "precision",
                "recall": "recall",
                "tpr": "recall",
            }
            return aliases.get(name, name)

        metric_map = {
            "log_loss": f"{card.format.indent_level(1)}- Our primary metric for training was log loss to ensure that the model produces well-calibrated probability estimates.\n{card.format.indent_level(1)}- Lower log loss is better, as it indicates more accurate and confident probability predictions.",
            "recall": f"{card.format.indent_level(1)}- Our primary metric for training was recall in order to ensure that we correctly identify as many students in need of support as possible.\n{card.format.indent_level(1)}- Higher recall is better, as it indicates fewer students in need are missed.",
            "precision": f"{card.format.indent_level(1)}- Our primary metric for training was precision to ensure that when the model identifies a student as needing support, it is likely to be correct.\n{card.format.indent_level(1)}- Higher precision is better, as it indicates fewer students are incorrectly flagged.",
            "roc_auc": f"{card.format.indent_level(1)}- Our primary metric for training was ROC AUC, which measures the model's ability to distinguish between students who need support and those who do not.\n{card.format.indent_level(1)}- Higher ROC AUC is better, as it indicates stronger overall classification performance across all thresholds.",
            "pr_auc": f"{card.format.indent_level(1)}- Our primary metric for training was PR AUC (Area under the Precision-Recall curve).\n{card.format.indent_level(1)}- A higher PR-AUC score indicates that the model performs well at distinguishing positive cases, especially when the positive class is rare.",
            "f1": f"{card.format.indent_level(1)}- Our primary metric for training was F1-score to balance the trade-off between precision and recall.\n{card.format.indent_level(1)}- A higher F1-score indicates that the model is effectively identifying students in need while minimizing both false positives and false negatives.",
        }
        # Safely fetch configured metric
        modeling = getattr(card.cfg, "modeling", None)
        training = getattr(modeling, "training", None)
        raw_metric = getattr(training, "primary_metric", None)
        metric = normalize_metric(raw_metric)

        return metric_map.get(
            metric, f"{card.format.indent_level(1)}- Default metric explanation."
        )

    @registry.register("sample_weight_section")
    def sample_weight():
        """
        Returns a markdown string describing the sample weights used for training the model. This section
        will still print out how many experiments were run but the sample weight details are optional, depending
        on where a column with a substring of "sample_weight" exists in the training data.
        """
        platform = "H2O AutoML"
        used_weights = any(
            col.startswith("sample_weight") for col in card.training_data.columns
        )
        sw_note = (
            f"{card.format.indent_level(1)}- Sample weights were used to stabilize training."
            if used_weights
            else None
        )
        num_runs = card.context.get("num_runs_in_experiment", None)
        if isinstance(num_runs, int) and num_runs > 0:
            mlops_note = (
                f"{card.format.indent_level(1)}- Utilizing {platform}, "
                f"we built a machine learning pipeline for data preprocessing, model experimentation, and evaluation. "
                f"We trained {num_runs} models before choosing one final, optimized model."
            )
        else:
            mlops_note = (
                f"{card.format.indent_level(1)}- Utilizing {platform}, "
                f"we built a machine learning pipeline for data preprocessing, model experimentation, and evaluation."
            )
        return "\n".join(filter(None, [mlops_note, sw_note]))

    @registry.register("classification_threshold_section")
    def classification_threshold():
        """
        Returns a markdown string describing the classification threshold used for predictions.
        The threshold is retrieved from MLflow run parameters (logged during training) or from config.
        """
        # Try to get threshold from MLflow run params first
        threshold = None
        try:
            if hasattr(card, "run_id") and card.run_id:
                run = card.client.get_run(card.run_id)
                if "classification_threshold" in run.data.params:
                    threshold = float(run.data.params["classification_threshold"])
        except Exception:
            pass

        # Fallback to config if not found in MLflow
        if threshold is None:
            modeling = getattr(card.cfg, "modeling", None)
            training = getattr(modeling, "training", None)
            if training:
                threshold = getattr(training, "classification_threshold", 0.5)
            else:
                threshold = 0.5

        threshold_note = (
            f"{card.format.indent_level(1)}- Classification Threshold: {threshold}\n"
            f"{card.format.indent_level(2)}- The classification threshold determines the probability cutoff for "
            f"predicting the positive class (students needing support).\n"
            f"{card.format.indent_level(2)}- Lower thresholds (e.g., 0.4) increase recall and sensitivity, "
            f"resulting in more students identified for intervention.\n"
            f"{card.format.indent_level(2)}- Higher thresholds (e.g., 0.6) increase precision and reduce false positives."
        )

        return threshold_note
