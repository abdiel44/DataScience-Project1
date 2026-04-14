"""
Building blocks for Phase E (modeling): subject-wise CV splits, metrics, saving predictions/figures.

This package does not run training pipelines; use notebooks or future `train_*.py` scripts.
"""

from modeling.artifacts import (
    save_confusion_matrix_figure,
    save_model_bundle,
    save_predictions_dataframe,
    save_roc_curve_figure,
    write_model_registry,
)
from modeling.cv_split import SubjectFoldConfig, subject_wise_fold_indices
from modeling.subject_id import DEFAULT_PREFER, ensure_subject_unit_column, subject_proxy_from_source_file
from modeling.metrics import (
    apnea_binary_metrics,
    cohen_kappa,
    macro_f1,
    mcnemar_exact,
    multiclass_sleep_metrics,
)

__all__ = [
    "DEFAULT_PREFER",
    "SubjectFoldConfig",
    "apnea_binary_metrics",
    "cohen_kappa",
    "macro_f1",
    "mcnemar_exact",
    "multiclass_sleep_metrics",
    "save_confusion_matrix_figure",
    "save_model_bundle",
    "save_predictions_dataframe",
    "save_roc_curve_figure",
    "write_model_registry",
    "subject_proxy_from_source_file",
    "subject_wise_fold_indices",
    "ensure_subject_unit_column",
]
