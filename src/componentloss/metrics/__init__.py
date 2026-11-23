"""Component-wise (object-level) evaluation of binary segmentations."""
from .detection import (SIZE_BINS, lesion_detection_metrics, lesion_size_profile, lesion_table, recall_by_size,
                        remove_small_components)

__all__ = ["SIZE_BINS", "lesion_detection_metrics", "lesion_size_profile", "lesion_table", "recall_by_size",
           "remove_small_components"]
