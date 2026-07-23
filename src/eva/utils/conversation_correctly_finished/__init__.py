"""Classifier for `conversation_correctly_finished` failures — one primary cause per record.

Modules: ``signals`` (contract), ``classifier`` (extract → classify → build sub-metrics),
``causes/*`` (per-cause detect + extract), ``final_turn`` (input-characteristic flags). Public API
re-exported below.
"""

from eva.utils.conversation_correctly_finished.classifier import (
    CATEGORY_PRIORITY,
    build_conv_finish_sub_metrics,
    build_final_turn_flag_sub_metrics,
    classify_conv_finish_failure,
    extract_conv_finish_signals,
)
from eva.utils.conversation_correctly_finished.final_turn import final_turn_input_flags
from eva.utils.conversation_correctly_finished.signals import Classification, ConvFinishSignals

__all__ = [
    "CATEGORY_PRIORITY",
    "Classification",
    "ConvFinishSignals",
    "build_conv_finish_sub_metrics",
    "build_final_turn_flag_sub_metrics",
    "classify_conv_finish_failure",
    "extract_conv_finish_signals",
    "final_turn_input_flags",
]
