"""Research baselines used only for like-for-like Pairwise comparisons.

The historical MM baseline pulls in the legacy v0.1 archive stack.  Import it
only when one of its public names is requested so independent evaluation
modules (for example the Rachel N=512 real evaluator) do not require that
unrelated stack merely because Python initializes this package.
"""

from importlib import import_module

__all__ = [
    "HISTORICAL_MM_BASELINE_ID",
    "HistoricalMMBaselineResult",
    "HistoricalMMSiamese",
    "evaluate_historical_mm_checkpoint",
    "load_historical_mm_checkpoint",
    "preprocess_historical_mm_mask",
    "score_historical_mm_validation",
]


def __getattr__(name: str):
    if name not in __all__:
        raise AttributeError("module {!r} has no attribute {!r}".format(__name__, name))
    module = import_module(__name__ + ".historical_mm_siamese")
    value = getattr(module, name)
    globals()[name] = value
    return value
