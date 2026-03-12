from __future__ import annotations

import logging
import os
import sys
import warnings


def local_rank() -> int:
    value = os.environ.get("LOCAL_RANK")
    if value is None:
        return -1
    try:
        return int(value)
    except ValueError:
        return -1


def is_rank0_or_single_process() -> bool:
    rank = local_rank()
    return rank in {-1, 0}


def silence_non_zero_local_ranks() -> None:
    """Silence logs/stdout on non-zero local ranks."""
    if is_rank0_or_single_process():
        return

    devnull = open(os.devnull, "w", encoding="utf-8")
    sys.stdout = devnull
    sys.stderr = devnull

    warnings.filterwarnings("ignore")
    logging.getLogger().setLevel(logging.ERROR)

    try:
        from datasets.utils.logging import disable_progress_bar, set_verbosity_error

        set_verbosity_error()
        disable_progress_bar()
    except Exception:
        pass

    try:
        from transformers.utils import logging as transformers_logging

        transformers_logging.set_verbosity_error()
    except Exception:
        pass
