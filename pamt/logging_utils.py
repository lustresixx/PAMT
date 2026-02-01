from __future__ import annotations

import logging
import os
import sys


def setup_logging() -> None:
    level_name = os.environ.get("PAMT_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=level,
        format="[%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
    )



