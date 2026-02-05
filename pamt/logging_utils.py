from __future__ import annotations

import logging
import os
import sys


class PrettyFormatter(logging.Formatter):
    def __init__(self, datefmt: str | None = None) -> None:
        super().__init__(datefmt=datefmt)
        self._header_fmt = "%(asctime)s.%(msecs)03d | %(levelname)-8s | %(name)s:%(lineno)d"

    def format(self, record: logging.LogRecord) -> str:
        record.message = record.getMessage()
        record.asctime = self.formatTime(record, self.datefmt)
        header = self._header_fmt % record.__dict__
        message = record.message or ""
        if record.exc_info:
            exc_text = self.formatException(record.exc_info)
            message = f"{message}\n{exc_text}" if message else exc_text
        if record.stack_info:
            message = f"{message}\n{record.stack_info}" if message else record.stack_info
        if not message:
            return header
        lines = message.splitlines()
        if len(lines) == 1:
            return f"{header} | {lines[0]}"
        indent = " " * (len(header) + 3)
        return "\n".join([f"{header} | {lines[0]}"] + [f"{indent}{line}" for line in lines[1:]])


def setup_logging() -> None:
    level_name = os.environ.get("PAMT_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    datefmt = os.environ.get("PAMT_LOG_DATEFMT", "%Y-%m-%d %H:%M:%S")
    formatter = PrettyFormatter(datefmt=datefmt)

    root = logging.getLogger()
    root.setLevel(level)

    if not root.handlers:
        handler = logging.StreamHandler(sys.stdout)
        root.addHandler(handler)

    for handler in root.handlers:
        handler.setLevel(level)
        handler.setFormatter(formatter)



