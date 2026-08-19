"""Internal command-line display helpers for the CMM workflow."""

from __future__ import annotations

import logging


NOISY_LOGGERS = ("micom", "cobra", "riptide", "optlang")


def configure_logging(verbose: bool = False) -> None:
    """Configure concise default output while preserving warnings and errors."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        force=True,
    )
    if not verbose:
        for logger_name in NOISY_LOGGERS:
            logging.getLogger(logger_name).setLevel(logging.WARNING)
    else:
        logging.getLogger("cmm").setLevel(logging.DEBUG)
