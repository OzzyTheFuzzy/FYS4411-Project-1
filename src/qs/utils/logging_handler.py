#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging

import colorlog

ALLOWED_LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


def setup_logger(name, level="INFO"):
    """Return a logger with a default ColoredFormatter."""

    formatter = colorlog.ColoredFormatter(
        "%(log_color)s%(levelname)s:%(name)-8s%(reset)s %(purple)s%(message)s",
        datefmt=None,
        reset=True,
        log_colors={
            "DEBUG": "cyan",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "red",
        },
    )

    logger = logging.getLogger(name)
    # Avoid adding multiple handlers to the same logger (causes duplicated output)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    # Prevent messages from being propagated to the root logger (which may also have handlers)
    logger.propagate = False

    level = level.upper()
    if level not in ALLOWED_LEVELS:
        msg = f"Logger level must be one of {ALLOWED_LEVELS}" " (not case-sensitive)"
        raise ValueError(msg)

    if level == "INFO":
        logger.setLevel(logging.INFO)
    elif level == "WARNING":
        logger.setLevel(logging.WARNING)
    elif level == "ERROR":
        logger.setLevel(logging.ERROR)
    elif level == "DEBUG":
        logger.setLevel(logging.DEBUG)
    elif level == "CRITICAL":
        logger.setLevel(logging.CRITICAL)

    return logger


if __name__ == "__main__":
    my_log = setup_logger("test", level="INFO")
    my_log.info("2+2=4")
    my_log.debug("Critters everywhere")
    my_log.warning("Is the oven on?")
    my_log.error("ALARM")
    my_log.critical("RUN!")
