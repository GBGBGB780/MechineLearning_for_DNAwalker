# coding=utf-8
"""
utils.logging_config — 统一日志配置 / Unified logging configuration.

调用 :func:`get_logger` 获取按需创建的 logger；首次调用时为根 logger 安装一个
带时戳的 StreamHandler，避免与 ``print()`` 行为差异过大。
Call :func:`get_logger` to obtain a lazily-configured logger; on first use a
timestamped StreamHandler is installed on the root logger so the visual output
stays close to the previous ``print()`` calls.
"""

import logging
import os

_DEFAULT_FORMAT = "[%(asctime)s] %(levelname)s %(name)s: %(message)s"
_DEFAULT_DATEFMT = "%H:%M:%S"
_CONFIGURED = False


_VALID_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}


def _configure_root_once(level=logging.INFO):
    global _CONFIGURED
    if _CONFIGURED:
        return
    root = logging.getLogger()
    # 仅当用户没有自己装 handler 时才安装，避免覆盖应用层配置。
    if not root.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_DEFAULT_FORMAT, _DEFAULT_DATEFMT))
        root.addHandler(handler)
    env_level = os.environ.get("DNAWALKER_LOG_LEVEL", "").upper()
    if env_level in _VALID_LEVELS:
        level = getattr(logging, env_level)
    root.setLevel(level)
    _CONFIGURED = True


def get_logger(name=None):
    """Return a configured logger.

    Args:
        name: Logger name (commonly ``__name__``). ``None`` returns the root.

    Returns:
        logging.Logger: configured logger ready to use.
    """
    _configure_root_once()
    return logging.getLogger(name)
