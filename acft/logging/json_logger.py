# acft/logging/json_logger.py

from __future__ import annotations

import sys
import json
import datetime
from typing import Any, Dict


class JsonLogger:
    """
    Very flexible JSON logger for ACFT.

    - Always outputs lines like:
        DebugReport: { ...JSON... }

    - Has a generic .log(event, **fields) method.
    - Any unknown method name (e.g. .debug(), .info(), .log_acft_result())
      is automatically mapped to .log(event_name, ...).
    """

    def __init__(
        self,
        stream=None,
        pretty: bool = True,
        enabled: bool = True,
    ) -> None:
        self.stream = stream or sys.stdout
        self.pretty = pretty
        self.enabled = enabled

    # ---------- internal writer ----------

    def _write(self, record: Dict[str, Any]) -> None:
        if not self.enabled:
            return

        if self.pretty:
            txt = json.dumps(record, indent=2, ensure_ascii=False)
        else:
            txt = json.dumps(record, ensure_ascii=False, separators=(",", ":"))

        # EXACT format you requested:
        self.stream.write(f"DebugReport: {txt}\n")
        self.stream.flush()

    # ---------- main logging API ----------

    def log(self, event: str, **fields: Any) -> None:
        """
        Generic logger entry point.

        Example:
            logger.log("acft_turn", prompt=..., decision=..., metrics=...)
        """
        record: Dict[str, Any] = {
            "event": event,
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        }
        record.update(fields)
        self._write(record)

    def log_acft_result(self, result: Any, **extra: Any) -> None:
        """
        Helper specifically for ACFTResult-like objects.

        Works even if result is a simple object with .decision/.answer/.metrics/.debug_report.
        """
        metrics_obj = getattr(result, "metrics", None)
        metrics_dict = None
        if metrics_obj is not None:
            metrics_dict = getattr(metrics_obj, "__dict__", None) or metrics_obj

        payload: Dict[str, Any] = {
            "decision": getattr(result, "decision", None),
            "answer": getattr(result, "answer", None),
            "metrics": metrics_dict,
            "debug": getattr(result, "debug_report", None),
        }
        payload.update(extra)
        self.log("acft_result", **payload)

    # ---------- magic: handle ANY other method name ----------

    def __getattr__(self, name: str):
        """
        If the CLI calls:
            logger.debug(...), logger.info(...),
            logger.log_debug_report(...), etc.

        We map that to:
            self.log(name, ...)
        """

        def _wrapper(*args, **kwargs):
            # If called like logger.method(obj)
            if args and not kwargs:
                self.log(name, data=args[0])
            else:
                self.log(name, **kwargs)

        return _wrapper


# =====================================================================
# Module-level default logger + log_run() helper used by engine.py
# =====================================================================

default_logger = JsonLogger()


def log_run(*args: Any, **kwargs: Any) -> None:
    """
    Flexible helper so engine.py can do:

        from acft.logging.json_logger import log_run

    and then call either:
        log_run("acft_run", prompt=..., decision=..., ...)
    or:
        log_run(result)

    We detect the usage pattern and forward to the default JsonLogger.
    """
    # Case 1: log_run(result)  (single non-str positional argument)
    if len(args) == 1 and not kwargs and not isinstance(args[0], str):
        default_logger.log_acft_result(args[0])
        return

    # Case 2: log_run("event_name", ...)  (first arg is event name)
    if len(args) >= 1 and isinstance(args[0], str):
        event = args[0]
        rest = args[1:]
        if rest and not kwargs:
            # e.g. log_run("acft_run", some_obj)
            default_logger.log(event, data=rest[0])
        else:
            default_logger.log(event, **kwargs)
        return

    # Fallback: just treat kwargs as a generic acft_run record
    default_logger.log("acft_run", **kwargs)