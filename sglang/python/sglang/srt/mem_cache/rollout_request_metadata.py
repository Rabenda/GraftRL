"""Bounded request metadata shared by rollout-time reuse features."""

from __future__ import annotations

import re
import threading
import time
from typing import Any, Dict, Optional


_MAX_REQUEST_META = 65536
_RID_TURN_RE = re.compile(r"_t(\d+)$")
_LOCK = threading.Lock()
_REQUEST_META: Dict[str, Dict[str, Any]] = {}


def register_request_meta(
    rid: str,
    *,
    agent_uid: Optional[str] = None,
    agent_turn: Optional[int] = None,
    agent_request_id: Optional[str] = None,
    global_step: Optional[int] = None,
) -> None:
    """Record the rollout identity needed to build a reuse group key."""
    if not rid:
        return

    meta: Dict[str, Any] = {}
    if agent_uid:
        meta["agent_uid"] = str(agent_uid)
    if agent_turn is not None:
        meta["agent_turn"] = int(agent_turn)
    if agent_request_id:
        meta["agent_request_id"] = str(agent_request_id)
    if global_step is not None:
        meta["global_step"] = int(global_step)
    if not meta:
        return

    meta["_created_at"] = time.time()
    with _LOCK:
        _REQUEST_META[str(rid)] = meta
        while len(_REQUEST_META) > _MAX_REQUEST_META:
            oldest_rid = min(
                _REQUEST_META,
                key=lambda key: float(_REQUEST_META[key].get("_created_at", 0.0)),
            )
            _REQUEST_META.pop(oldest_rid, None)


def lookup_request_meta(rid: Optional[str]) -> Optional[Dict[str, Any]]:
    if not rid:
        return None
    with _LOCK:
        return dict(_REQUEST_META.get(str(rid), {}) or {})


def parse_turn_from_rid(rid: str) -> Optional[int]:
    match = _RID_TURN_RE.search(str(rid))
    return int(match.group(1)) if match else None


def clear_request_meta() -> None:
    with _LOCK:
        _REQUEST_META.clear()
