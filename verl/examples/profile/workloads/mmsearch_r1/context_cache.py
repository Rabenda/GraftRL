"""Content-addressed artifacts and small process-local caches for MMSearch."""

from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Sequence

from PIL import Image


def text_content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def image_content_hash(image: Image.Image) -> str:
    """Hash decoded pixels so equivalent image objects share one artifact key."""
    rgb = image.convert("RGB")
    digest = hashlib.sha256()
    digest.update(b"mmsearch-image-rgb-v1\0")
    digest.update(str(rgb.size[0]).encode("ascii"))
    digest.update(b"x")
    digest.update(str(rgb.size[1]).encode("ascii"))
    digest.update(b"\0")
    digest.update(rgb.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class ContentArtifact:
    modality: str
    content_hash: str
    rank: int
    payload: Any
    label: str


@dataclass(frozen=True)
class ContextSelection:
    artifacts: tuple[ContentArtifact, ...]
    cache_hit: bool
    candidate_count: int
    unique_count: int

    @property
    def selected_count(self) -> int:
        return len(self.artifacts)

    @property
    def reduction_ratio(self) -> float:
        if self.candidate_count <= 0:
            return 0.0
        return 1.0 - self.selected_count / self.candidate_count


class ContentSelectionCache:
    """Cache deterministic static selection by content identity, not rollout id."""

    def __init__(self, max_entries: int = 4096):
        self.max_entries = max(1, int(max_entries))
        self._entries: OrderedDict[tuple[Any, ...], tuple[str, ...]] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def select(
        self,
        artifacts: Sequence[ContentArtifact],
        *,
        query_hash: str,
        topk: int,
        selector_version: str = "retrieval-rank-v1",
    ) -> ContextSelection:
        topk = max(1, int(topk))
        ordered_unique: list[ContentArtifact] = []
        artifact_by_hash: dict[str, ContentArtifact] = {}
        for artifact in sorted(artifacts, key=lambda value: value.rank):
            if artifact.content_hash in artifact_by_hash:
                continue
            artifact_by_hash[artifact.content_hash] = artifact
            ordered_unique.append(artifact)

        key = (
            selector_version,
            query_hash,
            topk,
            tuple((artifact.modality, artifact.content_hash) for artifact in ordered_unique),
        )
        with self._lock:
            selected_hashes = self._entries.get(key)
            cache_hit = selected_hashes is not None
            if selected_hashes is None:
                self.misses += 1
                selected_hashes = tuple(
                    artifact.content_hash for artifact in ordered_unique[:topk]
                )
                self._entries[key] = selected_hashes
                while len(self._entries) > self.max_entries:
                    self._entries.popitem(last=False)
            else:
                self.hits += 1
                self._entries.move_to_end(key)

        return ContextSelection(
            artifacts=tuple(artifact_by_hash[value] for value in selected_hashes),
            cache_hit=cache_hit,
            candidate_count=len(artifacts),
            unique_count=len(ordered_unique),
        )


class ContentValueCache:
    """Thread-safe LRU for position-independent derived values such as token ids."""

    def __init__(self, max_entries: int = 4096):
        self.max_entries = max(1, int(max_entries))
        self._entries: OrderedDict[tuple[Any, ...], Any] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key: tuple[Any, ...]) -> Any | None:
        with self._lock:
            value = self._entries.get(key)
            if value is None:
                self.misses += 1
                return None
            self.hits += 1
            self._entries.move_to_end(key)
            return value

    def set(self, key: tuple[Any, ...], value: Any) -> Any:
        with self._lock:
            existing = self._entries.get(key)
            if existing is not None:
                self._entries.move_to_end(key)
                return existing
            self._entries[key] = value
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)
            return value

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "entries": len(self._entries),
                "hits": self.hits,
                "misses": self.misses,
            }
