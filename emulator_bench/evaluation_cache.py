"""Process-safe persistent cache for deterministic RDChiral applications."""
from __future__ import annotations

import hashlib
import inspect
import json
import os
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path


CACHE_SCHEMA = 1


def _file_digest(value: object) -> str:
    try:
        path = Path(inspect.getsourcefile(value) or "")
        return hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    except OSError:
        return "unavailable"


def cache_namespace() -> str:
    """A namespace changes whenever RDKit/RDChiral/reactor behavior can change."""
    import rdkit
    from gln.common.reactor import Reactor
    from gln.common import cmd_args

    identity = {
        "schema": CACHE_SCHEMA,
        "rdkit": rdkit.__version__,
        "rdchiral": _file_digest(cmd_args.rdchiralRun),
        "reactor": _file_digest(Reactor.__class__),
    }
    encoded = json.dumps(identity, sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()[:20]


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    waits: int = 0
    computed: int = 0
    failures: int = 0

    def as_dict(self) -> dict[str, int]:
        return self.__dict__.copy()


def _owner_is_alive(owner: str | None) -> bool:
    """Best-effort liveness check for a lease owner on this shared host."""
    if not owner:
        return False
    try:
        pid = int(owner.split("-", 1)[0])
        os.kill(pid, 0)
    except (ValueError, ProcessLookupError):
        return False
    except PermissionError:
        # A different user owns a live evaluator; respect its lease.
        return True
    return True


class SQLiteReactionCache:
    """SQLite/WAL cache with lease claims, safe across evaluator processes."""

    def __init__(self, root: Path, *, lease_seconds: float = 900.0, rebuild: bool = False):
        self.namespace = cache_namespace()
        self.root = Path(root) / self.namespace
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "reaction_cache.sqlite3"
        if rebuild:
            for suffix in ("", "-wal", "-shm"):
                try:
                    (Path(str(self.path) + suffix)).unlink()
                except FileNotFoundError:
                    pass
        self.lease_seconds = lease_seconds
        self.owner = f"{os.getpid()}-{time.time_ns()}"
        self.stats = CacheStats()
        self.connection = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=NORMAL")
        self.connection.execute("PRAGMA busy_timeout=30000")
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS reactions ("
            "key TEXT PRIMARY KEY, state TEXT NOT NULL, outcomes TEXT, "
            "owner TEXT, lease_until REAL, updated REAL NOT NULL)"
        )

    @staticmethod
    def key(raw_product: str, template: str) -> str:
        return hashlib.sha256((raw_product + "\0" + template).encode()).hexdigest()

    def acquire(self, raw_product: str, template: str) -> tuple[str, list[str] | None]:
        """Return ``ready``, ``claimed``, or ``wait`` and any cached outcomes."""
        key = self.key(raw_product, template)
        now = time.time()
        lease_until = now + self.lease_seconds
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            row = self.connection.execute(
                "SELECT state, outcomes, owner, lease_until FROM reactions WHERE key=?", (key,)
            ).fetchone()
            if row is not None and row[0] == "ready":
                self.connection.execute("COMMIT")
                self.stats.hits += 1
                return "ready", json.loads(row[1])
            if row is None:
                self.connection.execute(
                    "INSERT INTO reactions VALUES (?, 'leased', NULL, ?, ?, ?)",
                    (key, self.owner, lease_until, now),
                )
                self.connection.execute("COMMIT")
                self.stats.misses += 1
                return "claimed", None
            # A killed evaluator must not make all resumed jobs wait for the
            # full lease timeout.  The PID is host-local by design: this cache
            # is only shared among benchmark jobs on this machine.
            if row[3] is None or row[3] <= now or not _owner_is_alive(row[2]):
                self.connection.execute(
                    "UPDATE reactions SET owner=?, lease_until=?, updated=? WHERE key=?",
                    (self.owner, lease_until, now, key),
                )
                self.connection.execute("COMMIT")
                self.stats.misses += 1
                return "claimed", None
            self.connection.execute("COMMIT")
            self.stats.waits += 1
            return "wait", None
        except Exception:
            self.connection.execute("ROLLBACK")
            raise

    def wait_ready(self, raw_product: str, template: str) -> list[str] | None:
        key = self.key(raw_product, template)
        while True:
            state, outcomes = self.acquire(raw_product, template)
            if state == "ready":
                return outcomes
            if state == "claimed":
                # The preceding owner died or its lease expired.  The caller computes it.
                return _CLAIMED
            time.sleep(0.05)

    def store(self, raw_product: str, template: str, outcomes: list[str] | None) -> None:
        key = self.key(raw_product, template)
        self.connection.execute(
            "UPDATE reactions SET state='ready', outcomes=?, owner=NULL, lease_until=NULL, updated=? "
            "WHERE key=? AND owner=?",
            (json.dumps(outcomes), time.time(), key, self.owner),
        )
        self.stats.computed += 1
        if outcomes is None:
            self.stats.failures += 1

    def write_manifest(self) -> Path:
        manifest = self.root / "manifest.json"
        temporary = manifest.with_suffix(".tmp")
        temporary.write_text(json.dumps({
            "cache_schema": CACHE_SCHEMA,
            "namespace": self.namespace,
            "database": str(self.path),
            "stats": self.stats.as_dict(),
            "updated_at": time.time(),
        }, indent=2) + "\n")
        temporary.replace(manifest)
        return manifest

    def close(self) -> None:
        self.write_manifest()
        self.connection.close()


# Sentinel: an expiring lease was reclaimed, so the caller must calculate it.
_CLAIMED = object()
