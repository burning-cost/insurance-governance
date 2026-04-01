"""ExplainabilityAuditLog — append-only JSONL audit log for model decisions.

The log is intentionally simple: each line is a JSON-serialised
ExplainabilityAuditEntry. There is no schema migration, no database, and no
external dependency. The file can be read by any tool that understands JSONL.

Append-only means append-only: this class has no delete or overwrite method.
If you need to redact an entry for data protection reasons, that is a separate
process involving the original file and a documented redaction log — not
something this library should automate.

For regulatory submission, :meth:`export_period` writes a filtered copy of
the log to a new file. The export preserves all entry hashes so the recipient
can independently verify integrity.

Thread safety: this class does not use file locking. If you are writing from
multiple processes, use a single writer process or add external locking.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .entry import ExplainabilityAuditEntry


class ExplainabilityAuditLog:
    """Append-only JSONL audit log over a local file.

    Each call to :meth:`append` writes one line to the file immediately.
    Reads re-parse the entire file on demand — there is no in-memory cache.
    This keeps the implementation simple and ensures reads always reflect
    the file as it exists on disk.

    Parameters
    ----------
    path:
        Path to the JSONL log file. The file will be created if it does not
        exist. The parent directory must already exist.
    model_id:
        The model_id this log belongs to. Used to validate entries on append
        and to label exports.
    model_version:
        The model version this log covers. Entries with a different version
        are still accepted (challenger runs, version migrations) but the
        version is recorded in export metadata.
    """

    def __init__(
        self,
        path: str | Path,
        model_id: str,
        model_version: str,
    ) -> None:
        self._path = Path(path)
        self._model_id = model_id
        self._model_version = model_version

        # Create the file if it does not exist
        if not self._path.exists():
            self._path.touch()

    @property
    def path(self) -> Path:
        """The path to the underlying JSONL file."""
        return self._path

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def model_version(self) -> str:
        return self._model_version

    def append(self, entry: ExplainabilityAuditEntry) -> None:
        """Append an entry to the log.

        The entry is serialised to JSON and written as a single line. The
        entry hash is included so integrity can be verified later.

        Args:
            entry: The entry to append. Must be an
                :class:`~.entry.ExplainabilityAuditEntry`.

        Raises:
            TypeError: If entry is not an ExplainabilityAuditEntry.
        """
        if not isinstance(entry, ExplainabilityAuditEntry):
            raise TypeError(
                f"entry must be an ExplainabilityAuditEntry, got {type(entry)}"
            )
        line = json.dumps(entry.to_dict(), separators=(",", ":"), ensure_ascii=True)
        with open(self._path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    def read_all(self) -> list[ExplainabilityAuditEntry]:
        """Read and deserialise every entry in the log.

        Returns an empty list if the log file is empty.

        Returns:
            List of :class:`~.entry.ExplainabilityAuditEntry` in the order
            they were appended.

        Raises:
            ValueError: If a line in the log cannot be parsed as JSON.
        """
        entries: list[ExplainabilityAuditEntry] = []
        with open(self._path, "r", encoding="utf-8") as fh:
            for lineno, raw in enumerate(fh, start=1):
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    d = json.loads(raw)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Corrupt JSONL at line {lineno} in {self._path}: {exc}"
                    ) from exc
                entries.append(ExplainabilityAuditEntry.from_dict(d))
        return entries

    def read_since(self, cutoff: datetime) -> list[ExplainabilityAuditEntry]:
        """Return entries with timestamps at or after ``cutoff``.

        Args:
            cutoff: Datetime threshold. Entries with ``timestamp_utc``
                equal to or later than this value are included. If the
                datetime is naive, it is assumed to be UTC.

        Returns:
            Filtered list of entries, in append order.
        """
        if cutoff.tzinfo is None:
            cutoff = cutoff.replace(tzinfo=timezone.utc)

        result: list[ExplainabilityAuditEntry] = []
        for entry in self.read_all():
            ts = datetime.fromisoformat(entry.timestamp_utc)
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            if ts >= cutoff:
                result.append(entry)
        return result

    def verify_chain(self) -> list[dict[str, Any]]:
        """Verify hash integrity for every entry in the log.

        Recomputes the SHA-256 hash for each entry and compares it against
        the stored value. Returns a list of dicts describing any failures.
        An empty list means all entries are intact.

        Returns:
            List of dicts, one per failed entry, with keys:
            - ``entry_id``: the entry identifier
            - ``line``: 1-based line number in the file
            - ``reason``: description of the failure
        """
        failures: list[dict[str, Any]] = []
        with open(self._path, "r", encoding="utf-8") as fh:
            for lineno, raw in enumerate(fh, start=1):
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    d = json.loads(raw)
                except json.JSONDecodeError:
                    failures.append({
                        "entry_id": f"<line {lineno}>",
                        "line": lineno,
                        "reason": "JSON parse failure",
                    })
                    continue

                try:
                    entry = ExplainabilityAuditEntry.from_dict(d)
                except Exception as exc:
                    failures.append({
                        "entry_id": d.get("entry_id", f"<line {lineno}>"),
                        "line": lineno,
                        "reason": f"Deserialisation error: {exc}",
                    })
                    continue

                if not entry.verify_integrity():
                    failures.append({
                        "entry_id": entry.entry_id,
                        "line": lineno,
                        "reason": "Hash mismatch — entry may have been tampered with",
                    })

        return failures

    def export_period(
        self,
        start: datetime,
        end: datetime,
        path: str | Path,
    ) -> Path:
        """Export entries within a time window to a new JSONL file.

        Intended for regulatory submission. The output file contains only the
        entries whose ``timestamp_utc`` falls within [start, end] inclusive.
        A metadata header is prepended as the first line (prefixed ``#``).

        Args:
            start: Start of the period (inclusive). Naive datetimes assumed UTC.
            end: End of the period (inclusive). Naive datetimes assumed UTC.
            path: Output file path. Will be overwritten if it exists.

        Returns:
            The resolved output :class:`~pathlib.Path`.

        Raises:
            ValueError: If start is after end.
        """
        if start.tzinfo is None:
            start = start.replace(tzinfo=timezone.utc)
        if end.tzinfo is None:
            end = end.replace(tzinfo=timezone.utc)
        if start > end:
            raise ValueError(
                f"start must be before or equal to end; got start={start!r}, end={end!r}"
            )

        out_path = Path(path)
        entries_in_window: list[ExplainabilityAuditEntry] = []
        for entry in self.read_all():
            ts = datetime.fromisoformat(entry.timestamp_utc)
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            if start <= ts <= end:
                entries_in_window.append(entry)

        meta = {
            "export_type": "explainability_audit",
            "model_id": self._model_id,
            "model_version": self._model_version,
            "period_start": start.isoformat(),
            "period_end": end.isoformat(),
            "entry_count": len(entries_in_window),
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "source_file": str(self._path),
        }

        with open(out_path, "w", encoding="utf-8") as fh:
            # Metadata header as a comment line so JSONL parsers skip it
            fh.write("# " + json.dumps(meta, separators=(",", ":")) + "\n")
            for entry in entries_in_window:
                line = json.dumps(
                    entry.to_dict(), separators=(",", ":"), ensure_ascii=True
                )
                fh.write(line + "\n")

        return out_path
