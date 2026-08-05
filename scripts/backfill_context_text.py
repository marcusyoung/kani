#!/usr/bin/env python3
"""Backfill context_text in routing logs that predate TASK-037.02.

Routing logs written before the dual-embedding change have
classification_context.text and last_user_message but no context_text.
This script reconstructs context_text by stripping the [conversation]
header and the last user message line from the text field -- the exact
inverse of what build_classification_input does at runtime.

Usage:
    uv run python scripts/backfill_context_text.py [--log-dir DIR] [--dry-run]

By default operates on all routing-*.jsonl files in the kani log directory.
Creates .bak backups before modifying. Use --dry-run to preview changes.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from kani.dirs import log_dir

CONVERSATION_HEADER = "[conversation]\n"
MAX_CONTEXT_CHARS = 3500


def reconstruct_context_text(text: str, last_user_message: str) -> str:
    """Reconstruct context_text from classification text and last user message.

    Mirrors the logic in build_classification_input: the text field is
    "[conversation]\\n" followed by role-prefixed lines, where the final
    line is "user: {last_user_message}".  context_text is everything
    between the header and the last user message.
    """
    if not text or not text.startswith(CONVERSATION_HEADER):
        return ""

    without_header = text[len(CONVERSATION_HEADER) :]

    if not last_user_message:
        return ""

    # The last user message may span multiple lines, so match by suffix
    # rather than line-by-line.
    expected_suffix = f"user: {last_user_message}"
    if without_header.endswith(expected_suffix):
        context = without_header[: -len(expected_suffix)].rstrip("\n")
    else:
        # Fallback: single-line match (last_user_message had no newlines)
        lines = without_header.split("\n")
        if lines and lines[-1] == expected_suffix:
            context = "\n".join(lines[:-1])
        else:
            return ""

    if len(context) > MAX_CONTEXT_CHARS:
        context = context[-MAX_CONTEXT_CHARS:]
    return context


def process_file(path: Path, *, dry_run: bool) -> tuple[int, int, int]:
    """Process one routing log file. Returns (total, backfilled, skipped)."""
    records: list[str] = []
    backfilled = 0
    skipped = 0
    total = 0

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                records.append(line)
                skipped += 1
                continue

            cc = record.get("classification_context")
            if not isinstance(cc, dict):
                records.append(line)
                skipped += 1
                continue

            # Already has context_text -- leave untouched
            if cc.get("context_text"):
                records.append(line)
                continue

            text = str(cc.get("text") or "")
            last_user = str(cc.get("last_user_message") or "")

            reconstructed = reconstruct_context_text(text, last_user)
            cc["context_text"] = reconstructed
            backfilled += 1
            records.append(json.dumps(record, ensure_ascii=False))

    if dry_run:
        print(
            f"  [DRY RUN] {path.name}: {total} records, "
            f"{backfilled} would backfill, {skipped} skipped"
        )
    else:
        backup = path.with_suffix(".jsonl.bak")
        shutil.copy2(path, backup)
        with path.open("w", encoding="utf-8") as f:
            for record in records:
                f.write(record + "\n")
        print(
            f"  {path.name}: {total} records, "
            f"{backfilled} backfilled, {skipped} skipped (backup: {backup.name})"
        )

    return total, backfilled, skipped


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Backfill context_text in routing logs that predate TASK-037.02"
    )
    parser.add_argument(
        "paths",
        nargs="*",
        help="Specific routing log files. If omitted, scan --log-dir.",
    )
    parser.add_argument(
        "--log-dir",
        default=str(log_dir()),
        help="Directory containing routing-*.jsonl files (default: kani log dir)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without modifying files",
    )
    args = parser.parse_args(argv)

    if args.paths:
        log_paths = [Path(p).expanduser() for p in args.paths]
    else:
        log_dir_path = Path(args.log_dir).expanduser()
        log_paths = sorted(log_dir_path.glob("routing-*.jsonl"))

    if not log_paths:
        parser.error("No routing log files found")

    print(
        f"Processing {len(log_paths)} file(s)"
        f"{' (dry run)' if args.dry_run else ''}\n"
    )

    grand_total = 0
    grand_backfilled = 0
    grand_skipped = 0

    for path in log_paths:
        if not path.is_file():
            print(f"  SKIP (not a file): {path}")
            continue
        total, backfilled, skipped = process_file(path, dry_run=args.dry_run)
        grand_total += total
        grand_backfilled += backfilled
        grand_skipped += skipped

    print(
        f"\nTotal: {grand_total} records, "
        f"{grand_backfilled} backfilled, "
        f"{grand_skipped} skipped"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
