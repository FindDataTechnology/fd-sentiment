#!/usr/bin/env python3
"""Export a gold set from the SQLite store to ``gold/<set-version>.json``.

The JSON is the committed, reviewable artifact: it carries each item's
``doc_id`` / ``as_of`` / ``expected_band`` plus a ``text_sha256`` and length, but
NOT the announcement body — the body is re-fetched at evaluation time and the
hash proves it still matches what was labelled.

Run:
    python3 scripts/export_gold_set.py --set-version gold-v8 --db /tmp/gold-v8.db
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ALL_BANDS = ("strong_bullish", "bullish", "neutral", "bearish", "strong_bearish")
WEAK_BAND_MIN = 10


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--set-version", required=True)
    ap.add_argument("--db", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    conn = sqlite3.connect(args.db)
    rows = conn.execute(
        "SELECT doc_id, as_of, expected_band, text FROM gold_set "
        "WHERE set_version=? ORDER BY doc_id",
        (args.set_version,),
    ).fetchall()
    if not rows:
        print(f"error: no rows for {args.set_version} in {args.db}", file=sys.stderr)
        return 1

    items = [
        {
            "doc_id": doc_id,
            "as_of": as_of,
            "expected_band": band,
            "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "text_chars": len(text),
        }
        for doc_id, as_of, band, text in rows
    ]
    counts = Counter(i["expected_band"] for i in items)

    missing = [b for b in ALL_BANDS if counts.get(b, 0) == 0]
    if missing:
        print(
            f"error: {args.set_version} is missing band(s): {', '.join(missing)} — "
            "a missing band makes every correct prediction in that band count as wrong",
            file=sys.stderr,
        )
        return 1

    out = Path(args.out) if args.out else ROOT / "gold" / f"{args.set_version}.json"
    payload = {
        "set_version": args.set_version,
        "built_at": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "items": items,
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"wrote {len(items)} items to {out}")
    print("bands: " + " | ".join(f"{b}={counts.get(b, 0)}" for b in ALL_BANDS))
    thin = [b for b in ALL_BANDS if counts.get(b, 0) < WEAK_BAND_MIN]
    if thin:
        print(
            f"warning: band(s) below {WEAK_BAND_MIN} samples carry no statistical "
            f"significance: {', '.join(thin)}",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
