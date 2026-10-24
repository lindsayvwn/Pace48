"""Save / restore the user's peak tags and include/exclude choices as a small CSV."""
from __future__ import annotations

import io

import pandas as pd

SEP = "|"  # peak names can contain commas ("Wildcat, A peak"), so don't split on commas
COLUMNS = ["key", "date", "title", "distance_mi", "nh4000_peaks", "include"]


def tags_to_csv(tags: dict) -> bytes:
    rows = [{
        "key": k,
        "date": v.get("date", ""),
        "title": v.get("title", ""),
        "distance_mi": v.get("distance_mi", ""),
        "nh4000_peaks": SEP.join(v.get("peaks", [])),
        "include": bool(v.get("include", True)),
    } for k, v in sorted(tags.items())]
    return pd.DataFrame(rows, columns=COLUMNS).to_csv(index=False).encode("utf-8")


def tags_from_csv(data: bytes) -> dict:
    df = pd.read_csv(io.BytesIO(data), dtype=str, keep_default_na=False)
    if "key" not in df.columns:
        raise ValueError("This doesn't look like a saved-tags file (no 'key' column).")
    out = {}
    for _, r in df.iterrows():
        peaks = [p for p in r.get("nh4000_peaks", "").split(SEP) if p]
        out[r["key"]] = {
            "peaks": peaks,
            "include": str(r.get("include", "True")).strip().lower() not in ("false", "0", "no"),
            "date": r.get("date", ""), "title": r.get("title", ""), "distance_mi": r.get("distance_mi", ""),
        }
    return out
