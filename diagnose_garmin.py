"""Check your Garmin Personal API setup, one step at a time.   Run:  python diagnose_garmin.py

Prints what works and what doesn't. It never prints your key or token, and it prints no hike titles.
"""
from __future__ import annotations

import collections
import os
import sys
from pathlib import Path

import garmin_personal as gp

HERE = Path(__file__).parent
SECRETS = HERE / ".streamlit" / "secrets.toml"
OK, BAD, NOTE = "[ OK ]", "[FAIL]", "[ .. ]"


def find_key() -> str:
    print("Step 1: finding your key")
    key, where = "", ""
    if SECRETS.exists():
        try:
            import tomllib
            data = tomllib.loads(SECRETS.read_text(encoding="utf-8"))
            sect = data.get("garmin_personal")
            if sect is None:
                print(f"  {BAD} {SECRETS} exists but has no [garmin_personal] section (check the spelling and the brackets).")
                print(f"       Sections found: {list(data) or 'none'}")
            else:
                key, where = str(sect.get("api_key", "")), "secrets.toml"
                if str(sect.get("enabled", True)).lower() in ("false", "0"):
                    print(f"  {BAD} enabled = false in secrets.toml, so the app ignores the key. Remove that line or set it to true.")
        except Exception as e:  # noqa: BLE001
            print(f"  {BAD} Couldn't read {SECRETS}: {type(e).__name__}: {e}")
            print("       Check for a missing quote or bracket.")
    else:
        print(f"  {NOTE} No file at {SECRETS}")
        print("       (it must be named secrets.toml, inside the .streamlit folder; check it isn't secrets.toml.txt)")
    if not key and os.environ.get(gp.KEY_ENV):
        key, where = os.environ[gp.KEY_ENV], f"{gp.KEY_ENV} environment variable"
    key = key.strip()
    if not key:
        print(f"  {BAD} No key found. Put it in .streamlit/secrets.toml under [garmin_personal] as api_key = \"...\"")
        sys.exit(1)
    if "PASTE" in key.upper():
        print(f"  {BAD} The key is still the placeholder text. Replace it with your real key.")
        sys.exit(1)
    print(f"  {OK} Key found in {where} ({len(key)} characters).")
    return key


def main() -> None:
    key = find_key()

    print("\nStep 2: exchanging the key for a token")
    try:
        token = gp.get_token(key, force=True)
    except gp.PersonalApiError as e:
        print(f"  {BAD} {e}")
        sys.exit(1)
    print(f"  {OK} Garmin accepted the key.")

    print("\nStep 3: listing your activities (all types)")
    try:
        everything = gp.list_activities(token, types=())
    except gp.PersonalApiError as e:
        print(f"  {BAD} {e}")
        sys.exit(1)
    if not everything:
        print(f"  {BAD} Garmin returned 0 activities. The Personal API may only return recent activities, or your")
        print("       account has none it can see. Open Garmin Connect and check a recent activity exists.")
        sys.exit(1)
    types = collections.Counter(gp._type_key(i) for i in everything)
    starts = sorted(str(i.get("startTimeLocal", ""))[:10] for i in everything if i.get("startTimeLocal"))
    print(f"  {OK} {len(everything)} activities found, dated {starts[0] if starts else '?'} to {starts[-1] if starts else '?'}.")
    print("       By type: " + ", ".join(f"{k} x{v}" for k, v in types.most_common(12)))

    first = everything[0]
    want = ["activityId", "activityName", "startTimeLocal", "activityType", "distance", "duration", "elapsedDuration",
            "elevationGain", "averageHR"]
    missing = [k for k in want if k not in first]
    print(f"  {OK if not missing else NOTE} Fields the app reads are {'all present' if not missing else 'MISSING: ' + ', '.join(missing)}.")
    if missing:
        print(f"       Fields Garmin actually sent: {', '.join(sorted(first))}")

    print("\nStep 4: which activities the app will use")
    cfg_types = gp.DEFAULT_TYPES
    if SECRETS.exists():
        try:
            import tomllib
            t = tomllib.loads(SECRETS.read_text(encoding="utf-8")).get("garmin_personal", {}).get("activity_types", cfg_types)
            cfg_types = tuple([t] if isinstance(t, str) else t)
        except Exception:  # noqa: BLE001
            pass
    chosen = [i for i in everything if any(w.lower() in gp._type_key(i) for w in cfg_types)]
    if chosen:
        print(f"  {OK} {len(chosen)} match activity_types = {list(cfg_types)}.")
    else:
        print(f"  {BAD} 0 activities match activity_types = {list(cfg_types)}. Your types are listed above;")
        print("       add the right one in secrets.toml, e.g.  activity_types = [\"hiking\", \"walking\"]")
        sys.exit(1)

    print("\nStep 5: downloading and reading the newest matching activity")
    newest = max(chosen, key=lambda i: str(i.get("startTimeLocal", "")))
    try:
        z = gp.download_zip(token, newest["activityId"])
    except gp.PersonalApiError as e:
        print(f"  {BAD} {e}")
        sys.exit(1)
    print(f"  {OK} Downloaded {len(z):,} bytes.")
    from garmin_io import read_fit_uploads
    df, notes = read_fit_uploads([(f"{newest['activityId']}.zip", z)])
    if df.empty:
        print(f"  {BAD} The file downloaded but couldn't be read as an activity. Notes: {notes}")
        sys.exit(1)
    r = df.iloc[0]
    print(f"  {OK} Read it: {r.distance_mi:.1f} mi, {r.ascent_ft:,.0f} ft climb, total {r.time_hr:.2f} h, "
          f"moving {r.moving_hr:.2f} h, stopped {r.stopped_hr:.2f} h.")

    print("\nStep 6: cache")
    print(f"  {NOTE} Cache file: {gp.CACHE_FILE} ({'exists' if gp.CACHE_FILE.exists() else 'not created yet'}).")
    print("       If hikes still look wrong in the app, delete the .cache folder and reload.")
    print("\nAll steps passed. The app should show your hikes by default when nothing is uploaded.")


if __name__ == "__main__":
    main()
