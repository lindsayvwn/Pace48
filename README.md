# ⛰️ NH48 Pace Prediction

In 1957, the Appalachian Mountain Club (AMC) established the New Hampshire 48 hiking challenge. To complete the challenge, you must hike all 48 peaks in the white mountains that stand at least 4,0000 feet above sea level (_https://4000footers.com/nh.shtml_)

This project focuses on creating a streamlit app that will ingest your Garmin hiking data to predict how long it will take you to complete each NH48 (See the [**Notes**](#notes) section for current limitations and the roadmap for the app.)

**Live app:** _https://pace48.streamlit.app/_

## How it works
1. **Upload** your Garmin `Activities.csv` (required: supplies activity titles) plus the `.zip`/`.fit` files for your hikes
   (recommended: Garmin's CSV often leaves climb blank). Rows are matched by start time.
2. **Tag & exclude**: untick *Use* to leave an activity out; pick NH 4000 peaks (auto-suggested from titles).
3. **Save my hikes** (download button): one condensed file with your hike numbers, tags and exclusions. Next visit, upload it
   as step ⓪ plus only your *new* hikes (Activities.csv + new .zip/.fit files); no need to re-upload everything.
4. **Explore** columns with line / scatter / histogram charts.
5. **Your pace profile** (needs the .zip/.fit files): moving vs stopped time, flat-ground pace, climb rate on uphill and
   steep (15%+) sections, descent rate, and an elevation + heart-rate profile with your stops marked for each hike.
6. **Model**: pick two inputs + a target, fit a linear regression, type values, get a prediction.
7. **Predict a peak**: the simple model is `distance_mi + ascent_ft → time_hr`, applied to each peak's reference route.
   With 5+ hikes that have track data you can switch to a *moving + stops* model (moving time from distance and climb, plus
   your typical stopped time). The app compares the two with a leave-one-out test and defaults to the one that misses by less.

With fewer than 5 hikes the app falls back to the Naismith rule scaled to your pace and says so on screen.

## Privacy
- Your Garmin data is **never committed** (`.gitignore` blocks `.fit` and `Activities.csv`) and is processed in memory only.
- From `.fit` files only distance, climb, time, date, heart-rate summaries and a small (~80 point) elevation/heart-rate profile are kept: no GPS coordinates, name or weight.
- `data/demo_activities_SYNTHETIC.csv` is **fake** data for trying the app.

## Data
- `data/nh4000_peaks.csv`: peak, elevation and prominence (from my NH 4000 checklist) plus an *approximate* single-route
  round-trip distance and elevation gain gathered from public hiking sites (Trailspotting / AllTrails). Peaks that share a
  route share the same numbers. Double-check against a current guidebook.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Files
`app.py` UI · `garmin_io.py` .fit/CSV parsing, matching, cleaning · `tags_io.py` save/restore tags · `track_features.py` per-hike moving/stopped, grade-based paces and profile · `model.py` features, regression, moving+stops model, fallback
