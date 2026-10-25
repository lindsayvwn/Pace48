# ⛰️ NH 4000-Footer Pace Predictor

A Streamlit app that learns **your** hiking pace from Garmin files and predicts how long each of New Hampshire's
48 four-thousand-footers will take you.

**Live app:** _https://pace48.streamlit.app/_

## How it works
1. **Upload** your Garmin `Activities.csv` (required: supplies activity titles) plus the `.zip`/`.fit` files for your hikes
   (recommended: Garmin's CSV often leaves climb blank). Rows are matched by start time.
2. **Tag & exclude**: untick *Use* to leave an activity out; pick NH 4000 peaks (auto-suggested from titles).
3. **Save your tags** (download button) and upload that file next visit so you don't have to re-tag.
4. **Explore** columns with line / scatter / histogram charts.
5. **Model**: pick two inputs + a target, fit a linear regression, type values, get a prediction.
6. **Predict a peak**: the default model is `distance_mi + ascent_ft → time_hr`, applied to each peak's reference route.

With fewer than 5 hikes the app falls back to the Naismith rule scaled to your pace and says so on screen.

## Privacy
- Your Garmin data is **never committed** (`.gitignore` blocks `.fit` and `Activities.csv`) and is processed in memory only.
- From `.fit` files only distance, climb, time, date and heart-rate summaries are read: no GPS track, name or weight.
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
`app.py` UI · `garmin_io.py` .fit/CSV parsing, matching, cleaning · `tags_io.py` save/restore tags · `model.py` features, regression, fallback
