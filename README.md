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

## How stops are detected
Every hike is judged against itself, not against fixed numbers. Your progress is measured over a **1 minute moving window** (ground distance plus height
gained, Naismith-style: 1 m climbed counts like 8 m walked). A stretch is *slow* when progress is under about 40% of your typical moving speed on that hike,
**unless** you were climbing or dropping fast (about 950 ft/hr or more): slow on the ground but steep vertical effort counts as moving.

A slow stretch of 1+ minute (the default; adjustable) is a *break* if any of these hold: the watch timer paused; your GPS points stayed inside a tight
**5 metre circle** for a minute or more (a DBSCAN cluster, see below); or your heart rate fell at least 2 standard deviations below where it was while moving
(the standard deviation is how much your heart rate normally wobbles from minute to minute on that hike). Slow stretches where heart rate stayed up (ice, rock)
are shown as *slow technical* and count as moving. The shortest-break and heart-rate numbers live at the top of the **Your pace profile** tab under
"Stop detection (advanced)"; the raw slow stretches are saved with each hike, so changing them is instant. This is unsupervised: no labelled breaks, so treat the default as a starting point.

**Stationary clusters.** While a `.fit` file is being read, its own GPS positions are converted to metres and clustered with DBSCAN (eps 5 m). A cluster only counts
if it holds you for at least a minute, is compact (90% of points within 8 m of its centre) and you were not climbing quickly. Those seconds are forced to "not moving" and the stretch is a break.
Only that hike's own coordinates are ever looked at, they stay in memory for the length of the calculation, and nothing saved or shown contains a coordinate.

## Peak tags
**From your GPS track (preferred).** While a `.fit` file is read, the app checks whether the track came within 250 m of each of the 48 summits (positions in `data/nh4000_summits.csv`,
published coordinates) while also being at that peak's elevation (within about 100 ft). Every summit reached is tagged, in the order reached, so "Wildcat A and D" or a Franconia Ridge day
tags itself, and a hike whose title says Lafayette but whose track went over Liberty and Flume is tagged Liberty and Flume. Only the peak *names* are saved with the hike; coordinates are used in memory and discarded.
GPS-based tags replace title-based guesses (and older saved tags) unless you edit that hike's peaks by hand in the table.

**From the title (fallback, for files without GPS).** Titles are matched to the 48 peaks automatically, forgiving of "Mt."/"Mount" and small spelling slips ("Waumbeck"). Shorthand such as
"Wildcat A", "Tripyramid" or "Franconia Ridge" lives in `data/peak_aliases.csv` (columns `alias,peaks`; separate several peaks with `|`):
add your own lines there. Trail names that contain a peak's name but aren't the peak ("Lincoln Woods", "Lincoln Brook", "Zealand Falls") are listed in `data/peak_exclusions.csv` (`phrase,peak`) and ignored when tagging.

## Default dashboard from your own Garmin (optional, owner only)
Want the app to open on *your* hikes instead of an empty upload page? Add a Garmin **Personal API** key:

1. Create a key at <https://connect.garmin.com/app/settings/personalApiToken> (shown once; it expires, so you'll renew it).
2. Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` (or paste it into Streamlit Cloud's *Secrets*) and put the key in.
   You can also set the `GARMIN_PERSONAL_API_KEY` environment variable instead.
3. Run the app. With no uploads it shows your hikes; **anything a visitor uploads replaces them**, and visitors never see or use your key.

How it works: `garmin_personal.py` exchanges the key for a token, lists your activities, downloads each activity's original
`.zip`, and runs it through the same reader as an upload, so you get the same moving/stopped time, paces and profiles.
Parsed hikes are cached in `.cache/` (gitignored) so later loads only download new activities.

**Hiding hikes:** list hikes visitors must not see under `[garmin_personal]` in secrets (not in the repo): `hidden_dates = ["2024-05-31", "2025-01-05"]` (calendar day of the start) and/or `hidden_activity_ids = [...]`. They are removed on the server before anything reaches a visitor, including the "Save my hikes" download. Set `show_hidden = true` locally to see everything.

**Refresh button:** the banner has a "Refresh Garmin data" button that re-checks Garmin for new hikes. Anyone can press it; it is limited to once every 2 minutes for the whole app.

Things to know: everyone who opens your deployed app sees your default hikes (dates, titles, heart rate, elevation profiles),
so turn it off with `enabled = false`, or don't set a key, if that's not what you want. Check Garmin's terms for the Personal
API before exposing your data on a public site. On hosts with a temporary disk the cache is rebuilt after each restart.

## Privacy
- Your Garmin data is **never committed** (`.gitignore` blocks `.fit` and `Activities.csv`) and is processed in memory only.
- From `.fit` files only distance, climb, time, date, heart-rate summaries and a small (~80 point) elevation/heart-rate profile are kept: no GPS coordinates (positions are used in memory only to spot rests, then discarded), name or weight.
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
`app.py` UI · `garmin_io.py` .fit/CSV parsing, matching, cleaning · `tags_io.py` save/restore tags · `garmin_personal.py` optional owner dashboard via Garmin Personal API · `track_features.py` per-hike moving/stopped, grade-based paces and profile · `model.py` features, regression, moving+stops model, fallback
