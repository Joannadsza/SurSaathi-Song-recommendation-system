# SurSaathi — सुर साथी

A Hindi song recommendation system that suggests tracks by **lyric similarity**
(TF‑IDF over cleaned lyrics) and by **mood** (Happy, Sad, Romantic, Party,
Dance,  Motivational, Calm, Emotional, Patriotic), plus a karaoke
view for reading along to a song's lyrics.

## Project structure

```
sursaathi-project/
├── backend/              Flask API — serves songs, search, recommendations
│   ├── app.py
│   ├── requirements.txt
│   └── data/sursaathi_data.json   (974 songs, precomputed lyric-similarity)
├── model_train/           NLP pipeline, training, evaluation, similarity builder
└── frontend/              React + Vite app
    ├── src/
    │   ├── components/    Header, SearchBar, MoodChips, SongCard, SongRow, Logo
    │   ├── pages/          Home, SongDetail, Karaoke
    │   └── lib/            api.js, moods.js
    └── public/favicon.svg
```

## Running it

**1. Backend** (from `backend/`):
```bash
pip install -r requirements.txt
python app.py
```
Starts on `http://localhost:5001`.

**2. Frontend** (from `frontend/`, in a second terminal):
```bash
npm install
npm run dev
```
Starts on `http://localhost:5173` — open that in your browser.

## What's on each page

- **Home** — logo + tagline, a search bar, mood filter chips, and a small
  curated row of 4 songs (shuffles on request — never dumps the whole
  catalog on screen).
- **Song page** (`/song/:id`) — full song info, a lyric preview, and two
  four-song recommendation rows: "More like this · lyrics" and
  "More like this · mood".
- **Karaoke** (`/song/:id/karaoke`) — a clean, lyrics-only view with a
  "Follow along" button that steps through the lines at a steady pace
  (click any line to jump to it manually).

## How recommendations work

* **Lyric similarity** — `model_train/build_similarity.py` cleans lyrics,
  builds word 1-2-gram TF-IDF (+ a small theme block), computes cosine
  similarity once and stores each song's top 8 neighbours in
  `backend/data/sursaathi_data.json` (`sims`). The API only reads them.
* **Mood** — `model_train/train_mood_model.py` trains a Linear SVM / Logistic
  Regression on word + character TF-IDF plus energy/theme/occasion features.
  `backend/app.py` loads it, predicts a song's mood and returns songs tagged
  with that mood.
* Shared preprocessing: `model_train/nlp_pipeline.py` (used by training,
  evaluation, similarity and the API).
* Evaluation, ablations, error analysis: `model_train/evaluate_models.py` ->
  `model_train/results/`. Full write-up: `VIVA_NOTES.md`.

## Notes

- The frontend expects the backend at `http://localhost:5001` — change
  `BASE_URL` in `frontend/src/lib/api.js` if you deploy them separately.
- `node_modules/` isn't included — run `npm install` first.
