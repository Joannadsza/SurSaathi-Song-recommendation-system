"""
SurSaathi backend API
----------------------
Serves song metadata, mood filters, search, and two kinds of
recommendations (lyric-similarity via precomputed TF-IDF, and
mood-overlap) from a precomputed dataset.

Run:
    pip install -r requirements.txt
    python app.py
Server starts on http://localhost:5001
"""
import json
import random
from pathlib import Path

from flask import Flask, jsonify, request
from flask_cors import CORS

DATA_PATH = Path(__file__).parent / "data" / "sursaathi_data.json"

app = Flask(__name__)
CORS(app)

with open(DATA_PATH, "r", encoding="utf-8") as f:
    RAW = json.load(f)

SONGS = RAW["songs"]
BY_ID = {s["id"]: s for s in SONGS}

ALL_MOODS = [
    "Happy 😊", "Sad 😢", "Romantic ❤️", "Party 🎉", "Dance 💃",
    "Devotional 🙏", "Motivational 💪", "Calm 🌿", "Emotional 💙", "Patriotic 🇮🇳",
]


def to_card(song):
    """Slim representation used in lists / grids — no full lyrics payload."""
    return {
        "id": song["id"],
        "name": song["name"],
        "nameDev": song["nameDev"],
        "singer": song["singer"],
        "singerDev": song["singerDev"],
        "year": song["year"],
        "thumb": song["thumb"],
        "moods": song["moods"],
        "theme": song["theme"],
        "energy": song["energy"],
    }


def to_detail(song):
    return {
        **to_card(song),
        "occasion": song["occasion"],
        "sentiment": song["sentiment"],
        "primaryEmotion": song["primaryEmotion"],
        "lyrics": song["lyrics"],
        "lyricsDevSnippet": song["lyricsDevSnippet"],
    }


@app.get("/api/moods")
def get_moods():
    return jsonify(ALL_MOODS)


@app.get("/api/songs/featured")
def featured():
    """Small curated set for the homepage — never the whole catalog."""
    mood = request.args.get("mood")
    limit = int(request.args.get("limit", 4))
    pool = SONGS
    if mood:
        pool = [s for s in pool if mood in s["moods"]]
    sample = random.sample(pool, min(limit, len(pool))) if pool else []
    return jsonify([to_card(s) for s in sample])


@app.get("/api/search")
def search():
    q = request.args.get("q", "").strip().lower()
    limit = int(request.args.get("limit", 8))
    if not q:
        return jsonify([])
    results = [
        s for s in SONGS
        if q in s["name"].lower()
        or q in s["singer"].lower()
        or q in s["theme"].lower()
        or q in s["nameDev"]
        or q in s["lyrics"].lower()
    ][:limit]
    return jsonify([to_card(s) for s in results])


@app.get("/api/songs/<int:song_id>")
def song_detail(song_id):
    song = BY_ID.get(song_id)
    if not song:
        return jsonify({"error": "not found"}), 404
    return jsonify(to_detail(song))


@app.get("/api/songs/<int:song_id>/recommendations")
def recommendations(song_id):
    song = BY_ID.get(song_id)
    if not song:
        return jsonify({"error": "not found"}), 404
    limit = int(request.args.get("limit", 4))

    lyric_recs = [
        {**to_card(BY_ID[x["id"]]), "matchScore": x["score"]}
        for x in song["sims"][:limit]
        if x["id"] in BY_ID
    ]

    mood_pool = [
        (s, len(set(s["moods"]) & set(song["moods"])))
        for s in SONGS if s["id"] != song_id
    ]
    mood_pool = [p for p in mood_pool if p[1] > 0]
    mood_pool.sort(key=lambda p: (-p[1], -p[0]["year"]))
    mood_recs = [to_card(s) for s, _ in mood_pool[:limit]]

    return jsonify({"byLyrics": lyric_recs, "byMood": mood_recs})


@app.get("/api/songs/<int:song_id>/karaoke")
def karaoke(song_id):
    """Lyrics-only payload for the karaoke view."""
    song = BY_ID.get(song_id)
    if not song:
        return jsonify({"error": "not found"}), 404
    lines = [ln for ln in song["lyrics"].split("  ") if ln.strip()]
    return jsonify({
        "id": song["id"],
        "name": song["name"],
        "nameDev": song["nameDev"],
        "singer": song["singer"],
        "thumb": song["thumb"],
        "lines": lines,
    })


if __name__ == "__main__":
    app.run(port=5001, debug=True)