"""
SurSaathi Backend API
---------------------

Features:
    - Song metadata
    - Mood filters
    - Song search
    - Lyric-based recommendations
    - Mood-based recommendations
    - ML-based mood prediction
    - Trained Linear SVM / Logistic Regression integration

Server:
    http://localhost:5001

Run:
    python app.py
"""

import json
import random
import re
from pathlib import Path

import joblib
import pandas as pd

from flask import Flask, jsonify, request
from flask_cors import CORS

from scipy.sparse import hstack


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "sursaathi_data.json"

MODEL_DIR = BASE_DIR.parent / "model_train"

MODEL_PATH = MODEL_DIR / "mood_model.pkl"

VECTORIZER_PATH = MODEL_DIR / "tfidf_vectorizer.pkl"

METADATA_ENCODER_PATH = MODEL_DIR / "metadata_encoder.pkl"


# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)

CORS(app)


# ============================================================
# LOAD SONG DATA
# ============================================================

with open(
    DATA_PATH,
    "r",
    encoding="utf-8"
) as f:

    RAW = json.load(f)


SONGS = RAW["songs"]

BY_ID = {
    song["id"]: song
    for song in SONGS
}


# ============================================================
# MOODS
# ============================================================

ALL_MOODS = [

    "Happy 😊",
    "Sad 😢",
    "Romantic ❤️",
    "Party 🎉",
    "Dance 💃",
    "Motivational 💪",
    "Calm 🌿",
    "Emotional 💙",
    "Patriotic 🇮🇳",

]


# ============================================================
# STOPWORDS
# ============================================================

HINDI_ROMAN_STOPWORDS = {

    "hai", "hain", "ho", "hoon", "hun",
    "the", "thi", "tha",

    "ka", "ki", "ke", "ko",
    "se", "me", "mein", "main",

    "hum", "humne",
    "tum", "tumhe", "tumko",

    "aur",
    "na", "nahi",
    "toh", "to",
    "bhi",

    "ye", "yeh",
    "wo", "woh",

    "jo",
    "jaise",

    "kya",
    "kyun",
    "kyu",

    "koi",
    "kuch",

    "sa", "si",
    "hi", "bas",

    "par",
    "pe",

    "kar",
    "gaye",
    "gayi",
    "gaya",

    "diya",
    "diye",

    "liya"

}


# ============================================================
# LOAD ML MODEL
# ============================================================

MOOD_MODEL = None

TFIDF_VECTORIZER = None

METADATA_ENCODER = None

MODEL_READY = False

MODEL_NAME = "Not loaded"


def load_ml_model():

    global MOOD_MODEL
    global TFIDF_VECTORIZER
    global METADATA_ENCODER
    global MODEL_READY
    global MODEL_NAME

    try:

        if not MODEL_PATH.exists():

            print(
                "\nML model not found."
            )

            print(
                "Run model_train/train_mood_model.py first."
            )

            return

        if not VECTORIZER_PATH.exists():

            print(
                "\nTF-IDF vectorizer not found."
            )

            return

        MOOD_MODEL = joblib.load(
            MODEL_PATH
        )

        TFIDF_VECTORIZER = joblib.load(
            VECTORIZER_PATH
        )

        if METADATA_ENCODER_PATH.exists():

            METADATA_ENCODER = joblib.load(
                METADATA_ENCODER_PATH
            )

        MODEL_READY = True

        MODEL_NAME = (
            type(MOOD_MODEL).__name__
        )

        print("\nML model loaded successfully.")

        print(
            f"Model: {MODEL_NAME}"
        )

        print(
            f"Vectorizer: {type(TFIDF_VECTORIZER).__name__}"
        )

        if METADATA_ENCODER is not None:

            print(
                "Metadata encoder: Loaded"
            )

    except Exception as error:

        MODEL_READY = False

        print(
            "\nCould not load ML model."
        )

        print(
            f"Error: {error}"
        )


load_ml_model()


# ============================================================
# TEXT PREPROCESSING
# ============================================================

def clean_lyrics(text):

    """
    Apply the same preprocessing used during training.

    IMPORTANT:
    The preprocessing must be identical during training
    and prediction.
    """

    text = str(text).lower()

    text = re.sub(
        r"[^a-z\s]",
        " ",
        text
    )

    words = []

    for word in text.split():

        if word in HINDI_ROMAN_STOPWORDS:
            continue

        if len(word) <= 1:
            continue

        words.append(word)

    return " ".join(words)


# ============================================================
# ML MOOD PREDICTION
# ============================================================

def predict_mood(
    lyrics,
    energy=None,
    theme=None,
    occasion=None
):

    """
    Predict mood using the trained ML model.

    Input:
        lyrics
        energy
        theme
        occasion

    Output:
        predicted mood
        model name
    """

    if not MODEL_READY:

        return {
            "predictedMood": None,
            "model": None,
            "modelReady": False
        }

    try:

        cleaned_text = clean_lyrics(
            lyrics
        )

        # --------------------------------------------
        # TF-IDF
        # --------------------------------------------

        text_vector = (
            TFIDF_VECTORIZER.transform(
                [cleaned_text]
            )
        )

        # --------------------------------------------
        # Metadata
        # --------------------------------------------

        if METADATA_ENCODER is not None:

            metadata = pd.DataFrame(
                [[
                    energy or "",
                    theme or "",
                    occasion or ""
                ]],
                columns=[
                    "energy",
                    "theme",
                    "occasion"
                ]
            )

            metadata_vector = (
                METADATA_ENCODER.transform(
                    metadata
                )
            )

            final_vector = hstack([
                text_vector,
                metadata_vector
            ])

        else:

            final_vector = text_vector

        # --------------------------------------------
        # MODEL PREDICTION
        # --------------------------------------------

        prediction = MOOD_MODEL.predict(
            final_vector
        )

        predicted_mood = str(
            prediction[0]
        )

        # --------------------------------------------
        # DECISION SCORE
        # --------------------------------------------

        confidence = None

        try:

            decision_scores = (
                MOOD_MODEL.decision_function(
                    final_vector
                )
            )

            if hasattr(
                decision_scores,
                "__len__"
            ):

                scores = decision_scores[0]

                # Convert decision scores into
                # simple relative confidence.
                import numpy as np

                exp_scores = np.exp(
                    scores - np.max(scores)
                )

                probabilities = (
                    exp_scores /
                    exp_scores.sum()
                )

                confidence = round(
                    float(
                        np.max(probabilities)
                    ) * 100,
                    2
                )

        except Exception:

            confidence = None

        return {

            "predictedMood": predicted_mood,

            "model": MODEL_NAME,

            "confidence": confidence,

            "modelReady": True

        }

    except Exception as error:

        print(
            f"Prediction error: {error}"
        )

        return {

            "predictedMood": None,

            "model": MODEL_NAME,

            "confidence": None,

            "modelReady": False,

            "error": str(error)

        }


# ============================================================
# SONG CARD
# ============================================================

def to_card(song):

    """
    Slim representation used in lists/grids.
    """

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

        "energy": song["energy"]

    }


# ============================================================
# SONG DETAIL
# ============================================================

def to_detail(song):

    return {

        **to_card(song),

        "occasion": song["occasion"],

        "sentiment": song["sentiment"],

        "primaryEmotion": song["primaryEmotion"],

        "lyrics": song["lyrics"],

        "lyricsDevSnippet": song[
            "lyricsDevSnippet"
        ]

    }


# ============================================================
# GET MOODS
# ============================================================

@app.get("/api/moods")
def get_moods():

    return jsonify(
        ALL_MOODS
    )


# ============================================================
# FEATURED SONGS
# ============================================================

@app.get("/api/songs/featured")
def featured():

    mood = request.args.get(
        "mood"
    )

    limit = int(
        request.args.get(
            "limit",
            4
        )
    )

    pool = SONGS

    if mood:

        pool = [

            song
            for song in SONGS

            if song["moods"]
            and song["moods"][0] == mood

        ]

    sample = (

        random.sample(
            pool,
            min(
                limit,
                len(pool)
            )
        )

        if pool

        else []

    )

    return jsonify([
        to_card(song)
        for song in sample
    ])


# ============================================================
# SEARCH
# ============================================================

@app.get("/api/search")
def search():

    q = request.args.get(
        "q",
        ""
    ).strip().lower()

    limit = int(
        request.args.get(
            "limit",
            8
        )
    )

    if not q:

        return jsonify([])

    results = [

        song

        for song in SONGS

        if q in song["name"].lower()

        or q in song["singer"].lower()

        or q in song["theme"].lower()

        or q in song["nameDev"].lower()

        or q in song["lyrics"].lower()

    ][:limit]

    return jsonify([
        to_card(song)
        for song in results
    ])


# ============================================================
# SONG DETAILS
# ============================================================

@app.get("/api/songs/<int:song_id>")
def song_detail(song_id):

    song = BY_ID.get(
        song_id
    )

    if not song:

        return jsonify({
            "error": "not found"
        }), 404

    return jsonify(
        to_detail(song)
    )


# ============================================================
# ML MOOD PREDICTION FOR A SONG
# ============================================================

@app.get("/api/songs/<int:song_id>/predict-mood")
def predict_song_mood(song_id):

    song = BY_ID.get(
        song_id
    )

    if not song:

        return jsonify({
            "error": "not found"
        }), 404

    result = predict_mood(

        lyrics=song["lyrics"],

        energy=song.get(
            "energy"
        ),

        theme=song.get(
            "theme"
        ),

        occasion=song.get(
            "occasion"
        )

    )

    # Fallback if model has not been trained yet
    if not result["modelReady"]:

        fallback_mood = None

        if song.get("primaryEmotion"):

            fallback_mood = (
                song["primaryEmotion"]
            )

        elif song.get("moods"):

            fallback_mood = (
                song["moods"][0]
            )

        result["predictedMood"] = (
            fallback_mood
        )

    return jsonify({

        "songId": song_id,

        **result

    })


# ============================================================
# RECOMMENDATIONS
# ============================================================

@app.get(
    "/api/songs/<int:song_id>/recommendations"
)
def recommendations(song_id):

    song = BY_ID.get(
        song_id
    )

    if not song:

        return jsonify({
            "error": "not found"
        }), 404

    limit = int(
        request.args.get(
            "limit",
            4
        )
    )

    # --------------------------------------------------------
    # 1. LYRIC-BASED RECOMMENDATIONS
    #
    # Existing precomputed similarity scores are retained.
    # --------------------------------------------------------

    lyric_recs = [

        {
            **to_card(
                BY_ID[item["id"]]
            ),

            "matchScore": item[
                "score"
            ]

        }

        for item in song.get(
            "sims",
            []
        )[:limit]

        if item["id"] in BY_ID

    ]

    # --------------------------------------------------------
    # 2. PREDICT MOOD USING ML MODEL
    # --------------------------------------------------------

    prediction = predict_mood(

        lyrics=song["lyrics"],

        energy=song.get(
            "energy"
        ),

        theme=song.get(
            "theme"
        ),

        occasion=song.get(
            "occasion"
        )

    )

    predicted_mood = (
        prediction["predictedMood"]
    )

    # --------------------------------------------------------
    # 3. MOOD RECOMMENDATIONS
    #
    # First use predicted mood.
    # If ML model isn't available, use existing song moods.
    # --------------------------------------------------------

    if predicted_mood:

        mood_recs = [

            s

            for s in SONGS

            if s["id"] != song_id

            and predicted_mood in s.get(
                "moods",
                []
            )

        ]

    else:

        mood_recs = [

            s

            for s in SONGS

            if s["id"] != song_id

            and len(
                set(
                    s.get(
                        "moods",
                        []
                    )
                )

                & set(
                    song.get(
                        "moods",
                        []
                    )
                )
            ) > 0

        ]

    # Sort newer songs first
    mood_recs.sort(
        key=lambda s: -s["year"]
    )

    mood_recs = [
        to_card(song)
        for song in mood_recs[:limit]
    ]

    # --------------------------------------------------------
    # RESPONSE
    # --------------------------------------------------------

    return jsonify({

        "byLyrics": lyric_recs,

        "byMood": mood_recs,

        "predictedMood": predicted_mood,

        "model": prediction[
            "model"
        ],

        "confidence": prediction[
            "confidence"
        ],

        "modelReady": prediction[
            "modelReady"
        ]

    })


# ============================================================
# KARAOKE
# ============================================================

@app.get(
    "/api/songs/<int:song_id>/karaoke"
)
def karaoke(song_id):

    song = BY_ID.get(
        song_id
    )

    if not song:

        return jsonify({
            "error": "not found"
        }), 404

    lines = [

        line

        for line in song[
            "lyrics"
        ].split("  ")

        if line.strip()

    ]

    return jsonify({

        "id": song["id"],

        "name": song["name"],

        "nameDev": song["nameDev"],

        "singer": song["singer"],

        "thumb": song["thumb"],

        "lines": lines

    })


# ============================================================
# MODEL STATUS
# ============================================================

@app.get("/api/model-status")
def model_status():

    return jsonify({

        "modelReady": MODEL_READY,

        "model": MODEL_NAME,

        "vectorizerLoaded": (
            TFIDF_VECTORIZER is not None
        ),

        "metadataEncoderLoaded": (
            METADATA_ENCODER is not None
        )

    })


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    app.run(
        port=5001,
        debug=True
    )