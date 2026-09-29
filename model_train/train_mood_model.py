"""
SurSaathi — Mood Classification Model Training
------------------------------------------------
Pipeline:
    Song Lyrics
        ↓
    Text Preprocessing
        ↓
    Word TF-IDF + Character TF-IDF
        ↓
    Metadata Features (optional)
        ↓
    Linear SVM / Logistic Regression
        ↓
    Mood Prediction
        ↓
    Accuracy, Precision, Recall, F1, Confusion Matrix
        ↓
    Save trained model

Run:
    pip install pandas scikit-learn matplotlib seaborn joblib scipy
    python train_mood_model.py

Required input:
    song_cleaned.csv

Expected columns:
    lyrics
    label
    energy
    theme
    occasion

Output files:
    mood_model.pkl
    tfidf_vectorizer.pkl
    metadata_encoder.pkl
    confusion_matrix.png
    model_metrics.json
"""

import re
import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from scipy.sparse import hstack

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    StratifiedKFold
)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import FeatureUnion

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

CSV_PATH = BASE_DIR / "song_cleaned.csv"

MODEL_PATH = BASE_DIR / "mood_model.pkl"
VECTORIZER_PATH = BASE_DIR / "tfidf_vectorizer.pkl"
METADATA_ENCODER_PATH = BASE_DIR / "metadata_encoder.pkl"

CONFUSION_MATRIX_PATH = BASE_DIR / "confusion_matrix.png"
METRICS_PATH = BASE_DIR / "model_metrics.json"


# ============================================================
# SETTINGS
# ============================================================

RANDOM_STATE = 42

USE_METADATA_FEATURES = True

METADATA_COLUMNS = [
    "energy",
    "theme",
    "occasion"
]


# ============================================================
# ROMAN HINDI + ENGLISH STOPWORDS
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
    "hi",
    "bas",

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
# TEXT PREPROCESSING
# ============================================================

def clean_lyrics(text: str) -> str:
    """
    Preprocess song lyrics.

    Steps:
        1. Convert to lowercase
        2. Remove numbers and punctuation
        3. Tokenize using whitespace
        4. Remove Roman-Hindi stopwords
        5. Remove very short tokens

    Stemming and lemmatization are intentionally not used because
    the lyrics contain Romanized Hindi + English. Generic English
    stemming/lemmatization can incorrectly alter Roman-Hindi words.
    Character TF-IDF is used to handle spelling variations instead.
    """

    text = str(text).lower()

    # Keep only alphabetic characters and spaces
    text = re.sub(r"[^a-z\s]", " ", text)

    words = []

    for word in text.split():

        # Remove stopwords
        if word in HINDI_ROMAN_STOPWORDS:
            continue

        # Remove single-character words
        if len(word) <= 1:
            continue

        words.append(word)

    return " ".join(words)


# ============================================================
# MAIN TRAINING FUNCTION
# ============================================================

def main():

    print("=" * 70)
    print("SURSAATHI MOOD CLASSIFICATION MODEL TRAINING")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. LOAD DATASET
    # --------------------------------------------------------

    if not CSV_PATH.exists():
        print("\nERROR: song_cleaned.csv was not found.")
        print(f"Expected location: {CSV_PATH}")
        return

    df = pd.read_csv(CSV_PATH)

    required_columns = [
        "lyrics",
        "label"
    ]

    for column in required_columns:

        if column not in df.columns:
            print(f"\nERROR: Missing required column: {column}")
            return

    print(f"\nDataset loaded successfully.")
    print(f"Total songs: {len(df)}")
    print(f"Number of mood classes: {df['label'].nunique()}")

    print("\nMood distribution:")
    print(df["label"].value_counts())

    # --------------------------------------------------------
    # 2. CLEAN DATA
    # --------------------------------------------------------

    df = df.dropna(
        subset=["lyrics", "label"]
    ).reset_index(drop=True)

    # Make sure metadata columns exist
    if USE_METADATA_FEATURES:

        for column in METADATA_COLUMNS:

            if column not in df.columns:
                print(
                    f"\nWARNING: Metadata column '{column}' "
                    "not found. Metadata features disabled."
                )

                USE_METADATA = False
                break

        else:
            USE_METADATA = True

    else:
        USE_METADATA = False

    print(
        f"\nMetadata features enabled: "
        f"{USE_METADATA}"
    )

    # --------------------------------------------------------
    # 3. PREPROCESS LYRICS
    # --------------------------------------------------------

    print("\nPreprocessing lyrics...")

    df["clean_lyrics"] = df["lyrics"].apply(clean_lyrics)

    # Remove empty lyrics after preprocessing
    df = df[
        df["clean_lyrics"].str.strip().astype(bool)
    ].reset_index(drop=True)

    print(
        f"Songs remaining after preprocessing: "
        f"{len(df)}"
    )

    # --------------------------------------------------------
    # 4. TRAIN / TEST SPLIT
    # --------------------------------------------------------

    print("\nSplitting dataset into training and testing sets...")

    if USE_METADATA:

        text_train, text_test, meta_train, meta_test, y_train, y_test = (
            train_test_split(
                df["clean_lyrics"],
                df[METADATA_COLUMNS],
                df["label"],
                test_size=0.20,
                random_state=RANDOM_STATE,
                stratify=df["label"]
            )
        )

    else:

        text_train, text_test, y_train, y_test = (
            train_test_split(
                df["clean_lyrics"],
                df["label"],
                test_size=0.20,
                random_state=RANDOM_STATE,
                stratify=df["label"]
            )
        )

    print(f"Training samples: {len(text_train)}")
    print(f"Testing samples:  {len(text_test)}")

    # --------------------------------------------------------
    # 5. TF-IDF FEATURE EXTRACTION
    # --------------------------------------------------------

    print("\nCreating TF-IDF features...")

    # Word-level TF-IDF
    word_vec = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=8000,
        min_df=2,
        sublinear_tf=True
    )

    # Character-level TF-IDF
    # Helps with Roman-Hindi spelling variations
    char_vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        max_features=8000,
        min_df=2,
        sublinear_tf=True
    )

    # Combine word and character features
    vectorizer = FeatureUnion([
        ("word", word_vec),
        ("char", char_vec)
    ])

    # Fit ONLY on training data
    X_train_vec = vectorizer.fit_transform(text_train)

    # Apply learned vocabulary to test data
    X_test_vec = vectorizer.transform(text_test)

    print(
        f"TF-IDF feature count: "
        f"{X_train_vec.shape[1]}"
    )

    # --------------------------------------------------------
    # 6. METADATA FEATURES
    # --------------------------------------------------------

    meta_encoder = None

    if USE_METADATA:

        print("\nEncoding metadata features...")

        meta_encoder = OneHotEncoder(
            handle_unknown="ignore"
        )

        meta_train_vec = meta_encoder.fit_transform(
            meta_train
        )

        meta_test_vec = meta_encoder.transform(
            meta_test
        )

        X_train_vec = hstack([
            X_train_vec,
            meta_train_vec
        ])

        X_test_vec = hstack([
            X_test_vec,
            meta_test_vec
        ])

        print(
            f"Metadata features: "
            f"{meta_train_vec.shape[1]}"
        )

        print(
            f"Total combined features: "
            f"{X_train_vec.shape[1]}"
        )

    # --------------------------------------------------------
    # 7. TRAIN ML MODELS
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("TRAINING MACHINE LEARNING MODELS")
    print("=" * 70)

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    param_grid = {
        "C": [0.1, 1, 10]
    }

    candidates = {

        "Linear SVM": GridSearchCV(
            LinearSVC(
                class_weight="balanced",
                random_state=RANDOM_STATE,
                max_iter=5000
            ),
            param_grid,
            cv=cv,
            scoring="f1_weighted",
            n_jobs=-1
        ),

        "Logistic Regression": GridSearchCV(
            LogisticRegression(
                max_iter=3000,
                class_weight="balanced",
                random_state=RANDOM_STATE
            ),
            param_grid,
            cv=cv,
            scoring="f1_weighted",
            n_jobs=-1
        )
    }

    results = {}

    # --------------------------------------------------------
    # 8. EVALUATE EACH MODEL
    # --------------------------------------------------------

    for name, model in candidates.items():

        print("\n" + "-" * 70)
        print(name)
        print("-" * 70)

        model.fit(
            X_train_vec,
            y_train
        )

        predictions = model.predict(
            X_test_vec
        )

        accuracy = accuracy_score(
            y_test,
            predictions
        )

        precision = precision_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0
        )

        recall = recall_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0
        )

        results[name] = {
            "model": model,
            "predictions": predictions,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }

        print(
            f"Best parameters: "
            f"{model.best_params_}"
        )

        print(
            f"Accuracy : {accuracy:.4f}"
        )

        print(
            f"Precision: {precision:.4f}"
        )

        print(
            f"Recall   : {recall:.4f}"
        )

        print(
            f"F1 Score : {f1:.4f}"
        )

        print("\nClassification Report:")

        print(
            classification_report(
                y_test,
                predictions,
                zero_division=0
            )
        )

    # --------------------------------------------------------
    # 9. SELECT BEST MODEL
    # --------------------------------------------------------

    best_name = max(
        results,
        key=lambda name: results[name]["f1"]
    )

    best_model = results[best_name]["model"]
    best_predictions = results[best_name]["predictions"]

    best_accuracy = results[best_name]["accuracy"]
    best_precision = results[best_name]["precision"]
    best_recall = results[best_name]["recall"]
    best_f1 = results[best_name]["f1"]

    print("\n" + "=" * 70)
    print("BEST MODEL")
    print("=" * 70)

    print(f"Model     : {best_name}")
    print(f"Accuracy  : {best_accuracy:.4f}")
    print(f"Precision : {best_precision:.4f}")
    print(f"Recall    : {best_recall:.4f}")
    print(f"F1 Score  : {best_f1:.4f}")

    # --------------------------------------------------------
    # 10. CONFUSION MATRIX
    # --------------------------------------------------------

    labels_sorted = sorted(
        df["label"].unique()
    )

    cm = confusion_matrix(
        y_test,
        best_predictions,
        labels=labels_sorted
    )

    # Remove emoji from labels for matplotlib compatibility
    plot_labels = [
        re.sub(
            r"[^\x00-\x7F]+",
            "",
            str(label)
        ).strip()
        for label in labels_sorted
    ]

    plt.figure(
        figsize=(10, 8)
    )

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Purples",
        xticklabels=plot_labels,
        yticklabels=plot_labels
    )

    plt.title(
        f"Confusion Matrix - {best_name}"
    )

    plt.xlabel(
        "Predicted Mood"
    )

    plt.ylabel(
        "Actual Mood"
    )

    plt.xticks(
        rotation=45,
        ha="right"
    )

    plt.tight_layout()

    plt.savefig(
        CONFUSION_MATRIX_PATH,
        dpi=150
    )

    plt.close()

    print(
        f"\nConfusion matrix saved to:"
        f"\n{CONFUSION_MATRIX_PATH}"
    )

    # --------------------------------------------------------
    # 11. SAVE MODEL
    # --------------------------------------------------------

    print("\nSaving trained model...")

    joblib.dump(
        best_model,
        MODEL_PATH
    )

    joblib.dump(
        vectorizer,
        VECTORIZER_PATH
    )

    if USE_METADATA and meta_encoder is not None:

        joblib.dump(
            meta_encoder,
            METADATA_ENCODER_PATH
        )

    # --------------------------------------------------------
    # 12. SAVE METRICS
    # --------------------------------------------------------

    metrics = {

        "best_model": best_name,

        "accuracy": round(
            best_accuracy,
            4
        ),

        "precision": round(
            best_precision,
            4
        ),

        "recall": round(
            best_recall,
            4
        ),

        "f1_score": round(
            best_f1,
            4
        ),

        "training_samples": int(
            len(text_train)
        ),

        "testing_samples": int(
            len(text_test)
        ),

        "feature_count": int(
            X_train_vec.shape[1]
        ),

        "metadata_features": USE_METADATA
    }

    with open(
        METRICS_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metrics,
            f,
            indent=4
        )

    # --------------------------------------------------------
    # 13. FINAL OUTPUT
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 70)

    print("\nGenerated files:")

    print(f"1. {MODEL_PATH}")
    print(f"2. {VECTORIZER_PATH}")

    if USE_METADATA:
        print(f"3. {METADATA_ENCODER_PATH}")

    print(f"4. {CONFUSION_MATRIX_PATH}")
    print(f"5. {METRICS_PATH}")

    print("\nThe trained model can now be used by Flask.")


if __name__ == "__main__":
    main()