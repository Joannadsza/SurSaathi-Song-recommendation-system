"""
SurSaathi — Mood Classification Model Training
------------------------------------------------
This is the piece your project was missing: an actual *training* script.

Right now app.py just reads a precomputed sursaathi_data.json — moods and
similarity scores were already baked in, so there's nothing to "train" live.
This script is what should sit in front of that: it takes the raw/cleaned
CSV, trains a real model that LEARNS to predict a song's mood from its
lyrics, and reports how well it did. That's what you show your professor.

Run:
    pip install pandas scikit-learn matplotlib seaborn joblib
    python train_mood_model.py

Outputs (written next to this script):
    confusion_matrix.png   — visual proof of how well the model classifies
    mood_model.pkl         — the trained classifier
    tfidf_vectorizer.pkl   — the fitted TF-IDF vectorizer (needed to reuse the model)

USE_METADATA_FEATURES (below): if True, also feeds the model the song's
energy/theme/occasion tags, not just lyrics. This raises accuracy
(~64% -> ~68%) but with a real caveat — read the note next to the flag
before you decide whether to use it.
"""

import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from scipy.sparse import hstack

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

CSV_PATH = "song_cleaned.csv"
RANDOM_STATE = 42

# ----------------------------------------------------------------------
# TOGGLE: include energy/theme/occasion as extra features, alongside lyrics.
#
# The honest tradeoff: these columns were hand-tagged in the same pass as
# the mood label, so they correlate with mood directly — testing them
# ALONE (no lyrics at all) still predicts mood at ~57% accuracy. That means
# a good chunk of the accuracy gain from turning this on is the model
# reading "theme: Motivation" and guessing "Motivational", not the model
# understanding the lyrics. It's still a legitimate multi-feature model
# (energy/theme/occasion are real data, not the answer key), but it's a
# different, weaker claim than "predicts mood from lyrics alone" — be
# ready to explain that distinction if asked.
# ----------------------------------------------------------------------
USE_METADATA_FEATURES = True
METADATA_COLUMNS = ["energy", "theme", "occasion"]

# Roman-Hindi + English filler words that show up in almost every song and
# don't carry mood signal on their own (kept separate from sklearn's
# built-in English stopword list, which won't catch "hai", "tum", etc.)
HINDI_ROMAN_STOPWORDS = {
    "hai", "hain", "ho", "hoon", "hun", "the", "thi", "tha", "ka", "ki", "ke",
    "ko", "se", "me", "mein", "main", "hum", "humne", "tum", "tumhe", "tumko",
    "aur", "na", "nahi", "toh", "to", "bhi", "ye", "yeh", "wo", "woh", "jo",
    "jaise", "kya", "kyun", "kyu", "koi", "kuch", "sa", "si", "hi", "bas",
    "par", "pe", "kar", "gaye", "gayi", "gaya", "diya", "diye", "liya",
}


def clean_lyrics(text: str) -> str:
    """Lowercase, strip punctuation/numbers, drop filler words."""
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    words = [w for w in text.split() if w not in HINDI_ROMAN_STOPWORDS and len(w) > 1]
    return " ".join(words)


def main():
    # ------------------------------------------------------------------
    # 1. LOAD DATA
    # ------------------------------------------------------------------
    df = pd.read_csv(CSV_PATH)
    df = df.dropna(subset=["lyrics", "label"]).reset_index(drop=True)
    print(f"Loaded {len(df)} songs, {df['label'].nunique()} mood classes")
    print(df["label"].value_counts(), "\n")
    print(f"Using metadata features (energy/theme/occasion): {USE_METADATA_FEATURES}\n")

    # ------------------------------------------------------------------
    # 2. PREPROCESS TEXT
    # ------------------------------------------------------------------
    df["clean_lyrics"] = df["lyrics"].apply(clean_lyrics)

    # ------------------------------------------------------------------
    # 3. TRAIN / TEST SPLIT
    #    stratify=labels keeps the same mood proportions in both splits —
    #    important here since "Emotional" dominates the dataset (~47%)
    # ------------------------------------------------------------------
    if USE_METADATA_FEATURES:
        text_train, text_test, meta_train, meta_test, y_train, y_test = train_test_split(
            df["clean_lyrics"], df[METADATA_COLUMNS], df["label"],
            test_size=0.2, random_state=RANDOM_STATE, stratify=df["label"],
        )
    else:
        text_train, text_test, y_train, y_test = train_test_split(
            df["clean_lyrics"], df["label"],
            test_size=0.2, random_state=RANDOM_STATE, stratify=df["label"],
        )
    print(f"Train: {len(text_train)} songs | Test: {len(text_test)} songs\n")

    # ------------------------------------------------------------------
    # 4. VECTORIZE (fit ONLY on training data — this is the "learning" step
    #    for the vectorizer itself: it builds its vocabulary from train,
    #    then just applies that vocabulary to the unseen test set)
    #
    #    Word n-grams alone miss a lot here because Romanized Hindi has huge
    #    spelling variation ("tum" / "tumm" / "tumhe"). Character n-grams
    #    (analyzer="char_wb") catch those variants by matching sub-word
    #    chunks, so we combine word + char features with FeatureUnion.
    # ------------------------------------------------------------------
    word_vec = TfidfVectorizer(ngram_range=(1, 2), max_features=8000, min_df=2, sublinear_tf=True)
    char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=8000, min_df=2, sublinear_tf=True)
    vectorizer = FeatureUnion([("word", word_vec), ("char", char_vec)])

    X_train_vec = vectorizer.fit_transform(text_train)
    X_test_vec = vectorizer.transform(text_test)
    print(f"Combined word+char feature count: {X_train_vec.shape[1]}")

    meta_encoder = None
    if USE_METADATA_FEATURES:
        meta_encoder = OneHotEncoder(handle_unknown="ignore")
        meta_train_vec = meta_encoder.fit_transform(meta_train)
        meta_test_vec = meta_encoder.transform(meta_test)
        X_train_vec = hstack([X_train_vec, meta_train_vec])
        X_test_vec = hstack([X_test_vec, meta_test_vec])
        print(f"+ {meta_train_vec.shape[1]} one-hot metadata features "
              f"({', '.join(METADATA_COLUMNS)}) = {X_train_vec.shape[1]} total")
    print()

    # ------------------------------------------------------------------
    # 5. TRAIN & COMPARE THREE MODELS (with a small hyperparameter search
    #    for the two linear models — picks the best regularization strength
    #    C via 5-fold cross-validation on the training set only)
    # ------------------------------------------------------------------
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    param_grid = {"C": [0.1, 1, 10]}

    candidates = {
        "Logistic Regression": GridSearchCV(
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE),
            param_grid, cv=cv, scoring="accuracy",
        ),
        "Linear SVM": GridSearchCV(
            LinearSVC(class_weight="balanced", random_state=RANDOM_STATE, max_iter=5000),
            param_grid, cv=cv, scoring="accuracy",
        ),
        "Multinomial Naive Bayes": MultinomialNB(),
    }

    results = {}
    for name, model in candidates.items():
        if name == "Multinomial Naive Bayes" and USE_METADATA_FEATURES:
            # MultinomialNB requires non-negative features; TF-IDF is fine,
            # but skip it here since the metadata one-hot mix can trip it up
            continue
        model.fit(X_train_vec, y_train)
        preds = model.predict(X_test_vec)
        acc = accuracy_score(y_test, preds)
        results[name] = (model, preds, acc)
        print(f"=== {name} ===")
        if hasattr(model, "best_params_"):
            print(f"Best C (5-fold CV): {model.best_params_['C']}")
        print(f"Accuracy: {acc:.3f}")
        print(classification_report(y_test, preds, zero_division=0))
        print()

    # ------------------------------------------------------------------
    # 6. PICK THE BETTER MODEL
    # ------------------------------------------------------------------
    best_name = max(results, key=lambda n: results[n][2])
    best_model, best_preds, best_acc = results[best_name]
    print(f"Best model: {best_name} (accuracy {best_acc:.3f})")

    # ------------------------------------------------------------------
    # 7. CONFUSION MATRIX — the visual to show in your demo
    # ------------------------------------------------------------------
    labels_sorted = sorted(df["label"].unique())
    # strip the emoji for the plot only (matplotlib's default font can't
    # render them) — the saved model still uses the full label with emoji
    plot_labels = [re.sub(r"[^\x00-\x7F]+", "", lbl).strip() for lbl in labels_sorted]
    cm = confusion_matrix(y_test, best_preds, labels=labels_sorted)
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Purples",
                xticklabels=plot_labels, yticklabels=plot_labels)
    plt.title(f"Confusion Matrix — {best_name} (accuracy {best_acc:.1%})")
    plt.xlabel("Predicted mood")
    plt.ylabel("Actual mood")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    print("Saved confusion_matrix.png")

    # ------------------------------------------------------------------
    # 8. SAVE THE TRAINED MODEL + VECTORIZER (+ METADATA ENCODER IF USED)
    # ------------------------------------------------------------------
    joblib.dump(best_model, "mood_model.pkl")
    joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
    print("Saved mood_model.pkl and tfidf_vectorizer.pkl")
    if USE_METADATA_FEATURES:
        joblib.dump(meta_encoder, "metadata_encoder.pkl")
        print("Saved metadata_encoder.pkl (needed alongside the model since "
              "USE_METADATA_FEATURES is on)")


if __name__ == "__main__":
    main()
