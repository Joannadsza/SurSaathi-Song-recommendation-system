"""
SurSaathi — Lyric similarity builder (TF-IDF + cosine similarity)
-----------------------------------------------------------------
This is the script that produces the `sims` field stored for every song
in backend/data/sursaathi_data.json. The Flask API does NOT compute
similarity at request time; it just reads these precomputed neighbours.

Pipeline
  1. clean_roman_baseline()       lowercase, a-z only, stopwords, 1-letter tokens out
  2. TF-IDF (word 1-2 grams, min_df=2, sublinear_tf, L2-normalised rows)
  3. Theme block: one-hot(theme), L2-normalised, scaled by THEME_WEIGHT
       -> lyrics dominate; theme is a soft tie-breaker (weight chosen by
          evaluation, see results/recommender_eval.txt)
  4. Cosine similarity between all song vectors   S = X · Xᵀ
       (rows are unit-length, so the dot product IS the cosine)
  5. For each song keep the top 8 neighbours (excluding itself)
  6. Evaluate with precision@5 and write sims back into the JSON

Run:
    python build_similarity.py            # evaluate only (prints + results/)
    python build_similarity.py --write    # also overwrite sims in the JSON
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import OneHotEncoder, normalize

from nlp_pipeline import clean_roman_baseline

BASE = Path(__file__).resolve().parent
CSV = BASE / "song_cleaned.csv"
DATA = BASE.parent / "backend" / "data" / "sursaathi_data.json"
OUT = BASE / "results"
OUT.mkdir(exist_ok=True)

THEME_WEIGHT = 0.2
TOP_N = 8


def compute_similarity(df, theme_weight=THEME_WEIGHT):
    docs = df["lyrics"].map(clean_roman_baseline)
    tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    X = tfidf.fit_transform(docs)                       # (songs x terms)
    if theme_weight > 0:
        T = normalize(OneHotEncoder().fit_transform(df[["theme"]]))
        X = hstack([X, theme_weight * T]).tocsr()
    S = cosine_similarity(X)                            # (songs x songs)
    np.fill_diagonal(S, -1.0)                           # exclude itself
    return S, tfidf


def precision_at_k(top, tags, labels, themes, k=5):
    n = len(top)
    jac = np.mean([[len(tags[i] & tags[j]) / len(tags[i] | tags[j])
                    for j in top[i][:k]] for i in range(n)])
    lab = np.mean([[labels[i] == labels[j] for j in top[i][:k]]
                   for i in range(n)])
    thm = np.mean([[themes[i] == themes[j] for j in top[i][:k]]
                   for i in range(n)])
    return jac, lab, thm


def main(write):
    df = pd.read_csv(CSV)
    raw = json.load(open(DATA, encoding="utf-8"))
    songs = raw["songs"]
    assert len(songs) == len(df), "CSV and JSON must be the same songs"

    tags = [set(s["moods"]) for s in songs]
    labels, themes = df["label"].values, df["theme"].values

    S, tfidf = compute_similarity(df)
    top = np.argsort(-S, axis=1)[:, :TOP_N]
    print(f"TF-IDF vocabulary: {len(tfidf.vocabulary_)} terms")

    # ---- evaluation: random vs lyrics-only vs final
    n = len(songs)
    rng = np.random.default_rng(0)
    rand = np.array([rng.choice([j for j in range(n) if j != i], TOP_N,
                                replace=False) for i in range(n)])
    S0, _ = compute_similarity(df, theme_weight=0)
    top0 = np.argsort(-S0, axis=1)[:, :TOP_N]
    old = np.array([[x["id"] for x in s.get("sims", [])][:TOP_N]
                    for s in songs])

    lines = ["Recommender evaluation — precision@5 (higher is better)",
             "Metrics: mood-tag Jaccard | same primary mood | same theme",
             "(theme match is partly circular for the theme-weighted model)",
             ""]
    for name, t in [("Random neighbours", rand),
                    ("Lyrics-only TF-IDF cosine", top0),
                    (f"TF-IDF + theme weight {THEME_WEIGHT} (FINAL)", top),
                    ("Previously stored sims (script not in repo)", old)]:
        j, l, th = precision_at_k(t, tags, labels, themes)
        lines.append(f"{name:48s} {j:.3f} | {l:.3f} | {th:.3f}")
    report = "\n".join(lines)
    print(report)
    (OUT / "recommender_eval.txt").write_text(report, encoding="utf-8")

    # ---- worked example for the viva
    i = 0
    ex = [f"Example — most similar to '{songs[i]['name']}':"]
    for j in top[i][:5]:
        ex.append(f"  {S[i, j]:.3f}  {songs[j]['name']}")
    print("\n" + "\n".join(ex))
    (OUT / "similarity_example.txt").write_text("\n".join(ex), encoding="utf-8")

    if write:
        for i, s in enumerate(songs):
            s["sims"] = [{"id": int(j), "score": round(float(S[i, j]), 3)}
                         for j in top[i]]
        with open(DATA, "w", encoding="utf-8") as f:
            json.dump(raw, f, ensure_ascii=False)
        print(f"\nWrote new sims to {DATA}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    main(ap.parse_args().write)
