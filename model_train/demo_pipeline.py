"""
SurSaathi — step-by-step demo of the NLP pipeline for ONE song.
Shows: raw lyrics -> preprocessing -> TF-IDF weights -> cosine similarity
(with the words that drive the match) -> mood prediction.

Run:  python demo_pipeline.py "Tum Kahaan The"
"""
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from nlp_pipeline import clean_roman_baseline

warnings.filterwarnings("ignore")
BASE = Path(__file__).resolve().parent
df = pd.read_csv(BASE / "song_cleaned.csv")

query = " ".join(sys.argv[1:]) or "Ek Haseena Thi Ek Deewana Tha"
hit = df[df.music_name.str.lower() == query.lower()]
if hit.empty:
    hit = df[df.music_name.str.lower().str.contains(query.lower())]
i = hit.index[0]
print(f"SONG: {df.music_name[i]} — {df.singer[i]}\n")

print("1) RAW LYRICS (first 200 chars)\n  ", df.lyrics[i][:200], "\n")
docs = df.lyrics.map(clean_roman_baseline)
print("2) AFTER PREPROCESSING (lowercase, a-z only, stopwords out)\n  ",
      docs[i][:200], "\n")

vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
X = vec.fit_transform(docs)
terms = np.array(vec.get_feature_names_out())
row = X[i].toarray().ravel()
top = row.argsort()[::-1][:10]
print("3) TOP TF-IDF TERMS FOR THIS SONG (weight = tf × idf, L2-normalised)")
for t in top:
    print(f"   {terms[t]:22s} {row[t]:.3f}")

S = cosine_similarity(X[i], X).ravel()
S[i] = -1
print("\n4) COSINE SIMILARITY  cos(a,b) = a·b / (|a||b|)  (rows are unit length)")
for j in S.argsort()[::-1][:5]:
    shared = (X[i].multiply(X[j])).toarray().ravel()
    why = ", ".join(terms[shared.argsort()[::-1][:4]])
    print(f"   {S[j]:.3f}  {df.music_name[j]:35s} shared terms: {why}")

model = joblib.load(BASE / "mood_model.pkl")
union = joblib.load(BASE / "tfidf_vectorizer.pkl")
enc = joblib.load(BASE / "metadata_encoder.pkl")
feats = hstack([union.transform([docs[i]]),
                enc.transform(df.loc[[i], ["energy", "theme", "occasion"]])])
scores = model.decision_function(feats)[0]
print(f"\n5) MOOD CLASSIFICATION ({type(model).__name__}) — picks the class with the highest score")
for k in scores.argsort()[::-1][:3]:
    print(f"   {model.classes_[k]:18s} score {scores[k]:+.2f}")
print(f"\n   Predicted: {model.predict(feats)[0]}   |   Dataset label: {df.label[i]}")
