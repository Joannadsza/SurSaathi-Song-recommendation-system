"""
SurSaathi — OPTIONAL: dense multilingual embeddings vs TF-IDF
-------------------------------------------------------------
NOT yet run (needs internet to download a Hugging Face model).
Compares sparse TF-IDF against dense sentence embeddings for
  (a) mood classification (5-fold CV, same protocol as evaluate_models.py)
  (b) lyric-similarity quality (precision@5, same metrics as build_similarity.py)

Install:  pip install sentence-transformers
Run:      python embedding_experiment.py
Models worth trying (Devanagari input works best for Hindi):
  intfloat/multilingual-e5-base      ai4bharat/IndicBERTv2-MLM-only
  google/muril-base-cased            sentence-transformers/LaBSE
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import normalize

from build_similarity import precision_at_k
from nlp_pipeline import clean_devanagari

BASE = Path(__file__).resolve().parent
MODEL = "intfloat/multilingual-e5-base"

df = pd.read_csv(BASE / "song_cleaned.csv")
songs = json.load(open(BASE.parent / "backend/data/sursaathi_data.json"))["songs"]
tags = [set(s["moods"]) for s in songs]

texts = ["passage: " + clean_devanagari(t)[:1500] for t in df.lyrics_devanagari]
emb = SentenceTransformer(MODEL).encode(texts, batch_size=16,
                                        show_progress_bar=True)
emb = normalize(emb)
np.save(BASE / "results" / "embeddings.npy", emb)

# (a) classification
clf = LogisticRegression(C=10, max_iter=3000, class_weight="balanced")
pred = cross_val_predict(clf, emb, df.label,
                         cv=StratifiedKFold(5, shuffle=True, random_state=42))
print("Embedding LR  acc %.3f  macro-F1 %.3f" % (
    accuracy_score(df.label, pred), f1_score(df.label, pred, average="macro")))

# (b) similarity
S = cosine_similarity(emb); np.fill_diagonal(S, -1)
top = np.argsort(-S, axis=1)[:, :5]
j, l, t = precision_at_k(top, tags, df.label.values, df.theme.values)
print("Embedding similarity  Jaccard %.3f | same mood %.3f | same theme %.3f" % (j, l, t))
