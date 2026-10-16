"""
SurSaathi — Preprocessing ablation (raw -> cleaned -> + variants -> + stemming
-> + repetition handling -> + data-driven stopwords), Roman AND Devanagari.

Run:  python ablation_preprocessing.py
Out:  results/preprocessing_ablation.md / .csv, results/data_driven_stopwords.txt
Protocol: LogisticRegression(C=10, balanced), word 1-2-gram + char_wb 3-5-gram
TF-IDF, 5-fold stratified CV x 2 repeats, vectorizers fitted inside folds.
NOTE: Devanagari vectorizers use token_pattern=r"\\S+". sklearn's default
pattern splits Devanagari words at vowel signs (kahaan -> 'kah'), which made
the first Devanagari experiment in evaluate_models.py invalid.
"""
import warnings
from collections import Counter
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from nlp_pipeline import (clean_roman_baseline, clean_roman_v2,
                          clean_devanagari, clean_devanagari_v2,
                          ROMAN_STOPWORDS)

warnings.filterwarnings("ignore")
BASE = Path(__file__).resolve().parent
OUT = BASE / "results"; OUT.mkdir(exist_ok=True)
SEED, META = 42, ["energy", "theme", "occasion"]

df = pd.read_csv(BASE / "song_cleaned.csv").dropna(subset=["lyrics", "label"]).reset_index(drop=True)
y = df["label"]

# ---------------- text variants
V = {
 "roman_raw":      df["lyrics"],
 "roman_clean":    df["lyrics"].map(clean_roman_baseline),
 "roman_variants": df["lyrics"].map(lambda t: clean_roman_v2(t, variants=True)),
 "roman_stem":     df["lyrics"].map(lambda t: clean_roman_v2(t, variants=True, stem=True)),
 "roman_repeat":   df["lyrics"].map(lambda t: clean_roman_v2(t, variants=False, repeat=True)),
 "roman_all":      df["lyrics"].map(lambda t: clean_roman_v2(t, variants=True, stem=True, repeat=True)),
 "dev_raw":        df["lyrics_devanagari"],
 "dev_clean":      df["lyrics_devanagari"].map(clean_devanagari),
 "dev_stem":       df["lyrics_devanagari"].map(lambda t: clean_devanagari_v2(t, stem=True)),
 "dev_stem_rep":   df["lyrics_devanagari"].map(lambda t: clean_devanagari_v2(t, stem=True, repeat=True)),
}
for k, v in V.items():
    df[k] = v

# ---------------- data-driven stopwords report (corpus statistics)
docs = [set(t.split()) for t in df["roman_clean"]]
dfreq = Counter(w for d in docs for w in d)
n = len(docs)
frequent = [(w, c / n) for w, c in dfreq.most_common(60) if w not in ROMAN_STOPWORDS]
(OUT / "data_driven_stopwords.txt").write_text(
    "Words appearing in the most songs (document frequency) that are NOT in the\n"
    "hand-made stopword list. Candidates for data-driven stopwords; in the\n"
    "ablation this is applied as max_df=0.3 (drop terms in >30% of songs).\n\n" +
    "\n".join(f"{w:14s} {p:5.1%}" for w, p in frequent[:40]), encoding="utf-8")
print("Top corpus-frequent non-stopwords:", [w for w, _ in frequent[:15]])


# ---------------- model builder
def word(max_df=1.0, dev=False):
    kw = dict(ngram_range=(1, 2), max_features=8000, min_df=2,
              sublinear_tf=True, max_df=max_df)
    if dev:
        kw["token_pattern"] = r"\S+"; kw["lowercase"] = False
    return TfidfVectorizer(**kw)


def char(dev=False):
    return TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                           max_features=8000, min_df=2, sublinear_tf=True,
                           lowercase=not dev)


def build(rom=None, dev=None, meta=False, max_df=1.0):
    parts = []
    if rom:
        parts += [("rw", word(max_df), rom), ("rc", char(), rom)]
    if dev:
        parts += [("dw", word(max_df, True), dev), ("dc", char(True), dev)]
    if meta:
        parts.append(("m", OneHotEncoder(handle_unknown="ignore"), META))
    return Pipeline([("f", ColumnTransformer(parts)),
                     ("c", LogisticRegression(C=10, max_iter=3000,
                          class_weight="balanced", random_state=SEED))])


EXP = {
 "R0 Roman: raw (lowercase only)":                    build("roman_raw"),
 "R1 Roman: cleaned (current baseline)":              build("roman_clean"),
 "R2 R1 + spelling variants (rules+dict+mined)":      build("roman_variants"),
 "R3 R2 + suffix stemming":                           build("roman_stem"),
 "R4 R1 + repetition handling (dedupe lines/runs)":   build("roman_repeat"),
 "R5 R1 + data-driven stopwords (max_df=0.3)":        build("roman_clean", max_df=0.3),
 "R6 all Roman steps combined":                       build("roman_all", max_df=0.3),
 "D0 Devanagari: raw":                                build(None, "dev_raw"),
 "D1 Devanagari: cleaned":                            build(None, "dev_clean"),
 "D2 D1 + suffix stemming":                           build(None, "dev_stem"),
 "D3 D2 + repetition handling":                       build(None, "dev_stem_rep"),
 "X1 Roman(all) + Devanagari(stem)":                  build("roman_all", "dev_stem", max_df=0.3),
 "R1 + metadata":                                     build("roman_clean", meta=True),
 "R6 + metadata":                                     build("roman_all", meta=True, max_df=0.3),
 "X1 + metadata":                                     build("roman_all", "dev_stem", meta=True, max_df=0.3),
}

cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=SEED)
sc = {"acc": "accuracy", "f1m": "f1_macro", "f1w": "f1_weighted"}
rows = []
for name, m in EXP.items():
    r = cross_validate(m, df, y, cv=cv, scoring=sc)
    row = {"experiment": name}
    for k in sc:
        row[k] = f"{r['test_'+k].mean():.3f} ± {r['test_'+k].std():.3f}"
    rows.append(row); print(f"{name:52s} acc {row['acc']}  macroF1 {row['f1m']}", flush=True)

res = pd.DataFrame(rows)
res.to_csv(OUT / "preprocessing_ablation.csv", index=False)
with open(OUT / "preprocessing_ablation.md", "w", encoding="utf-8") as f:
    f.write("| Experiment | Accuracy | Macro-F1 | Weighted-F1 |\n|---|---|---|---|\n")
    for _, r in res.iterrows():
        f.write(f"| {r.experiment} | {r.acc} | {r.f1m} | {r.f1w} |\n")
print("DONE")
