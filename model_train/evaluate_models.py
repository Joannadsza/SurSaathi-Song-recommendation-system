"""
SurSaathi — Full evaluation & ablation study
--------------------------------------------
Run:  python evaluate_models.py
Needs: song_cleaned.csv, ../backend/data/sursaathi_data.json
Writes everything to results/

Experiments
  1. Baselines + ablations (repeated stratified 5-fold CV, 2 repeats)
  2. Per-class report, confusion matrix, error analysis (out-of-fold)
  3. Imbalance handling: macro-F1 + rare-class merge experiment
  4. Multi-label (one-vs-rest) formulation using all mood tags
All TF-IDF vectorizers are fitted INSIDE each CV fold (no leakage).
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix,
                             f1_score, accuracy_score)
from sklearn.model_selection import (RepeatedStratifiedKFold,
                                     StratifiedKFold, cross_validate,
                                     cross_val_predict)
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer
from sklearn.svm import LinearSVC

from nlp_pipeline import clean_roman, clean_roman_baseline, clean_devanagari

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent
OUT = BASE / "results"
OUT.mkdir(exist_ok=True)
SEED = 42
META = ["energy", "theme", "occasion"]


# ------------------------------------------------------------ data
df = pd.read_csv(BASE / "song_cleaned.csv").dropna(subset=["lyrics", "label"])
df = df.reset_index(drop=True)
df["roman_base"] = df["lyrics"].map(clean_roman_baseline)
df["roman_norm"] = df["lyrics"].map(clean_roman)
df["dev"] = df["lyrics_devanagari"].map(clean_devanagari)
y = df["label"]
print(f"{len(df)} songs, {y.nunique()} classes")


# ------------------------------------------------------------ builders
def word_vec():
    return TfidfVectorizer(ngram_range=(1, 2), max_features=8000,
                           min_df=2, sublinear_tf=True)


def char_vec():
    return TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                           max_features=8000, min_df=2, sublinear_tf=True)


def lr():
    return LogisticRegression(C=10, max_iter=3000,
                              class_weight="balanced", random_state=SEED)


def svm():
    return LinearSVC(C=1, class_weight="balanced", max_iter=5000,
                     random_state=SEED)


def build(text_col=None, char=True, dev=False, meta=False, clf=lr):
    parts = []
    if text_col:
        parts.append(("rw", word_vec(), text_col))
        if char:
            parts.append(("rc", char_vec(), text_col))
    if dev:
        # token_pattern=\S+ : sklearn's default pattern splits Devanagari
        # words at vowel signs (कहां -> 'कह'), so it must be overridden.
        parts.append(("dw", TfidfVectorizer(
            ngram_range=(1, 2), max_features=8000, min_df=2,
            sublinear_tf=True, token_pattern=r"\S+", lowercase=False), "dev"))
    if meta:
        parts.append(("m", OneHotEncoder(handle_unknown="ignore"), META))
    return Pipeline([
        ("features", ColumnTransformer(parts)),
        ("clf", clf()),
    ])


EXPERIMENTS = {
    "A. Majority-class baseline": Pipeline(
        [("features", ColumnTransformer(
            [("m", OneHotEncoder(handle_unknown="ignore"), META)])),
         ("clf", DummyClassifier(strategy="most_frequent"))]),
    "B. Metadata only (LR)":
        build(None, meta=True),
    "C. Lyrics only: word TF-IDF (LR)":
        build("roman_base", char=False),
    "D. Lyrics only: word+char TF-IDF (LR)":
        build("roman_base"),
    "E. D + metadata  [ORIGINAL SYSTEM] (LR)":
        build("roman_base", meta=True),
    "F. Original system with Linear SVM":
        build("roman_base", meta=True, clf=svm),
    "G. Lyrics only + spelling normalisation (LR)":
        build("roman_norm"),
    "H. G + metadata (LR)":
        build("roman_norm", meta=True),
    "I. Devanagari only: word TF-IDF (LR)":
        build(None, dev=True),
    "J. Roman(norm) + Devanagari, lyrics only (LR)":
        build("roman_norm", dev=True),
    "K. Roman(norm) + Devanagari + metadata (LR)":
        build("roman_norm", dev=True, meta=True),
}


# ------------------------------------------------------------ 1. CV
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=SEED)
scoring = {"acc": "accuracy", "f1_macro": "f1_macro",
           "f1_weighted": "f1_weighted"}
rows = []
for name, model in EXPERIMENTS.items():
    r = cross_validate(model, df, y, cv=cv, scoring=scoring, n_jobs=-1)
    row = {"experiment": name}
    for k in scoring:
        v = r[f"test_{k}"]
        row[k] = f"{v.mean():.3f} ± {v.std():.3f}"
        row[k + "_mean"] = v.mean()
    rows.append(row)
    print(f"{name:55s} acc {row['acc']}  macroF1 {row['f1_macro']}")

res = pd.DataFrame(rows)
res.drop(columns=[c for c in res if c.endswith("_mean")]).to_csv(
    OUT / "cv_results.csv", index=False)
with open(OUT / "cv_results.md", "w", encoding="utf-8") as f:
    f.write("| Experiment | Accuracy | Macro-F1 | Weighted-F1 |\n|---|---|---|---|\n")
    for _, r in res.iterrows():
        f.write(f"| {r.experiment} | {r.acc} | {r.f1_macro} | {r.f1_weighted} |\n")

best_name = res.sort_values("f1_macro_mean", ascending=False).iloc[0].experiment
print("\nBest by macro-F1:", best_name)
(OUT / "best_config.txt").write_text(best_name, encoding="utf-8")


# ------------------------------------------------------------ 2. per-class / errors
# Use the best lyrics+metadata-free-or-not config as ranked above
best_model = EXPERIMENTS[best_name]
skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
oof = cross_val_predict(best_model, df, y, cv=skf, n_jobs=-1)

report = classification_report(y, oof, zero_division=0)
(OUT / "per_class_report.txt").write_text(
    f"Config: {best_name}\n(out-of-fold predictions, 5-fold CV)\n\n{report}",
    encoding="utf-8")
print("\n" + report)

labels = sorted(y.unique())
plot_labels = [l.split(" ")[0] for l in labels]
cm = confusion_matrix(y, oof, labels=labels)
plt.figure(figsize=(9, 7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Purples",
            xticklabels=plot_labels, yticklabels=plot_labels)
plt.title(f"Confusion matrix (5-fold out-of-fold)\n{best_name}", fontsize=10)
plt.xlabel("Predicted"); plt.ylabel("Actual")
plt.xticks(rotation=45, ha="right"); plt.tight_layout()
plt.savefig(OUT / "confusion_matrix_cv.png", dpi=150); plt.close()

# error analysis: misclassified songs, most-confident wrong first
lr_for_conf = build("roman_norm", dev=True, meta=True)
proba = cross_val_predict(lr_for_conf, df, y, cv=skf, n_jobs=-1,
                          method="predict_proba")
classes = sorted(y.unique())
pred2 = np.array(classes)[proba.argmax(1)]
wrong = np.where(pred2 != y.values)[0]
err = pd.DataFrame({
    "song": df.music_name.values[wrong],
    "actual": y.values[wrong],
    "predicted": pred2[wrong],
    "confidence": proba.max(1)[wrong].round(3),
    "theme": df.theme.values[wrong],
    "primary_emotion": df.primary_emotion.values[wrong],
    "n_words": df.lyrics.str.split().str.len().values[wrong],
}).sort_values("confidence", ascending=False)
err.to_csv(OUT / "error_analysis_all.csv", index=False)
err.head(15).to_csv(OUT / "error_analysis_top15.csv", index=False)

pairs = (err.groupby(["actual", "predicted"]).size()
         .sort_values(ascending=False).head(8))
pairs.to_csv(OUT / "most_confused_pairs.csv")
print("\nMost confused pairs:\n", pairs)


# ------------------------------------------------------------ 3. imbalance
# Rare-class merge experiment (reported, not adopted: changes label set)
merge_map = {"Dance 💃": "Party 🎉", "Patriotic 🇮🇳": "Motivational 💪"}
y_merged = y.replace(merge_map)
r = cross_validate(build("roman_norm", meta=True), df, y_merged, cv=cv,
                   scoring=scoring, n_jobs=-1)
merge_txt = (f"Merging Dance->Party and Patriotic->Motivational "
             f"(7 classes): acc {r['test_acc'].mean():.3f}, "
             f"macro-F1 {r['test_f1_macro'].mean():.3f}")
(OUT / "class_merge_experiment.txt").write_text(merge_txt, encoding="utf-8")
print("\n" + merge_txt)


# ------------------------------------------------------------ 4. multi-label
songs = json.load(open(BASE.parent / "backend/data/sursaathi_data.json"))["songs"]
tags = [s["moods"] for s in songs]
mlb = MultiLabelBinarizer()
Y = mlb.fit_transform(tags)
ml = Pipeline([
    ("features", ColumnTransformer(
        [("rw", word_vec(), "roman_norm"), ("rc", char_vec(), "roman_norm"),
         ("m", OneHotEncoder(handle_unknown="ignore"), META)])),
    ("clf", OneVsRestClassifier(LogisticRegression(
        C=10, max_iter=3000, class_weight="balanced"))),
])
from sklearn.model_selection import KFold
oofY = np.zeros_like(Y)
for tr, te in KFold(5, shuffle=True, random_state=SEED).split(df):
    ml.fit(df.iloc[tr], Y[tr])
    oofY[te] = ml.predict(df.iloc[te])
ml_txt = (
    "Multi-label one-vs-rest LR over all %d mood tags (5-fold OOF)\n"
    "micro-F1  : %.3f\nmacro-F1  : %.3f\nsamples-F1: %.3f\n\nPer tag F1:\n" % (
        Y.shape[1], f1_score(Y, oofY, average="micro"),
        f1_score(Y, oofY, average="macro"),
        f1_score(Y, oofY, average="samples")))
for t, f in zip(mlb.classes_, f1_score(Y, oofY, average=None)):
    ml_txt += f"  {t:18s} {f:.3f}\n"
(OUT / "multilabel_results.txt").write_text(ml_txt, encoding="utf-8")
print("\n" + ml_txt)
print("Done. See results/")
