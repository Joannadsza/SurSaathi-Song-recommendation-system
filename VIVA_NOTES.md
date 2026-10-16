# SurSaathi — NLP Briefing Notes (for the NLP guide)

## 1. What the system does
Hybrid Hindi song recommender: (a) **lyric similarity** (TF-IDF + cosine),
(b) **mood classification** (supervised, 9 classes), (c) karaoke view.
Data: 974 Bollywood songs, Roman + Devanagari lyrics, annotated mood /
theme / energy / occasion.

## 2. ML algorithms and their role
| Component | Algorithm | Role |
|---|---|---|
| Mood classifier | **Linear SVM** (C=1) — compared with Logistic Regression | **Classifies** a song into one of 9 moods; it does not generate moods |
| Lyric recommender | TF-IDF + cosine similarity (no training) | Finds songs with similar lyrics |
| Mood recommender | Classifier output → songs carrying that mood tag | "More like this · mood" |

Model selection: GridSearchCV (C ∈ {0.1, 1, 10}, 5-fold) on the training set,
winner chosen by **CV score, not test score**. Linear SVM 0.637 vs Logistic
Regression 0.626 weighted-F1 — a difference within noise, so both are
reported as equivalent.

## 3. Preprocessing (`model_train/nlp_pipeline.py`, shared by training + API)
1. Lowercase → 2. keep a–z only (digits/punctuation/Devanagari removed) →
3. whitespace tokenisation → 4. Roman-Hindi stopword removal →
5. drop 1-letter tokens. **No stemming/lemmatisation**: English stemmers
corrupt Roman Hindi; character n-grams absorb spelling variation instead.
Further preprocessing (spelling variants, suffix stemming, repetition handling,
data-driven stopwords, Devanagari) was tested in §3b and gave no significant gain.

## 3b. Preprocessing ablation (`ablation_preprocessing.py`, 5-fold CV x 2, lyrics only unless stated)
| Step | Accuracy | Macro-F1 |
|---|---|---|
| R0 Roman raw (lowercase only) | 0.597 ± 0.038 | 0.363 |
| R1 Roman cleaned (baseline) | 0.593 ± 0.035 | 0.372 |
| R2 + spelling variants (rules + dictionary + mined, 138 merges) | 0.590 ± 0.030 | 0.376 |
| R3 + suffix stemming (-ein, -on, -ata, -ti, -ya ...) | 0.592 ± 0.029 | 0.378 |
| R4 R1 + repetition handling (dedupe lines + collapse runs) | 0.587 ± 0.040 | 0.353 |
| R5 R1 + data-driven stopwords (max_df = 0.3) | 0.596 ± 0.033 | 0.377 |
| R6 all Roman steps | 0.573 ± 0.033 | 0.354 |
| D0 Devanagari raw | 0.594 ± 0.027 | 0.359 |
| D1 Devanagari cleaned | 0.593 ± 0.029 | 0.375 |
| D2 + suffix stemming | 0.598 ± 0.028 | 0.378 |
| D3 + repetition handling | 0.587 ± 0.032 | 0.353 |
| X1 Roman(all) + Devanagari(stem) | 0.599 ± 0.032 | 0.380 |
| R1 + metadata | 0.652 ± 0.035 | 0.526 |
| X1 + metadata | 0.659 ± 0.022 | 0.533 |

**Reading:** every step is within one standard deviation (~±0.03), so none is a
demonstrated improvement. Why: sublinear TF-IDF already dampens repeated words,
and character n-grams already absorb spelling variation, so rule-based
normalisation adds little. Repetition handling is slightly *worse* — repeated
hooks carry mood signal. Devanagari is as informative as Roman (59%) but adds
nothing when combined. The deployed model keeps the simple baseline.
Data-driven stopword candidates (document frequency): tu 57%, dil 52%, tere 47%,
teri 40%, meri 39% — mostly pronouns plus "dil" (`results/data_driven_stopwords.txt`).
Mined spelling variants: `results/variant_map_preview.txt`.
**Correction:** an earlier run reported Devanagari as 47.8% — that was a
tokenisation bug (sklearn's default pattern splits Devanagari words at vowel
signs); fixed with `token_pattern=\S+`.

## 4. TF-IDF and cosine similarity
* TF-IDF weight = (1 + ln tf) × idf, rows L2-normalised.
* Classifier features: word 1–2-grams (8,000) + char_wb 3–5-grams (8,000)
  + one-hot energy/theme/occasion (20) = **16,020 features**.
* Similarity: word 1–2-gram TF-IDF (12,394 terms) + small theme block
  (weight 0.2). Because rows are unit length, cosine = dot product.
* **Pre-calculated**, not live. Built by `model_train/build_similarity.py`,
  stored in `backend/data/sursaathi_data.json` → `songs[i].sims`
  (top 8 `{id, score}`). `backend/app.py` only reads them.
* Live demo for any song: `python model_train/demo_pipeline.py "Tum Kahaan The"`
  (output saved in `results/demo_output.txt`; the demo shows lyrics-only
  scores, the stored scores add the 0.2 theme block).

## 5. Results (5-fold CV × 2 repeats, mean ± std; vectorizers fitted inside folds)
| Experiment | Accuracy | Macro-F1 | Weighted-F1 |
|---|---|---|---|
| A. Majority-class baseline | 0.473 ± 0.003 | 0.071 | 0.304 |
| B. Metadata only | 0.476 ± 0.035 | 0.400 | 0.490 |
| C. Lyrics only: word TF-IDF | 0.563 ± 0.026 | 0.319 | 0.510 |
| D. Lyrics only: word + char TF-IDF | 0.593 ± 0.035 | 0.372 | 0.554 |
| **E. D + metadata (LR)** | 0.652 ± 0.035 | 0.526 | 0.642 |
| **F. D + metadata (Linear SVM) — final** | 0.659 ± 0.026 | 0.522 | 0.647 |
| G. Lyrics + spelling normalisation | 0.585 ± 0.030 | 0.370 | 0.544 |
| H. G + metadata | 0.650 ± 0.030 | 0.525 | 0.639 |

**Honest headline: ~66% CV accuracy (macro-F1 0.52).** The single 80/20 test
split gives 68.7%, but that has about ±6 points of sampling error, so the CV
figure is the one to quote.

Findings
* Lyrics alone add ~12 points over the majority baseline (59% vs 47%).
* Char n-grams help (+3 pts) — they handle Roman-Hindi spelling noise.
* **Metadata (energy/theme/occasion) adds ~6 points accuracy and ~15 points
  macro-F1** — the model leans on annotations, not only lyrics. State this openly.
* Extra preprocessing gave **no significant improvement** (§3b).
* Weak classes (out-of-fold): Dance F1 0.10, Happy 0.23 (18 and 37 songs).
  See `results/per_class_report.txt`, `results/confusion_matrix_cv.png`.
* Merging rare classes (Dance→Party, Patriotic→Motivational, 7 classes):
  acc 0.663, macro-F1 0.571 (`class_merge_experiment.txt`; not adopted because it changes the label set).

## 6. Error analysis (`results/error_analysis_*.csv`, `most_confused_pairs.csv`)
Most confusion is Emotional ↔ Romantic (50 + 49 songs) and Emotional → Party /
Motivational / Sad / Calm. "Emotional" is a broad catch-all (47% of the data), so
many songs legitimately fit several moods. Confident errors (e.g. "Tere Liye",
"La Ilaaj") are songs whose labelled mood conflicts with lyric content —
evidence that single-label mood is an imperfect framing.

## 7. Multi-label formulation (`results/multilabel_results.txt`)
Songs carry several mood tags, so a one-vs-rest Logistic Regression over all
10 tags was also evaluated: micro-F1 0.784, macro-F1 0.746, samples-F1 0.792
(not directly comparable with the single-label scores; shown to justify the framing).

## 8. Recommender evaluation — precision@5 (`results/recommender_eval.txt`)
| System | Mood-tag Jaccard | Same primary mood | Same theme |
|---|---|---|---|
| Random neighbours | 0.287 | 0.278 | 0.128 |
| Lyrics-only TF-IDF cosine | 0.337 | 0.343 | 0.218 |
| **TF-IDF + theme 0.2 (final)** | **0.420** | **0.420** | 0.658 |
| Old stored sims (script lost) | 0.358 | 0.361 | 0.316 |
Lyrics-only similarity is only modestly better than random for mood, because
lyric overlap reflects shared hooks/vocabulary, not feeling. The theme block
fixes that. Theme-match is partly circular for the final model; read the first
two columns. Mood recommendations are tag-filtered by the *predicted* mood, which
is a filter, not a ranking.

## 9. Known limitations (say these before she does)
1. Mood labels come from dataset annotation, not human listening tests.
2. Metadata dependence (§5). Lyrics-only accuracy is 59%.
3. Small data; rare moods unreliable.
4. At runtime the API predicts moods for songs the model was trained on, so the
   live prediction is not a held-out accuracy demo — quote CV numbers instead.
5. The SVM "confidence" is a softmax over decision scores (relative, not calibrated).
6. Embedding comparison (IndicBERT/MuRIL/e5) is scripted in
   `embedding_experiment.py` but has not been run.
7. No human evaluation of recommendations.

## 10. Reproduce
```
cd model_train
python evaluate_models.py      # all experiments -> results/
python build_variant_map.py    # mine Roman spelling variants
python ablation_preprocessing.py   # raw/clean/stem/repeat/stopword ablation
python train_mood_model.py     # final model + metrics + confusion matrix
python build_similarity.py --write   # precompute cosine neighbours
python demo_pipeline.py "Song name"  # step-by-step demo
```
