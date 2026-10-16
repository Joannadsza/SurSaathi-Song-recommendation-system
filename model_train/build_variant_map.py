"""
Mine Roman-Hindi spelling variants from the corpus itself (unsupervised —
no labels used, so no leakage). Two words are merged when they
  * share first letter and identical form after folding interior e<->i and o<->u, rarer by >= 3x.
Writes variant_map.json (loaded by nlp_pipeline) and a preview file.
Run: python build_variant_map.py
"""
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from nlp_pipeline import clean_roman_baseline, normalize_roman_word

BASE = Path(__file__).resolve().parent
df = pd.read_csv(BASE / "song_cleaned.csv")
counts = Counter(w for t in df.lyrics.map(clean_roman_baseline)
                 for w in t.split())


def norm(w):
    w = re.sub(r"([bcdfghjklmnpqrstvwxyz])\1+", r"\1", normalize_roman_word(w))
    return w


def skeleton(w):
    return re.sub(r"[aeiou]", "", norm(w))


def edit(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[-1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


# Conservative rule: two forms are variants only if they become IDENTICAL
# after folding the vowel pairs that Roman-Hindi writers swap freely
# (e<->i, o<->u), the variant is >= 3x rarer than the canonical form, and
# the consonant skeleton has >= 3 letters. Different words (rang/ring,
# soch/sach) are never merged because a/o/i swaps are not folded.
def fold(w):
    # last letter is kept as-is: final e/i marks gender/tense
    # (ladke vs ladki, karne vs karni) and must NOT be merged
    w = norm(w)
    return w[:-1].replace("e", "i").replace("o", "u") + w[-1]


raw = Counter()
for w, c in counts.items():
    raw[norm(w)] += c

groups = defaultdict(list)
for form in raw:
    groups[fold(form)].append(form)

vmap = {}
for key, forms in groups.items():
    if len(forms) < 2 or len(skeleton(forms[0])) < 3:
        continue
    forms = sorted(forms, key=lambda f: -raw[f])
    canon = forms[0]
    for f in forms[1:]:
        if raw[f] * 3 <= raw[canon]:
            vmap[f] = canon

json.dump(vmap, open(BASE / "variant_map.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=0)
prev = sorted(vmap.items(), key=lambda kv: -counts.get(kv[0], 0))[:60]
(BASE / "results").mkdir(exist_ok=True)
(BASE / "results" / "variant_map_preview.txt").write_text(
    f"{len(vmap)} variant -> canonical merges mined. Most frequent:\n" +
    "\n".join(f"  {v:15s} -> {c}" for v, c in prev), encoding="utf-8")
print(f"{len(vmap)} merges; preview:")
print("\n".join(f"  {v} -> {c}" for v, c in prev[:25]))
