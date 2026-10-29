"""Headless sanity check of the mood math (no camera needed).

Builds a synthetic neutral face, then:
  1. checks neutral face -> ~0% meters and NEUTRAL mood
  2. opens the mouth    -> SURPRISED should dominate
  3. pulls a smile      -> HAPPY should dominate
Run:  python smoke_mood.py
"""
import types

import mood_detector as md


def lm(x, y):
    return types.SimpleNamespace(x=x, y=y)


def make_face():
    # sparse 478-point fake face; anything we didn't position stays at (0.5, 0.5)
    face = [lm(0.5, 0.5) for _ in range(478)]
    pts = {
        468: (0.10, 0.50), 473: (0.60, 0.50),          # iris centers (scale ref)
        33: (0.10, 0.42), 133: (0.18, 0.42),           # left eye corners
        159: (0.14, 0.40), 145: (0.14, 0.44),          # left lids
        362: (0.52, 0.42), 263: (0.60, 0.42),          # right eye corners
        386: (0.56, 0.40), 374: (0.56, 0.44),          # right lids
        105: (0.18, 0.28), 334: (0.58, 0.28),          # brow tops
        107: (0.30, 0.30), 336: (0.42, 0.30),          # inner brows
        61: (0.445, 0.552), 291: (0.535, 0.552),       # mouth corners
        13: (0.49, 0.550), 14: (0.49, 0.554),          # closed lips
    }
    for i, (x, y) in pts.items():
        face[i] = lm(x, y)
    return face


W = H = 1000.0


def judge(scores):
    """same rule as the live app"""
    top = max(scores, key=scores.get)
    return "NEUTRAL" if scores[top] < 0.30 else top


# 1. neutral face
base = md.extract_features(make_face(), W, H)
neutral = md.dev_to_pcts(base, base)
print("neutral meters:", {k: v for k, v in neutral.items()})
scores = md.score_moods({k: 0.0 for k in
                         ("brow", "furrow", "smile", "lift", "open", "eyes", "asym")})
print("neutral ->", judge(scores), "  PASS" if judge(scores) == "NEUTRAL" else "  FAIL")

# 1b. jaw open must NOT count as a smile ('lift' stays tiny)
f1b = make_face()
f1b[14] = lm(0.49, 0.62)
raw1b = md.extract_features(f1b, W, H)
pcts1b = md.dev_to_pcts(raw1b, base)
print("open-mouth lift meter:", pcts1b["lift"],
      "  PASS" if abs(pcts1b["lift"]) < 30 else "  FAIL (jaw reads as smile)")

# 2. mouth open -> surprised
f2 = make_face()
f2[14] = lm(0.49, 0.62)
pcts2 = md.dev_to_pcts(md.extract_features(f2, W, H), base)
scores2 = md.score_moods(pcts2)
top2 = judge(scores2)
print("open-mouth meters:", {k: v for k, v in pcts2.items() if abs(v) > 5})
print("open-mouth ->", top2, "  PASS" if top2 == "SURPRISED" else "  FAIL")

# 3. smile -> happy
f3 = make_face()
f3[61] = lm(0.42, 0.540)
f3[291] = lm(0.56, 0.540)
pcts3 = md.dev_to_pcts(md.extract_features(f3, W, H), base)
scores3 = md.score_moods(pcts3)
top3 = judge(scores3)
print("smile meters:", {k: v for k, v in pcts3.items() if abs(v) > 5})
print("smile ->", top3, "  PASS" if top3 == "HAPPY" else "  FAIL")

print("OK - mood math runs end-to-end.")
