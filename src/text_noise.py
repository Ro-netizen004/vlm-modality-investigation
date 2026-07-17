"""Graded TEXT corruption for the mirror arm (Phase 7) of the legibility experiment.

Symmetric counterpart to src/noise.py (image degradation): here we hold the IMAGE
clean and degrade the TEXT channel, to test whether modality preference tracks text
reliability the way a reliability-weighted observer would (shift toward the image as
the text becomes unreadable).

Degradation is character-level (random substitution / deletion), an OCR-error-style
analogue to image blur, applied deterministically from a fixed seed. Levels mirror the
image ladder's clean -> heavy intent (0/2/4/5) so the two arms are directly comparable.
"""
import random
import string

# Per-character corruption probability per level (0 clean ... 5 heavy).
TEXT_NOISE_LEVELS = {
    0: {"name": "clean",             "p": 0.00},
    2: {"name": "light_corruption",  "p": 0.08},
    4: {"name": "medium_corruption", "p": 0.18},
    5: {"name": "heavy_corruption",  "p": 0.35},
}

_ALPHABET = string.ascii_letters + string.digits


def degrade_text(text, level, seed=0):
    """Return `text` with each non-space character independently corrupted with prob p:
    half the time substituted with a random alphanumeric, half the time deleted.
    Whitespace and structure are preserved so the string stays a plausible (if noisy)
    problem statement. Deterministic given (text, level, seed)."""
    cfg = TEXT_NOISE_LEVELS[level]
    p = cfg["p"]
    if p <= 0:
        return text
    rng = random.Random(seed)
    out = []
    for ch in text:
        if ch.isspace() or rng.random() >= p:
            out.append(ch)
        elif rng.random() < 0.5:
            out.append(rng.choice(_ALPHABET))  # substitution
        # else: deletion (append nothing)
    return "".join(out)


def get_text_level_names():
    return {lvl: cfg["name"] for lvl, cfg in TEXT_NOISE_LEVELS.items()}
