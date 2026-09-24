"""Cheap heuristics that flag transcripts which look broken.

These are warnings, not proof: they catch the common failure modes of Whisper
(wrong language detected, hallucination loops, near-empty output) without any
extra model or dependency.
"""

import re
import unicodedata
from collections import Counter

# Most frequent function words. In normal speech they make up roughly 35-50%
# of all words; garbled or wrong-language output falls far below that.
STOPWORDS = {
    "en": set(
        "the a an and or but of to in on at for with from by as is are was were be been "
        "it its this that these those i you he she we they me him her us them my your our "
        "their not no do does did have has had will would can could so if then just like "
        "what which who how there here about all more very really know think going yeah".split()
    ),
    "id": set(
        "yang dan di ke dari ini itu dengan untuk tidak ada saya kita kami kamu anda dia "
        "mereka akan sudah bisa juga atau pada dalam jadi karena kalau apa ya nah gitu "
        "aku tapi lebih seperti harus banyak sangat orang satu begitu".split()
    ),
}

# Languages normally written in Latin script; other scripts showing up in
# their transcripts usually means the decoder went off the rails.
LATIN_LANGS = {
    "en", "id", "ms", "es", "fr", "de", "it", "pt", "nl", "sv", "da", "no", "fi",
    "pl", "cs", "ro", "tr", "vi", "tl", "sw", "hu", "hr", "sk", "sl", "ca", "cy",
}

MIN_STOPWORD_RATIO = 0.20
MAX_NON_LATIN_RATIO = 0.02
MIN_WORDS_PER_MINUTE = 20
MAX_REPEATED_LINE_RATIO = 0.20
MIN_AVG_LOGPROB = -1.0
MIN_LANGUAGE_PROBABILITY = 0.80

_WORD = re.compile(r"[^\W\d_]+(?:'[^\W\d_]+)?")


def _letters(text):
    return [c for c in text if c.isalpha()]


def _is_latin(c):
    return "LATIN" in unicodedata.name(c, "")


def check_transcript(
    texts,
    duration,
    language,
    avg_logprobs=None,
    detected_language=None,
    language_probability=None,
):
    """Return a list of human-readable warnings (empty list = looks fine).

    texts: segment texts. duration: audio length in seconds. language: the
    language the transcript is supposed to be in. detected_language /
    language_probability: set only when the language was auto-detected.
    """
    warnings = []
    joined = " ".join(texts)
    words = _WORD.findall(joined.lower())

    if detected_language is not None and language_probability is not None:
        if language_probability < MIN_LANGUAGE_PROBABILITY:
            warnings.append(
                f"language auto-detected as '{detected_language}' with low confidence "
                f"({language_probability:.2f}); consider forcing it with --language"
            )

    if not words:
        if duration > 10:
            warnings.append("no text recognized")
        return warnings

    minutes = duration / 60
    if minutes >= 1 and len(words) / minutes < MIN_WORDS_PER_MINUTE:
        warnings.append(
            f"very little text: {len(words) / minutes:.0f} words/min "
            "(silence, music, or failed recognition?)"
        )

    letters = _letters(joined)
    if language in LATIN_LANGS and letters:
        non_latin = sum(1 for c in letters if not _is_latin(c)) / len(letters)
        if non_latin > MAX_NON_LATIN_RATIO:
            warnings.append(
                f"{non_latin:.0%} of letters are non-Latin, unexpected for '{language}'"
            )

    stop = STOPWORDS.get(language)
    if stop and len(words) >= 50:
        ratio = sum(1 for w in words if w in stop) / len(words)
        if ratio < MIN_STOPWORD_RATIO:
            warnings.append(
                f"text does not look like '{language}' "
                f"(common-word ratio {ratio:.0%}, expected >= {MIN_STOPWORD_RATIO:.0%}); "
                "wrong language or garbled output?"
            )

    lines = [t.strip().lower() for t in texts if t.strip()]
    if len(lines) >= 10:
        line, count = Counter(lines).most_common(1)[0]
        if count / len(lines) > MAX_REPEATED_LINE_RATIO:
            warnings.append(
                f"{count}/{len(lines)} segments are the same line ({line[:40]!r}); "
                "possible hallucination loop"
            )

    if avg_logprobs:
        mean = sum(avg_logprobs) / len(avg_logprobs)
        if mean < MIN_AVG_LOGPROB:
            warnings.append(f"low model confidence (mean avg_logprob {mean:.2f})")

    return warnings
