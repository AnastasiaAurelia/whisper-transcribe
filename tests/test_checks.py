import unittest

from whisper_transcribe.checks import check_transcript

ENGLISH = [
    "So today we are going to talk about how the build system works and why it is slow.",
    "And I think the first thing you have to know is that there are two kinds of cache.",
    "One of them lives on your laptop, and the other one is shared by the whole team.",
    "If you change a single file, it should only rebuild what depends on that file.",
    "That is the idea, but in practice it does not always work that way, so let me show you.",
] * 4

# Synthetic stand-in for real Whisper failure output: English speech decoded as
# Welsh ('cy') yields pseudo-words, a few stray English words and odd scripts.
GARBLED = [
    "ydrwyn felgoch aethryd hynny, cyfrawd yn mellwyr drostion alrightwyd ynaf",
    "yn ffadwen ceirol am ysgrathod ar dd increasing brawnog hynny'r quaint",
    "yw'r amlodde f fancy nhw ydeg y cael un gwenfryn i chwad漢ol ond yn yr",
    "aedwyn erfen ni te ddadlo rhabin",
    "Lorwenió a tagodyl yn ungwaith",
]


class CheckTests(unittest.TestCase):
    def test_clean_english_has_no_warnings(self):
        self.assertEqual(check_transcript(ENGLISH, duration=60, language="en"), [])

    def test_garbled_output_flagged_as_not_english(self):
        # Vary the lines so only the language check can fire.
        texts = [f"{line} {i}" for i in range(4) for line in GARBLED]
        warnings = check_transcript(texts, duration=60, language="en")
        self.assertTrue(any("does not look like 'en'" in w for w in warnings), warnings)

    def test_low_confidence_auto_detection(self):
        warnings = check_transcript(ENGLISH, 60, "cy", detected_language="cy",
                                    language_probability=0.63)
        self.assertTrue(any("low confidence" in w for w in warnings), warnings)

    def test_non_latin_script(self):
        warnings = check_transcript(["これは日本語のテキストです"] * 3, 30, "en")
        self.assertTrue(any("non-Latin" in w for w in warnings), warnings)

    def test_hallucination_loop(self):
        texts = ENGLISH[:5] + ["Thank you for watching."] * 10
        warnings = check_transcript(texts, 60, "en")
        self.assertTrue(any("hallucination" in w for w in warnings), warnings)

    def test_empty_and_sparse_output(self):
        self.assertIn("no text recognized", check_transcript([], 120, "en"))
        warnings = check_transcript(["Hello."], 600, "en")
        self.assertTrue(any("very little text" in w for w in warnings), warnings)

    def test_low_logprob(self):
        warnings = check_transcript(ENGLISH, 60, "en", avg_logprobs=[-1.4, -1.2])
        self.assertTrue(any("low model confidence" in w for w in warnings), warnings)

    def test_unknown_language_skips_word_check(self):
        self.assertEqual(check_transcript(ENGLISH, 60, "xx"), [])


if __name__ == "__main__":
    unittest.main()
