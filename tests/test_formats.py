import tempfile
import unittest
from pathlib import Path

from whisper_transcribe.formats import to_srt, to_txt, ts, write_outputs

SEGMENTS = [(0.0, 2.5, "Hello there."), (3661.25, 3663.0, "An hour later.")]


class FormatTests(unittest.TestCase):
    def test_ts(self):
        self.assertEqual(ts(0), "00:00:00,000")
        self.assertEqual(ts(3661.25), "01:01:01,250")
        self.assertEqual(ts(59.5, "."), "00:00:59.500")

    def test_srt(self):
        self.assertEqual(
            to_srt(SEGMENTS),
            "1\n00:00:00,000 --> 00:00:02,500\nHello there.\n\n"
            "2\n01:01:01,250 --> 01:01:03,000\nAn hour later.\n\n",
        )

    def test_txt(self):
        self.assertEqual(to_txt(SEGMENTS), "[00:00:00] Hello there.\n[01:01:01] An hour later.\n")

    def test_write_outputs_creates_folder(self):
        with tempfile.TemporaryDirectory() as d:
            srt, txt = write_outputs(SEGMENTS, Path(d) / "out", "talk")
            self.assertEqual(srt.read_text(encoding="utf-8"), to_srt(SEGMENTS))
            self.assertEqual(txt.read_text(encoding="utf-8"), to_txt(SEGMENTS))


if __name__ == "__main__":
    unittest.main()
