"""Model-free input-layout contracts for the streaming codec experiment."""

import struct
import unittest

from run_codec_replay import transpose


class CodecReplayTests(unittest.TestCase):
    def test_frame_major_to_codebook_major(self):
        payload = struct.pack("<48i", *range(48))
        self.assertEqual(struct.unpack("<48i", transpose(payload)),
                         tuple(t * 16 + k for k in range(16) for t in range(3)))

    def test_incomplete_or_out_of_vocabulary_rejected(self):
        for payload in (b"", b"a", b"\0" * 63, struct.pack("<16i", *([2048] * 16)),
                        struct.pack("<16i", *([-1] * 16))):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                transpose(payload)


if __name__ == "__main__":
    unittest.main()
