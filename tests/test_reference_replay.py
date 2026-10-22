"""One test per bundled Octave reference case; absent local fixtures are explicit skips.

Unlike this partial-download-friendly test suite, ``validate.py --replay``
requires every bundled reference pair and fails if one is absent or empty.
"""
import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import validate


class ReferenceReplayTests(unittest.TestCase):
    pass


def reference_test(tag):
    def test(self):
        files = [Path(validate.HERE, 'validation', 'reference', f'{prefix}_{tag}.mat')
                 for prefix in ('rec', 'res')]
        unavailable = [path.name for path in files if not path.is_file() or path.stat().st_size == 0]
        if unavailable:
            self.skipTest('missing or empty local fixture: ' + ', '.join(unavailable))
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(validate, 'REPLAY_CASES', (tag,)):
            for path in files:
                Path(directory, path.name).symlink_to(path)
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                passed = validate.replay(directory)
            self.assertTrue(passed, output.getvalue())
    return test


for _tag in validate.REPLAY_CASES:
    setattr(ReferenceReplayTests, 'test_' + _tag, reference_test(_tag))


if __name__ == '__main__':
    unittest.main()
