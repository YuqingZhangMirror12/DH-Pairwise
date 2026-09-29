import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from tools.materialize_research_snapshot import materialize, checked_relative


class TestMaterialize(unittest.TestCase):
    def fixture(self, root):
        (root/'base/pkg').mkdir(parents=True)
        (root/'overlays/demo/pkg').mkdir(parents=True)
        (root/'manifests').mkdir()
        (root/'base/pkg/a.py').write_text('a = 1\n')
        (root/'overlays/demo/pkg/b.py').write_text('b = 2\n')
        files = {'pkg/a.py':hashlib.sha256(b'a = 1\n').hexdigest(),
                 'pkg/b.py':hashlib.sha256(b'b = 2\n').hexdigest()}
        (root/'manifests/demo.json').write_text(json.dumps(dict(
            schema='research-source-capsule/1',variant='demo',
            source_subdirectory='overlays/demo',source_files=files)))

    def test_exact_overlay(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'release';root.mkdir();self.fixture(root)
            out=Path(d)/'new';r=materialize(root,'demo',out)
            self.assertEqual(r['files'],2)
            self.assertEqual((out/'pkg/b.py').read_text(),'b = 2\n')
            self.assertFalse(r['training_started'])

    def test_refuse_existing(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(FileExistsError):materialize(Path(d),'demo',Path(d))

    def test_tamper_before_writes(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'release';root.mkdir();self.fixture(root)
            (root/'base/pkg/a.py').write_text('a = 3\n')
            with self.assertRaises(ValueError):materialize(root,'demo',Path(d)/'new')
            self.assertFalse((Path(d)/'new').exists())

    def test_unsafe_path(self):
        for p in ['../private','/etc/passwd']:
            with self.assertRaises(ValueError):checked_relative(p)


if __name__=='__main__':unittest.main()
