import json
from pathlib import Path
import tempfile
import unittest
from .heldout_extend_build import catalog_payload
from .heldout_augment import digest


class CatalogByteAdmissionTests(unittest.TestCase):
    def test_copied_and_original_payloads_are_checked_not_only_catalog(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory).resolve()
            copied=root/"copy.png";original=root/"original.png"
            copied.write_bytes(b"exact-native-mask")
            original.write_bytes(b"exact-native-mask")
            cat=root/"catalog.json"
            cat.write_text(json.dumps(dict(release_root=str(root),copied_files_sha256={"copy.png":digest(copied)},
                                          original_files_sha256={str(original):digest(original)})))
            source=dict(catalog_path=str(cat),catalog_sha256=digest(cat))
            admitted=catalog_payload(source)
            self.assertEqual(admitted["copied_count"],1)
            self.assertEqual(admitted["original_count"],1)
            self.assertEqual(set(admitted["files"]),{str(copied),str(original)})
            for path in (copied,original):
                path.write_bytes(b"changed")
                with self.assertRaises(ValueError):catalog_payload(source)
                path.write_bytes(b"exact-native-mask")
            cat.write_text("{}")
            with self.assertRaises(ValueError):catalog_payload(source)


if __name__=="__main__":unittest.main()
