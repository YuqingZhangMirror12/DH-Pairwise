import json
from pathlib import Path
import tempfile
import unittest

from .revision_pipeline import validate_human_approval
from ..s7_compound_v1.materialize import digest


class ApprovalTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.pilot=self.root/'pilot';self.pilot.mkdir()
        self.profile=self.root/'profile.json';self.profile.write_text('{}')
        for name,record in dict(status=dict(sample_count=800),validation=dict(status='passed'),
            independent_pixel_audit=dict(status='passed'),protocol=dict(version=14),
            pipeline_status=dict(stage='data_ready',status='complete')).items():
            (self.pilot/(name+'.json')).write_text(json.dumps(record))
        self.record=dict(schema='s7-reviewed-pilot-approval/1',user_approved=True,
            approved_train_pairs=24000,approved_validation_pairs=6000,pilot_root=str(self.pilot),
            profile_sha256=digest(self.profile),pilot_protocol_sha256=digest(self.pilot/'protocol.json'),
            pilot_validation_sha256=digest(self.pilot/'validation.json'),
            pilot_pixel_audit_sha256=digest(self.pilot/'independent_pixel_audit.json'))
        self.approval=self.root/'approval.json'
        self.approval.write_text(json.dumps(self.record))

    def test_explicit_matching_approval(self):
        self.assertEqual(validate_human_approval(self.profile,self.pilot,self.approval),self.record)

    def test_no_or_different_approval_rejected(self):
        with self.assertRaises(ValueError):validate_human_approval(self.profile,self.pilot,None)
        self.profile.write_text('{"revision":15}')
        with self.assertRaises(ValueError):validate_human_approval(self.profile,self.pilot,self.approval)

    def test_incomplete_pilot_rejected(self):
        (self.pilot/'status.json').write_text('{"sample_count":798}')
        with self.assertRaises(ValueError):validate_human_approval(self.profile,self.pilot,self.approval)


if __name__=='__main__':unittest.main()
