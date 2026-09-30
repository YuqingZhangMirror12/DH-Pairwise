import json,tempfile,unittest
from pathlib import Path
from serve_review import ReviewStore,SCHEMA,key_for
class Persistence(unittest.TestCase):
    def setUp(self):
        self.t=tempfile.TemporaryDirectory();self.root=Path(self.t.name);self.c=dict(dataset='敦煌',pair_id=7,case_name='sample',checkpoint_sha256='a'*64,fingerprint='b'*64)
        p=self.root/'snapshot.json';p.write_text(json.dumps(dict(queries=dict(cases=dict(rows=[self.c]),model=dict(rows=[dict(id='frozen')])))))
        self.s=ReviewStore(self.root/'annotations.json',p)
    def tearDown(self):self.t.cleanup()
    def p(self,v='success',rev=0):return dict(record=dict(self.c,schema=SCHEMA,verdict=v,updated_at='2026-09-28T00:00:00Z'),expected_revision=rev)
    def test_unreviewed(self):self.assertEqual(self.s.read()['records'],{})
    def test_save_reload_clear(self):
        self.assertEqual(self.s.save(self.p())[0],200);self.assertEqual(self.s.read()['records'][key_for(self.c)]['revision'],1)
        self.assertEqual(self.s.save(self.p('failure',1))[0],200);self.assertEqual(self.s.save(self.p(None,2))[0],200)
        self.assertIsNone(self.s.read()['records'][key_for(self.c)]['verdict'])
    def test_conflict(self):
        self.s.save(self.p());self.assertEqual(self.s.save(self.p('failure'))[0],409);self.assertEqual(self.s.read()['records'][key_for(self.c)]['verdict'],'success')
    def test_fingerprint(self):
        x=self.p();x['record']['fingerprint']='c'*64;self.assertEqual(self.s.save(x)[0],400)
    def test_checkpoint(self):
        x=self.p();x['record']['checkpoint_sha256']='c'*64;self.assertEqual(self.s.save(x)[0],400)
    def test_unknown_case(self):
        x=self.p();x['record']['pair_id']=8;self.assertEqual(self.s.save(x)[0],400)
    def test_corrupt_store_preserved(self):
        self.s.path.write_text('{broken')
        with self.assertRaises(ValueError):self.s.save(self.p())
        self.assertEqual(self.s.path.read_text(),'{broken')
if __name__=='__main__':unittest.main()
