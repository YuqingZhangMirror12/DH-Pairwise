import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import scheduler as s
import common as c

class ScheduleTests(unittest.TestCase):
    def test_order_and_parallel_dispatch(self):
        tasks={k:dict(status='pending') for k in c.ORDER}
        for task in c.ORDER:
            self.assertEqual(s.next_task(tasks),task);tasks[task]['status']='running'
        self.assertIsNone(s.next_task(tasks))
    def test_failure_does_not_launch_lower_priority(self):
        tasks={k:dict(status='pending') for k in c.ORDER};tasks['binary_patch']['status']='failed'
        self.assertIsNone(s.next_task(tasks))
    def test_process_identity_not_pid_only(self):
        row=dict(pid=1,starttime=3,cmdline='train')
        for actual in (dict(row,state='Z'),dict(row,state='R',starttime=4),dict(row,state='R',cmdline='different')):
            with patch.object(c,'identity',return_value=actual):self.assertFalse(c.live(row))
        with patch.object(c,'identity',return_value=dict(row,state='R')):self.assertTrue(c.live(row))
    def test_occupied_card_is_not_free(self):
        with patch.object(c.subprocess,'check_output',side_effect=['0, GPU-0\n1, GPU-1\n','GPU-1, 12\n']):
            self.assertFalse(c.free('0,1'))
        with patch.object(c.subprocess,'check_output',side_effect=['0, GPU-0\n1, GPU-1\n','']):
            self.assertTrue(c.free('0,1'))
    def test_existing_branch_terminal_not_process_disappearance(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);formal=root/'formal_m12';formal.mkdir()
            c.save(root/'driver_m12.json',dict(status='running_formal'))
            c.save(root/'formal_launch_m12.json',dict(job={'pid':1},controller={'pid':2}))
            lane=dict(arm='m12')
            with patch.object(s,'old_root',return_value=root),patch.object(s,'live',return_value=False):
                self.assertEqual(s.old_status(lane),'unexpected_exit')
                c.save(root/'driver_m12.json',dict(status='training_complete_evaluation_pending'))
                c.save(root/'m12_formal_exit.json',dict(returncode=0))
                c.save(formal/'training_complete.json',dict(status='training_complete'))
                self.assertEqual(s.old_status(lane),'terminal')
                c.save(formal/'failure.json',dict(error='known'))
                self.assertEqual(s.old_status(lane),'failed')
    def test_failure_priority_in_wait(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);c.save(p/'done.json',dict(status='complete'));c.save(p/'failure.json',{})
            with self.assertRaises(RuntimeError):c.wait_terminal(p/'done.json',{},[p/'failure.json'],'complete')
    def test_no_terminal_on_unexplained_exit(self):
        with tempfile.TemporaryDirectory() as t,patch.object(c,'live',return_value=False):
            with self.assertRaises(RuntimeError):c.wait_terminal(Path(t)/'done.json',{},[],'complete')
    def test_cuda_hidden_by_default_but_explicit_gate_card_visible(self):
        self.assertEqual(c.env()['CUDA_VISIBLE_DEVICES'],'')
        self.assertEqual(c.env('2')['CUDA_VISIBLE_DEVICES'],'2')
    def test_atomic_receipt(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'nested/x.json';c.save(p,dict(status='passed'));self.assertEqual(c.read(p)['status'],'passed')
            self.assertEqual(list(p.parent.glob('*.tmp*')),[])
if __name__=='__main__':unittest.main()
