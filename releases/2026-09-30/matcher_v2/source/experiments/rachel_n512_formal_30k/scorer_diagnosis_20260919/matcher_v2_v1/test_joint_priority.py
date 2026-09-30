import copy
import unittest
from unittest.mock import patch

from . import joint_priority as p


class JointPriorityTests(unittest.TestCase):
    def checkpoint(self):
        return dict(model={},optimizer=dict(state={0:{}},param_groups=[{},{}]),rng=[{},{}],
            plateau={},curve=[],epoch=7,offset=20000,updates=5200,exposures=166400,
            binding={'fixture':True},stage='scorer',world_size=2)

    def test_complete_state_and_identity_required(self):
        value=self.checkpoint();p.check_checkpoint(value,value['binding'])
        for key in ('model','optimizer','rng','plateau','curve','epoch','offset','updates','exposures','binding'):
            broken=copy.deepcopy(value);del broken[key]
            with self.assertRaises(ValueError):p.check_checkpoint(broken,value['binding'])
        with self.assertRaises(ValueError):p.check_checkpoint(value,{'different':True})
        for key,replacement in [('rng',[{}]),('world_size',1),('stage','matcher'),('updates',0)]:
            broken=copy.deepcopy(value);broken[key]=replacement
            with self.assertRaises(ValueError):p.check_checkpoint(broken,value['binding'])

    def test_reused_pid_and_zombie_are_never_signalled(self):
        record=dict(pid=123,starttime=1,cmdline='expected',state='S')
        with patch.object(p,'identity',return_value=record):self.assertTrue(p.alive(record))
        for field,value in [('starttime',2),('cmdline','other'),('state','Z')]:
            with patch.object(p,'identity',return_value=dict(record,**{field:value})):
                self.assertFalse(p.alive(record))
        with patch.object(p,'identity',side_effect=FileNotFoundError):self.assertFalse(p.alive(record))

    def test_unknown_controller_rejected(self):
        for command in ('python unrelated.py', 'python -m '+p.__package__+'.joint_priority',
                        'python -m '+p.__package__+'.joint_priority --source /tmp/no --out-new /tmp/no'):
            with self.assertRaises(ValueError):p.check_controller({'cmdline':command})


if __name__=='__main__':unittest.main()
