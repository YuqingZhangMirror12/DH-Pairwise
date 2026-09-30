from copy import deepcopy
import unittest

from .config import TrainingConfig
from .migration import validate_bindings,validate_resume_state,rebase_plateau


class MigrationTests(unittest.TestCase):
    def setUp(self):
        self.config=TrainingConfig()
        self.new=dict(schema='s7-consensus-experiment/1',reference_checkpoint_sha256='reference',
            data_contract_sha256='contract',geometry_calibration_sha256='geometry',
            validation_contract={'cal':'cal','select':'select'},validation_design={'kind':'mixed'},
            config=self.config.record(),implementation_sha256={'pose_consensus.py':'new',
                'legacy_pose_consensus.py':'old','train.py':'new','matcher.py':'protected','losses.py':'loss'})
        self.old=deepcopy(self.new)
        self.old.update(arm='scratch',formal_training=True)
        for key in ('proposal_revision','merge_repair_policy','validation_checkpoint_archive'):
            del self.old['config'][key]
        self.old['implementation_sha256']={'pose_consensus.py':'old','train.py':'old',
            'matcher.py':'protected','losses.py':'loss'}
        self.saved=dict(epoch=26,offset=24000,updates=19500,exposures=624000,world_size=2,
            optimizer={'param_groups':[{'lr':2.5e-5}]},rng=[{},{}],
            plateau={'best':1.,'bad':1,'reductions':2,'since_reduction':1})

    def test_only_reviewed_source_changes_are_allowed(self):
        validate_bindings(self.old,self.new)

    def test_protected_matcher_or_loss_source_cannot_change(self):
        for name in ('matcher.py','losses.py'):
            new=deepcopy(self.new);new['implementation_sha256'][name]='other'
            with self.assertRaises(ValueError):validate_bindings(self.old,new)

    def test_data_calibration_and_budget_cannot_change(self):
        for name in ('data_contract_sha256','geometry_calibration_sha256','validation_contract'):
            new=deepcopy(self.new);new[name]='other'
            with self.assertRaises(ValueError):validate_bindings(self.old,new)
        new=deepcopy(self.new);new['config']['maximum_epochs']=60
        with self.assertRaises(ValueError):validate_bindings(self.old,new)

    def test_original_algorithm_is_preserved(self):
        new=deepcopy(self.new);new['implementation_sha256']['legacy_pose_consensus.py']='other'
        with self.assertRaises(ValueError):validate_bindings(self.old,new)

    def test_complete_optimizer_state_and_counters_survive(self):
        validate_resume_state(self.saved,self.config)
        self.assertEqual(self.saved['updates'],19500)
        self.assertEqual(self.saved['optimizer']['param_groups'][0]['lr'],2.5e-5)

    def test_incomplete_epoch_or_missing_optimizer_cannot_be_relabelled(self):
        for key,value in (('offset',23968),('updates',19499),('exposures',623968),('world_size',4)):
            saved=deepcopy(self.saved);saved[key]=value
            with self.assertRaises(ValueError):validate_resume_state(saved,self.config)
        saved=deepcopy(self.saved);del saved['optimizer']
        with self.assertRaises(ValueError):validate_resume_state(saved,self.config)

    def test_lr_reset_or_budget_expansion_rejected(self):
        saved=deepcopy(self.saved);saved['optimizer']['param_groups'][0]['lr']=1e-4
        with self.assertRaises(ValueError):validate_resume_state(saved,self.config)
        saved=deepcopy(self.saved);saved.update(epoch=48,updates=36000,exposures=1152000)
        with self.assertRaises(ValueError):validate_resume_state(saved,self.config)

    def test_new_metric_observations_keep_consumed_lr_reductions(self):
        plateau=rebase_plateau(self.saved['plateau'],.8)
        self.assertEqual((plateau.bad,plateau.reductions,plateau.since_reduction),(0,2,0))
        self.assertEqual(plateau.observe(.8,28,self.config),'continue')
        self.assertEqual(plateau.observe(.8,30,self.config),'continue')
        self.assertEqual(plateau.observe(.8,32,self.config),'simulation_plateau_after_lr_reductions')

    def test_new_improvements_are_not_forced_to_stop_at32(self):
        plateau=rebase_plateau(self.saved['plateau'],.8)
        for epoch,value in ((28,.81),(30,.82),(32,.83)):
            self.assertEqual(plateau.observe(value,epoch,self.config),'continue')
        self.assertEqual(plateau.reductions,2)


if __name__=='__main__':unittest.main()
