"""The probe adapter must not silently inspect only the frozen Matcher."""
import unittest
from ..probe_external import attention_role


class IndependentProbeRoles(unittest.TestCase):
    def test_matcher_and_scorer_have_distinct_roles(self):
        for layer in (0,3):
            suffix=f'arc_context.blocks.{layer}.cross_attn'
            self.assertEqual(attention_role(suffix),'matcher_context')
            self.assertEqual(attention_role('scorer_features.'+suffix),'scorer_context')

    def test_verifier_is_not_the_scorer_encoder(self):
        self.assertEqual(attention_role('verifier.blocks.0.cross_attn'),'verifier')
        self.assertEqual(attention_role('verifier.pool'),'verifier')
        self.assertIsNone(attention_role('scorer_features.patch_encoder'))
        self.assertIsNone(attention_role('arc_context.blocks.0.self_attn'))


if __name__=='__main__':unittest.main()
