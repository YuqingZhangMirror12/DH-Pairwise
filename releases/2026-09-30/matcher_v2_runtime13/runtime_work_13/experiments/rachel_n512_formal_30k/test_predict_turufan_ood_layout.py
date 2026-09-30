"""CPU-only contract tests, without requiring a local PyTorch installation.

Extract only the three pure serialization/resume functions from the actual
entry-point AST. The parent's separate remote smoke verifies model loading and
the GPU/native decoder paths; these tests do not pretend to cover inference.
"""
import ast
from pathlib import Path
import unittest
import numpy as np


SOURCE = Path(__file__).with_name("predict_turufan_ood_layout.py")
tree = ast.parse(SOURCE.read_text())
pure = ast.Module(body=[node for node in tree.body if isinstance(node, ast.FunctionDef)
    and node.name in {"json_safe", "layout_row", "validate_completed"}], type_ignores=[])
namespace = {"np": np}
exec(compile(pure, str(SOURCE), "exec"), namespace)
layout_row = namespace["layout_row"]
validate_completed = namespace["validate_completed"]


class LayoutContractTests(unittest.TestCase):
    def test_translation_sign_and_units(self):
        row = layout_row("p", "native", [12.5, -31], True, reason="ok", score=.2)
        self.assertEqual(row["offset_b_in_a_rc"], [-12.5, 31])
        self.assertEqual(row["t_a_to_b_rc"], [12.5, -31])
        self.assertFalse(row["score_is_canonical"])

    def test_invalid_layout_is_null_not_identity(self):
        row = layout_row("p", "native", [np.nan, np.nan], False, reason="native_decoder_invalid")
        self.assertIsNone(row["offset_b_in_a_rc"])
        self.assertIsNone(row["t_a_to_b_rc"])
        self.assertFalse(row["valid"])

    def test_nonfinite_valid_translation_is_structural_error(self):
        with self.assertRaises(ValueError):
            layout_row("p", "native", [np.nan, 0], True, reason="bad")

    def test_nonfinite_probability_is_error(self):
        with self.assertRaises(ValueError):
            layout_row("p", "native", None, False, reason="bad", score=np.nan)

    def test_rejected_probability_does_not_prevent_layout(self):
        row = layout_row("p", "native", [1, 2], True, reason="ok", score=.001, decision_valid=True)
        self.assertTrue(row["valid"])
        self.assertTrue(row["layout_decoded_independently_of_pair_probability"])

    def test_resume_requires_exact_binding_order_and_sign(self):
        rows = [layout_row("p", "native", [1, 2], True, reason="ok")]
        payload = dict(status="complete", method="m", binding={"hash": "a"}, predictions=rows)
        self.assertEqual(validate_completed(payload, "m", ["p"], {"hash": "a"}), rows)
        with self.assertRaises(ValueError):
            validate_completed(payload, "m", ["q"], {"hash": "a"})
        with self.assertRaises(ValueError):
            validate_completed(payload, "m", ["p"], {"hash": "b"})
        rows[0]["offset_b_in_a_rc"] = [1, 2]
        with self.assertRaises(ValueError):
            validate_completed(payload, "m", ["p"], {"hash": "a"})

    def test_nonfinite_diagnostic_is_json_null(self):
        row = layout_row("p", "native", [1, 2], True, reason="ok", diagnostics={"residual": np.inf})
        self.assertIsNone(row["diagnostics"]["residual"])


if __name__ == "__main__":
    unittest.main()
