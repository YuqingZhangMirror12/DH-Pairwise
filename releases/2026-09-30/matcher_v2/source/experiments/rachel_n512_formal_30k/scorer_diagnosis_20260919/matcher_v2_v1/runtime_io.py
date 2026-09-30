"""New-arm exports using the shared committed-state writer and new selection."""
from ..curriculum_training_v1 import runtime_io as core
from ..curriculum_training_v1.model_adapter import require
from .model_runtime import EXPERIMENT_SCHEMA, EVALUATION_POLICY, check_schedule, check_export
from .validation import validate_rule, select_history


def export_completed(root, checkpoints, plan, binding):
    validate_rule(plan)
    experiment = binding.get('matcher_v2_experiment', {})
    require(experiment.get('schema') == EXPERIMENT_SCHEMA
            and experiment.get('evaluation_policy') == EVALUATION_POLICY, 'explicit new-arm export policy required')
    check_schedule(plan, experiment['schedule'])
    return core.export_completed(root, checkpoints, plan, binding, selection_replay=select_history)


def selected_matcher(root, process_return, expected_plan_sha256, arm):
    """Successful complete process + native SIM selection, never a live checkpoint."""
    selected = core.selected_curriculum_matcher(root, process_return, expected_plan_sha256)
    require(selected['source_binding'].get('matcher_v2_experiment', {}).get('arm') == arm,
            'completed Matcher belongs to a different new arm')
    import torch
    check_export(torch.load(selected['path'], map_location='cpu', weights_only=False))
    return selected
