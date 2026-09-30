"""Copy the immutable experiment1/2 source; adapt only experiment3's engine."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

REL = Path('experiments/rachel_n512_formal_30k/scorer_diagnosis_20260919')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inventory(root):
    return {str(p.relative_to(root)): sha(p) for p in root.rglob('*.py')}


def once(text, before, after):
    if text.count(before) != 1:
        raise ValueError('unrecognized engine revision: ' + before[:70])
    return text.replace(before, after)


def adapt_engine(text):
    start = text.index('def make_binding(')
    end = text.index('def make_caches(', start)
    text = text[:start] + 'from ..aggressive_binary_v1.runtime import make_binding\n\n\n' + text[end:]
    text = text[:text.index('def run(args):')] + (
        "from ..aggressive_binary_v1.runtime import run, main\n\n"
        "if __name__ == '__main__':\n    main()\n")
    edits = [
        ("frozen_hash=state_digest(model.matcher) if stage=='scorer' else None",
         "initial_matcher_hash=state_digest(model.matcher)\n    initial_head_hash=state_digest(model.head)\n    frozen_hash=initial_matcher_hash if stage=='scorer' else None"),
        ("development=None if args.preflight_steps else RealDevelopment(args.real_split,binding['real_development'])",
         "development=None if args.preflight_steps or stage=='matcher' else RealDevelopment(args.real_split,binding['real_development'])"),
        ("real_report,real_predictions=development.evaluate(model,device,config)\n        report['real_development']=real_report\n        predictions['real_development']=real_predictions\n        real_improve=epoch_number>0 and (best_real is None or tuple(real_report['key'])>tuple(best_real['key']))\n        if real_improve:\n            best_real=dict(epoch=epoch_number,key=real_report['key'],thresholds=real_report['thresholds'])",
         "real_improve=False\n        if development is not None:\n            real_report,real_predictions=development.evaluate(model,device,config)\n            report['real_development']=real_report\n            predictions['real_development']=real_predictions\n            real_improve=epoch_number>0 and (best_real is None or tuple(real_report['key'])>tuple(best_real['key']))\n            if real_improve:\n                best_real=dict(epoch=epoch_number,key=real_report['key'],thresholds=real_report['thresholds'])"),
        ("improve=best is None or tuple(report['key'])>tuple(best['key'])",
         "eligible=stage!='matcher' or epoch_number>0\n        improve=eligible and (best is None or tuple(report['key'])>tuple(best['key']))"),
        ("layout_improve=best_layout is None or layout_key>best_layout['layout']",
         "layout_improve=eligible and (best_layout is None or layout_key>best_layout['layout'])"),
        ("best_real=best_real,binding=binding,selection_on_real=True,test_used=False)",
         "best_real=best_real,binding=binding,selection_on_real=stage=='scorer',test_used=False)"),
        ("model_state_hashes=hashes,matcher_unchanged=frozen_unchanged,",
         "model_state_hashes=hashes,matcher_unchanged=state_digest(model.matcher)==initial_matcher_hash,\n                        head_unchanged=state_digest(model.head)==initial_head_hash,\n                        matcher_frozen_expected=stage=='scorer',"),
        ("if best_real is None:raise AssertionError('no trained REAL-SELECT checkpoint')",
         "if stage=='scorer' and best_real is None:raise AssertionError('no trained REAL-SELECT checkpoint')\n    if stage=='matcher' and state_digest(model.head)!=initial_head_hash:\n        raise AssertionError('binary Scorer was changed during Matcher-only training')"),
        ("selection_on_real=True,test_used=False,best_real=best_real,best_real_sha256=digest(root/'best_real.pt'),",
         "selection_on_real=stage=='scorer',test_used=False,best_real=best_real,\n        best_real_sha256=digest(root/'best_real.pt') if stage=='scorer' else None,"),
    ]
    for old, new in edits:
        text = once(text, old, new)
    return text.replace(text[:text.index('import argparse')],
                        '"""Experiment3 stage engine: new random Matcher, then fresh binary head."""\n', 1)


def prepare(baseline, out):
    baseline = Path(baseline).resolve(); out = Path(out).resolve()
    if out.exists():
        raise ValueError('new independent preparation root required')
    before = inventory(baseline)
    train_path = REL / 's7_consensus_v1' / 'train.py'
    engine = adapt_engine((baseline / train_path).read_text())
    config_path = REL / 's7_consensus_v1' / 'config.py'
    config = once((baseline / config_path).read_text(),
                  "schema: str = 'binary-cluster-scorer/1'", "schema: str = 'aggressive-binary-training/1'")
    shutil.copytree(baseline, out, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copytree(Path(__file__).parent, out / REL / 'aggressive_binary_v1',
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    (out / train_path).write_text(engine)
    (out / config_path).write_text(config)
    (out / REL / 's7_consensus_v1' / 'launch.py').write_text(
        "raise RuntimeError('Experiment3 requires full human-approved data and its dedicated external launcher.')\n")
    if inventory(baseline) != before:
        raise AssertionError('experiment1/2 baseline source changed')
    receipt = dict(schema='aggressive-binary-preparation/1', status='prepared_not_validated',
        source_sha256=inventory(out), baseline_sha256=before, baseline_unchanged=True,
        full_generation_authorized=False, gpu_preflight=False, formal_training_started=False)
    receipt_path = out.parent / ('preparation_' + out.name + '.json')
    if receipt_path.exists():
        raise ValueError('preserve previous preparation receipt')
    receipt_path.write_text(json.dumps(receipt, indent=2) + '\n')
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', required=True); parser.add_argument('--out', required=True)
    args = parser.parse_args()
    receipt = prepare(args.baseline, args.out)
    print(json.dumps(dict(status=receipt['status'], files=len(receipt['source_sha256']))))


if __name__ == '__main__':
    main()
