"""Finalize report metadata only after recorded local/UI checks; never run inference."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('project', 'binding', 'annotations', 'ui-verification', 'receipt'):
        p.add_argument('--' + name, type=Path, required=True)
    args = p.parse_args()
    require(not args.receipt.exists(), 'preserve prior completion receipt')
    binding = json.loads(args.binding.read_text())
    ui = json.loads(args.ui_verification.read_text())
    require(binding['status'] == 'passed' and ui['status'] == 'passed', 'unfinished checks')
    target = args.project / 'src/data.json'
    initial_sha = sha(target)
    require(initial_sha == binding['snapshot_sha256'], 'report changed after binding')
    require(sha(args.annotations) == binding['annotations_sha256'], 'annotations changed')
    data = json.loads(target.read_text())
    require(data['id'] == binding['report_id'], 'report identity changed')
    require(data['buildStatus'] == 'updating', 'unexpected report lifecycle')
    qs = data['queries']
    require(len(qs['threshold_e32_metrics']['rows']) == 144, 'wrong metric rows')
    require(len(qs['threshold_e32_c10']['rows']) == 11, 'wrong fixed cases')
    require(sum(len(c['clusters']) for c in qs['threshold_e32_c10']['rows']) == 38,
            'missing captured clusters')
    require(len(qs['matched_real_test_baselines']['rows']) == 16, 'wrong matched baseline rows')
    for q in qs.values():
        for path, expected in q.get('source', {}).get('inputSha256', {}).items():
            require(sha(path) == expected, 'frozen source changed: ' + path)
    for path, expected in ui['authored_source_sha256'].items():
        require(sha(path) == expected, 'authored content changed after UI check: ' + path)
    data['buildStatus'] = 'complete'
    data['generatedAt'] = datetime.now(timezone.utc).isoformat()
    tmp = target.with_suffix('.json.complete.tmp')
    require(not tmp.exists(), 'unfinished earlier metadata write')
    with tmp.open('x') as stream:
        json.dump(data, stream, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
        stream.write('\n')
    require(sha(target) == initial_sha and sha(args.annotations) == binding['annotations_sha256'],
            'concurrent source or human annotation change')
    tmp.replace(target)
    actual = json.loads(target.read_text())
    require(actual == data, 'metadata write verification failed')
    receipt = dict(status='passed', build_status='complete', report_id=data['id'],
                   before_snapshot_sha256=initial_sha, snapshot_sha256=sha(target),
                   annotations_sha256=sha(args.annotations),
                   ui_verification_sha256=sha(args.ui_verification),
                   changed_fields=['buildStatus', 'generatedAt'],
                   query_rows_unchanged=True, neural_forward_passes=0,
                   remote_state_writes=0, source_sha256=sha(__file__))
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    with args.receipt.open('x') as stream:
        json.dump(receipt, stream, indent=2)
        stream.write('\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
