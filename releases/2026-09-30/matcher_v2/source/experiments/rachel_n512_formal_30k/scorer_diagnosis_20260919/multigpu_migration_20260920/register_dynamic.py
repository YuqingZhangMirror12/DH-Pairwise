"""Transfer untouched, already-authorized GPU experiments from static lanes.

Only scheduling metadata is added. Original CLI, source roots, budget, outputs,
and all completion criteria remain unchanged. This is a one-shot registration.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path

from transfer_registry import register_experiments


def packages_from_plan(plan):
    lanes = {lane['name']: lane for lane in plan['lanes']}
    root = Path(plan['output_root'])
    result = []
    for priority, arm in enumerate(('edge_multi', 'matched_edges'), 10):
        lane = lanes['lane1']
        stages = [deepcopy(s) for s in lane['stages']
                  if s['name'].startswith(arm + '_C')]
        if len(stages) != 7 or stages[0]['name'] != arm + '_C16_train':
            raise ValueError('unexpected original direct-scoring package: ' + arm)
        smoke = next(s for s in lane['stages'] if s['name'] == arm + '_discard32')
        # These smokes already completed; do not rerun or count them as training.
        stages[0]['prerequisites'].append(dict(path=smoke['completion'],
            expect=deepcopy(smoke['completion_expect']), producer_lane='lane1'))
        result.append(dict(name=arm, origin_lane='lane1', stages=stages,
                           priority=priority))

    lane = lanes['lane6']
    stages = [deepcopy(s) for s in lane['stages']
              if s['name'].startswith('M20_all_tokens_')]
    if len(stages) != 5 or not all(s['gpu'] for s in stages):
        raise ValueError('unexpected M20 GPU package')
    # Static lane ordering used to imply these dependencies. Make them explicit
    # when dispatching elsewhere; the CPU preparation stays with its live lane.
    for name in ('M20_hard_SIMVAL6000', 'M20_train_cache', 'M20_val_cache'):
        producer = next(s for s in lane['stages'] if s['name'] == name)
        stages[0]['prerequisites'].append(dict(path=producer['completion'],
            expect=deepcopy(producer['completion_expect']), producer_lane='lane6'))
        stages[0]['prerequisites'].extend(deepcopy(producer.get('additional_completions', [])))
    result.append(dict(name='M20_all_tokens', origin_lane='lane6', stages=stages,
                       priority=20))
    for package in result:
        for stage in package['stages']:
            stage['original_runtime_receipt'] = str(root / package['origin_lane'] /
                ('runtime_' + stage['name'] + '.json'))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    plan = json.loads(Path(args.plan).read_text())
    packages = packages_from_plan(plan)
    if args.execute:
        registered = register_experiments(Path(plan['output_root']) / 'dynamic_pool', packages)
        print(json.dumps({'registered': registered}))
    else:
        print(json.dumps([dict(name=p['name'], origin_lane=p['origin_lane'],
            stages=[s['name'] for s in p['stages']]) for p in packages], indent=2))


if __name__ == '__main__':
    main()
