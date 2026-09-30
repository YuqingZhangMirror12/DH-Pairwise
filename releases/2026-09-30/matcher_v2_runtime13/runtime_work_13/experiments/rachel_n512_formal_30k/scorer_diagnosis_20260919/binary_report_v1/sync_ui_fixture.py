"""Update only a private visual-test app's fixture data, preserving its identity."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixture', type=Path, required=True)
    parser.add_argument('--app', type=Path, required=True)
    parser.add_argument('--complete', action='store_true')
    args = parser.parse_args()
    fixture = json.loads(args.fixture.read_text())
    query = fixture['queries']['binary_ui_fixtures']
    if query['source']['kind'] != 'synthetic' or not all(row.get('fixture') is True for row in query['rows']):
        raise ValueError('Only synthetic fixtures belong in this private test app')
    target = args.app / 'src/data.json'
    data = json.loads(target.read_text())
    if data.get('id') != 'report:ab92bece-b138-4939-9077-29f7d6bbd8f7':
        raise ValueError('Refusing to update a different app')
    for field in ('title', 'status', 'filters', 'queries'):
        data[field] = fixture[field]
    # This status concerns the internal display harness, never the experiments.
    data['buildStatus'] = 'complete' if args.complete else 'creating'
    target.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(status='private_fixture_synced', app_id=data['id'],
                          synthetic=True, real_results=False, cases=len(query['rows']))))


if __name__ == '__main__':
    main()
