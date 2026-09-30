"""Small exact-mask export; avoid transferring unused feature/attention arrays."""
import argparse
import json
from pathlib import Path
import numpy as np


def main(root):
    root = Path(root)
    protocol = json.loads((root / 's4/protocol.json').read_text())
    if protocol['status'] != 'complete':
        raise ValueError('S4 probe must finish before exporting its masks')
    output = root / 's4_masks'
    output.mkdir(exist_ok=False)
    for row in json.loads((root / 's4/cases.json').read_text()):
        with np.load(root / 's4' / row['arrays_path'], allow_pickle=False) as arrays:
            np.savez_compressed(output / row['arrays_path'], mask_a=arrays['mask_a'], mask_b=arrays['mask_b'])
    print(json.dumps({'status': 'complete', 'masks': protocol['completed_count'] * 2}))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('root')
    main(p.parse_args().root)
