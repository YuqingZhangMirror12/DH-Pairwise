"""Read only: distinguish diagonal pixels from genuinely detached material."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
from scipy import ndimage
import torch

from experiments.rachel_n512_formal_30k.train_partial_seam import read_pair_metadata
from experiments.rachel_n512_formal_30k.train_realism_data_ablation import make_training_dataset
from staging.pairwise_v0_2.pairwise_data.rachel_guided_partial_dataset import GuidedPartialDataset


def run(args):
    torch.set_num_threads(1)
    base, _ = make_training_dataset(args.dataset, args.train_manifest)
    dataset = GuidedPartialDataset(base, pair_metadata=read_pair_metadata(args.train_manifest),
        bank=args.outline_bank, epoch=1)
    counts, records = Counter(), []
    for group_index in range(args.groups):
        for i in dataset._groups[group_index]:
            sample = base[i]
            for side in "ab":
                mask = np.asarray(getattr(sample, "mask_" + side))[0].astype(bool)
                c4, c8 = (ndimage.label(mask, ndimage.generate_binary_structure(2, c)) for c in (1, 2))
                sizes = np.sort(np.bincount(c8[0].ravel())[1:])[::-1]
                category = "one_4connected" if c4[1] == 1 else ("one_8connected" if c8[1] == 1 else "detached_8connected")
                counts[category] += 1
                records.append(dict(pair_id=sample.pair_id, label=bool(sample.label), side=side,
                    components4=c4[1], components8=c8[1], sizes8=sizes[:8].tolist(),
                    nondominant_pixels=int(sizes[1:].sum()), total_pixels=int(mask.sum())))
    result = dict(status="complete", rows=args.groups * 2, endpoint_exposures=len(records),
        counts=dict(counts), records=records, original_files_modified=False)
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps(dict(counts=dict(counts), endpoint_exposures=len(records))))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for name in ("dataset", "train-manifest", "outline-bank", "output"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--groups", type=int, default=128)
    run(p.parse_args())
