"""Discard-only paired-density GPU wiring test; NOT formal training.

Exactly twelve completed source-density smoke pairs, in order0..11+0..3,
provide16 exposures and one effective16 update. This deliberately does not
call the formal trainer/population validator and cannot create checkpoints.
"""
import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import platform
import sys
from types import SimpleNamespace

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch

from experiments.rachel_n512_formal_30k.train_score_density import build_training_model,create_optimizer
from experiments.rachel_n512_formal_30k.train_score_design import train_score_segment,SEED
from experiments.rachel_n512_formal_30k.train_joint_damage import state_digest
from experiments.rachel_n512_formal_30k.train_realism_data_ablation import save_json
from staging.pairwise_v0_2.pairwise_data.rachel_paired_density_dataset import (
    PairedSourceDensityWeatheredDataset,file_sha256)
from staging.pairwise_v0_2.training import rachel_n512_runner as runner
from staging.pairwise_v0_2.training.rachel_weathering_training import make_weathering_loader


def run(args):
    if platform.system()!='Linux' or not torch.cuda.is_available():
        raise RuntimeError('discard smoke requires remote Linux CUDA')
    torch.set_num_threads(1)
    root=Path(args.output).resolve();root.mkdir(parents=True,exist_ok=False)
    source_path=Path(args.checkpoint).resolve(strict=True)
    dataset=PairedSourceDensityWeatheredDataset(args.manifest)
    if len(dataset)!=12 or dataset.contour_cap!=args.contour_cap:
        raise ValueError('this bounded wiring smoke requires the explicit12-pair manifest at the requested cap')
    indices=list(range(12))+list(range(4))
    pair_ids=[dataset.entries[i]['pair_id'] for i in indices]
    protocol=dict(schema_version='paired-density-discard-gpu-smoke/1',
        formal_training=False,checkpoint_created=False,weights_discarded=True,
        architecture='candidate_dual',contour_cap=args.contour_cap,
        unique_pair_count=12,exposures=16,expected_optimizer_updates=1,
        source_indices=indices,exposure_pair_ids=pair_ids,
        source_manifest=str(dataset.manifest_path),source_manifest_sha256=file_sha256(dataset.manifest_path),
        source_density_identity=dataset.identity,source_density_protocol=dataset.protocol,
        metadata_checkpoint=str(source_path),metadata_checkpoint_sha256=file_sha256(source_path),
        source_weights_loaded=False,seed=SEED,precision='fp32',microbatch=4,effective_batch=16,
        optimizer='AdamW',learning_rate=1e-4,weight_decay=1e-4,grad_clip_norm=5.,
        matching_loss='existing train_score_segment/compute_weathering_loss unchanged',
        candidate_R_weight=.5,candidate_R_tolerance_px=20.,
        validation_or_test_or_real_or_ood_opened=False,
        device_name=torch.cuda.get_device_name(0),pytorch=torch.__version__)
    save_json(root/'protocol.json',protocol)
    save_json(root/'status.json',dict(status='running',global_exposure=0,optimizer_updates=0))
    try:
        source=torch.load(source_path,map_location='cpu',weights_only=False)
        runner._set_determinism(SEED)
        built,loss_config,digests=build_training_model(source,args.contour_cap,'candidate_dual')
        del source
        protocol.update(model_metadata=built.metadata,loss_config=asdict(loss_config),initialization=digests)
        save_json(root/'protocol.json',protocol)
        device=torch.device('cuda:0');model=built.model.to(device)
        optimizer=create_optimizer(model,'candidate_dual')
        loader=make_weathering_loader(dataset,indices,batch_size=4,num_workers=0,seed=SEED,contour_cap=args.contour_cap)
        torch.cuda.reset_peak_memory_stats(device)
        report=train_score_segment(model,loader,optimizer,loss_config,device,
            SimpleNamespace(output=str(root),architecture='candidate_dual',log_every=4),epoch=1)
        torch.cuda.synchronize(device)
        if report['samples']!=16 or report['optimizer_updates']!=1:
            raise ValueError('smoke exposure/update contract changed')
        if not all(math.isfinite(v) for v in report['loss_components'].values()):
            raise FloatingPointError('nonfinite component loss')
        final_digest=state_digest(model)
        if final_digest==digests['initial_weights_sha256']:
            raise ValueError('optimizer update did not change model state')
        report.update(status='complete',formal_training=False,weights_discarded=True,
            unique_pair_count=12,exposures=16,contour_cap=args.contour_cap,
            manifest_identity=dataset.identity,initialization=digests,
            final_state_digest_for_audit_only=final_digest,checkpoint_created=False,
            peak_reserved_gpu_bytes=torch.cuda.max_memory_reserved(device),
            all_component_losses_finite=True)
        save_json(root/'smoke_report.json',report)
        save_json(root/'status.json',dict(status='complete_discard_smoke',global_exposure=16,
            optimizer_updates=1,formal_training=False,weights_discarded=True))
        del optimizer,model,built
        torch.cuda.empty_cache()
        print(json.dumps(report,indent=2,allow_nan=False),flush=True)
        return 0
    except Exception as error:
        failure=dict(status='failed',formal_training=False,weights_discarded=True,checkpoint_created=False,
            error_type=type(error).__name__,error=str(error),contour_cap=args.contour_cap)
        save_json(root/'smoke_report.json',failure)
        save_json(root/'status.json',failure)
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',required=True);p.add_argument('--checkpoint',required=True)
    p.add_argument('--contour-cap',required=True,type=int,choices=(512,1024));p.add_argument('--output',required=True)
    sys.exit(run(p.parse_args()))
