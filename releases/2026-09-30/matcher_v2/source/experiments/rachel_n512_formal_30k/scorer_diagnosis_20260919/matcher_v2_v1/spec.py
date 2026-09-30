"""Explicit architecture identity for new exports, never guessed from weights."""
from dataclasses import asdict
import json

from staging.pairwise_v0_2.models.rachel_n512 import RachelN512Config
from .adapter import fresh_matcher_v2
from .network import MatcherV2Config

SCHEMA = 's7-matcher-v2-model-spec/1'


def model_spec(model, seed):
    if type(seed) is not int or seed < 0:
        raise ValueError('explicit nonnegative initialization seed required')
    return json.loads(json.dumps(dict(schema=SCHEMA, architecture=asdict(model.base.config),
        matcher_v2=asdict(model.v2_config), model_seed=seed), allow_nan=False))


def from_model_spec(spec, *, frozen, state=None):
    if set(spec) != {'schema', 'architecture', 'matcher_v2', 'model_seed'} or spec['schema'] != SCHEMA:
        raise ValueError('explicit v2 model spec required; legacy state must not be guessed')
    cfg = MatcherV2Config(**spec['matcher_v2'])
    model = fresh_matcher_v2(RachelN512Config(**spec['architecture']), config=cfg, seed=spec['model_seed'])
    if model_spec(model, spec['model_seed']) != spec:
        raise ValueError('noncanonical model specification')
    if state is not None:
        model.load_state_dict(state, strict=True)
    model.set_frozen(frozen)
    return model
