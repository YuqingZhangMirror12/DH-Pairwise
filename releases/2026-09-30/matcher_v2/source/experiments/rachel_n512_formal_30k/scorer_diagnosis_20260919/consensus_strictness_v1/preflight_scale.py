"""One simulation row, all policies; never a tuning/selection run."""
from pathlib import Path
import json
import scale_sweep as s

out = Path(__file__).resolve().parent/'preflight_one_sim'
out.mkdir(exist_ok=False)
for name, _ in s.VARIANTS:
    (out/name/'sim_select').mkdir(parents=True)
s.initialize(s.PHASE1, out, s.FORMAL)
result = s.worker(('sim_select', 0))
print(json.dumps({name:dict(pair_id=r['pair_id'],clusters=r['retained_cluster_count']) for name,r in result.items()}))
