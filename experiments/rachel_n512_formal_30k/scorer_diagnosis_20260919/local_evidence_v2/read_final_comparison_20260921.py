"""Recompute aggregate metrics from an owner's external completed run."""
import argparse
import json,pathlib,collections,math
parser=argparse.ArgumentParser()
parser.add_argument("--run-root",required=True)
r=pathlib.Path(parser.parse_args().run_root)
arms=["reference512_h4","cap128_h4","cap256_h4","cap512_h8","gcn_pairing_h4","gcn_shredding_h4","joint_D_h4","stable_h4"]
out={}
ids=None
for a in arms:
 f=json.loads((r/"training"/a/"freeze.json").read_text())
 ts=f["operating_points"]["thresholds"]
 rr=[json.loads(l) for l in (r/"evaluation"/a/"real/pair_results.jsonl").open()]
 rr=[x for x in rr if not x["label"] or x["review_status"]=="keep"]
 assert len(rr)==803 and sum(x["label"] for x in rr)==295
 if ids is None: ids=set(x["pair_id"] for x in rr)
 else: assert ids==set(x["pair_id"] for x in rr)
 oo=[json.loads(l) for l in (r/"evaluation"/a/"ood/pair_results.jsonl").open()]
 assert len(oo)==301 and all(x["label"] for x in oo)
 out[a]={}
 for op in ["max_f1","recall_99"]:
  t=ts[op]
  def accepted(x): return x["decision_valid"] and x["classification"]["fused"]>=t
  def good(x):
   l=x["layouts"]["full_top2_mode"]
   return bool(x["label"] and l["valid"] and l["translation_l2_px"] is not None and l["translation_l2_px"]<=20)
  tp=sum(x["label"] and accepted(x) for x in rr)
  fp=sum(not x["label"] and accepted(x) for x in rr)
  low=[x for x in rr if good(x) and min(x["candidate_details"]["selected_token_count_a"],x["candidate_details"]["selected_token_count_b"])<=32]
  allgood=[x for x in rr if good(x)]
  out[a][op]={"threshold":t,"accuracy":(tp+508-fp)/803,"recall":tp/295,"f1":2*tp/(295+tp+fp),"fp":fp,"tp":tp,"good_layout_accepted":sum(accepted(x) for x in allgood),"good_layout_total":len(allgood),"low_support_good_total":len(low),"low_support_good_accepted":sum(accepted(x) for x in low),"ood_accepted":sum(accepted(x) for x in oo)}
  if op=="max_f1":
   s=json.loads((r/"evaluation"/a/"real/summary.json").read_text())["groups"]["kept_plus_all_negative"]["classification"]["fused"]["max_f1"]
   for k in ["accuracy","recall","f1","tp","fp"]: assert abs(out[a][op][k]-s[k])<1e-10,(a,k)
   out[a]["auroc"]=s["auroc"]
 s_test=json.loads((r/"evaluation"/a/"test/summary.json").read_text())["groups"]["all"]["classification"]["fused"]["max_f1"]
 out[a]["simtest"]={k:s_test[k] for k in ["accuracy","precision","recall","f1","auroc","tp","fp","fn","tn"]}
 status=json.loads((r/"training"/a/"status.json").read_text())
 out[a]["training_elapsed_seconds"]=status["elapsed_seconds"]
 print(a,json.dumps(out[a]))
