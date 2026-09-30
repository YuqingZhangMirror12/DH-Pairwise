import json

from experiments.rachel_n512_formal_30k import summarize_pairability_overnight as target


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def metric(threshold=.5, f1=.8):
    return dict(threshold=threshold, accuracy=.75, precision=.8, recall=.8, f1=f1,
                auroc=.85, auprc=.9, tp=400, fp=100, fn=100, tn=400)


def population(n=3000, positives=1500):
    return dict(sample_count=n, positive_count=positives,
                classification={branch:{"at_0_5":metric(), "at_original_frozen_threshold":metric(.7),
                            "at_validation_row_f1_threshold":metric(.7)} for branch in ("coarse","local","fused")},
                layout={"top2_mode":dict(positive_pose_coverage=1., median_px_conditional=2., p90_px_conditional=11.,
                    recall={str(t):.6 for t in (2,5,8,10)},
                    assembly={str(t):dict(precision=.3,recall=.2,f1=.24,tp=100,fp=200,fn=400) for t in (2,5,8,10)})})


def test_absent_or_provisional_sources_stay_pending_not_zero(tmp_path):
    save(tmp_path/"realism_training/original24k/train_val_freeze.json",dict(status="provisional",validation=population()))
    report=target.summarize(tmp_path)
    assert report["status"]=="partial"
    assert len(report["results"])==32
    assert all(row["status"]=="pending" and row["sample_count"] is None and row["classification"]==[] for row in report["results"])
    assert report["winner_selection_performed"] is False
    assert "pending" in target.markdown(report)


def test_head_paths_original_vs_recalibrated_and_strict_are_separate(tmp_path):
    policies={"existing_coarse":dict(branch="existing_coarse",threshold=.7),
              "existing_coarse_original_frozen":dict(branch="existing_coarse",threshold=.2),
              "matrix_only":dict(branch="matrix_only",threshold=.6)}
    save(tmp_path/"heads/matrix_soft/validation_freeze.json",dict(status="complete",matcher_checkpoint_id="fixed",policies=policies))
    def values(n,p,ap):
        rows={k:metric(v["threshold"]) for k,v in policies.items()}
        for row in rows.values():row["auprc"]=ap
        return dict(sample_count=n,positive_count=p,policies=rows)
    save(tmp_path/"heads/matrix_soft_real_native/real/metrics.json",dict(status="complete",split="real",test_or_real_used_for_fit=False,
        native_probability_reporting=True,
        cache={"matcher_checkpoint_id":"fixed"},classification=values(1016,508,.8),strict_classification=values(547,508,.98)))
    full=target._head(tmp_path,"matrix_soft","real_balanced1016")
    strict=target._head(tmp_path,"matrix_soft","real_strict547")
    assert full["status"]==strict["status"]=="complete"
    assert full["classification"][0]["ap"]==.8 and strict["classification"][0]["ap"]==.98
    assert full["classification"][0]["threshold_regime"]=="val_recalibrated"
    assert full["classification"][1]["threshold_regime"]=="original_frozen"
    assert full["classification"][1]["threshold"]==.2
    assert not full["layout"] and not full["joint"]


def test_missing_native_correction_does_not_fall_back_to_old_ap(tmp_path):
    save(tmp_path/"heads/matrix_soft_test/test/metrics.json",dict(status="complete",split="test"))
    save(tmp_path/"score_fusion/test/metrics.json",dict(status="complete",split="test"))
    assert target._head(tmp_path,"matrix_soft","test")["status"]=="pending"
    assert target._score_fusion(tmp_path,"test")["classification"]==[]


def test_common_layout_and_joint_do_not_replace_pair_f1_or_select_real_winner(tmp_path):
    source=dict(population(1016,508),status="complete",split="real",selected_full_decoder="top2_mode",
                strict_summary=population(547,508))
    source["layout"]["top1_mode"]={**source["layout"]["top2_mode"],"recall":{str(t):.99 for t in (2,5,8,10)}}
    save(tmp_path/"assignment/real/summary.json",source)
    row=target._common(tmp_path,"assignment","real_balanced1016")
    assert row["classification"][0]["f1"]==.8
    assert row["joint"][0]["f1"]==.24
    assert next(x for x in row["layout"] if x["predeclared_selected_decoder"])["decoder"]=="top2_mode"
    assert target._common(tmp_path,"assignment","real_strict547")["sample_count"]==547


def test_data_model_original_threshold_key_is_own_val_alias_and_output_is_new(tmp_path):
    root=tmp_path/"run"
    save(root/"realism_evaluation/realism60k/test/summary.json",dict(population(),status="complete",split="test",selected_full_decoder="top2_mode"))
    row=target._common(root,"realism60k","test")
    frozen=[x for x in row["classification"] if x["policy"]=="at_original_frozen_threshold"]
    assert all(x["threshold_regime"]=="own_model_val_frozen" and x["duplicate_of_own_val_fused"] for x in frozen)
    out=tmp_path/"report"
    target.write_report(root,out)
    assert (out/"summary.json").is_file() and (out/"summary.md").is_file()
    import pytest
    with pytest.raises(FileExistsError):target.write_report(root,out)


def test_fixed_top2_joint_counts_wrong_pose_as_fp_and_fn_and_uses_gate(tmp_path):
    score_dir=tmp_path/"heads/matrix_soft_test/test"
    pose_dir=tmp_path/"assignment/test"
    score_dir.mkdir(parents=True)
    pose_dir.mkdir(parents=True)
    policies={"matrix_only":dict(branch="matrix_only",threshold=.5,gate_threshold=None),
              "gated":dict(branch="matrix_only",threshold=.5,gate_threshold=.9)}
    freeze=tmp_path/"heads/matrix_soft/validation_freeze.json"
    save(freeze,dict(status="complete",matcher_checkpoint_id="fixed",policies=policies))
    save(pose_dir/"protocol.json",dict(status="complete",matcher_checkpoint_id="fixed"))
    score_rows=[]
    pose_rows=[]
    for i in range(3000):
        score_rows.append(dict(pair_id=str(i),label=i<1500,
            scores={"matrix_only":.8,"existing_coarse":.95 if i<500 else .1}))
        pose_rows.append(dict(pair_id=str(i),label=i<1500,layouts={"top2_mode":dict(valid=i<1000,
            translation_l2_px=5 if i<1000 else 0)}))
    (score_dir/"pair_scores.jsonl").write_text("\n".join(json.dumps(row) for row in score_rows))
    (pose_dir/"pair_results.jsonl").write_text("\n".join(json.dumps(row) for row in pose_rows))
    node=dict(target._base("matrix_soft","test",score_dir/"metrics.json"),status="complete",
        provenance=dict(validation_freeze=str(freeze),matcher_checkpoint_id="fixed"),
        classification=[dict(policy=name,threshold_regime="val_recalibrated") for name in policies])
    target._derive_fixed_top2(tmp_path,node,dict(status="complete"),{})
    assert node["derived_joint_status"]=="complete"
    at10=next(x for x in node["derived_joint"] if x["policy"]=="matrix_only" and x["tolerance_px"]==10)
    assert (at10["tp"],at10["fp"],at10["fn"])==(1000,2000,500)
    gated=next(x for x in node["derived_joint"] if x["policy"]=="gated" and x["tolerance_px"]==10)
    assert (gated["tp"],gated["fp"],gated["fn"])==(500,0,1000)
    at2=next(x for x in node["derived_joint"] if x["policy"]=="matrix_only" and x["tolerance_px"]==2)
    assert at2["tp"]==0
    pose_rows[-1]["pair_id"]="unmatched"
    (pose_dir/"pair_results.jsonl").write_text("\n".join(json.dumps(row) for row in pose_rows))
    target._derive_fixed_top2(tmp_path,node,dict(status="complete"),{})
    assert node["derived_joint_status"]=="needs_attention" and node["derived_joint"]==[]
    assert "bijection" in node["derived_joint_issues"][0]


def test_scalar_fusion_joint_uses_policy_named_score_and_preserves_strict_denominator(tmp_path):
    score_dir=tmp_path/"score_fusion/real"
    pose_dir=tmp_path/"assignment/real"
    score_dir.mkdir(parents=True)
    pose_dir.mkdir(parents=True)
    freeze=tmp_path/"score_fusion/val/validation_freeze.json"
    save(freeze,dict(status="complete",matcher_checkpoint_id="fixed",
                    policies={"affine_free":dict(kind="affine_logits",threshold=.5)}))
    save(pose_dir/"protocol.json",dict(status="complete",matcher_checkpoint_id="fixed"))
    score_rows=[dict(pair_id=str(i),label=i<508,strict_member=i<547,scores={"affine_free":.9}) for i in range(1016)]
    pose_rows=[dict(pair_id=str(i),label=i<508,strict_member=i<547,
        layouts={"top2_mode":dict(valid=i<250,translation_l2_px=1 if i<250 else None)}) for i in range(1016)]
    (score_dir/"pair_scores.jsonl").write_text("\n".join(json.dumps(row) for row in score_rows))
    (pose_dir/"pair_results.jsonl").write_text("\n".join(json.dumps(row) for row in pose_rows))
    cache={}
    for population,n,fp in (("real_balanced1016",1016,766),("real_strict547",547,297)):
        node=dict(target._base("score_fusion",population,score_dir/"metrics.json"),status="complete",
            provenance=dict(validation_freeze=str(freeze),matcher_checkpoint_id="fixed"),
            classification=[dict(policy="affine_free",threshold_regime="val_recalibrated")])
        target._derive_fixed_top2(tmp_path,node,dict(status="complete"),cache)
        assert node["derived_joint_status"]=="complete"
        row=node["derived_joint"][0]
        assert (row["sample_count"],row["positive_count"],row["tp"],row["fp"],row["fn"])==(n,508,250,fp,258)


def infonce_fixture(root):
    def scores(n, p, threshold):
        tp, fp, fn = p - 1, 2, 1
        tn = n - p - fp
        return dict(threshold=threshold, accuracy=(tp+tn)/n, precision=tp/(tp+fp),
                    recall=tp/p, f1=2*tp/(2*tp+fp+fn), auroc=.85, auprc=.9,
                    tp=tp, fp=fp, fn=fn, tn=tn)
    freeze = dict(schema_version=target.INFONCE_SCHEMA, status="complete", selected_on="val",
        test_or_real_used_for_fit=False, encoder_frozen=True, train_count=24000, val_count=3000,
        completed_epochs=10, selected_epoch=3, matcher_checkpoint_id="fixed", config={"seed":260909},
        thresholds={"infonce":-.2, "frozen_feature_cosine":-.6},
        validation=scores(3000,1500,-.2), baseline_validation=scores(3000,1500,-.6),
        history=[dict(epoch=10,validation=metric(f1=.1))])
    save(root/"coarse_infonce/training/validation_freeze.json",freeze)
    for split,n,p in (("test",3000,1500),("real",1016,508)):
        def payload(n,p):
            return dict(sample_count=n,positive_count=p,
                        policies={name:scores(n,p,t) for name,t in freeze["thresholds"].items()})
        data=dict(status="complete",split=split,test_or_real_used_for_fit=False,
            matcher_checkpoint_id="fixed",selected_epoch=3,encoder_frozen=True,
            projection_trained_with_infonce=True,classification=payload(n,p))
        if split=="real":data["strict_classification"]=payload(547,508)
        save(root/"coarse_infonce/evaluation"/split/"metrics.json",data)
    return freeze


def test_optional_infonce_requires_complete_freeze_before_metrics(tmp_path):
    infonce_fixture(tmp_path)
    path=tmp_path/"coarse_infonce/training/validation_freeze.json"
    path.unlink()
    for pop in target.POPULATIONS:
        row=target._coarse_infonce(tmp_path,pop)
        assert row["status"]=="pending" and not row["classification"]
    save(path,dict(status="running"))
    assert target._coarse_infonce(tmp_path,"test")["status"]=="pending"
    infonce_fixture(tmp_path)
    (tmp_path/"coarse_infonce/evaluation/test/metrics.json").unlink()
    assert target._coarse_infonce(tmp_path,"test")["status"]=="pending"
    assert target._coarse_infonce(tmp_path,"val")["status"]=="complete"


def test_infonce_selected_val_and_external_metrics_preserve_cosine_contract(tmp_path):
    freeze=infonce_fixture(tmp_path)
    for pop in target.POPULATIONS:
        row=target._coarse_infonce(tmp_path,pop)
        assert row["status"]=="complete"
        assert row["sample_count"]==target.COUNTS[pop][0]
        assert row["provenance"]["selected_epoch"]==3
        assert row["provenance"]["encoder_frozen"] is True
        assert row["provenance"]["paper_reproduction"] is False
        assert row["provenance"]["gallery_retrieval_evaluated"] is False
        assert [m["policy"] for m in row["classification"]]==list(target.INFONCE_POLICIES)
        assert [m["threshold"] for m in row["classification"]]==[-.2,-.6]
        assert all(m["threshold_regime"]=="own_model_val_frozen" for m in row["classification"])
        assert not row["layout"] and not row["joint"]
    assert target._coarse_infonce(tmp_path,"val")["classification"][0]["f1"]==freeze["validation"]["f1"]
    report=target.summarize(tmp_path)
    text=target.markdown(report)
    assert "不是完整 PairingNet" in text and "不是概率" in text
    assert "| coarse_infonce | infonce | infonce | -0.200 |" in text


def test_infonce_external_cannot_substitute_epoch_threshold_or_checkpoint(tmp_path):
    for key,value in (("selected_epoch",10),("matcher_checkpoint_id","other")):
        infonce_fixture(tmp_path)
        path=tmp_path/"coarse_infonce/evaluation/test/metrics.json"
        data=target.read_json(path)
        data[key]=value
        save(path,data)
        assert target._coarse_infonce(tmp_path,"test")["status"]=="needs_attention"
    infonce_fixture(tmp_path)
    path=tmp_path/"coarse_infonce/evaluation/test/metrics.json"
    data=target.read_json(path)
    data["classification"]["policies"]["infonce"]["threshold"]=-.1
    save(path,data)
    row=target._coarse_infonce(tmp_path,"test")
    assert row["status"]=="needs_attention" and row["classification"]==[]


def test_infonce_joint_reuses_frozen_negative_cosine_thresholds_and_top2(tmp_path):
    infonce_fixture(tmp_path)
    score_dir=tmp_path/"coarse_infonce/evaluation/test"
    pose_dir=tmp_path/"assignment/test"
    save(pose_dir/"protocol.json",dict(status="complete",matcher_checkpoint_id="fixed"))
    score_rows=[dict(pair_id=str(i),label=i<1500,
        scores={"infonce":.1 if i<1000 else -.5,"frozen_feature_cosine":.2}) for i in range(3000)]
    pose_rows=[dict(pair_id=str(i),label=i<1500,
        layouts={"top2_mode":dict(valid=i<500,translation_l2_px=1 if i<500 else None)}) for i in range(3000)]
    (score_dir/"pair_scores.jsonl").write_text("\n".join(json.dumps(row) for row in score_rows))
    (pose_dir/"pair_results.jsonl").write_text("\n".join(json.dumps(row) for row in pose_rows))
    node=target._coarse_infonce(tmp_path,"test")
    target._derive_fixed_top2(tmp_path,node,dict(status="complete"),{})
    assert node["derived_joint_status"]=="complete"
    for name,fp in (("infonce",500),("frozen_feature_cosine",2500)):
        row=next(row for row in node["derived_joint"] if row["policy"]==name and row["tolerance_px"]==10)
        assert (row["tp"],row["fp"],row["fn"])==(500,fp,1000)
        assert row["decoder"]=="top2_mode"


def data_training_freeze(root, arm="realism60k", *, status="complete", selected=48000):
    pool = 60000 if arm=="realism60k" else 24000
    path = root/"realism_training"/arm/"train_val_freeze.json"
    thresholds = dict(coarse=.3,local=.6,fused=.8)
    value = dict(status=status, unique_count=pool, completed_global_exposures=120000,
        completed_optimizer_updates=7500, completed_dataset_epochs=120000//pool,
        completed_validation_events=5, selected_global_exposure=selected,
        selected_optimizer_updates=selected//16, selected_epoch=(selected-1)//pool+1,
        selected_validation_event=selected//24000, checkpoint=str(path.parent/"winner.pt"),
        initial_checkpoint="/shared/warm_start/winner.pt", selection_rule="VAL F1; AP then earlier",
        test_or_real_used_for_fit=False, classifier_thresholds=thresholds,
        validation=dict(sample_count=3000,positive_count=1500,
            methods={name:metric(threshold) for name,threshold in thresholds.items()}))
    save(path,value)
    return path,value


def test_complete_60k_metadata_separates_full_run_from_selected_48k(tmp_path):
    path,_=data_training_freeze(tmp_path)
    report=target.summarize(tmp_path)
    metadata=report["data_training_checkpoint_metadata"]
    row=next(row for row in metadata["arms"] if row["arm"]=="realism60k")
    assert row["status"]=="complete" and row["source"]==str(path)
    assert (row["unique_count"],row["completed_global_exposures"],row["completed_optimizer_updates"],
            row["completed_dataset_epochs"],row["completed_validation_events"])==(60000,120000,7500,2,5)
    assert (row["selected_global_exposure"],row["selected_optimizer_updates"],
            row["selected_epoch"],row["selected_validation_event"])==(48000,3000,1,2)
    assert row["selected_checkpoint_completed_one_pool_pass_in_this_run"] is False
    assert row["selected_checkpoint_is_end_of_run_state"] is False
    assert metadata["warm_start_history_included_in_exposures"] is False
    text=target.markdown(report)
    assert "| realism60k | complete | 60000 | 120000 / 7500 | 2 / 5 | 48000 / 3000 | 1 / 2 |" in text
    assert "该权重在本轮尚未遍历完整训练池" in text
    assert "不包含其历史训练" in text


def test_provisional_training_metadata_never_reports_complete_or_final_selection(tmp_path):
    # Even plausible completed-looking counters cannot override provisional.
    data_training_freeze(tmp_path,status="provisional")
    report=target.summarize(tmp_path)
    row=next(row for row in report["data_training_checkpoint_metadata"]["arms"] if row["arm"]=="realism60k")
    assert row["status"]=="pending"
    assert row["unique_count"] is None and row["completed_global_exposures"] is None
    assert row["selected_global_exposure"] is None and row["selected_epoch"] is None
    assert row["selected_checkpoint_completed_one_pool_pass_in_this_run"] is None
    validation=next(row for row in report["results"] if row["stage"]=="realism60k" and row["population"]=="val")
    assert validation["status"]=="pending" and validation["classification"]==[]
    assert "| realism60k | pending | — | — / — | — / — | — / — | — / — |" in target.markdown(report)


def test_missing_training_freezes_have_metadata_sources_but_no_budget_inference(tmp_path):
    report=target.summarize(tmp_path)
    rows=report["data_training_checkpoint_metadata"]["arms"]
    assert [row["arm"] for row in rows]==list(target.ARMS)
    for row in rows:
        assert row["status"]=="pending" and row["unique_count"] is None
        assert row["completed_optimizer_updates"] is None and row["selected_validation_event"] is None
        assert row["source"]==str(tmp_path/"realism_training"/row["arm"]/"train_val_freeze.json")


def test_training_metadata_never_substitutes_last_weights_or_changes_frozen_metrics(tmp_path):
    paths=[]
    for arm,selected in (("original24k",24000),("matched24k",96000),("realism60k",120000)):
        path,_=data_training_freeze(tmp_path,arm,selected=selected)
        paths.append((path,path.read_bytes()))
        # These are deliberately contradictory non-authority files. Metadata
        # must come only from train_val_freeze, not last/protocol/epoch files.
        save(path.parent/"protocol.json",dict(selected_global_exposure=120000,history_exposures=900000))
        save(path.parent/"last.pt",dict(selected_global_exposure=120000))
    original=target._data_validation(tmp_path,"original24k")
    report=target.summarize(tmp_path)
    validation=next(row for row in report["results"] if row["stage"]=="original24k" and row["population"]=="val")
    assert validation==original
    rows={row["arm"]:row for row in report["data_training_checkpoint_metadata"]["arms"]}
    assert rows["original24k"]["selected_global_exposure"]==24000
    assert rows["matched24k"]["selected_global_exposure"]==96000
    assert rows["original24k"]["selected_checkpoint_is_end_of_run_state"] is False
    assert rows["matched24k"]["selected_checkpoint_is_end_of_run_state"] is False
    assert rows["realism60k"]["selected_checkpoint_is_end_of_run_state"] is True
    assert all(row["checkpoint_artifact_kind"]=="validation_selected_winner" for row in rows.values())
    assert all(row["last_checkpoint_saved_by_trainer"] is False for row in rows.values())
    assert all(row["warm_start_history_included_in_exposures"] is False for row in rows.values())
    assert all(path.read_bytes()==before for path,before in paths)
    assert "不是固定 120k 末权重对照" in target.markdown(report)
