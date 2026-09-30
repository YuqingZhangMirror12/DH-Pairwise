"""Pure local queue transformation tests; no process actions or remote calls."""
from copy import deepcopy
from contextlib import contextmanager
import unittest

from experiments.rachel_n512_formal_30k.reconnect_decoupled_matrix_revision import prepare_configs, option, SUFFIX


@contextmanager
def raises(error, text):
    with unittest.TestCase().assertRaisesRegex(error,text):
        yield


def snapshots():
    result=[]
    for index in range(4):
        root="/root/queues/q%d"%index
        deps=[dict(name="finished_other",pid=10,marker="other.json",status_path="other_state.json")]
        if index:deps.append(dict(name="previous",pid=20+index,marker="/root/queues/q%d/config.json"%(index-1),status_path="old_state.json"))
        stages=[dict(name="tail_%d_%d"%(index,i),command=["python","noop",str(i)]) for i in range(5)]
        if index==0:
            stages=[dict(name="s3_matrix_smoke32",command=["python","train","--output","/run/s3/smoke32","--smoke","32"]),
                    dict(name="s3_matrix_M12_C8",command=["python","train","--output","/run/s3/training"],resume_arguments=["--resume"])]+[
                dict(name="untouched_%d"%i,command=["python","unchanged",str(i)]) for i in range(43)]
        config=dict(root=root,source="/source%d"%index,dependencies=deps,stages=stages)
        state=dict(config=deepcopy(config),stages=[dict(s,status="queued") for s in stages],dependency_pid_overrides={"finished_other":999})
        if index==0:
            state["stages"][0]["status"]="complete";state["stages"][1]["status"]="running"
        result.append(dict(path=root+"/config.json",config=config,state=state))
    return result


def test_only_two_S3_commands_and_dependency_registrations_change():
    original=snapshots();before=deepcopy(original)
    configs,bindings=prepare_configs(original,"/new/source","/archive/training")
    assert original==before
    assert bindings==[None,"previous","previous","previous"]
    assert len(configs[0]["stages"])==45
    assert configs[0]["stages"][2:]==original[0]["config"]["stages"][2:]
    for item in configs[0]["stages"][:2]:
        assert option(item["command"],"--matcher-checkpoint")=="/archive/training/epoch_012.pt"
        assert "--resume" not in item["command"]
    assert option(configs[0]["stages"][0]["command"],"--smoke-phase")=="classifier"
    for i,c in enumerate(configs):
        assert c["root"]==original[i]["config"]["root"]+SUFFIX
        assert c["dependencies"][0]["pid"]==999
        if i:
            assert c["source"]==original[i]["config"]["source"]
            assert c["stages"]==original[i]["config"]["stages"]
            assert c["dependencies"][1]["pid"] is None
            assert c["dependencies"][1]["marker"]==configs[i-1]["root"]+"/config.json"


def test_started_tail_or_changed_config_rejected_instead_of_replaying():
    values=snapshots();values[2]["state"]["stages"][0]["status"]="complete"
    with raises(ValueError,"pure waiting"):
        prepare_configs(values,"/new/source","/archive/training")
    values=snapshots();values[0]["config"]["stages"][0]["command"].append("unexpected")
    with raises(ValueError,"differs from frozen"):
        prepare_configs(values,"/new/source","/archive/training")


def test_existing_matcher_or_resume_command_not_silently_replaced():
    for args in (["--matcher-checkpoint","other.pt"],["--resume"]):
        values=snapshots()
        values[0]["config"]["stages"][1]["command"]+=args
        values[0]["state"]["config"]=deepcopy(values[0]["config"])
        with raises(ValueError,"silently replace|must not resume"):
            prepare_configs(values,"/new/source","/archive/training")
