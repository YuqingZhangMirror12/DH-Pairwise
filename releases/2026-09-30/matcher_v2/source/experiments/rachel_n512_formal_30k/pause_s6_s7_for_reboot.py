"""User-requested reboot pause: preserve the last committed state; never relaunch."""
import json
import os
import shutil
import signal
import time
from pathlib import Path

R = Path("/root/autodl-tmp/rachel_score_design_20260913_001")
WORK = R / "s6_s7_20260915/manual_pause_20260916"
TRAIN = R / "attention_depth_20260915/s4_cross_attention_depth2/training"
HANDLES = [
    (1598, "894339636", "s6_depth_2_4"),
    (1599, "894339661", "s7_augmented_full24"),
    (1600, "894339687", "original_tail_1"),
    (1601, "894339712", "original_tail_2"),
    (1602, "894339737", "original_tail_3"),
]
TRAINER = (28810, "899429942")

def proc(pid):
    try:
        f = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        argv = Path(f"/proc/{pid}/cmdline").read_bytes().decode().split(chr(0))
        return dict(pid=pid, state=f[0], ppid=int(f[1]), start_ticks=f[19], argv=argv)
    except FileNotFoundError:
        return None

def same_live(pid, start):
    p = proc(pid)
    return bool(p and p["start_ticks"] == start and p["state"] not in ("Z", "X"))

def write_receipt(data):
    tmp = WORK / "pause_receipt.tmp"
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    os.replace(tmp, WORK / "pause_receipt.json")

def wait_gone(handles, seconds):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if not any(same_live(pid, start) for pid, start in handles):
            return True
        time.sleep(.2)
    return False

def main():
    queues = []
    for pid, start, name in HANDLES:
        p = proc(pid)
        if not p or p["start_ticks"] != start or "--config" not in p["argv"]:
            raise RuntimeError(f"Queue identity changed: {pid}")
        config_path = Path(p["argv"][p["argv"].index("--config") + 1])
        if config_path.parent.name != name:
            raise RuntimeError(f"Unexpected queue config: {pid}")
        config = json.loads(config_path.read_text())
        state = json.loads((Path(config["root"]) / "queue_state.json").read_text())
        queues.append(dict(pid=pid, start_ticks=start, name=name,
                           config=str(config_path), source=config["source"],
                           root=config["root"], argv=p["argv"], state_before=state.get("status"),
                           active_stage=state.get("active_stage")))
        if pid == 1598 and state.get("active_child", {}).get("pid") != TRAINER[0]:
            raise RuntimeError("Active trainer changed before pause")
    p = proc(TRAINER[0])
    if not p or p["start_ticks"] != TRAINER[1] or str(TRAIN) not in p["argv"]:
        raise RuntimeError("Trainer identity changed")
    WORK.mkdir(exist_ok=False)
    receipt = dict(schema="user-requested-reboot-pause/1", status="pausing",
                   requested_at_unix=time.time(), automatic_resume_allowed=False,
                   resume_requires_explicit_user_message=True, queues=queues,
                   trainer=p, training_run=str(TRAIN))
    write_receipt(receipt)
    stopped = []
    try:
        for row in reversed(queues):
            os.kill(row["pid"], signal.SIGSTOP)
            stopped.append(row["pid"])
        os.kill(TRAINER[0], signal.SIGSTOP)
        stopped.append(TRAINER[0])
        time.sleep(.2)
        shutil.copy2(TRAIN / "last.pt", WORK / "s6_depth2_last.pt")
        for name in ("protocol.json", "status.json", "matcher_pretraining_receipt.json"):
            shutil.copy2(TRAIN / name, WORK / name)
        for row in queues:
            shutil.copy2(Path(row["root"]) / "queue_state.json", WORK / (row["name"] + "_queue_state.json"))
            shutil.copy2(row["config"], WORK / (row["name"] + "_config.json"))
        import torch
        ck = torch.load(WORK / "s6_depth2_last.pt", map_location="cpu", weights_only=False)
        required = ("model_state_dict", "optimizer_state_dict", "rng_state", "resume_identity")
        if any(k not in ck for k in required):
            raise RuntimeError("Checkpoint lacks required resume state")
        if ck["global_exposure"] != ck["completed_segments"] * 6000:
            raise RuntimeError("Invalid committed exposure")
        receipt["checkpoint"] = dict(path=str(WORK / "s6_depth2_last.pt"),
            completed_segments=ck["completed_segments"], global_exposure=ck["global_exposure"],
            epoch=ck["epoch"], phase=ck["phase"], optimizer_updates=ck["optimizer_updates"],
            saved_model_optimizer_rng=True, size_bytes=(WORK / "s6_depth2_last.pt").stat().st_size,
            unfinished_segment_may_be_replayed=True)
        del ck
        write_receipt(receipt)
        os.kill(TRAINER[0], signal.SIGINT)
        os.kill(TRAINER[0], signal.SIGCONT)
        if not wait_gone([TRAINER], 15):
            os.kill(TRAINER[0], signal.SIGTERM)
            if not wait_gone([TRAINER], 10):
                raise RuntimeError("Trainer did not terminate; checkpoint is preserved")
        for row in reversed(queues):
            if same_live(row["pid"], row["start_ticks"]):
                os.kill(row["pid"], signal.SIGTERM)
                os.kill(row["pid"], signal.SIGCONT)
        all_handles = [TRAINER] + [(row["pid"], row["start_ticks"]) for row in queues]
        if not wait_gone(all_handles, 10):
            raise RuntimeError("A queue did not terminate")
        receipt.update(status="paused_for_user_server_restart", paused_at_unix=time.time(),
                       all_target_processes_stopped=True, queue_configs_unchanged=True,
                       resume_note="Do not auto-restart. After explicit user readiness, resume S6 with --resume, then rebind S7 and three tails to new predecessor PIDs.")
        receipt["training_status_after"] = json.loads((TRAIN / "status.json").read_text())
        write_receipt(receipt)
        print(json.dumps(receipt, ensure_ascii=False))
    except BaseException as error:
        receipt.update(status="pause_needs_attention", error=repr(error),
                       stopped_pids=stopped, automatic_resume_allowed=False)
        write_receipt(receipt)
        raise

if __name__ == "__main__":
    main()

