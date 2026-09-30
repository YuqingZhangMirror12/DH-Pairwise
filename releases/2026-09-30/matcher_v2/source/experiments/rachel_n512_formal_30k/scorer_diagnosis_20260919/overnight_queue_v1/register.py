"""Register the latest direct user authorization; does not launch any GPU job."""
from common import *
USER_TEXT='我要去睡觉了，请你确保GPU后续队列任务一直接上训练，尽量不要让GPU空置（如果任务之间有验证顺序，可以让后续某个任务用4张GPU+2张GPU，不要空置GPU）。另外调整一下排队训练任务的顺序，先“两组冻结E32的轻量Scorer对照”，再“新数据从零训练Matcher及轻量Scorer”，最后“E32小学习率联合微调 ”'
def main():
    if (ROOT/'plan.json').exists() or (ROOT/'authorization.json').exists():raise ValueError('already registered')
    ROOT.mkdir(parents=True,exist_ok=True)
    authorization=dict(status='user_authorized',evidence_kind='explicit_user_message_in_current_thread',
        user_text=USER_TEXT,user_text_sha256=hashlib.sha256(USER_TEXT.encode()).hexdigest(),
        order=ORDER,keep_existing_training_unchanged=True,dependency_semantics='priority launch order; independent arms may overlap',
        required_post_training_evaluation=True,automatic_retries=0,registered_unix=time.time(),
        current_topology='two+two+two; four+two permitted only with separately verified same-effective-batch preparation')
    save(ROOT/'authorization.json',authorization)
    old=[]
    for lane in LANES:
        root=old_root(lane);old.append(dict(lane=lane,launch=read(root/('formal_launch_'+lane['arm']+'.json')),
            original_config_sha256=sha(root/('formal_'+lane['arm'])/'CONFIG.json')))
    save(ROOT/'plan.json',dict(schema='scorer-event-queue/1',status='registered',authorization_sha256=sha(ROOT/'authorization.json'),
        source_sha256={p.name:sha(p) for p in (ROOT/'source').glob('*.py')},existing_training=old,
        poll_seconds=60,desktop_report_minutes=120,automatic_retries=0,training_source_mutation=False,
        binary_controller=str(BINARY/'controller_source_03/launch_training.py'),
        later_admissions='aggressive_scratch then joint_e32; no launch until verified dedicated command registered'))
    print(json.dumps(dict(status='registered',plan_sha256=sha(ROOT/'plan.json'),order=ORDER)))
if __name__=='__main__':main()
