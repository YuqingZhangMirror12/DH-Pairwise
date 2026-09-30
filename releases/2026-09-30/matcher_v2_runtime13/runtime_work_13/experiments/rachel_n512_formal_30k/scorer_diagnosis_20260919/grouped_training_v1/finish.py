"""Finite collector. Wait for both authorized jobs; no new training work."""
import argparse
import time
from support import *

def run(root):
    root=Path(root);start=time.time()
    while True:
        states=[read(root/f'worker_{i}.json') if (root/f'worker_{i}.json').exists() else {} for i in range(2)]
        if any(s.get('status')=='failed' for s in states):
            save(root/'status.json',dict(status='needs_attention',workers=states));return
        if all(s.get('status')=='complete' for s in states):break
        if time.time()-start>12*3600:
            save(root/'status.json',dict(status='collector_timeout',workers=states));return
        time.sleep(30)
    rows={arm:read(root/'evaluation'/arm/'summary.json') for arm in ARMS}
    old=read(R/'bounded_real_calibration_v2_20260921/results.json')
    lines=['# 同锚点成组训练：GS0／GS1','',
        '固定S7 M12 Matcher、两层96维4 heads局部Cross-Attention、C16。每组1正＋3负，12K组＝48K Pair。',
        'GS0和GS1使用相同组、顺序、batch48、初始化及768K Pair训练曝光。GS1唯一增加0.5倍组内CE。',
        '两组BCE均按正／负各50%权重；组内softmax只用于训练，推理仍独立sigmoid，可全部拒绝。','',
        '| 模型 | SIMTEST Accuracy@0.3 | SIMTEST Recall@0.3 | SIMTEST F1@0.3 | SIMVAL组内Top1 | SIMVAL组内MRR |',
        '|---|---:|---:|---:|---:|---:|']
    def pc(x):return f'{100*x:.2f}%'
    for arm,item in rows.items():
        m=item['test_fixed03'];g=read(root/'training'/arm/'group_validation_016.json')
        lines.append(f"| {arm} | {pc(m['accuracy'])} | {pc(m['recall'])} | {pc(m['f1'])} | {pc(g['group_top1'])} | {g['group_mrr']:.4f} |")
    for split,title in (('real','敦煌'),('ood','Turufan')):
        lines+=['',f'## {title}','',
            '| 模型 | Accuracy@0.3 | Precision@0.3 | Recall@0.3 | F1@0.3 | CV阈值中位数 | 独立折F1 |',
            '|---|---:|---:|---:|---:|---:|---:|']
        for key in ('reference512_h4','old_matched_tokens','old_edge_multi',*ARMS):
            if key in rows:
                m=rows[key][split]['fixed03'];cv=rows[key][split]['bounded_cv']
            else:
                item=old['results'][split]['models'][key];m=item['fixed_03'];cv=item['cv']['bounded_max_f1']
            lines.append(f"| {key} | {pc(m['accuracy'])} | {pc(m['precision'])} | {pc(m['recall'])} | {pc(m['f1'])} | {cv['threshold_median']:.2f} | {pc(cv['pooled_out_of_fold']['f1'])} |")
    lines+=['','## 解释边界','',
        '- GS1对GS0才是组内竞争的受控效果；不能将对旧24K结果的差别全归因于组内loss。',
        '- 本轮使用原S7正例及同增强类型的跨来源干扰碎片；没有新生成图片，也没有增加同来源非邻接组。',
        '- 成组训练不会修复Matcher未找到真实接缝的问题；没有可用候选的正例未被删除或改成负例。',
        '- 真实域保留原分折、原构造负例；每次4折校准1折测试，阈值限定0.20–0.80。真实域不选epoch。',
        '- 真实数据已参与历史架构分析和人工筛选，本轮不是新盲测。Turufan仍无Layout GT。','']
    (root/'RESULTS.md').write_text('\n'.join(lines))
    save(root/'results.json',dict(status='complete',results=rows))
    save(root/'status.json',dict(status='complete',workers=states,summary=str(root/'RESULTS.md')))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default=str(ROOT));run(p.parse_args().root)
