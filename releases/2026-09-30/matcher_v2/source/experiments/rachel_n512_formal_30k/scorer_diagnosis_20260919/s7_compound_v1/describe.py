"""Generate a data card from a completed, loader-validated S7-C export.

No model inference, real-label fitting, or sample modification is performed.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import numpy as np
from .materialize import read, save_json


NAMES = {
    'clean_anchor':'干净锚点', 'seam_local_deep':'局部深腐蚀',
    'partial_curve':'曲线Partial', 'wave_gaps':'波状内缩＋深缺口',
    'partial_wave':'Partial＋波状内缩',
    'partial_wave_gaps':'Partial＋波状内缩＋深缺口',
}


def quantiles(values):
    v=np.array([x for x in values if x is not None],float)
    if not len(v):return dict(n=0,mean=None,median=None,p10=None,p90=None)
    return dict(n=len(v),mean=float(v.mean()),median=float(np.median(v)),
        p10=float(np.quantile(v,.1)),p90=float(np.quantile(v,.9)))


def describe(root, reference, real_pairs=None, exclusions=None):
    root=Path(root)
    status=read(root/'status.json');validation=read(root/'validation.json')
    if status['status']!='complete' or validation['status']!='passed':
        raise ValueError('only completed formal data may be described as ready')
    if (status['sample_count'],status['positive_count'],status['negative_count'])!=(24000,12000,12000):
        raise ValueError('formal sample budget not fulfilled')
    summary=read(root/'summary.json');protocol=read(root/'protocol.json')
    reference=read(reference)['cohorts']
    revision_note='此处敦煌参照仍为配方设计时295对的旧口径。'
    if bool(real_pairs) != bool(exclusions):
        raise ValueError('real_pairs and exclusions must be provided together')
    if real_pairs:
        original=[r for r in read(real_pairs) if r['dataset']=='dunhuang_cv' and r['label']]
        excluded=read(exclusions)
        if excluded['schema']!='user-confirmed-gt-exclusions-v1':
            raise ValueError('explicit user exclusions required')
        index={r['pair_id']:r for r in original}
        ids=set()
        for e in excluded['records']:
            if (e['dataset']!='敦煌' or e['pair_id'] in ids or e['pair_id'] not in index
                    or index[e['pair_id']]['case_name'].split(' · ')[0]!=e['case_folder']):
                raise ValueError('incorrect GT exclusion identity')
            ids.add(e['pair_id'])
        retained=[r for r in original if r['pair_id'] not in ids]
        if len(original)!=295 or len(retained)!=292:
            raise ValueError('unexpected revised Dunhuang reference population')
        keys=('d20_length_px','d20_longest_px','d40_gap_mean_px',
              'd10_breaks_mean','mean_fragment_area_px')
        reference['Dunhuang retained']=dict(pair_count=len(retained),
            metrics={key:quantiles(r.get(key) for r in retained) for key in keys})
        missing=sum(r.get('d40_gap_mean_px') is None for r in retained)
        revision_note=(f'配方设计时使用295对；本次展示按用户新确认的3个错误GT排除后重计为292对，'
                       f'未反向调整正在生成的数据配方。间隙缺失{missing}对，不填0。')
    rows=[]
    with (root/'pair_metrics.jsonl').open() as f:
        for line in f:rows.append(json.loads(line))
    if len(rows)!=24000 or len({r['pair_id'] for r in rows})!=24000:
        raise ValueError('missing or duplicated measured records')
    grouped=defaultdict(list)
    for r in rows:grouped[(r['recipe'],r['label'])].append(r)
    recipes=[]
    for recipe,name in NAMES.items():
        positive,negative=grouped[(recipe,True)],grouped[(recipe,False)]
        if len(positive)!=len(negative):raise ValueError('class-conditional recipe imbalance')
        record=dict(recipe=recipe,name=name,positive=len(positive),negative=len(negative),
            fraction_each_class=len(positive)/12000)
        for label,rr in [('positive',positive),('negative',negative)]:
            peaks=[];notches=[];retentions=[]
            for r in rr:
                augmentation=r['augmentation']
                if augmentation.get('partial'):
                    retentions.append(augmentation['partial']['source_seam_retention'])
                damage=augmentation.get('damage',{})
                details=[d for d in damage.values() if d.get('applied')]
                if details:peaks.append(max(d['applied_max_depth_px'] for d in details))
                notches.append(sum(d.get('notch_count',0) for d in details))
            record[label+'_damage_depth_px']=quantiles(peaks)
            record[label+'_notch_count']=quantiles(notches)
            # The retained source seam is defined on the positive member; the
            # negative shares the recorded crop/material recipe, not a GT seam.
            if label=='positive':record['source_seam_retention']=quantiles(retentions)
        for key in ('d20_length_px','d20_longest_px','d40_gap_mean_px','d10_breaks_mean',
                    'mean_fragment_area_px','area_ratio','matched_tokens'):
            record[key]=quantiles(r.get(key) for r in positive)
        recipes.append(record)
    save_json(root/'recipe_metrics.json',recipes)
    comparison=[]
    for key,name in [('S7 positive','原S7'),('S7-H positive','S7-H困难子集'),
                     ('NEW','新S7-C'),('Dunhuang retained','敦煌人工保留集')]:
        m=summary['positive_metrics'] if key=='NEW' else reference[key]['metrics']
        n=12000 if key=='NEW' else reference[key]['pair_count']
        comparison.append(dict(dataset=name,positive_pairs=n,
            mean_contact_length_px=m['d20_length_px']['mean'],
            median_contact_length_px=m['d20_length_px']['median'],
            mean_longest_contact_px=m['d20_longest_px']['mean'],
            median_pair_mean_gap_px=m['d40_gap_mean_px']['median'],
            mean_breaks=m['d10_breaks_mean']['mean'],
            mean_fragment_area_px2=m['mean_fragment_area_px']['mean']))
    notes=[
        '近接长度是GT位置下≤20px且法线相向的两侧边界长度平均，不是人工语义接缝长度。',
        '间隙取≤40px近接带内每对平均距离的中位数，不等于原始材料腐蚀深度。',
        '断开次数取10px近接带内8–100px中断的两侧平均，不把不同采样点当独立样本。',
        '敦煌统计已用于配方开发，因此不能把该集合再称为完全未见的最终测试。',
        'Turufan人工审核已完成，但本数据配方未使用其摆放接缝统计，也未将模型摆放冒充GT。',
        '新S7-C尚未用于模型训练；形态更接近不等于性能已提高。',
        '新增强版本可重复使用同一个原始TRAIN对，24K并非24K独立来源。',
        revision_note,
    ]
    previous=[]
    for key,name in [('S7 reference_e1 applied','旧轻腐蚀（实际生效）'),
                     ('S7 wave applied','旧10–30px波状内缩'),
                     ('S7 local applied','旧局部深腐蚀'),
                     ('S7 seam_gaps applied','旧接缝缺口'),
                     ('S7 partial_curve applied','旧曲线Partial'),
                     ('Dunhuang retained','敦煌人工保留集')]:
        m=reference[key]['metrics']
        previous.append(dict(recipe=name,count=reference[key]['pair_count'],
            mean_contact_length_px=m['d20_length_px']['mean'],
            median_pair_mean_gap_px=m['d40_gap_mean_px']['median'],
            mean_breaks=m['d10_breaks_mean']['mean']))
    save_json(root/'comparison.json',dict(rows=comparison,previous_recipe_comparison=previous,notes=notes))
    counts=[f'| {r["name"]} | {r["positive"]:,} | {r["negative"]:,} | {r["fraction_each_class"]:.0%} |' for r in recipes]
    table=[f'| {r["dataset"]} | {r["mean_contact_length_px"]:.1f} | {r["median_contact_length_px"]:.1f} | {r["median_pair_mean_gap_px"]:.2f} | {r["mean_breaks"]:.2f} | {r["mean_fragment_area_px2"]:.0f} |' for r in comparison]
    oldtable=[f'| {r["recipe"]} | {r["count"]:,} | {r["mean_contact_length_px"]:.1f} | {r["median_pair_mean_gap_px"]:.2f} | {r["mean_breaks"]:.2f} |' for r in previous]
    text='\n'.join([
        '# S7-C v1 组合腐蚀训练数据', '',
        '**状态：24,000对已物化并通过加载检查；新模型训练尚未启动。**', '',
        '## 已生效配方', '',
        '| 类型 | 正例 | 负例 | 每类占比 |','|---|---:|---:|---:|',*counts,'',
        f'实际Partial比例{summary["actual_partial_fraction"]:.0%}，两种以上组合{summary["actual_compound_fraction"]:.0%}。干净10%，不单列2/4px轻腐蚀。',
        '宽段波状内缩3–10px；局部深峰10–30px；多个缺口宽15–50px、1–5处。双侧宽段按总量分配，组合深度取max、不简单叠加。',
        '正例损伤限原TRAIN监督接缝，负例在自身整圈随机位置施加；两类共用配方与强度参数，共同通过再收录。',
        '曲线Partial来自TRAIN轮廓donor，斜向裁切并保留25%–65%源接缝；正例新裁边/深缺口不作为正对应。',
        '30%共同镜像，水平/垂直各15%；同时变换两片坐标及GT位移。未来采用此固定数据时，不机械重复增加同一镜像配额。', '',
        '## 全量实测分布（正例）','',
        '| 数据 | 均长px | 长度中位px | 每对平均gap中位px | 平均断开 | 平均碎片面积px² |',
        '|---|---:|---:|---:|---:|---:|',*table,'',
        '本次长度配额是32–256px占25%、256–512px占50%、512–800px占25%。800px仅是本困难数据的长尾控制，不是模型预测限制。', '',
        '## 原单项增强为何还不够', '',
        '| 原配方/真实集 | 正例数 | 均长px | 每对平均gap中位px | 平均断开 |',
        '|---|---:|---:|---:|---:|',*oldtable,'',
        '旧轻腐蚀对本次间隙分布的补充不足，不等于已经证明它对训练完全无效。旧10–30px宽段内缩的典型间隙反而过大；它测得近接长度短，部分是20px测量阈值排除了过宽的间隙，不能当成真正缩短了源接缝。',
        '旧局部深腐蚀/缺口仍留下过长的连续边界；曲线Partial的长度最接近敦煌，但单独施加仍偏贴合、断开少。因此新版本加强Partial并与波状间隙、深缺口组合，不把每一处都统一内缩10–30px。', '',
        '## 数据与监督', '',
        f'- 独立原始Pair数：{summary["unique_base_pair_count"]:,}；单个原始Pair最多复用{summary["max_base_pair_reuse"]}个增强版本。',
        '- 保留原S7的普通、大小悬殊、Gen4/Gen5合并及Gen5分组、原负例来源类别；没有加入新的真实标签。',
        '- 所有片保持单连通，只删除材料；不补洞、不保留最大分量来掩盖断裂，不静默回退干净样本充数。',
        '- 原GT位移不因腐蚀而变；对应由原TRAIN源关系继承，至少4个互反正对应。负例不制造对应。',
        '- 兼容旧S7 materialized加载器和v3 train.json；800px框架、512点上限和原窗口设定不变。',
        '- 历史兼容标记对损伤例禁用精确pose回归，但保留translation_valid与GT，候选质量监督仍可使用；不擅自改损失。',
        f'- 生成过程中每个archive已回读检查；完成后另以两个实际加载器覆盖{validation["actual_loader_and_metadata_checked_count"]}种配方/标签/镜像组合。', '',
        '## 文件', '',
        f'- 远端目录：`{root}`',
        '- `train_s7c_24k.json`：旧S7训练接口；`train.json`：v3训练接口。',
        '- `samples/`：掩膜、坐标、标签；`targets/`：可靠接缝连通/断开辅助监督。',
        '- `summary.json`、`recipe_metrics.json`、`pair_metrics.jsonl`：实际分布和逐例测量。',
        '- `protocol.json`、`validation.json`、`generator_snapshot/`：配置、检查记录和生成代码。', '',
        '## 解释限制', '',*[f'- {s}' for s in notes], '',
        '原S7、原CAL/SELECT、F/I训练及人工审核HTML均未修改。', ''])
    (root/'DATA_CARD.md').write_text(text)
    source=dict(label='S7-C物化记录与同口径分布统计',
        files=[dict(label=x) for x in ('S7-C summary.json','recipe_metrics.json','comparison.json',
                                      'validation.json','reference_distribution_summary.json')],
        metricDefinitions=[dict(label='形态统计',definition=notes[0]),
                           dict(label='间隙',definition=notes[1])],
        filters=['接缝指标仅统计正例','新数据为TRAIN来源24K物化版本'],caveats=notes[3:])
    save_json(root/'inline_sources.json',dict(schemaVersion=1,items=[
        dict(id='actual-mixture',title='实际完成的正负增强配额',queries=[dict(id='mixture',source=source,
            rows=[{k:r[k] for k in ('name','positive','negative','fraction_each_class')} for r in recipes],
            columns=['name','positive','negative','fraction_each_class'],preview=dict(kind='aggregate',note='全量24K统计。'))]),
        dict(id='shape-comparison',title='新旧仿真与敦煌的实测形态',queries=[dict(id='comparison',source=source,
            rows=comparison,columns=list(comparison[0]),preview=dict(kind='aggregate',note='正例全量分组汇总。'))]),
        dict(id='previous-recipes',title='旧单项增强与敦煌的差距',queries=[dict(id='previous',source=source,
            rows=previous,columns=list(previous[0]),preview=dict(kind='aggregate',note='旧增强只统计实际生效的正例；敦煌为人工保留集。'))])]))
    print(json.dumps(dict(status='data_card_ready',root=str(root),comparison=comparison),ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True);p.add_argument('--reference',required=True)
    p.add_argument('--real-pairs');p.add_argument('--exclusions')
    a=p.parse_args();describe(a.root,a.reference,a.real_pairs,a.exclusions)
