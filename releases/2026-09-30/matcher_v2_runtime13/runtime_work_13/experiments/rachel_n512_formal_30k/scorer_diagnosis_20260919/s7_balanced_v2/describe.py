"""Data-only receipt for a completed S7-B v2 export; never trains a model."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import numpy as np
from ..s7_compound_v1.materialize import read, save_json


def quantiles(values):
    values=np.asarray([v for v in values if v is not None],float)
    if not len(values):
        return dict(n=0,mean=None,median=None,p10=None,p90=None)
    return dict(n=len(values),mean=float(values.mean()),median=float(np.median(values)),
                p10=float(np.quantile(values,.1)),p90=float(np.quantile(values,.9)))


def describe(root):
    root=Path(root)
    status,checked,summary,protocol=[read(root/f) for f in
        ('status.json','validation.json','summary.json','protocol.json')]
    if status['status']!='complete' or checked['status']!='passed' or summary['sample_count']!=24000:
        raise ValueError('full generated and validated data required')
    manifest=read(root/'train_s7b_24k.json')
    anchor_groups=defaultdict(list)
    for e in manifest['entries']:
        if e.get('anchor_group_id'):
            anchor_groups[e['anchor_group_id']].append(dict(
                pair_id=e['pair_id'],source_pair_id=e['source_pair_id'],
                anchor_fragment_token=e['anchor_fragment_token'],negative_kind=e['negative_kind']))
    save_json(root/'negative_anchor_groups.json',dict(schema='negative-anchor-groups/1',
        original_anchor_identity=True,augmented_views_may_differ=True,groups=dict(anchor_groups)))
    rows=[json.loads(line) for line in (root/'pair_metrics.jsonl').open()]
    if len(rows)!=24000 or len({r['pair_id'] for r in rows})!=24000:
        raise ValueError('measured population mismatch')
    recipes=[]
    for recipe in protocol['recipe_percent']:
        classes={str(label):[r for r in rows if r['recipe']==recipe and r['label']==label]
                 for label in (True,False)}
        peaks={};notches={}
        for label,items in classes.items():
            peak=[];notch=[]
            for r in items:
                damage=[d for d in r['augmentation']['damage'].values() if d.get('applied')]
                if damage:peak.append(max(d['applied_max_depth_px'] for d in damage))
                notch.append(sum(d.get('notch_count',0) for d in damage))
            peaks[label]=quantiles(peak);notches[label]=quantiles(notch)
        recipes.append(dict(recipe=recipe,positive=len(classes['True']),negative=len(classes['False']),
            actual_damage_peak_px=peaks,notch_count=notches,
            positive_contact_length_px=quantiles(r['d20_length_px'] for r in classes['True'])))
    sources=defaultdict(list)
    for e in manifest['entries']:
        source=e['source_row'];a,b=(source['fragment_'+s] for s in 'ab')
        aa,ab=float(a['foreground_area']),float(b['foreground_area'])
        sources['positive' if e['label'] else e['negative_kind']].append(
            dict(mean_area=(aa+ab)/2,ratio=min(aa,ab)/max(aa,ab)))
    before={name:dict(pairs=len(items),mean_fragment_area_px=quantiles(r['mean_area'] for r in items),
                     area_ratio=quantiles(r['ratio'] for r in items)) for name,items in sources.items()}
    save_json(root/'recipe_metrics.json',recipes)
    save_json(root/'source_size_metrics.json',before)
    names={'clean':'无腐蚀','wave':'波状内缩','local_deep':'局部深腐蚀','gaps':'局部缺口',
           'wave_gaps':'波状内缩＋缺口','wave_local':'波状内缩＋局部深腐蚀','wave_local_gaps':'波状内缩＋局部深腐蚀＋缺口'}
    recipe_table=[f'| {names[r["recipe"]]} | {r["positive"]:,} | {r["negative"]:,} | {r["positive"]/12000:.0%} |'
                  for r in recipes]
    negative_table=[f'| {label} | {summary["negative_kind"][key]:,} | {summary["negative_kind"][key]/12000:.0%} |'
        for key,label in [('cross_gen','跨Gen'),('cross_parent_same_gen','同Gen跨parent'),
                          ('same_parent_nonadjacent','同parent非相邻')]]
    area_table=[]
    for label,key in [('正例','True'),('负例','False')]:
        metric=summary['metrics'][key]
        area_table.append(f'| {label} | {metric["mean_fragment_area_px"]["mean"]:.0f} | '
                          f'{metric["mean_fragment_area_px"]["median"]:.0f} | '
                          f'{metric["area_ratio"]["median"]:.3f} |')
    notes=[
        '24K为增强后Pair数量，并非24K独立原图；重复来源必须按原图lineage归组划分。',
        '无腐蚀30%只描述腐蚀轴，可包含合并、遗漏或Partial；幸存的源接缝未施加腐蚀。',
        'Partial独立占70%，既有Gen4/Gen5合并与Gen5遗漏来源保留，不算入35%组合腐蚀。',
        '镜像为两片共同翻转15%（水平/垂直各7.5%），已变换GT位移；以后加载不叠加旧镜像率。',
        '局部深腐蚀/缺口10–30px，波状宽段3–10px；组合取最大深度，不叠加成更深切除。',
        '正例损伤在TRAIN原接缝；负例损伤随机落在自身外轮廓；两类配方/强度配额相同。',
        '源对应继承、人工新切边不补成正匹配；不足4个可靠对应或碎片断裂时重采样，不回退干净充数。',
        '2000个锚碎片各有3个不同负伙伴，共6000负Pair；增强视图可不同，成组元数据不自动改变训练loss。',
        '跨Gen指Gen2/3/4/5大类不同；同Gen跨parent要求相同生成器目录且不同原图lineage。',
        '同parent负例由CSV非邻接和独立无接缝共同确认；未知邻接不直接当负例。',
        '尺寸匹配降低大小捷径风险，但同parent负例主要来自碎片数更多的Gen家族，不能宣称全部捷径已消除。',
        '训练来源来自冻结TRAIN，未读取CAL/SELECT或真实掩膜用于本轮采样；形态配方此前参考过敦煌开发分析。',
        '原S7、S7-C、运行中F/I和真实标注未修改；这批数据尚未用于新的模型训练。',
    ]
    profile=protocol.get('distribution_revision')
    if profile:
        notes[4]='局部深腐蚀/缺口10–30px；宽段波状范围及概率记录于protocol.distribution_revision，组合取最大深度，不相加。'
        notes[11]='训练掩膜仅来自TRAIN；面积/接缝聚合分布参考了敦煌292有效GT，因此敦煌不再是未参与配方开发的测试集。Turufan未用于拟合。'
        notes.append('成对同尺度变换同步更新坐标和GT；输出面积受画布适配与后续腐蚀影响，以全量实测为准。')
        notes.append('新版GT距离支持仅用于腐蚀施加位置；不据此填补匹配标签。人工新切边和未知位置仍不作正对应监督。')
        if profile.get('heterogeneous_strong'):
            notes[4]='宽段非均匀退蚀10–30px，含连续1–4px浅谷；局部深腐蚀/缺口10–30px。组合在波状深度上叠加并限制每侧30px，每个缺口须独立额外删除材料。'
            notes.append('长度配额以腐蚀前近接长度为准；腐蚀后全支持段法线投影另存latent目录，不以最终≤20/40px筛掉远段。该投影不作训练对应标签。')
    title=profile['name'] if profile else 'S7-B v2 数据配额重配'
    text='\n'.join(['# '+title,'',
        '**24,000对已生成并通过加载和配额检查；尚未用这批数据训练新模型。**','',
        '## 腐蚀配方（每类12K）','',
        '| 配方 | 正例 | 负例 | 每类比例 |','|---|---:|---:|---:|',*recipe_table,'',
        '合计无腐蚀30%、单一腐蚀35%、组合腐蚀35%。','',
        '## 负例来源','', '| 来源 | 数量 | 负例中比例 |','|---|---:|---:|',*negative_table,'',
        '其中一半负例属于2000个“三个负伙伴共享同一原锚碎片”的组。',
        '保留1200个原Gen5衍生负例；正例来源为9000个native、1800个合并大小悬殊、1200个Gen5分组/遗漏。','',
        '## 增强后尺寸','',
        '| 标签 | 平均碎片面积px² | Pair内平均面积的中位px² | 面积比中位 |','|---|---:|---:|---:|',*area_table,'',
        f'正例d20近接均长{summary["metrics"]["True"]["d20_length_px"]["mean"]:.1f}px；'
        '该值是GT下≤20px且法线相向的边界长度代理，不是人工语义接缝长度。',
        f'独立原Pair共{summary["unique_base_pairs"]:,}，其中负例{summary["unique_negative_base_pairs"]:,}；'
        f'正例原Pair最多复用{summary["max_positive_reuse"]}次。','',
        '## 使用及限制','',*[f'- {n}' for n in notes],'',
        '## 文件','',f'- 远端目录：`{root}`',
        '- `train_s7b_24k.json`兼容旧S7加载；`train.json`兼容v3。',
        '- `negative_anchor_groups.json`列出2000个原始锚碎片的三负伙伴分组，便于后续组采样。',
        '- `summary.json`、`recipe_metrics.json`、`source_size_metrics.json`为完整统计。',
        '- `protocol.json`、`validation.json`、`generator_snapshot/`保留配方和验证依据。',''])
    (root/'DATA_CARD.md').write_text(text)
    save_json(root/'inline_sources.json',dict(schemaVersion=1,items=[dict(
        id='data-mixture',title='S7-B v2实际生成配额',queries=[dict(id='mixture',
            source=dict(label='S7-B v2物化与加载记录',files=[dict(label=x) for x in
                ('summary.json','validation.json','recipe_metrics.json')],
                filters=['训练数据：正负各12,000对'],caveats=[notes[0],notes[-1]]),
            rows=[dict(recipe=names[r['recipe']],positive=r['positive'],negative=r['negative']) for r in recipes],
            columns=['recipe','positive','negative'],preview=dict(kind='aggregate',note='完整24K数据配额。'))]),
        dict(id='negative-sources',title='负例来源与多负伙伴',queries=[dict(id='negative',
            source=dict(label='新负例来源计划与已生成清单',files=[dict(label='summary.json'),dict(label='sources_v2.json')],
                        caveats=[notes[7],notes[10]]),
            rows=[dict(source=k,pairs=v) for k,v in summary['negative_kind'].items()],
            columns=['source','pairs'],preview=dict(kind='aggregate',note='全部12K负例。'))])]))
    print(json.dumps(dict(status='data_card_ready',pairs=summary['sample_count'],
                         root=str(root)),ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True)
    describe(p.parse_args().root)
