"""Local presentation adapter; bound remote data sources stay immutable."""
import argparse,json,hashlib,shutil
from pathlib import Path
from datetime import datetime,timezone
import numpy as np

REVISION='curriculum-v17p5-v18/5-crop20-before-erosion-whole-light'


def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False).encode()).hexdigest()


def prepare(artifact,receipt,data,source_name):
    if data['id']!='report:1b87f48a-d200-4451-96aa-32b1bdf5fcaa':raise ValueError('wrong app')
    if artifact['protocol']['revision']!=REVISION:raise ValueError('wrong source revision')
    if receipt['status'] not in ('probe_complete','complete'):raise ValueError('completed actual generation required')
    complete=bool(artifact['complete'] and receipt['status']=='complete')
    rows=artifact['rows'];pop=artifact['population'];groups=artifact['groups']
    if len({r['id'] for r in rows})!=len(rows):raise ValueError('duplicate displayed identity')
    lookup={r['id']:r for r in rows}
    for row in rows:
        s=row['detail']['spec'];d=row['detail'];floor=d['crop_only_seam_floor']
        if row['label']!='正例' or row['audit']['status']!='passed':raise ValueError('positive audited examples only')
        if s['light']!=[1.,4.] or s['light_scope']!='whole_postprimary_contour':raise ValueError('wrong final light')
        if s['pristine_protection_enabled'] or s['pristine_min_fraction']!=0:raise ValueError('hidden pristine rule')
        if floor!=row['audit']['crop_only_seam_floor'] or not floor['passed']:raise ValueError('crop audit mismatch')
        if floor['erosion_and_light_subject_to_this_gate']:raise ValueError('crop gate incorrectly extends to corrosion')
        if floor['originally_short']:
            if floor['changed'] or d['trim']['applied']:raise ValueError('short source was cropped')
        elif floor['after']['common_over_smaller_perimeter']<.20:raise ValueError('below20 after structural cut')
        if d['trim']['applied'] and not .25<=1-d['trim']['retained_fraction']<=.40:raise ValueError('crop target range')
        for light in d['background'].values():
            if not light['whole_postprimary_contour'] or light['contact_protection_enabled'] or light['applied_max_depth_px']>4.000001:
                raise ValueError('wrong actual light scope/depth')
    for g in groups:
        if len(set(g['ids']))!=len(g['ids']) or any(i not in lookup for i in g['ids']):raise ValueError('invalid group references')
        if complete and (len(g['ids'])!=10 or g['count']!=10):raise ValueError('each corrosion needs10 positive examples')
    progress=[];summary=[]
    for version in ('v17.5','v18'):
        vp=[r for r in pop if r['version']==version];pos=[r for r in vp if r['label']]
        vg=[g for g in groups if g['version']==version]
        shown=[r for r in rows if r['version']==version]
        if len(vp)!=receipt['versions'][version]['pairs']:raise ValueError('manifest and receipt do not reconcile')
        if complete and len(vg)!=12:raise ValueError('missing primary/compound/control category')
        progress.append(dict(version=version,status='每类10个正样本已完成' if complete else '像素预检通过；每类10个正样本准备中',
            audited_pairs=len(vp),audited_positives=len(pos),shown_positives=len(shown),groups=len(vg),
            slots=sum(g['count'] for g in vg),per_type=10,
            applied_crops=sum(r['crop_applied'] for r in pos),skipped_crops=sum(not r['crop_applied'] for r in pos)))
        metrics=[('裁前：公共接缝／较小片完整周长','%',100,[r['crop_floor']['before']['common_over_smaller_perimeter'] for r in pos]),
            ('裁后：公共接缝／较小片完整周长（不含后续腐蚀）','%',100,[r['crop_floor']['after']['common_over_smaller_perimeter'] for r in pos]),
            ('实际执行的新增裁短比例','%',100,[r['trim_fraction'] for r in pos if r['crop_applied']]),
            ('主损伤双侧间隙峰值','px',1,[r['gap_peak'] for r in pos if r['gap_peak'] is not None]),
            ('继承GT点对数','条',1,[r['inherited'] for r in pos]),
            ('展示样本轻退化实测最大深度','px',1,[b['applied_max_depth_px'] for r in shown for b in r['detail']['background'].values()])]
        for name,unit,scale,values in metrics:
            if not values:continue
            quant=np.quantile(np.asarray(values)*scale,[0,.1,.5,.9,1])
            summary.append(dict(version=version,metric=name,unit=unit,n=len(values),
                **dict(zip(('min','p10','p50','p90','max'),map(float,quant)))))
    protocol=[
        dict(item='裁切阶段接缝下限',v175='公共接缝≥较小片总周长20%',v18='同左',note='只在裁后、主腐蚀和轻退化之前检查；较小片按裁后像素面积决定'),
        dict(item='原始比例已经不足20%',v175='跳过新增结构裁切',v18='同左',note='裁前像素保留；仍可施加规定腐蚀及最后轻退化，不补造材料'),
        dict(item='新增端部裁短／面积',v175='目标25–40%；被裁片面积损失≤20%',v18='同左',note='自然TRAIN轮廓、一端或两端；找不到同时可行方案则跳过，不强切'),
        dict(item='裁切请求侧',v175='70%较小片／30%较大片',v18='同左',note='预先随机分配；实际执行和跳过分列，不能拿请求配额冒充实际切割配额'),
        dict(item='连续弱腐蚀峰值',v175='4–8px',v18='5–8px',note='是主接缝损伤配方，不受20%裁切下限约束'),
        dict(item='起伏／突变／渐进峰值',v175='7–15px',v18='10–15px',note='主层覆盖剩余公共弧35–50%／40–60%；20%不用于筛腐蚀'),
        dict(item='独立缺口',v175='1–4处，各5–15px',v18='同左',note='组合弱腐蚀仍单侧主＋弱封顶15px'),
        dict(item='最后轻微退化',v175='当前全轮廓70±2%；峰值1–4px',v18='同左',note='包含公共接缝、已有腐蚀边缘与新裁边；没有保护弧，clean对照不施加'),
        dict(item='双侧主损伤gap峰值',v175='5–35px',v18='同左',note='最终固定GT下实际投影；渐变肩部可低于5px'),
        dict(item='未损伤接缝与监督',v175='不设25%未损伤配额；至少4个继承GT对应',v18='同左',note='不将最终完好长度与裁切前后的公共弧混为一谈')]
    provenance=dict(label='裁切20%修订实际掩膜、结构裁切门控及逐行像素/监督审计',
        tables=[source_name,artifact['source_root']+'/pipeline_complete.json' if complete else artifact['source_root']+'/probe_complete.json',
            artifact['source_root']+'/v17.5/pixel_audit.json',artifact['source_root']+'/v18/pixel_audit.json'],
        executedAt=datetime.fromtimestamp(receipt['updated_unix'],timezone.utc).isoformat(),
        notes=['每类10例专指正样本；负例成对生成审计，不占展示名额。','20%只适用于裁切阶段，原始<20%的例外保留像素并明确标记。','全轮廓轻退化最后执行，峰值1–4px；不修改历史v14/v17或GPU训练。'],
        metricDefinitions=[dict(label='公共接缝／较小片周长',definition='原始GT双侧共同存活弧长的较小值，除以裁切阶段较小像素面积片的完整外周长；新裁边只进分母，不进公共弧。',componentIds=['curriculum-distribution','curriculum-gallery']),
            dict(label='每类10个正样本',definition='每版每个腐蚀/组合/对照分类10个不同正例ID；跨分类复用不增加唯一样本数。',componentIds=['curriculum-progress','curriculum-gallery'])])
    old={k:digest(v) for k,v in data['queries'].items() if not k.startswith('curriculum_')}
    for key,values in [('curriculum_cases',rows),('curriculum_groups',groups),('curriculum_summary',summary),('curriculum_protocol',protocol),('curriculum_progress',progress)]:
        data['queries'][key]=dict(rows=values,source=provenance)
    data.update(generatedAt=datetime.now(timezone.utc).isoformat(),buildStatus='complete' if complete else 'updating',
        curriculumReview=dict(complete=complete,record_complete=True,revision=REVISION,
            source_binding_sha256=artifact['protocol']['source_binding_sha256'],
            full_generation_authorized=True,full_generation_started=False,scale_confirmation_pending=True,
            unique_display_pairs=len(rows),display_slots=artifact['display_slots']))
    data['report']['asOf']=provenance['executedAt'][:10]
    if old!={k:digest(v) for k,v in data['queries'].items() if not k.startswith('curriculum_')}:raise ValueError('archive data changed')
    return data,dict(revision=REVISION,complete=complete,unique_positive_examples=len(rows),groups=len(groups),
        display_slots=artifact['display_slots'],archive_queries_sha256=old,archive_queries_preserved=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--rendered',required=True);p.add_argument('--receipt',required=True);p.add_argument('--app',required=True);a=p.parse_args()
    source=Path(a.rendered);target=Path(a.app)/'src/data.json'
    artifact=json.loads(source.read_text());receipt=json.loads(Path(a.receipt).read_text());old=json.loads(target.read_text())
    data,verification=prepare(artifact,receipt,old,str(source))
    backup=source.parent/'previous_review_snapshot.json'
    if not backup.exists():shutil.copy2(target,backup)
    target.write_text(json.dumps(data,ensure_ascii=False,separators=(',',':'))+'\n')
    verification.update(rendered_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),snapshot_sha256=hashlib.sha256(target.read_bytes()).hexdigest())
    (source.parent/'snapshot_receipt.json').write_text(json.dumps(verification,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(verification,ensure_ascii=False))

if __name__=='__main__':main()
