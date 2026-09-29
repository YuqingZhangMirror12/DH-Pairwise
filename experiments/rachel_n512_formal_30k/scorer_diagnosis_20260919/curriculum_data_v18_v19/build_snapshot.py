"""Add source-backed review rows without overwriting archived v17 evidence."""
import argparse,hashlib,json,shutil
from datetime import datetime,timezone
from pathlib import Path
import numpy as np

def main():
 p=argparse.ArgumentParser();p.add_argument('--rendered',required=True);p.add_argument('--app',required=True)
 p.add_argument('--handoff',action='store_true');a=p.parse_args()
 source=Path(a.rendered);artifact=json.loads(source.read_text());app=Path(a.app);path=app/'src/data.json'
 data=json.loads(path.read_text());expected='report:1b87f48a-d200-4451-96aa-32b1bdf5fcaa'
 if data['id']!=expected:raise ValueError('wrong review app identity')
 if artifact['protocol']['revision']!='curriculum-v17p5-v18-review/3-pristine25':raise ValueError('superseded unprotected preview')
 backup=source.parent/'v17_original_snapshot.json'
 if not backup.exists():shutil.copy2(path,backup)
 rows=artifact['rows'];pop=artifact['population'];groups=artifact['groups']
 complete=artifact['complete']
 if complete and any(g['count']!=10 for g in groups):raise ValueError('incomplete per-type display')
 if any(not .25<=1-r['detail']['trim']['retained_fraction']<=.40 for r in rows if r['label']=='正例'):
  raise ValueError('actual25–40% range required')
 if any(r['pristine_fraction']<.25 for r in pop if r['label']):raise ValueError('strict25% pristine original arc required')
 keys=[{r['source_base_key'] for r in pop if r['version']==v} for v in ('v17.5','v18')]
 if not keys[0].isdisjoint(keys[1]):raise ValueError('cross-version base identity overlap')
 summaries=[];progress=[]
 for v in ('v17.5','v18'):
  population=[r for r in pop if r['version']==v];local_groups=[g for g in groups if g['version']==v]
  progress.append(dict(version=v,status='各类10例完成，待人工审核' if complete else '已审计预检，完整分组准备中',
    audited_pairs=len(population),groups=len(local_groups),slots=sum(g['count'] for g in local_groups),per_type=10))
  metrics=[('新增裁短比例','%',100,[r['trim_fraction'] for r in population if r['label']]),
    ('双侧严格未损伤／原始裁前公共接缝','%',100,[r['pristine_fraction'] for r in population if r['label']]),
    ('新增裁切面积损失','%',100,[r['cut_area_loss'] for r in population]),
    ('主损伤双侧gap峰值','px',1,[r['gap_peak'] for r in population if r['gap_peak'] is not None]),
    ('正例继承对应数','条',1,[r['inherited'] for r in population if r['label']])]
  for name,unit,scale,values in metrics:
   if not values:continue
   q=np.quantile(np.asarray(values)*scale,[0,.1,.5,.9,1])
   summaries.append(dict(version=v,metric=name,unit=unit,n=len(values),**dict(zip(('min','p10','p50','p90','max'),map(float,q)))))
 protocol=[
  dict(item='连续弱腐蚀峰值',v175='4–8px',v18='5–8px',note='指定接缝区域连续渐进；肩部允许更浅'),
  dict(item='起伏／突变／渐进峰值',v175='7–15px',v18='10–15px',note='只在已选一侧施加主层'),
  dict(item='独立缺口',v175='1–4处，各5–15px',v18='1–4处，各5–15px',note='逐处实际像素生效；主＋弱合计封顶15px'),
  dict(item='主层覆盖公共接缝',v175='目标35–50%',v18='目标40–60%',note='裁短后保留原始公共弧为分母；场与实际移除另列'),
  dict(item='新增端部裁短',v175='25–40%',v18='25–40%',note='同一裁前参考；不是逐版累加'),
  dict(item='裁切侧／面积上限',v175='70%小片／30%大片；≤20%',v18='同左',note='两类侧别均按自身裁前像素面积计算'),
  dict(item='双侧主损伤gap峰值',v175='5–35px',v18='5–35px',note='不变GT下双侧投影；不是整缝最小间隙'),
  dict(item='双侧严格未损伤共同弧',v175='至少原始裁前接缝25%',v18='至少原始裁前接缝25%',note='两侧端点及各自3×3邻域像素不变；轻退化也避开'),
  dict(item='其余轮廓轻退化',v175='可退化轮廓70%，1–3px',v18='同左',note='排除已损伤、新裁边及保护弧段；全轮廓实际占比另列'),
  dict(item='分散证据分组',v175='2–4段',v18='2–4段',note='两侧每段≥8px且保留≥1条GT对应；不等同严格未损伤弧'),
  dict(item='扩量与课程训练',v175='未授权／未开始',v18='未授权／未开始',note='人工审核后再决定；v19已弃用，仅保留历史')]
 provenance=dict(label='远端CPU实测掩膜、固定源伙伴gap、逐像素与实际监督审计',
   tables=[str(source),artifact['source_root']+'/protocol.json',artifact['source_root']+'/v17.5/pixel_audit.json',artifact['source_root']+'/v18/pixel_audit.json'],
   notes=['只有审计通过的数据进入页面；按类型抽样，不是训练配额分布。',
          'v17.5/v18底样身份互斥；新增裁短目标与实测都在25–40%；双侧严格未损伤弧至少原始裁前25%。',
          '未扩量6K／3K；旧v14/v17与GPU训练未更改。'],
   sha256=hashlib.sha256(source.read_bytes()).hexdigest())
 for key,values in [('curriculum_cases',rows),('curriculum_groups',groups),('curriculum_summary',summaries),('curriculum_protocol',protocol),('curriculum_progress',progress)]:
  data['queries'][key]=dict(rows=values,source=provenance)
 data.update(title='v17.5／v18：保留25%严丝合缝弧段的课程数据审核',generatedAt=datetime.now(timezone.utc).isoformat(),
   buildStatus='complete' if complete else ('paused' if a.handoff else 'updating'),
   curriculumReview=dict(complete=complete,revision=artifact['protocol']['revision'],
     source_binding_sha256=artifact['protocol']['source_binding_sha256'],
     rendered_sha256=provenance['sha256'],full_generation_authorized=False,
     unique_display_pairs=len(rows),display_slots=artifact['display_slots']))
 data['report']['asOf']='2026-09-28'
 path.write_text(json.dumps(data,ensure_ascii=False,separators=(',',':'))+'\n')
 receipt=dict(id=data['id'],complete=complete,progress=progress,source_sha256=provenance['sha256'],
    snapshot_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),archive_queries_preserved=True)
 (source.parent/'snapshot_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps(receipt,ensure_ascii=False))
if __name__=='__main__':main()
