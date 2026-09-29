"""Attach a completed read-only requalification without replacing old pictures."""
import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


def attach(data, summary, rows):
    data=deepcopy(data)
    if data['id']!='report:1b87f48a-d200-4451-96aa-32b1bdf5fcaa':raise ValueError('wrong report')
    if summary['status']!='complete' or summary['contract']!='curriculum-original20-final30-light4/2-common-arc':raise ValueError('wrong contract or incomplete audit')
    if len(rows)!=summary['actual_positive_pixels_audited'] or len({r['id'] for r in rows})!=len(rows):raise ValueError('duplicate/missing audit rows')
    if any(not r['actual_pixels_checked'] for r in rows):raise ValueError('pixels not checked')
    lookup={r['id']:r for r in rows}
    if any(r['id'] not in lookup for r in data['queries']['curriculum_cases']['rows']):raise ValueError('displayed case lacks new audit')
    totals=[];types=[]
    for v in ('v17.5','v18'):
        r=[x for x in rows if x['version']==v];s=summary['versions'][v]
        counts=dict(positive=len(r),original20_excluded=sum(not x['original20_pass'] for x in r),
                    extra_final30_excluded=sum(x['original20_pass'] and not x['final30_pass'] for x in r),eligible=sum(x['eligible'] for x in r))
        if any(s[k]!=n for k,n in counts.items()):raise ValueError('summary does not reconcile')
        if counts['positive']!=sum(counts[k] for k in ('original20_excluded','extra_final30_excluded','eligible')):raise ValueError('exclusive counts do not sum')
        totals.append(dict(version=v,**counts))
        types.extend(dict(version=v,recipe=k,**a) for k,a in s['recipes'].items())
    source=dict(label='原始20%／最终30%逐像素复核',files=[dict(label='connectable30_audit_02/summary.json'),dict(label='connectable30_audit_02/rows.json')],
        notes=['固定原始共同弧分母，不额外增加单侧30%条件；旧图保留以便复查。','仅检查原审核集，不是全量课程数据或新生成结果。'],
        metricDefinitions=[dict(label='原始20%',definition='原始双侧公共弧的较小长度 / 原始较小像素面积片的完整外周长，至少20%。'),
                           dict(label='最终30%',definition='最终仍可连接的双侧公共弧较小长度 / 原始公共弧较小长度，至少30%；主腐蚀段不计，只允许最后每侧1–4px轻退化。')])
    protocol=[dict(item='原始公共接缝',v175='≥原始较小片完整周长20%',v18='同左',note='不足直接排除，不再保留“不裁但腐蚀”例外'),
        dict(item='最终可连接公共弧',v175='≥原始未裁切、未腐蚀公共弧30%',v18='同左',note='主腐蚀损伤段不计；轻退化可保留；不重定义分母、不跨缺口补长度'),
        dict(item='最后轻退化',v175='全轮廓70±2%，每侧1–4px',v18='同左',note='不恢复完全零损伤保护区；最终实际掩膜仍须可回读原始对应'),
        dict(item='原有裁切和深度',v175='其他已确认规则保留',v18='同左',note='自然轮廓、25–40%裁短、面积≤20%；不为凑数降低新门槛')]
    old={k:digest(v) for k,v in data['queries'].items() if not k.startswith('curriculum_contract30_')}
    for name,value in [('summary',totals),('cases',rows),('types',types),('protocol',protocol)]:
        data['queries']['curriculum_contract30_'+name]=dict(rows=value,source=source)
    data['curriculumQualification']=dict(contract=summary['contract'],complete=True,new_samples_generated=0,
        original_images_preserved=True,new_per_type10_complete=False,
        summary_rows=totals,rows_sha256=summary['rows_sha256'])
    data['buildStatus']='complete'
    data['generatedAt']=datetime.now(timezone.utc).isoformat()
    if old!={k:digest(v) for k,v in data['queries'].items() if not k.startswith('curriculum_contract30_')}:raise ValueError('original data or image changed')
    return data,dict(old_queries_sha256=old,original_queries_unchanged=True,displayed_cases_audited=len(data['queries']['curriculum_cases']['rows']),all_positive_audited=len(rows))


def main():
    p=argparse.ArgumentParser();p.add_argument('--audit',type=Path,required=True);p.add_argument('--app',type=Path,required=True);a=p.parse_args()
    summary=read(a.audit/'summary.json')
    if sha(a.audit/'rows.json')!=summary['rows_sha256']:raise ValueError('audit rows SHA mismatch')
    target=a.app/'src/data.json';backup=a.audit/'previous_review_snapshot.json'
    if backup.exists():raise ValueError('snapshot update already applied; verify instead of repeat')
    data,receipt=attach(read(target),summary,read(a.audit/'rows.json'))
    shutil.copy2(target,backup)
    target.write_text(json.dumps(data,ensure_ascii=False,separators=(',',':'))+'\n')
    receipt.update(snapshot_sha256=sha(target),audit_summary_sha256=sha(a.audit/'summary.json'))
    (a.audit/'report_binding.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(receipt))


if __name__=='__main__':main()
