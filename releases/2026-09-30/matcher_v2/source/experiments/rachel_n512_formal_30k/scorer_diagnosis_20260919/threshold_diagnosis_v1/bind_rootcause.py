"""Append the root-cause follow-up, never replace existing results or reviews."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from .analyze import read, sha


def digest(x):
    return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def main(complete=False):
    out=Path('artifacts/threshold_rootcause_20260927')
    path=Path('reports/seam_v3_real_analysis_20260923/src/data.json')
    annotations=Path('artifacts/human_layout_review_20260923/annotations.json')
    before=sha(annotations);data=read(path);r=read(out/'rootcause.json');distribution=read(out/'distributions.json')
    old={k:digest(v) for k,v in data['queries'].items() if not k.startswith('td_rc_')}
    assert r['status']=='complete'
    summaries=[]
    for row in r['input_distributions']:
        summaries.append(dict(dataset=row['dataset'],group=row['group'],samples=row['samples'],
            **{k:v['median'] for k,v in row['measures'].items()},distribution=row['measures']))
    groups=[dict(group=k,cases=v['cases'],**{key:stat['median'] for key,stat in v['statistics'].items()},
                 pair_ids=v['pair_ids']) for k,v in r['correct_rejected_subgroups'].items()]
    provenance=dict(label='固定E8评分解析＋相同误报预算对照＋已保存Matcher Q/簇成员拆解',
        files=[str(out/'rootcause.json'),str(out/'joined_candidate_counts.json'),str(out/'dun_operating_decisions.json')],
        sha256=r['sources'],caveats=r['caveats'],
        evidenceFlow=[dict(title='评分干预',detail='固定全部候选、最终位姿、Q与支持概率；只修改指定读出项。4402对、20006簇，未训练或再次推理。'),
            dict(title='相同误报预算',detail='从已见数据的负例分数确定截点，不使用正例优化截点；事后开发诊断，非独立测试收益，不写回CAL/模型配置。'),
            dict(title='点数与Q',detail='已有2905对phase1缓存；真实簇与E8输出先核对质量/位姿，不一致9簇不并入逐例归因；C10实际并集成员再次一致核验。')])
    queries=dict(td_rc_summary=[r],td_rc_operating=r['operating_points'],td_rc_inputs=summaries,
                 td_rc_rejected=groups,td_rc_examples=r['c10_examples'],
                 td_rc_distribution=[distribution],td_rc_bins=distribution['bins'],
                 td_rc_quantiles=distribution['quantiles'],td_rc_coverage=distribution['coverage'],
                 td_rc_q_strata=distribution['q_within_count_strata'])
    provenance['files'].append(str(out/'distributions.json'))
    provenance['sha256'][str(out/'distributions.json')]=sha(out/'distributions.json')
    for key,rows in queries.items():data['queries'][key]=dict(rows=rows,source=provenance)
    data['buildStatus']='complete' if complete else 'updating'
    data['generatedAt']=datetime.now(timezone.utc).isoformat()
    assert all(digest(data['queries'][k])==v for k,v in old.items())
    path.write_text(json.dumps(data,ensure_ascii=False,allow_nan=False,separators=(',',':'))+'\n')
    assert sha(annotations)==before
    receipt=dict(status='passed',old_queries_unchanged=list(old),annotation_sha256=before,
        added_queries=list(queries),snapshot_sha256=sha(path),rootcause_sha256=sha(out/'rootcause.json'))
    (out/'report_binding.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--complete',action='store_true');args=p.parse_args();main(args.complete)
