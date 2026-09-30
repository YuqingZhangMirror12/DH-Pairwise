"""Local presentation adapter for source05 evidence; never alters generation source."""
import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

REVISION = 'curriculum-v17p5-v18/4-unprotected-light1to4'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def prepare(artifact, probe, data, source_name):
    if data['id'] != 'report:1b87f48a-d200-4451-96aa-32b1bdf5fcaa':
        raise ValueError('wrong review app identity')
    if artifact['protocol']['revision'] != REVISION or probe['status'] != 'probe_complete':
        raise ValueError('completed source05 probe required')
    rows, population, groups = artifact['rows'], artifact['population'], artifact['groups']
    if len({r['id'] for r in rows}) != len(rows) or len({r['id'] for r in population}) != len(population):
        raise ValueError('duplicate evidence identities')
    for row in rows:
        spec = row['detail']['spec']
        if spec['pristine_protection_enabled'] or spec['pristine_min_fraction'] != 0 or spec['light'] != [1., 4.]:
            raise ValueError('stale protected or light-depth contract')
        if row['audit']['status'] != 'passed':
            raise ValueError('unaudited evidence')
        for background in row['detail']['background'].values():
            if background['contact_protection_enabled'] or background['applied_max_depth_px'] > 4.000001:
                raise ValueError('hidden protection or light depth above 4px')
        if row['label'] == '正例' and not .25 <= 1-row['detail']['trim']['retained_fraction'] <= .40:
            raise ValueError('actual trimming outside 25–40%')
    identities = [{r['source_base_key'] for r in population if r['version'] == v} for v in ('v17.5', 'v18')]
    if not identities[0].isdisjoint(identities[1]):
        raise ValueError('cross-version source identity overlap')
    progress, summary = [], []
    for version in ('v17.5', 'v18'):
        pop = [r for r in population if r['version'] == version]
        shown = [r for r in rows if r['version'] == version]
        vg = [g for g in groups if g['version'] == version]
        if len(pop) != probe['versions'][version]['pairs'] or len(shown) != len(pop):
            raise ValueError('probe population and displayed examples must reconcile')
        progress.append(dict(version=version, status='新规范小试已完成；全量规模待确认',
            audited_pairs=len(pop), groups=len(vg), slots=sum(g['count'] for g in vg), per_type=None))
        metrics = [
            ('新增裁短比例', '%', 100, [r['trim_fraction'] for r in pop if r['label']]),
            ('双侧严格未损伤／原始裁前公共接缝（仅统计）', '%', 100, [r['pristine_fraction'] for r in pop if r['label']]),
            ('新增裁切面积损失', '%', 100, [r['cut_area_loss'] for r in pop]),
            ('主损伤双侧gap峰值', 'px', 1, [r['gap_peak'] for r in pop if r['gap_peak'] is not None]),
            ('正例继承对应数', '条', 1, [r['inherited'] for r in pop if r['label']]),
            ('轻退化实测最大深度（每片）', 'px', 1, [b['applied_max_depth_px'] for r in shown for b in r['detail']['background'].values()])]
        for name, unit, scale, values in metrics:
            if not values:
                continue
            q = np.quantile(np.asarray(values)*scale, [0, .1, .5, .9, 1])
            summary.append(dict(version=version, metric=name, unit=unit, n=len(values),
                **dict(zip(('min', 'p10', 'p50', 'p90', 'max'), map(float, q)))))
    protocol = [
        dict(item='连续弱腐蚀峰值', v175='4–8px', v18='5–8px', note='指定接缝区域连续渐进；肩部可更浅'),
        dict(item='起伏／突变／渐进峰值', v175='7–15px', v18='10–15px', note='仅已选一侧施加主层'),
        dict(item='独立缺口', v175='1–4处，各5–15px', v18='1–4处，各5–15px', note='逐处像素生效；同侧主＋弱封顶15px'),
        dict(item='主层覆盖公共接缝', v175='目标35–50%', v18='目标40–60%', note='裁短后留存原始公共弧为分母'),
        dict(item='新增端部裁短', v175='25–40%', v18='25–40%', note='原始裁前参考，不是从v17累加裁切'),
        dict(item='裁切侧／面积上限', v175='70%小片／30%大片；≤20%', v18='同左', note='全量成功样本配额；本22对小试不冒称满足全量配额'),
        dict(item='双侧主损伤gap峰值', v175='5–35px', v18='5–35px', note='固定GT下双侧投影，不是整缝最小间隙'),
        dict(item='双侧严格未损伤共同弧', v175='无最低比例、无保护区', v18='同左', note='仍按原始裁前接缝计算；只统计，不筛选'),
        dict(item='背景轻微退化', v175='可退化原始轮廓70%；1–4px', v18='同左', note='排除新裁边及主损伤；不再保护公共接缝；峰值范围，渐变肩部可更浅'),
        dict(item='保留有效监督', v175='至少4条继承GT对应', v18='同左', note='Partial保留15%较小片周长下限；不是25%未损伤接缝要求'),
        dict(item='扩量与课程训练', v175='全量已授权，规模待确认', v18='同左', note='本页是已完成小试记录；课程GPU训练未启动；v19已弃用')]
    provenance = dict(label='source05实际掩膜及独立像素/监督审计，22对预检',
        tables=[source_name, artifact['source_root']+'/probe_complete.json',
            artifact['source_root']+'/v17.5/pixel_audit.json', artifact['source_root']+'/v18/pixel_audit.json'],
        executedAt=datetime.fromtimestamp(probe['updated_unix'], timezone.utc).isoformat(),
        notes=['全部22对实际小试均已审计；不是全量，也不是各类10例配额。',
            '两版不保留25%未损伤弧最低值；轻退化请求峰值1–4px且实际深度不超过4px。',
            'v14/v17历史归档和GPU训练未改动；全量批准不等于已经生成。'],
        metricDefinitions=[
            dict(label='审计通过样本对', definition='各版manifest的唯一pair id数，与pixel_audit和probe_complete行数一致。', componentIds=['curriculum-progress']),
            dict(label='展示槽', definition='各分类引用样本对的次数总和；同一对可能跨分类复用，不是独立样本量。', componentIds=['curriculum-progress']),
            dict(label='未损伤比例', definition='原始裁前共同弧中，两侧原始伙伴及其3×3像素邻域均未变化的弧长比例；仅统计，无门槛。', componentIds=['curriculum-distribution', 'curriculum-gallery']),
            dict(label='轻退化实测最大深度', definition='相对主腐蚀后掩膜，实际新增移除像素的最大内退深度；每片一个观测。', componentIds=['curriculum-distribution', 'curriculum-gallery'])])
    old_queries = {k: digest(v) for k, v in data['queries'].items() if not k.startswith('curriculum_')}
    for key, values in [('curriculum_cases', rows), ('curriculum_groups', groups), ('curriculum_summary', summary), ('curriculum_protocol', protocol), ('curriculum_progress', progress)]:
        data['queries'][key] = dict(rows=values, source=provenance)
    data.update(title='v17.5／v18：取消接缝保护、轻退化1–4px的课程数据记录',
        generatedAt=datetime.now(timezone.utc).isoformat(), buildStatus='complete',
        curriculumReview=dict(complete=False, record_complete=True, revision=REVISION,
            source_binding_sha256=artifact['protocol']['source_binding_sha256'],
            full_generation_authorized=True, full_generation_started=False, scale_confirmation_pending=True,
            unique_display_pairs=len(rows), display_slots=artifact['display_slots']))
    data['report']['asOf'] = provenance['executedAt'][:10]
    if old_queries != {k: digest(v) for k, v in data['queries'].items() if not k.startswith('curriculum_')}:
        raise ValueError('unrelated archived queries changed')
    return data, dict(id=data['id'], revision=REVISION, unique_pairs=len(rows), display_slots=artifact['display_slots'],
        archived_queries_sha256=old_queries, archive_queries_preserved=True, source_root=artifact['source_root'])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--rendered', required=True)
    parser.add_argument('--probe-receipt', required=True);parser.add_argument('--app', required=True)
    parser.add_argument('--build-status', choices=('updating','complete'), default='updating')
    a=parser.parse_args();source=Path(a.rendered);target=Path(a.app)/'src/data.json'
    artifact=json.loads(source.read_text());probe=json.loads(Path(a.probe_receipt).read_text())
    old=json.loads(target.read_text());data, receipt=prepare(artifact, probe, old, str(source))
    data['buildStatus']=a.build_status
    backup=source.parent/'prior_pristine25_snapshot.json'
    if not backup.exists():shutil.copy2(target, backup)
    target.write_text(json.dumps(data, ensure_ascii=False, separators=(',', ':'))+'\n')
    receipt.update(rendered_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        snapshot_sha256=hashlib.sha256(target.read_bytes()).hexdigest())
    (source.parent/'snapshot_receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == '__main__':main()
