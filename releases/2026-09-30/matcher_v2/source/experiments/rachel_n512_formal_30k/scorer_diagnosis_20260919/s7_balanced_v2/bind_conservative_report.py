"""Bind reviewed conservative evidence without deleting historical app data."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path


def bind(project,evidence_dir=None,complete=False):
    project=Path(project);path=project/'src/data.json'
    snapshot=json.loads(path.read_text())
    snapshot['buildStatus']='complete' if complete else 'updating'
    snapshot.setdefault('methodology',{}).update(strongRejected=True,conservativePending=evidence_dir is None,
        newTrainingStarted=False,newStrongTrainingStarted=False)
    snapshot['title']='分层增强800对试产：待生成结果审核' if not evidence_dir else '保守腐蚀新版：增强配额与每类10例审核'
    if evidence_dir:
        evidence_dir=Path(evidence_dir)
        evidence=json.loads((evidence_dir/'evidence.json').read_text())
        cases=json.loads((evidence_dir/'showcases.json').read_text())
        group_names={
            'structure_native_positive':'原始相邻碎片',
            'structure_union_positive_tiny':'Gen4/Gen5合并与大小碎片',
            'structure_gen5_partition_positive':'Gen5分组合并／丢片',
        }
        for group in evidence['caseGroups']:
            group['name']=group_names.get(group['id'],group['name'])
        assert evidence['validation']['status']=='passed' and evidence['numericalAudit']['status']=='passed'
        assert len(cases)==evidence['uniqueCases']
        layered=evidence.get('layered',False)
        version='v14' if evidence.get('partialSubtypeRows') else 'v13'
        if complete:
            if layered:
                assert evidence['pairs']==800 and evidence['status']=='pilot_complete'
                assert evidence['pixelAudit']['status']=='passed' and evidence['pixelAudit']['all_actual_masks_reconstructed']
                assert all(g['count']==20 and len(set(g['caseIds']))==20 for g in evidence['caseGroups'])
            else:
                assert evidence['pairs']==24000 and evidence['status']=='complete'
                assert all(g['count']==10 and len(set(g['caseIds']))==10 for g in evidence['caseGroups'])
        if layered:
            snapshot['title']=f'分层增强 {version}：800对试产 · 每组20例人工审核'
            snapshot['methodology'].update(fullDataGenerationRunning=False,newPilotPending=False,
                waitingForHumanPilotApproval=True,newFullGenerationStarted=False)
        snapshot['conservativeEvidence']=evidence;snapshot['conservativeCases']=cases
        now=datetime.now(timezone.utc).isoformat();snapshot['generatedAt']=now
        def source(ids,definition):
            return dict(label=f'分层{version}实际生成、全800对Mask检查与腐蚀深度场重建审计' if layered else '保守v12实际生成、逐archive深度场审计及真实Mask',executedAt=now,
                files=[dict(label=str(evidence_dir/f)) for f in ('evidence.json','showcases.json')],
                metricDefinitions=[dict(label='口径',definition=definition,componentIds=ids)],
                caveats=['合成训练数据；尚未训练，不代表模型性能。','完整800对试产的实际计数，非24K。Partial属于互斥主损伤，轻退化为明确例外。' if layered else '固定比例是全量实际计数；Partial/镜像为独立轴。',
                    '面积/长度聚合参考过敦煌开发分析；负例无GT seam。'])
        recipes=[dict(r,peakMedian=r['actualPeak']['median'],coverageMedian=r['affectedFraction']['median'],
            gapMean=r['gapMean']['mean'],sourceLengthMedian=r['sourceLength']['median'],
            sourceLengthP10=r['sourceLength']['p10'],sourceLengthP90=r['sourceLength']['p90']) for r in evidence['recipeRows']]
        snapshot['queries']['conservativeRecipes']=dict(rows=recipes,source=source(['conservative-mixture'],
            '正、负各自为分母；峰值为每对实际删除材料最大内缩的组内中位，覆盖含弱叠加。'))
        snapshot['queries']['conservativeNegative']=dict(rows=evidence['negativeSources'],source=source(['conservative-negatives'],
            '明确负例按源parent和Gen标签计数；不是人工摆放失败。'))
        n=evidence['positives'];summary=evidence['summary']
        rows=[dict(name='共同水平镜像',count=summary['mirror_counts'].get('horizontal',0),denominator=n),
              dict(name='共同垂直镜像',count=summary['mirror_counts'].get('vertical',0),denominator=n),
              dict(name='多负伙伴样本',count=summary['grouped_negative_pairs'],denominator=evidence['negatives'])]
        if not layered:rows.insert(0,dict(name='曲线Partial',count=round(n*summary['partial_fraction']),denominator=n))
        rows.extend(dict(name=group_names.get('structure_'+key.split(':',1)[1],key),count=value,denominator=n)
                    for key,value in summary['source_strata'].items() if key.startswith('positive:'))
        for row in rows:row['share']=row['count']/row['denominator']
        snapshot['queries']['conservativeAxes']=dict(rows=rows,source=source(['conservative-axes'],
            '除多负伙伴以负例为分母，其余以正例为分母。碎片来源、镜像为不同轴，不能把全表相加；Partial已归腐蚀层级。' if layered else '除多负伙伴以负例为分母，其余以正例为分母。镜像、Partial、来源为不同轴，不能把全表相加。'))
        rows=[dict(name=f'缺口 {key} 处',count=value,denominator=sum(evidence['gapCounts']['True'].values()))
              for key,value in sorted(evidence['gapCounts']['True'].items())]
        rows.extend(dict(name={'weak_gradual':'弱腐蚀：渐进波状','weak_inset':'弱腐蚀：整段平滑内收'}[key],
                         count=value,denominator=sum(evidence['numericalAudit']['weak_modes'].values()))
                    for key,value in evidence['numericalAudit']['weak_modes'].items())
        for row in rows:row['share']=row['count']/row['denominator']
        snapshot['queries']['conservativeForms']=dict(rows=rows,source=source(['conservative-forms'],
            '缺口按含缺口正例统计；弱腐蚀形式按含弱腐蚀的全部正负样本统计，各自条件分母单列。'))
        snapshot['queries']['conservativeCases']=dict(rows=[{k:v for k,v in c.items() if k!='image'} for c in cases],
            source=source(['conservative-gallery'],
                '每配方10正10负；Partial端部/中段各20正例；碎片增强20正例；负例来源20负例。按面积分位选例，类别可共享案例；GT仅用于正例。' if layered else '每配方5正5负，按面积分位确定性选例；结构与镜像10正例，类别可能共享同一案例。GT位置仅适用于正例。'))
        if layered:
            snapshot['queries']['conservativeLayers']=dict(rows=evidence['corrosionLayerRows'],source=source(['layered-types'],
                '主损伤互斥，Partial25/其他60/clean15。弱、强、缺口分类均为主损伤；新增1–3px层单列不重复计入。'))
            light=[dict(r,coverageMean=r['coverage']['mean'],eligibleLengthMean=r['eligibleLength']['mean'],peakMedian=r['peak']['median']) for r in evidence['backgroundRows']]
            snapshot['queries']['conservativeLight']=dict(rows=light,source=source(['layered-light'],
                '每侧实际退化的连续原轮廓弧长／剩余未腐蚀原轮廓；排除人工新切边，实际70%±2pp。碎片计出现次数，不是唯一碎片。'))
            snapshot['queries']['layeredLengths']=dict(rows=recipes,source=source(['layered-lengths'],
                '正例主损伤前潜在支持段的长度；Partial用裁切后保留段。主支持长度配额与受损后实际近接长度分开。'))
            if evidence.get('partialSubtypeRows'):
                partial=[dict(r,ratioMedian=r['ratio']['median'],commonLengthMedian=r['commonLength']['median'],
                    smallerPerimeterMedian=r['smallerPerimeter']['median'],flankMedian=r['minFlank']['median'],
                    removedMedian=r['removedMiddle']['median']) for r in evidence['partialSubtypeRows']]
                snapshot['queries']['layeredPartial']=dict(rows=partial,source=source(['layered-partial'],
                    '裁切后、轻退化前：双方保留原TRAIN支持弧长之较小值／裁切后按像素面积较小碎片的完整新周长。只对正例定义，删去中段和新切边不计为接缝；逐个正例≥15%。端/中段各占Partial的一半是本轮审核配比。'))
        # Historical queries have their own cutoff/provenance. Do not stamp
        # the entire retained report as newly measured on this date.
        snapshot['report'].pop('asOf',None)
    path.write_text(json.dumps(snapshot,ensure_ascii=False,separators=(',',':')))
    print(json.dumps(dict(id=snapshot['id'],status=snapshot['buildStatus'],evidence=bool(evidence_dir))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--project',required=True);p.add_argument('--evidence');p.add_argument('--complete',action='store_true')
    a=p.parse_args();bind(a.project,a.evidence,a.complete)
