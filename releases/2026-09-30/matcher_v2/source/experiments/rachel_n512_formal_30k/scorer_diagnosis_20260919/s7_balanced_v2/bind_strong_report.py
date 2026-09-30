"""Bind completed generated-data evidence into the existing local review app."""
import argparse
from pathlib import Path
from datetime import datetime,timezone
from ..s7_compound_v1.materialize import read,save_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--evidence',required=True);p.add_argument('--project',required=True)
    p.add_argument('--full-running',action='store_true')
    p.add_argument('--build-status',choices=('updating','complete'),default='updating')
    p.add_argument('--full-failed',help='A recorded full-generation failure; pilot evidence stays unchanged')
    a=p.parse_args();folder=Path(a.evidence);project=Path(a.project)
    ev=read(folder/'evidence.json');cases=read(folder/'showcases.json')
    if ev['status'] not in ('pilot_complete','complete') or ev['validation']['status']!='passed':raise ValueError('incomplete data')
    data=read(project/'src/data.json')
    # Existing report identity/presentation remains; old baseline queries stay.
    data['title']='强腐蚀仿真数据：实际配额与逐类案例审核'
    data['buildStatus']=a.build_status;data['generatedAt']=datetime.now(timezone.utc).isoformat()
    # Historical real/S7 baselines and this new export have different cutoffs.
    # Use the shell's prepared timestamp instead of an obsolete common as-of.
    data.setdefault('report',{}).pop('asOf',None)
    data['strongEvidence']={k:v for k,v in ev.items() if k!='cases'}
    data['strongCases']=cases
    data['methodology'].update(fullDataGenerationRunning=False,newPilotPending=False,
        strongFullDataRunning=a.full_running,newStrongTrainingStarted=False)
    if a.full_failed:
        if a.full_running:raise ValueError('full generation cannot be both running and failed')
        failure=read(a.full_failed)
        if failure.get('status')!='failed':raise ValueError('expected an actual failure record')
        data['methodology']['strongFullDataFailure']=dict(failure,source=str(Path(a.full_failed).resolve()))
        data['buildStatus']='paused'
    else:
        data['methodology'].pop('strongFullDataFailure',None)
    def query(rows,definition,ids):
        return dict(rows=rows,source=dict(label='已完成强腐蚀生成记录与未截断接缝诊断',
            executedAt=datetime.fromtimestamp((folder/'evidence.json').stat().st_mtime,timezone.utc).isoformat(),
            files=[dict(label=str((folder/'evidence.json').resolve())),dict(label=str((folder/'showcases.json').resolve()))],
            metricDefinitions=[dict(label='口径',definition=definition,componentIds=ids)],
            caveats=['未训练新模型；敦煌聚合面积参与配方开发，非未触及测试。',
                     '潜在接缝来自腐蚀前TRAIN GT支持段；法线投影是诊断代理，不是新增对应标签。']))
    recipes=[dict(r,mixtureShare=r['share'],peakMedian=r['peak']['median'],gapMean=r['gapMean']['mean'],lengthMean=r['latentLength']['mean']) for r in ev['recipeRows']]
    data['queries']['strongRecipes']=query(recipes,'每类正/负分别为分母；配额为已完成样本的实际数量，不是计划值。',['strong-summary','strong-recipes','strong-recipe-bars'])
    data['queries']['strongLatent']=query([dict(r,gapShare=r['share']) for r in ev['latentHist']],'每对等权、对内按原接缝弧长加权；不按最终gap截断；法线找不到幸存材料的部分计入未解析比例。',['strong-gap-chart'])
    data['queries']['strongSeamSummary']=query([dict(r,resolved=r['resolvedMean'],meanGap=r['meanGap']['mean'],lengthMean=r['sourceLength']['mean']) for r in ev['latentSummaries']],
        '近远并存：同一GT支持段至少5px原弧长gap≤5、至少10px原弧长gap≥15；不是最近两点的距离。',['strong-seam-summary'])
    data['queries']['strongNotches']=query(ev['notches'],'仅含gaps的样本为分母；K为真正施加、并额外删除材料的边缘缺口数，不是新造内部孔洞。',['strong-notch-bars'])
    negative_scope=(f"试产是冻结来源计划的前{ev['negatives']}个负例，未冒充全量35/35/30已落实。"
                    if ev['status']=='pilot_complete' else '正式全量负例；35/35/30须由实际计数及来源校验确认。')
    data['queries']['strongNegative']=query(ev['negativeSources'],'训练负例分母；'+negative_scope,['strong-negative-table'])
    placement=[dict(r,coreMedian=r['coreWidthPx']['median'],coreP10=r['coreWidthPx']['p10'],
                    coreP90=r['coreWidthPx']['p90'],attemptsMean=r['planAttempts']['mean'])
               for r in ev.get('placementSummary',[])]
    data['queries']['strongPlacement']=query(placement,
        '浅段核心宽度按实际受腐蚀碎片为单位，不含两侧各8px平滑过渡；正例围绕已有TRAIN锚点，负例同物理宽度随机放置。计划尝试数按最终保留裁切后的获准计划统计，不是所有外层拒绝次数。',
        ['strong-placement-table'])
    data['queries']['strongCases']=query([{k:v for k,v in c.items() if k not in ('image','gapProfile')} for c in cases],
        '每配方各展示带/不带Partial的典型样本：按该组平均gap接近中位选取；额外展示结构、镜像、负例与大小差异。图是生成Mask/GT，不是模型预测。',['strong-cases'])
    profiles=[dict(r,caseId=c['id']) for c in cases for r in c['gapProfile']]
    data['queries']['strongProfiles']=query(profiles,'腐蚀前支持段累计弧长；每例至多等索引抽200点展示，完整原数组保留。断开的未知区不插值成真接缝。',['strong-profile'])
    save_json(project/'src/data.json',data)
    print(dict(pairs=ev['pairs'],cases=len(cases),report_id=data['id']))


if __name__=='__main__':main()
