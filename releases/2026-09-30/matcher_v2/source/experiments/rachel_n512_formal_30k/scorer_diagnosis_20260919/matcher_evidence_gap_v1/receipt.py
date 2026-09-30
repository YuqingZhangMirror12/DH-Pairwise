"""Project only reviewed, answer-scoped aggregates into an inline receipt."""
import json
from pathlib import Path


ROOT = Path('artifacts/matcher_evidence_gap_20260927')
g = json.loads((ROOT/'full/analysis.json').read_text())
r = json.loads((ROOT/'ranking/summary.json').read_text())

geometry_rows=[]
for split, label in [('sim_select','仿真 SELECT'),('dunhuang_cv','敦煌')]:
    for key, metric in [('length_40_px','近接轮廓长度 L40（px）'),
            ('gap40_over10_share','L40 中间隙大于 10px 的长度比例'),
            ('correct_edges_per100px_L40','每 100px 的正确簇点对数')]:
        v=g['distributions'][split][key]
        geometry_rows.append(dict(dataset=label,metric=metric,**{k:v[k] for k in ('n','p10','p25','p50','p75','p90')}))
threshold_rows=[]
for split,label in [('sim_select','仿真 SELECT'),('dunhuang_cv','敦煌')]:
    for key,diameter in [('threshold16',16),('threshold20',20)]:
        v=g['threshold_change'][split][key]
        threshold_rows.append(dict(dataset=label,diameter_px=diameter,
            correct_candidate_pairs=v['coverage_retained'],top_mass_correct=v['top_correct'],
            correct_cluster_n_median=v['n_when_correct']['p50'],negative_maxmass_median=v['negative_max_mass']['p50']))

ranking_rows=[]; operating_rows=[]
for job, label in [('m12_sim_test_v14','仿真 TEST'),('m12_dunhuang_cv','敦煌'),('m12_turufan','Turufan')]:
    d=r['datasets'][job]; x=d['ranking']['by_label']; q=d['ranking']
    ranking_rows.append(dict(dataset=label,positive_pairs=d['positives'],
        multi_positive=x['positive_n'],changed_top=x['positive_n']-x['positive_same_top'],
        raw_layout=d['methods']['observed_mass_length_px']['layout_correct_final_poses'],
        scorer_layout=d['methods']['logit']['layout_correct_final_poses'],
        wrong_to_correct=(q['positive_layout_transitions'] or {}).get('wrong_to_correct'),
        correct_to_wrong=(q['positive_layout_transitions'] or {}).get('correct_to_wrong')))
    for method,name in [('observed_mass_length_px','原始 Q×弧长质量'),('logit','实际 Scorer'),
                        ('perfect_local_score','全部局部判支持，保留重叠'),('no_conflict_score','仅删除最终冲突扣分')]:
        z=d['methods'][method]; p=z['matched_fp'][-1]
        operating_rows.append(dict(dataset=label,method=name,auc=z['auc'],
            fp=p['fp'],tp=p['tp'],joint_accepted=p['joint_accepted'],retrospective=True))


def item(id,title,source,rows):
    return dict(id=id,title=title,queries=[dict(id=id+'-calculation',source=source,
        rows=rows,columns=list(rows[0]),preview=dict(kind='aggregate',note='全量记录的聚合结果。'))])


items=[
    item('geometry-distributions','短接缝和间隙的分布与归一化点数',dict(
        label='同一冻结 M12 的几何诊断',files=['full/analysis.json','geometry_diagnostics.py','PROTOCOL.md'],
        metricDefinitions=[dict(id='l40',definition='L40 是 GT 摆放下相向、外侧、间隙不超过 40px 的实际轮廓弧长，两侧平均，不使用 Q。'),
            dict(id='density',definition='点数取原始质量最大的 GT20 正确簇，无正确候选计零，再除以可观测 L40。')],
        filters=['SIM SELECT：750 个正增强实例','敦煌：292 个有效 GT 正例；零长度的 1 对不能计算密度'],
        caveats=['L40 不是完整祖先接缝；间隙不是已知腐蚀深度。',
                 '两域输入像素不是统一物理尺度，写卷和碎片重复也使样本并不独立。'],
        evidenceFlow=[dict(kind='validation',title='全量覆盖',detail='2303 对完成；750 与 292 正例分母核对，逐文件 SHA 保留。'),
            dict(kind='calculation',title='粗分层',detail='近接长度 256–512px 且 L10/L40≥80% 时，SIM 170 对点数中位数 256，敦煌 127 对为 123；不是严格因果匹配。')]),geometry_rows),
    item('pose-threshold','16→20px 只作冻结 proposal 对照',dict(
        label='原始局部候选的完整连接 CPU 重放',files=['full/analysis.json','full/complete.json','run.py'],
        metricDefinitions=[dict(id='diameter',definition='16/20px 均指全部原始成员平移的最大两两距离，不是链式连通或移动中心半径。')],
        filters=['最多保留 8 个簇','GT 布局正确性仍为 20px','未重跑 Matcher 或 Scorer'],
        caveats=['这不是 20px 新模型的分类结果，生产配置未改。','负例最大簇质量也会随着聚类放宽而增加。'],
        evidenceFlow=[dict(kind='validation',title='边界与完整性',detail='全量 2303 对重放完成，原生成员最大直径断言通过，4 个几何单元测试通过。')]),threshold_rows),
    item('scorer-rank','Scorer 重排确实存在，但布局净收益不稳定',dict(
        label='阈值 M12 E8 已冻结逐候选预测',files=['ranking/summary.json','pair_predictions.jsonl','ranking.py','model.py','consensus_head.py'],
        metricDefinitions=[dict(id='mass',definition='原始 M 是去重并集 Q 乘观测弧长的质量，尚未经过支持/冲突概率门控。'),
            dict(id='rerank',definition='第一名改变率只统计多候选正例；布局正确数统计全部正例，固定同一组最终精修位姿。')],
        filters=['模型：冻结阈值 M12 E8','敦煌：排除 GT 错误 16、389、563'],
        caveats=['布局统计不要求最终分类接受。','Turufan 没有布局 GT。',
            '原始质量基线仍沿用现有神经精修后的位姿，不是完全无神经网络的系统。'],
        evidenceFlow=[dict(kind='validation',title='与不可变输出核对',detail='核对预测 SHA、全部候选原值、模型未变回执和 argmax；3000/800/602 分母一致。'),
            dict(kind='validation',title='纯 sum(Q) 复核',detail='敦煌 797 个全候选缓存对齐的完整对中，292 个正例全部保留，仍为 14 个改对、23 个改错。排除的 3 对均为负例。')]),ranking_rows),
    item('scorer-classification','分类收益与布局排序分开评价',dict(
        label='相同误报预算下的冻结读出诊断',files=['ranking/summary.json','ranking.py','threshold_diagnosis/summary.json'],
        metricDefinitions=[dict(id='equal-fp',definition='只使用已有负例分数确定相同误报预算的操作点；边界并列整体拒绝，再计算正例接受与摆对且接受。')],
        filters=['敦煌误报预算 32/508','Turufan 误报预算 19/301','仿真 TEST 误报预算 94/1500'],
        caveats=['这些是真实数据已曝光后的开发性回顾诊断，不是部署阈值或新模型成绩。',
            '当前模型正式分类阈值仍为仿真 CAL 选出的 0.21，Dun 接受正例 144、误报 1。',
            '热图、权重及局部概率说明信息流，但本次没有证明网络依赖何种输入特征的因果机制。'],
        evidenceFlow=[dict(kind='validation',title='计算与边界测试',detail='5 个单测核对 AUC ties、单调逆序、同误报边界 ties、预算及稳定 argmax。'),
            dict(kind='calculation',title='同一局部读出',detail='实际分数由支持 P 加分、冲突 C 扣分和材料重叠扣分构成；全部判支持与删除冲突均保留现有候选和位姿。')]),operating_rows)]

items.append(dict(id='current-head-design',title='现有输入已包含 Patch 与 Q；新建议简化的是读出',queries=[dict(
    id='head-code',source=dict(label='阈值版绑定实现的本地相同源码快照',
        files=['threshold_evidence.py','consensus_head.py','model.py','losses.py','RESULTS.md'],
        metricDefinitions=[dict(id='head',definition='现有头以端点为 token，使用自身与 Q 加权对侧的 Patch/Context 特征和 17 项标量，经两层四头 Attention 后预测局部支持/未知/冲突及定位可靠性。'),
            dict(id='readout',definition='整簇读出由局部支持质量加分、冲突质量扣分、重叠扣分组成，并非直接簇级 MLP。'),
            dict(id='supervision',definition='现有训练已有候选 BCE 与候选排序损失；并不是完全没有排序监督。')],
        caveats=['直接池化加小 MLP 的评分头是新对照建议，尚未实现或部署。',
            '现有简化版实验简化 Layout 聚类，不是简化 Scorer 网络。',
            '仅替换读出时会保留既有定位模块，不能冒称已经完全移除旧网络。']),
    summary='输入目标与用户设想接近；建议验证取消局部三分类—显式惩罚瓶颈，保持相同候选与位姿进行公平对照。')]))
(ROOT/'answer-sources.json').write_text(json.dumps(dict(schemaVersion=1,items=items),ensure_ascii=False,indent=2)+'\n')
print('answer-sources.json: 5 findings, reviewed aggregates and code only')
