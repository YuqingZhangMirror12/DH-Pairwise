"""Compact, reproducible supporting tables; no HTML or model deployment."""
import argparse
from pathlib import Path
from common import read,save

LABELS={
    'old_s6_d2_c16':'C0／S6两层续训', 'old_c1_c16':'C1：全局＋最终端点残差',
    'old_c2_c16':'C2：全局＋全量点残差', 'old_s7_c16':'S7原全轮廓头续训',
    'old_all_tokens':'S7：全轮廓新头', 'old_matched_tokens':'S7：最终匹配端点',
    'old_edge_seed':'S7：单初始候选点对', 'old_edge_multi':'S7：最多5候选点对',
    'old_matched_edges':'S7：最终解算点对', 'old_G0':'G0：同事思路／冻结特征',
    'old_G1':'G1：同事思路／训练特征', 'old_s4_d1_c16':'Attention一层续训',
    'old_s6_d4_c16':'Attention四层续训', 'old_zero_c16':'谱对照：关闭摘要',
    'old_mass_c16':'谱对照：匹配总量', 'old_mass_spectral_c16':'谱对照：总量＋谱特征',
    'old_m16_all_tokens':'Matcher M16＋全轮廓新头', 'old_m20_all_tokens':'Matcher M20＋全轮廓新头',
    'reference512_h4':'新512点对／4 heads', 'cap128_h4':'新128点对／4 heads',
    'cap256_h4':'新256点对／4 heads', 'cap512_h8':'新512点对／8 heads',
    'gcn_pairing_h4':'新PairingNet式GCN', 'gcn_shredding_h4':'新ShreddingNet式GCN',
    'joint_D_h4':'新联合D', 'stable_h4':'新稳定输入',
}


def percent(v):return f'{100*v:.2f}%'


def write(root):
    root=Path(root);data=read(root/'results.json')
    lines=['# 非极端阈值：昨天与本轮模型的真实域五折比较','',
        f"状态：{data['status']}。已登记{data['registered_models']}个固定检查点；没有重新训练模型。",'',
        '## 本轮约束','',
        '- 原始score不变，不用温度缩放、Platt或其它变换把低分改成0.3。',
        '- 允许阈值仅0.20–0.80，步长0.01。每次4折按最大F1选阈值、1折独立测试，轮换全部5折。',
        '- 校准F1并列时依次选最接近0.30、Precision更高、阈值更高；规则在重算前固定。',
        '- 额外报告所有模型统一固定0.30的结果；这列没有做阈值搜索。',
        '- 来源分组、正例和新构造负例完全复用上一轮；已知写本正反面和共享碎片不跨校准／测试折。',
        '- 敦煌295正＋508负；Turufan301正＋301跨来源负。跨来源负例按用户来源规则构造，不是逐对人工核验。',
        '- 历史正例／同来源负例复用原检查点分数；旧模型只补推新跨来源负例。旧推理batch1保持一致；不是重新以batch1训练。',
        '- 18组历史端点按原预算固定C16，不在真实数据挑选epoch；另有本轮8组C16。GCN两组是我们的Scorer变体，不是原论文完整模型。',
        '- 模型、阈值规则和人工筛选曾受到这些真实案例反馈。本轮是分组交叉校准诊断，不是新的未经观察盲测；不把26组最高分视为无偏冠军估计。','',
        '阈值为5折中位数[最小,最大]；各指标由独立测试折预测合并计算，不是5折百分比的简单平均。','']
    for split,title in (('real','敦煌'),('ood','Turufan')):
        models=data['results'][split]['models']
        lines += [f'## {title}：受限五折最大F1','',
            '| 模型 | 原SIM阈值F1 | CV阈值[范围] | Accuracy | Precision | Recall | F1 | FP | 固定0.3的F1 |',
            '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
        for key,label in LABELS.items():
            if key not in models:
                lines.append(f'| {label} | 待完成 | — | — | — | — | — | — | — |');continue
            item=models[key];cv=item['cv']['bounded_max_f1'];m=cv['pooled_out_of_fold']
            lines.append(f"| {label} | {percent(item['old_sim']['f1'])} | {cv['threshold_median']:.2f} [{cv['threshold_min']:.2f}, {cv['threshold_max']:.2f}] | {percent(m['accuracy'])} | {percent(m['precision'])} | {percent(m['recall'])} | {percent(m['f1'])} | {m['fp']} | {percent(item['fixed_03']['f1'])} |")
        lines += ['',f'### {title}：统一固定0.30','',
            '| 模型 | Accuracy | Precision | Recall | F1 | FP | 原始score AUROC |',
            '|---|---:|---:|---:|---:|---:|---:|']
        for key,label in LABELS.items():
            if key not in models:continue
            m=models[key]['fixed_03']
            lines.append(f"| {label} | {percent(m['accuracy'])} | {percent(m['precision'])} | {percent(m['recall'])} | {percent(m['f1'])} | {m['fp']} | {m['auroc']:.4f} |")
        lines += ['',f'### {title}：95%召回目标在允许范围内是否可达','',
            '| 模型 | 校准达到目标折数/5 | 阈值中位数 | 独立折Recall | Precision | FP |',
            '|---|---:|---:|---:|---:|---:|']
        for key,label in LABELS.items():
            if key not in models:continue
            cv=models[key]['cv']['bounded_recall95'];m=cv['pooled_out_of_fold']
            met=sum(f['target_met'] for f in cv['folds'])
            lines.append(f"| {label} | {met}/5 | {cv['threshold_median']:.2f} | {percent(m['recall'])} | {percent(m['precision'])} | {m['fp']} |")
        lines += ['','达不到95%时保持阈值下限0.20，并明确标记未达到，不越界降阈值。','']
    lines += ['## 敦煌已正确Layout被分类放行的数量','',
        '| 模型 | 正确Layout总数 | 原SIM阈值放行 | 受限CV放行 | 固定0.3放行 |',
        '|---|---:|---:|---:|---:|']
    for key,label in LABELS.items():
        item=data['results']['real']['models'].get(key)
        if item is None:continue
        m=item['cv']['bounded_max_f1']['pooled_out_of_fold']
        lines.append(f"| {label} | {m['layout_correct_total']} | {item['old_sim']['layout_correct_accepted']} | {m['layout_correct_accepted']} | {item['fixed_03']['layout_correct_accepted']} |")
    lines += ['','Layout成功口径沿用有效位移且误差≤20px；这里只改分类接受，未重排位移候选。Turufan无Layout GT。','',
        '## 可复算文件','',
        '- `results.json`：精确指标与每折校准／测试混淆矩阵。',
        '- `registry.json`：固定模型、原始评估位置、源代码快照和检查点绑定。',
        '- `oof/<split>/<model>/`：每对独立折阈值与接受结果。',
        '- `predictions/`：新负例推理；本轮8组则引用先前完成输出。',
        '- `<split>/manifest.json`：来源隔离的同一分折及负例清单。',
        '- `common.py`、`calibrate.py`、`infer.py`、`prepare.py`：完整可复算实现；未改变旧freeze。','']
    (root/'RESULTS.md').write_text('\n'.join(lines))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);write(p.parse_args().root)
