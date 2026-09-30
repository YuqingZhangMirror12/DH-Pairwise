"""Project only aggregate, non-identifying evidence to the inline receipt."""
import argparse
from pathlib import Path
from common import read,save
from report import LABELS


def build(root):
    root=Path(root);data=read(root/'results.json')
    if data['status']!='complete':raise ValueError('final receipt requires completed evaluation')
    items=[]
    for split,title in (('real','敦煌'),('ood','Turufan')):
        cohort=data['results'][split];records=[]
        for key,label in LABELS.items():
            item=cohort['models'][key];cv=item['cv']['bounded_max_f1'];m=cv['pooled_out_of_fold']
            records.append(dict(model=label,threshold_median=cv['threshold_median'],
                threshold_min=cv['threshold_min'],threshold_max=cv['threshold_max'],
                accuracy_pct=round(100*m['accuracy'],2),precision_pct=round(100*m['precision'],2),
                recall_pct=round(100*m['recall'],2),f1_pct=round(100*m['f1'],2),
                fixed_03_f1_pct=round(100*item['fixed_03']['f1'],2),
                old_sim_f1_pct=round(100*item['old_sim']['f1'],2),tp=m['tp'],fp=m['fp'],fn=m['fn'],tn=m['tn']))
        columns=[{'field':field,'label':label} for field,label in (
            ('model','模型'),('threshold_median','五折阈值中位数'),('threshold_min','阈值最小'),
            ('threshold_max','阈值最大'),('accuracy_pct','Accuracy (%)'),('precision_pct','Precision (%)'),
            ('recall_pct','Recall (%)'),('f1_pct','F1 (%)'),('fixed_03_f1_pct','固定0.3 F1 (%)'),
            ('old_sim_f1_pct','原SIM阈值 F1 (%)'),('tp','TP'),('fp','FP'),('fn','FN'),('tn','TN'))]
        items.append(dict(id=split+'-bounded-cv',title=title+'：26组模型的受限阈值比较',queries=[dict(
            id=split+'-oof-counts',source=dict(label='固定检查点逐对独立折预测',
                filters=[f"正例：{cohort['positive']}对",f"负例：{cohort['negative']}对",
                    '阈值候选：0.20–0.80，步长0.01','每次4折校准、1折测试，全部5折轮换'],
                caveats=['真实样本参与过此前人工筛选和模型分析，本轮不是新的未见盲测。',
                    '跨来源负例按来源规则构造，并非逐对人工核验。',
                    '表中阈值为各折中位数，指标使用每对各自独立折阈值计算。']),
            rows=records,columns=columns,reportingPeriod='2026年9月20日至21日固定模型的真实域阈值复算',
            evidenceFlow=[dict(kind='calculation',title='阈值选择',
                detail='在校准折最大化F1；并列时优先最接近0.30，再比较Precision和更高阈值。未对原始分数缩放。'),
                dict(kind='validation',title='来源隔离与计数',
                detail='每次校准与测试的已知来源组、碎片ID交集均为0；每对仅计入一次独立测试折。')])]))
    save(root/'source-receipt.json',dict(schemaVersion=1,items=items))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);build(p.parse_args().root)
