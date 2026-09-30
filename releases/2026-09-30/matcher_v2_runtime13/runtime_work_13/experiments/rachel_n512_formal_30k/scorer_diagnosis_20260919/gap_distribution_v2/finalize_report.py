"""Bind actual case figures and evidence to the existing report identity."""
import argparse
from pathlib import Path
import nbformat
from nbclient import NotebookClient
from .measure import read,save


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--project',required=True)
    p.add_argument('--new',required=True);p.add_argument('--complete',action='store_true')
    p.add_argument('--full-running',action='store_true',help='Set only after verifying the actual full-data controller')
    p.add_argument('--pilot-pending',action='store_true',help='Figures from completed samples only; no complete new-dataset distribution')
    a=p.parse_args();root=Path(a.root).resolve();project=Path(a.project)
    data=read(root/'reviewed_snapshot.json');old=read(project/'src/data.json')
    data['id']=old['id'];data['buildStatus']='paused' if a.pilot_pending else 'complete' if a.complete else 'updating'
    data['methodology']['fullDataGenerationRunning']=a.full_running
    data['methodology']['newPilotPending']=a.pilot_pending
    scale_path=root/a.new/'scale_summary.json'
    if scale_path.exists():
        scale=read(scale_path);data['newScale']=scale
        rr=[dict(label=k,**{stat:r['common_scale'][stat] for stat in ('p10','p50','p90')},
            window32Median=r['windows_original_simulation_px']['32']['p50'],
            window32P10=r['windows_original_simulation_px']['32']['p10'],
            window32P90=r['windows_original_simulation_px']['32']['p90'],
            clipped=r['clipped_fraction'],backoff=r['topology_backoff_fraction'])
            for k,r in [('正例',scale['positive']),('负例',scale['negative'])]]
        data['queries']['newScale']=dict(rows=rr,source=dict(label='新数据实际共同尺度记录',
            files=[dict(label=str(scale_path))],metricDefinitions=[dict(label='实际窗口',
                definition='32/common_scale，换算至缩放前的仿真规范坐标；不是原纸物理尺度。',componentIds=['new-scale-table'])]))
    data['showcases']=read(root/a.new/'showcases/showcases.json')+read(root/'real_showcases/showcases.json')
    evidence=[{k:v for k,v in r.items() if k!='image'} for r in data['showcases']]
    data['queries']['showcases']=dict(rows=evidence,source=dict(label='实际生成Mask与原GT摆放',
        files=[dict(label=str(root/a.new/'showcases/showcases.json')),dict(label=str(root/'real_showcases/showcases.json'))],
        metricDefinitions=[dict(label='案例选择',definition='各类可用样本中选择面积接近中位的样本；Partial演示优先要求裁切材料保留≤85%，以看清裁切作用；敦煌按平均近接距离约2/5/12px选择。仅定性示例，不推断频率。',componentIds=['showcases'])]))
    receipt_path=root/a.new/'generation_summary.json'
    if receipt_path.exists():
        receipt=read(receipt_path)
        data['generationReceipt']={k:receipt[k] for k in ('sample_count','positive_count','negative_count',
            'corrosion_recipe_counts','corrosion_category_counts','partial_fraction','mirror_counts',
            'negative_kind','grouped_negative_pairs','source_strata','unique_base_pairs',
            'positive_source_replacements','max_positive_reuse')}
        names={'clean':'无腐蚀','wave':'单一波状','wave_gaps':'波状＋缺口','wave_local':'波状＋局部深腐蚀'}
        rr=[dict(recipe=names[k],positive=v,negative=v,share=v/receipt['positive_count'])
            for k,v in receipt['corrosion_recipe_counts'].items()]
        data['queries']['recipes']=dict(rows=rr,source=dict(label='已完成试产/全量的实际生成记录',
            files=[dict(label=str(receipt_path)),dict(label=str(root/a.new/'validation.json'))],
            metricDefinitions=[dict(label='正负配方',definition='每个已生成组含一对正例和一对负例，配方相同；实际数量与加载验证对齐。',componentIds=['recipe-table'])]))
    save(project/'src/data.json',data)
    comparison=read(root/'comparison.json');metrics=comparison['metrics']
    rows=['| 数据 | 正例 | 0–2px | 2–8px | 8–15px | 平均面积px² | gap TV↓ |',
          '|---|---:|---:|---:|---:|---:|---:|']
    rows += [f'| {r["dataset"]} | {r["positivePairs"]} | {r["gapUnder2"]:.2%} | {r["gap2to8"]:.2%} | {r["gap8to15"]:.2%} | {r["areaMean"]:.0f} | {r["gapTV"]:.4f} |' for r in metrics]
    text='\n'.join(['# 仿真与敦煌的点级接缝分布','',
        '原295对全部测量；主对照为剔除用户确认错误GT（16、389、563）后的292对。',
        '下表为法线相向、距离≤40px的几何近接带，逐对等权、两侧等权；不是语义接缝GT。','',*rows,'',
        '全轮廓、不截断的双向距离也保存在各summary.json，报告可切换查看。',
        '面积为800px模型输入上的前景像素数；每个正例两次碎片观测，增强重复来源未冒充独立文档。','',
        '## Patch尺度','',
        '旧S7/S7-C与敦煌窗口均为7/16/32/64输入像素，统一采样到16×16；网格间隔为0.4/1/2.067/4.2px。',
        '敦煌32px窗口对应原扫描图宽度中位114.28px，P10–P90为50.48–258px；其缩放按同Case的父画布共同进行。',
        '新S7-D额外应用双片共同尺度变化，原仿真坐标的窗口宽度为window/common_scale；GT与对应坐标同步变换。',
        '缺少DPI，不能从扫描像素差推出物理尺度差；真正应控制的是网络输入尺度及形状相对窗口的比例。','',
        '## 新数据及限制','',
        '无腐蚀30%，单一波状35%，波状＋缺口20%，波状＋局部深腐蚀15%；Partial独立70%，镜像15%。',
        '负例35%跨Gen、35%同Gen跨parent、30%同parent非邻接；2000个原锚碎片各3个负伙伴。',
        'Gen4/Gen5结构来源保留。新训练尚未启动，运行中的F/I不变。',
        'Dunhuang聚合分布已参与配方开发，因此它不再是未参与开发的测试集；Turufan未参与本次拟合。',
        '统计更接近不等于已证明Recall改善，需后续受控训练验证。','',
        f'新配方案例来源：{a.new}。含pilot表示计划800对试产，不是24K全量统计。',
        '当前仅展示新版已生成样本的定性案例，完整试产未完成，未计算/引用完成子集的分布冒充总体。' if a.pilot_pending else '当前新配方统计为已完整生成的试产或正式集。',
        '正式24K正在独立生成，完成后另行通知并更新全量统计。' if a.full_running else '完整生成状态以远端pipeline_status/validation及数据卡为准。','',
        '13个仿真实际案例和3个敦煌GT示例见report和showcases目录；图中摆放是GT，不是模型预测。',''])
    (root/'RESULTS.md').write_text(text)
    nb=nbformat.v4.new_notebook(cells=[nbformat.v4.new_markdown_cell(text),
        nbformat.v4.new_code_cell("from pathlib import Path\nimport json, numpy as np\nroot=Path.cwd()\ncomparison=json.loads((root/'comparison.json').read_text())\ncomparison['metrics']"),
        nbformat.v4.new_code_cell("def coarse_gap(name):\n s=json.loads((root/name/'summary.json').read_text())\n e=np.array([np.inf if v is None else v for v in s['fine_edges']])\n h=np.array(s['histograms']['seam40']['pair_equal_share'])\n bounds=[0,2,4,8,15,20,30,40,np.inf]\n return np.array([h[(e[:-1]>=a)&(e[:-1]<b)].sum() for a,b in zip(bounds[:-1],bounds[1:])])\nreference=coarse_gap('Dunhuang292')\nfor r in comparison['metrics']:\n v=coarse_gap(r['key'])\n assert abs(v.sum()-1)<1e-8\n assert abs(abs(v-reference).sum()/2-r['gapTV'])<1e-8\n print(r['dataset'],np.round(v*100,3),'TV=',r['gapTV'])"),
        nbformat.v4.new_code_cell("import matplotlib.pyplot as plt\nfig,ax=plt.subplots(figsize=(9,4))\nfor r in comparison['metrics']:\n if r['key']=='Dunhuang295':continue\n s=json.loads((root/r['key']/'summary.json').read_text())\n c=np.cumsum(s['histograms']['seam40']['pair_equal_share'])[:160]\n ax.plot(s['fine_edges'][1:161],c,label=r['key'])\nax.set(xlabel='GT nearest boundary distance (px)',ylabel='Pair-equal cumulative share',xlim=(0,40),ylim=(0,1))\nax.legend();plt.show()"),
        nbformat.v4.new_markdown_cell('Raw-mask measurement is implemented in gap_distribution_v2/measure.py. This notebook recomputes the reviewed histogram/CDF statistics from its measured summaries. The S7-D trial is not a model result. Geometry GT and labels are never inferred from a prediction.')])
    nb.metadata['kernelspec']=dict(name='python3',display_name='Python3',language='python')
    NotebookClient(nb,timeout=120,resources={'metadata':{'path':str(root)}}).execute()
    nbformat.write(nb,root/'distribution_analysis.ipynb')
    print({'report_bound':str(project),'cases':len(data['showcases']),'notebook_executed':True})


if __name__=='__main__':main()
