"""Create the portable analysis notebook and a concise Chinese result record."""
from pathlib import Path
import argparse
import json
import nbformat as nbf
from nbclient import NotebookClient


def run(root):
    root=Path(root).resolve();s=json.loads((root/'summary.json').read_text());cs=s['cohorts']
    names=['S7 positive','S7-H positive','Dunhuang retained','Dunhuang full','Turufan positive']
    zh={'S7 positive':'S7 仿真','S7-H positive':'S7-H 困难子集','Dunhuang retained':'敦煌保留集','Dunhuang full':'敦煌原始集','Turufan positive':'Turufan'}
    def m(n,k,f='mean'):return cs[n]['metrics'][k].get(f)
    def percent(n,k):return f"{100*cs[n]['rates'][k]['fraction']:.2f}%"
    def fmt(x):return '待标注/不可测' if x is None else f'{x:,.2f}'
    def table(keys):
        lines=['| 指标 | S7 仿真 | S7-H | 敦煌保留 | 敦煌原始 | Turufan |','|---|---:|---:|---:|---:|---:|']
        for title,key,field in keys:lines.append('| '+title+' | '+' | '.join(fmt(m(n,key,field)) for n in names)+' |')
        return '\n'.join(lines)
    qlines=['| 数据 | 接近长度中位数 [P25, P75] px | 每对平均间隙中位数 [P25, P75] px | 长度 P10–P90 px |',
        '|---|---:|---:|---:|']
    for n in names[:-1]:
        l=cs[n]['metrics']['d20_length_px'];g=cs[n]['metrics']['d40_gap_mean_px']
        qlines.append(f"| {zh[n]} | {l['median']:.1f} [{l['p25']:.1f}, {l['p75']:.1f}] | {g['median']:.2f} [{g['p25']:.2f}, {g['p75']:.2f}] | {l['p10']:.1f}–{l['p90']:.1f} |")
    ar=['| S7 配方 | 分配正例数/比例 | 实际改变正例数/总正例比例 | 实际改变负例数 | 说明 |','|---|---:|---:|---:|---|']
    desc={
        'reference_e1':('原 E1 干净/轻腐蚀','保留原配方：每碎片 clean/2px/4px 概率70%/25%/5%；本槽生效正例最大设定2px 889对、4px 215对。'),
        'wave':('整圈起伏内缩','10–30px 连续内缩，相关长度30–90px；生效对平均最大实测深度29.79px。'),
        'local':('局部深腐蚀','1–3处、深10–30px、宽30–90px；可发生于非接缝边缘；生效对平均最大深度23.70px。'),
        'seam_gaps':('接缝缺口','每个被处理碎片1–5处、宽15–50px、深10–30px；生效对平均最大深度23.68px。'),
        'partial_curve':('曲线 Partial seam','从训练轮廓库取非轴对齐曲线，裁去较小碎片一部分；请求保留原监督源弧段25%–75%。'),
        'gen5_partition':('Gen5 三组组合','1200对结构增强已使用；625个3-1-1、575个1-2-2。changed=false只表示未再腐蚀，不是增强失败。')}
    for recipe in ['reference_e1','wave','local','seam_gaps','partial_curve','gen5_partition']:
        p=next(a for a in s['augmentation'] if a['recipe']==recipe and a['label']);n=next(a for a in s['augmentation'] if a['recipe']==recipe and not a['label'])
        val=f"{p['changed']} / {p['fraction_of_all_12k']:.2%}" if recipe!='gen5_partition' else '1200 / 10.00%（结构）'
        ar.append(f"| {desc[recipe][0]} | {p['requested']} / {p['requested']/12000:.0%} | {val} | {n['changed'] if recipe!='gen5_partition' else '1200（结构）'} | {desc[recipe][1]} |")
    report=f'''# 仿真与真实轮廓分布统计 · 2026-09-23

## 结论

当前数据存在明确的**形态分布差异**，但这不是“所有真实边缘都需要30px腐蚀”，也不能单凭分布统计认定分类失败完全由数据造成。

- S7 的几何近接边界平均534.1px，保留敦煌为380.8px；中位数477.3对367.4px。最长连续近接段也偏长：均值469.0对299.5px。
- 每对平均接缝间隙的总体均值看似接近（4.34对4.86px），中位数却是1.33对3.94px。S7混合了大量近乎贴合样本和一批间隙很大的wave样本，中等间隙覆盖不足。
- S7-H提高了断开频率，却仍然有长接缝、低平均间隙；它不是已经贴近真实分布的“困难集”。
- 对目前人工保留的真实集，极端面积比不再是最大的覆盖缺口。Turufan输入碎片显著更大，但部分原因来自既有归一化规则不同，不能解释为实物更大。

## 口径与边界

S7共24,000对、正负各12,000；S7-H是其中7,832对、正负各3,916，并非独立新增样本。敦煌保留295正例；另列筛选前508正例，避免人工剔除难例掩盖原始分布。Turufan301正例仅统计面积、周长、面积比；**遵照用户要求，待全部人工标注完成再统计Turufan接缝**。没有读取尚未完成的人工标注，也没有将模型预测当成GT。

所有px来自已有模型800×800输入坐标。仿真在800画布生成；敦煌使用原整幅画布的共同缩放；Turufan使用同前缀两PNG的最大边长共同缩放到800（并非分别缩放）。不同写卷没有统一物理扫描尺度。

**可测量与不可测量必须分开：**

- 真实腐蚀深度无法从现存mask恢复，因为缺少腐蚀前的边界；报告的gap是现存两侧轮廓间距，不等于丢失材料的深度。
- 以下“接近长度”是GT摆放下距离≤20px、法线大致相向的外轮廓长度，取两侧平均，**不是人工标出的真实接缝总长度**。不会读取Matcher输出。
- 间隙在≤40px的同类近接边界内逐点测距，再逐对等权汇总；未检测到近接带的样本不以0填充gap。S7可测11,993/12,000，S7-H3,916/3,916，敦煌保留291/295、原始489/508。长度统计保留未检出为0，所以包含方法无法提取的样本，而不代表其物理接缝不存在。
- 轮廓按1px弧长采样、sigma=3px平滑用于测量；mask本身、训练输入、标注页面均未修改。对栅格/法线波动只跨越≤3px的孔隙，剔除<8px的孤立段。
- “断开次数”是10px近接口径下、最短接触包络内部8–100px不支持弧段的计数，取两侧平均；是几何代理，不是古代撕裂次数。>100px的分離单列在逐例记录，不直接算腐蚀断口。
- 真实Partial seam的原始完整接缝不可见，无法可靠赋予“被删掉了多少”或精确占比。仿真可统计增强执行记录；真实只能另看接触覆盖比例/不对称性，不能将它冒充Partial seam标签。

## 1. 已有增强、实际生效比例

底层为原Rachel撕碎数据，含Gen2、Gen3、Gen4/Gen5等；S7正例的来源构成为：原生正例9,000（75%）、多碎片合并的小大碎片正例1,800（15%）、新增Gen5分组正例1,200（10%）。原生/合并来源与下表腐蚀槽不是互斥的两个维度，不应相加成总样本数。

{chr(10).join(ar)}

强腐蚀/裁切请求未通过连通性、材料保留或监督支持要求时回退原样；没有在重试时偷偷减小深度。正负样本耦合接受。wave回退371组、seam_gaps745组、partial_curve738组、local1组。新强损伤各类从干净源生成，**不是Partial+深腐蚀+缺口自动同时叠加**。

额外Partial增强实际为1,662/12,000=13.85%，不是配置中的20%；这是“额外曲线裁切”的比例，不是全部仿真样本的天然部分接触比例。生效后保留的原监督源弧段比例均值55.03%、中位57.27%、P10–P90为35.05%–71.78%。源弧段继承长度与本文密集轮廓近接长度是不同指标，不能混用。

接缝缺口的1–5是**每个被处理碎片**的数量；两侧都处理时，一对可有1–10个施加事件，实际均值3.00个。它不能直接和真实“几何断开数”对比，因为两侧缺口可能重合，且缺口形成的间隙可能仍低于近接阈值。

S7-H只选local2,398对、seam_gaps2,110对、partial_curve3,324对的已生效样本；没有wave和干净回退。镜像增强：v3/F/I训练约10%，S7-H约30%，两片及GT共同翻转，故面积、间隙、接缝长度不变；不额外将镜像算作独立样本。旧实验另实现过矩形/直边hard-negative，但当前S7清单未显示专门的矩形替换来源，不把“实现过”算成“本轮正在使用”。S7负例包括同图不邻接与跨来源不同面积/尺度干扰，具体数量见summary.json。

## 2. 几何与面积对比

表中均值按正例pair等权；每对两片的面积均值等价于正例端点出现次数加权。不是去重碎片均值。

{table([('正例数','area_ratio','n'),('平均近接长度 ≤20px','d20_length_px','mean'),('最长连续近接段均值 px','d20_longest_px','mean'),('每对平均间隙的均值 px','d40_gap_mean_px','mean'),('每对平均间隙的中位数 px','d40_gap_mean_px','median'),('平均几何断开数','d10_breaks_mean','mean'),('平均单片面积 px²','endpoint_area_px','mean'),('平均单片周长 px','endpoint_perimeter_px','mean'),('较小/较大片面积比中位数','area_ratio','median')])}

{chr(10).join(qlines)}

面积去重后：敦煌保留567个不同碎片，平均151,068.74px²、中位146,069px²；Turufan602个不同碎片，平均258,652.85px²、中位267,264px²。面积分布及正负分组详见distribution_quantiles.csv；这两个去重均值不和训练重复曝光次数加权均值混称。

## 3. 更有诊断价值的分布差异

| 比例 | S7 | S7-H | 敦煌保留 | 敦煌原始 | Turufan |
|---|---:|---:|---:|---:|---:|
| 面积比<1:4 | {percent(names[0],'area_ratio_lt_1_4')} | {percent(names[1],'area_ratio_lt_1_4')} | {percent(names[2],'area_ratio_lt_1_4')} | {percent(names[3],'area_ratio_lt_1_4')} | {percent(names[4],'area_ratio_lt_1_4')} |
| 面积比<1:8 | {percent(names[0],'area_ratio_lt_1_8')} | {percent(names[1],'area_ratio_lt_1_8')} | {percent(names[2],'area_ratio_lt_1_8')} | {percent(names[3],'area_ratio_lt_1_8')} | {percent(names[4],'area_ratio_lt_1_8')} |
| 近接长度<256px | {percent(names[0],'contact20_lt256')} | {percent(names[1],'contact20_lt256')} | {percent(names[2],'contact20_lt256')} | {percent(names[3],'contact20_lt256')} | 待标注 |
| 最长连续近接段<128px | {percent(names[0],'contact20_longest_lt128')} | {percent(names[1],'contact20_longest_lt128')} | {percent(names[2],'contact20_longest_lt128')} | {percent(names[3],'contact20_longest_lt128')} | 待标注 |
| 存在几何断开 | {percent(names[0],'contact10_has_break')} | {percent(names[1],'contact10_has_break')} | {percent(names[2],'contact10_has_break')} | {percent(names[3],'contact10_has_break')} | 待标注 |
| 平均gap落在2–10px | {percent(names[0],'gap40_mean_2_to_10')} | {percent(names[1],'gap40_mean_2_to_10')} | {percent(names[2],'gap40_mean_2_to_10')} | {percent(names[3],'gap40_mean_2_to_10')} | 待标注 |

**长且过于完整的接近带，比单纯的最大腐蚀深度更值得关注。**S7-H虽然断开率53.24%接近敦煌57.63%，其平均近接长度538.2px仍比敦煌380.8px长。S7-H每对gap的中位2.01px也低于3.94px。它的强腐蚀经常只影响短局部，其他长边仍基本吻合。只筛“确实被腐蚀过”并不能保证困难形态覆盖。

**Partial曲线裁切本身方向有效，但混合占比与联合形态仍不足。**已生效Partial子组平均近接长度395.4px、中位362.1px，已较接近保留敦煌；但只占全部正例13.85%，且不与强损伤叠加。这是下一轮可利用的信号，不是分类性能提升的证明。

**不能用平均值掩盖两头分布。**S7平均gap4.34px和敦煌4.86px相近，但S7中位仅1.33px，P90为18.07px；敦煌中位3.94px、P90为9.17px。S7更偏“很贴合 / 整圈较大间隙”，真实多处于中等间隙。换20/64px测距上限时，中等gap占比方向仍一致，但具体比例变化明显，因此这些值只是显式口径下的代理统计。

**面积悬殊不是当前保留集最主要缺口。**S7中<1:4占19.52%，已高于保留敦煌15.93%和Turufan11.63%；S7-H进一步达到24.59%。不建议不加区别地继续增加极小碎片。人工清单中仍有13对在现有mask下<1:8，本次忠实保留其标记，不自动更改筛选结果。

## 4. 下一轮数据方向（建议，尚未修改训练）

1. 由“抽中了哪种增强”转为检查增强后的**联合几何指标**：接触长度/全轮廓比例、最长连续段、平均间隙及分布、断开数量、面积比。满足实际目标区间后才计入对应增强配额，单独记录回退。
2. 提高实际生效的Partial比例，并补Partial+非均匀接缝内缩+多个缺口的组合；允许多个不连续但支持同一GT位移的短段。不能仅增加整圈30px内缩。
3. 保留现有10–30px局部峰值，同时增加更宽的中等间隙带及变化幅度，缩短剩余完美贴合段。具体配比需用开发集统计确定，当前不能从未完成的Turufan标注推定。
4. 检查短接缝正例的生成接受条件是否过度筛掉它们。当前至少32px保留源弧段、至少4个监督对应以及连通性限制是必要约束，但可能产生选择偏差；应改生成位置/宽度，而不是把无有效对应的样本伪装成可靠正例。
5. 将输入尺度也纳入采样：Turufan平均模型输入面积为S7约1.95倍、周长约1.31倍。可以在仿真中共同缩放两片，保持相对尺寸和GT；不要分别缩放碎片。这个建议尚未做训练验证。
6. 真实数据的统计用于设计配方后，这一分析属于开发性证据。应按写卷来源留出未参与配方设计的最终评估组，避免再用同一批案例宣称无偏泛化提升。

目前结论是“确有数据分布错配，且已定位到长接缝、间隙形态、实际增强生效率与输入尺度”，**不是已经排除了模型结构、监督或解码问题**。没有因本统计重启/新增训练或修改F/I配置。

## 文件与复现

- `distribution_analysis.ipynb`：已执行的计算与分布图；附图为描述性分布，不是置信区间。
- `distribution_quantiles.csv`、`distribution_rates.csv`：含均值、中位数、P10/P25/P75/P90和正负分组。
- `pair_metrics.csv`：逐对指标；Turufan接缝字段留空，绝不填0。
- `summary.json`、`sim_receipt.json`、`real_receipt.json`：完整汇总、源清单和口径。
- `measure.py`、`summarize.py`：测量与汇总代码。矩形已知间隙/交换对称/远离/内部断口测试通过；六类仿真GT轮廓叠图检查支持带位置。对真实GT没有重新优化或平移修正。

异常口径：敦煌保留4例无法提取40px相向带，其中2例最近边法线不相向，另2例相向边整体存在约6px/19px穿入；本次没有假造接缝或更改GT。完整逐例表保留这些记录，gap为空。自动几何代理并非语义接缝真值，尤其在GT穿入、角接触和缺失邻片时应谨慎。
'''
    (root/'RESULTS.md').write_text(report)
    notebook=nbf.v4.new_notebook()
    notebook.metadata['kernelspec']={'display_name':'Python 3','language':'python','name':'python3'}
    notebook.cells=[
        nbf.v4.new_markdown_cell('# 仿真—真实形态分布诊断\n## tl;dr\nS7接近边界更长，典型gap更小；S7-H提高断开率，但并未完全弥合差距。Turufan接缝等待用户完成标注，仅统计面积。详见同目录RESULTS.md。'),
        nbf.v4.new_markdown_cell('## Context & Methods\n### Key Assumptions\n- 基于原输入mask与敦煌GT位移，不使用模型候选或人工未完成标记。\n- 近接长度：≤20px相向边界的两侧平均；gap：≤40px近接带内的每对均值。几何代理不等于语义接缝或腐蚀深度。\n- 面积以800输入px²计；不同数据集共同缩放规则有区别。\n- S7-H为S7子集；Dunhuang full与retained也重叠，曲线不是独立试验组。\n- 数据源：sim_receipt.json、real_receipt.json以及RESULTS.md中的复现说明。'),
        nbf.v4.new_markdown_cell('## Data\n### 1. Load measured records'),
        nbf.v4.new_code_cell("%matplotlib inline\nfrom pathlib import Path\nimport json\nimport numpy as np\nimport pandas as pd\nimport matplotlib.pyplot as plt\nfrom IPython.display import display\nroot=Path.cwd()\nsummary=json.loads((root/'summary.json').read_text())\nplots=json.loads((root/'plot_data.json').read_text())\nassert summary['cohorts']['S7 positive']['pair_count']==12000\nassert summary['cohorts']['S7-H positive']['pair_count']==3916\nassert all(x['d20_length_px'] is None for x in plots['Turufan positive'])\nprint({k:len(v) for k,v in plots.items()})"),
        nbf.v4.new_markdown_cell('## Results\n### 2. Comparison table (means; area in input px²)\nGap missingness is retained; no-seam cases are not filled with zero gap.'),
        nbf.v4.new_code_cell("metrics={'Contact length ≤20px':'d20_length_px','Longest contact segment':'d20_longest_px','Mean gap ≤40px':'d40_gap_mean_px','Break count proxy':'d10_breaks_mean','Fragment area':'endpoint_area_px'}\nrows=[]\nfor name in plots:\n c=summary['cohorts'][name]\n rows.append({'Dataset':name,'Pairs':c['pair_count'],**{k:c['metrics'][v].get('mean') for k,v in metrics.items()}})\ndisplay(pd.DataFrame(rows).set_index('Dataset').round(2))"),
        nbf.v4.new_markdown_cell('### 3. Distribution, not only averages\nThe cumulative curves use all measured pairs. A curve further left means shorter/smaller values; missing gap observations are excluded only from the gap curve.'),
        nbf.v4.new_code_cell("plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':False})\ncolors={'S7 positive':'#2867A0','S7-H positive':'#B36A13','Dunhuang retained':'#734F8D','Dunhuang full':'#6B747D','Turufan positive':'#727A24'}\nstyles={'S7 positive':'-','S7-H positive':'--','Dunhuang retained':'-','Dunhuang full':':','Turufan positive':'-.'}\nfig,axes=plt.subplots(2,2,figsize=(13,9),layout='constrained')\nspecs=[('d20_length_px','Near-contact length (px; d≤20)','GT-aligned near-contact length'),('d40_gap_mean_px','Per-pair mean gap (px; d≤40)','GT-aligned boundary gap'),('d10_breaks_mean','Internal break count proxy','Fragmented contact support'),('mean_fragment_area_px','Mean fragment area per pair (input px²)','Fragment size in model inputs')]\nfor ax,(key,label,title) in zip(axes.flat,specs):\n for name,rows in plots.items():\n  values=np.sort([r[key] for r in rows if r.get(key) is not None])\n  if not len(values):continue\n  ax.step(values,100*np.arange(1,len(values)+1)/len(values),where='post',color=colors[name],ls=styles[name],lw=1.8,label=f'{name} (n={len(values):,})')\n ax.set(xlabel=label,ylabel='Cumulative share (%)',title=title,xlim=(0,None),ylim=(0,100))\n ax.grid(axis='y',color='#D9DDE1',lw=.6);ax.legend(loc='lower right',fontsize=8.5,frameon=False)\nfig.savefig(root/'distributions.png',dpi=160)\nplt.show()"),
        nbf.v4.new_markdown_cell('S7的gap均值接近敦煌，但中位数与尾部明显不同。S7-H的断开数接近真实，但接近长度仍偏长。面积图包含Turufan；其余图按用户要求不包含Turufan。'),
        nbf.v4.new_markdown_cell('### 4. Actual vs requested augmentation\nOnly positive pair slots are counted here. Gen5 is structural augmentation; its zero erosion flag is not a failed augmentation.'),
        nbf.v4.new_code_cell("aug=[a for a in summary['augmentation'] if a['label']]\ndisplay(pd.DataFrame([{'Recipe':a['recipe'],'Requested':a['requested'],'Applied damage':a['changed'],'Actual share of positives':a['fraction_of_all_12k'],'Mean max measured erosion px':a['measured_max_depth'].get('mean')} for a in aug]).round(3))\nrecipes=[a for a in aug if a['recipe']!='gen5_partition']\nfig,ax=plt.subplots(figsize=(9,4),layout='constrained')\ny=np.arange(len(recipes));ax.barh(y-.18,[a['requested']/120 for a in recipes],height=.34,label='Requested slots',color='#A7B8C8');ax.barh(y+.18,[a['changed']/120 for a in recipes],height=.34,label='Actually changed',color='#2867A0')\nax.set(yticks=y,yticklabels=[a['recipe'] for a in recipes],xlabel='Share of 12,000 positive pairs (%)',xlim=(0,33),title='S7 requested vs actual damage')\nax.invert_yaxis();ax.legend(frameon=False);fig.savefig(root/'augmentation_coverage.png',dpi=160);plt.show()"),
        nbf.v4.new_markdown_cell('## Takeaways\n1. 新数据应关注剩余近接长度、典型gap及断开的联合分布，不只看最大腐蚀深度。\n2. 实际生效Partial子组已经更接近敦煌长度，但只占13.85%额外裁切；可以优先验证提高其实际占比、与缺口叠加。\n3. 真实Partial率与腐蚀前损失深度不可由现有mask唯一恢复；不填虚构数字。\n4. 这是分布错配证据，不是模型结构无误的因果证明。\n5. Turufan接缝统计待用户确认标注完成。')]
    path=root/'distribution_analysis.ipynb';nbf.write(notebook,path)
    client=NotebookClient(notebook,timeout=120,kernel_name='python3',resources={'metadata':{'path':str(root)}})
    client.execute();nbf.validate(notebook);nbf.write(notebook,path)
    source_rows=[dict(dataset=zh[n],positive_pairs=cs[n]['pair_count'],
        mean_contact_length_px=m(n,'d20_length_px'),median_pair_gap_px=m(n,'d40_gap_mean_px','median'),
        mean_breaks=m(n,'d10_breaks_mean'),mean_area_px2=m(n,'endpoint_area_px')) for n in names]
    source=dict(schemaVersion=1,items=[dict(id='seam-distribution',title='仿真与真实的形态分布',queries=[dict(id='measured-distributions',
        source=dict(label='原mask与GT位移的几何统计',files=[dict(label='summary.json'),dict(label='distribution_quantiles.csv')],
            metricDefinitions=[dict(label='近接长度',definition='GT摆放下距离≤20px且法线相向的边界长度，取两侧平均；不是人工接缝标签。'),
                dict(label='间隙',definition='≤40px近接带内先求每对平均间距，再汇总分布；不能解释为原始腐蚀深度。')],
            filters=['接缝只统计正例','Turufan接缝等待人工标注完成'],
            caveats=['S7-H嵌套于S7，敦煌保留集嵌套于原始集。','各数据均为模型输入像素，缩放规则不同。','缺少近接带的gap留空，不按0计。']),
        rows=source_rows,columns=list(source_rows[0]),preview=dict(kind='aggregate',note='全量测量的分组汇总。'))]),
        dict(id='augmentation-coverage',title='S7增强实际生效情况',queries=[dict(id='augmentation-receipts',
            source=dict(label='S7物化样本与增强记录',files=[dict(label='train_s7_24k.json'),dict(label='report_json')],
                caveats=['Gen5是结构增强，未再次腐蚀不能当作增强失败。','Partial裁切生效比例不是所有天然部分接触的总比例。']),
            rows=[dict(recipe=a['recipe'],requested_positive_slots=a['requested'],actual_damaged_positive_pairs=a['changed']) for a in s['augmentation'] if a['label']],
            columns=['recipe','requested_positive_slots','actual_damaged_positive_pairs'])])])
    (root/'inline_sources.json').write_text(json.dumps(source,ensure_ascii=False,indent=2,allow_nan=False))
    print(path)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);run(p.parse_args().root)
