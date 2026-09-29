"""Keep the old preview inspectable but withdraw its endpoint acceptance."""
import argparse,json
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('--app',required=True);a=p.parse_args()
path=Path(a.app)/'src/data.json';d=json.loads(path.read_text())
d['buildStatus']='updating';d['status']='curved-preview-endpoint-correction'
d['reviewStatus']='旧预览的端部约束未通过，已停止该批CPU生成；正在补做仅从一端／两端裁短的新小试'
d['curveRevision']['endpoint_only_verified']=False
d['queries']['pilot_cases']['source']['notes'].append('后续端部审计发现01111_0移除了中间接缝；旧逐像素审计未覆盖此语义，不能据此扩量。')
path.write_text(json.dumps(d,ensure_ascii=False)+'\n')
