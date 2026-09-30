"""One-time user-requested report status change; retain old evidence."""
import argparse,json
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--app',required=True);a=p.parse_args();f=Path(a.app)/'src/data.json'
    d=json.loads(f.read_text());d.update(status='superseded-awaiting-curved-revision',buildStatus='creating',
        reviewStatus='旧直线裁短小试不再用于扩量；正在按70/30裁切侧、20%接缝、两侧类型均≤20%面积及真实写卷曲线重新生成')
    f.write_text(json.dumps(d,ensure_ascii=False)+'\n')

if __name__=='__main__':main()
