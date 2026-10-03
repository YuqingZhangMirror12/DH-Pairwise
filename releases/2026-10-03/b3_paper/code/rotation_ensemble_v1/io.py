import hashlib
import json
import os
from pathlib import Path


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def digest(obj):return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def read(path):return json.loads(Path(path).read_text())


def ref(path):return dict(path=str(Path(path).resolve()),sha256=sha(path))


def checked(record):
    if sha(record['path'])!=record['sha256']:raise ValueError('bound file changed: '+record['path'])
    return read(record['path'])


def save(path,obj,replace=False):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if not replace and path.exists():raise ValueError('refuse overwrite: '+str(path))
    temporary=path.with_suffix(path.suffix+'.tmp')
    with temporary.open('x') as f:
        json.dump(obj,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    temporary.replace(path)


def loaded_sources(root,prefix):
    import sys
    out={}
    for name,m in list(sys.modules.items()):
        if name.startswith(prefix) or name.startswith('staging.pairwise_v0_2'):
            p=getattr(m,'__file__',None)
            if p:
                path=Path(p).resolve()
                if Path(root).resolve() not in path.parents:raise ValueError('unfrozen native dependency: '+name)
                out[str(path)]=sha(path)
    return out
