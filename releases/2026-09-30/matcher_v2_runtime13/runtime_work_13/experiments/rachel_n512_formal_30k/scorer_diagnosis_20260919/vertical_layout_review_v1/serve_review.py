"""Loopback-only review UI and independent atomic, versioned manual labels."""
import argparse,json,os,tempfile,threading
from pathlib import Path
from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
from urllib.parse import urlsplit
SCHEMA='assembly-review-v1'
def key_for(r):return json.dumps([r['dataset'],r['pair_id']],ensure_ascii=False,separators=(',',':'))
class ReviewStore:
    def __init__(self,path,snapshot):
        self.path=Path(path);self.lock=threading.Lock();d=json.loads(Path(snapshot).read_text())
        self.cases={key_for(c):c for c in d['queries']['cases']['rows']};self.model=d['queries']['model']['rows'][0]
    def read(self):
        if not self.path.exists():return dict(schema=SCHEMA,model=self.model,records={})
        d=json.loads(self.path.read_text())
        if d.get('schema')!=SCHEMA or d.get('model')!=self.model or not isinstance(d.get('records'),dict):raise ValueError('Existing labels do not match snapshot; never overwrite')
        for k,r in d['records'].items():
            if k not in self.cases or r.get('fingerprint')!=self.cases[k]['fingerprint']:raise ValueError('Saved review identity differs')
        return d
    def save(self,payload):
        try:
            r=payload['record'];key=key_for(r);c=self.cases[key];expected=payload['expected_revision']
            if (r['schema']!=SCHEMA or r['checkpoint_sha256']!=c['checkpoint_sha256'] or r['fingerprint']!=c['fingerprint']
                or r['verdict'] not in ('success','failure',None) or type(expected) is not int or expected<0
                or not isinstance(r['updated_at'],str) or len(r['updated_at'])>40):raise ValueError()
        except (KeyError,TypeError,ValueError):return 400,{'error':'Invalid case, model, fingerprint, or review'}
        with self.lock:
            d=self.read();old=d['records'].get(key,{})
            if old.get('revision',0)!=expected:return 409,{'error':'Concurrent review changed','record':old}
            result={k:r[k] for k in ('schema','dataset','pair_id','checkpoint_sha256','fingerprint','verdict','updated_at')}
            result.update(revision=expected+1,case_name=c['case_name']);d['records'][key]=result;self.path.parent.mkdir(parents=True,exist_ok=True)
            fd,name=tempfile.mkstemp(prefix='.assembly-review-',suffix='.tmp',dir=self.path.parent)
            try:
                with os.fdopen(fd,'w') as f:json.dump(d,f,ensure_ascii=False,indent=2);f.flush();os.fsync(f.fileno())
                os.replace(name,self.path)
            finally:
                if os.path.exists(name):os.unlink(name)
            return 200,{'record':result}
def handler(directory,store):
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self,*a,**kw):super().__init__(*a,directory=str(directory),**kw)
        def reply(self,code,data):
            c=json.dumps(data,ensure_ascii=False).encode();self.send_response(code);self.send_header('Content-Type','application/json; charset=utf-8');self.send_header('Content-Length',str(len(c)));self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(c)
        def valid_host(self):return self.headers.get('Host') in ('127.0.0.1:'+str(self.server.server_port),'localhost:'+str(self.server.server_port))
        def do_GET(self):
            if not self.valid_host():return self.reply(403,{'error':'Loopback host only'})
            if urlsplit(self.path).path=='/api/assembly-reviews':
                with store.lock:self.reply(200,store.read())
            else:super().do_GET()
        def do_POST(self):
            if not self.valid_host():return self.reply(403,{'error':'Loopback host only'})
            if urlsplit(self.path).path!='/api/assembly-reviews':return self.reply(404,{'error':'Unknown route'})
            origin=self.headers.get('Origin')
            if origin and origin!='http://'+self.headers.get('Host',''):return self.reply(403,{'error':'Same origin only'})
            if self.headers.get('Content-Type','').split(';')[0]!='application/json':return self.reply(415,{'error':'JSON required'})
            try:
                size=int(self.headers.get('Content-Length','0'))
                if not 0<size<=8192:raise ValueError()
                code,body=store.save(json.loads(self.rfile.read(size)))
            except (ValueError,TypeError):return self.reply(400,{'error':'Invalid payload'})
            self.reply(code,body)
    return Handler
def main():
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--store',type=Path,required=True);p.add_argument('--port',type=int,default=8783);a=p.parse_args()
    store=ReviewStore(a.store,a.project/'src/data.json');store.read();server=ThreadingHTTPServer(('127.0.0.1',a.port),handler(a.project/'dist',store))
    print('Assembly review: http://127.0.0.1:'+str(a.port),flush=True);server.serve_forever()
if __name__=='__main__':main()
