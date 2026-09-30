"""Loopback-only static report plus independent human layout annotations.

Does not modify predictions, GT, training, or the report's protected runtime.
The report remains usable offline via browser backup and CSV export.
"""
import argparse
import json
import math
import os
from pathlib import Path
import tempfile
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

MODELS = ("S7_M12_matched_C16", "v3_B22", "S7_H")
SCHEMA = "layout-human-v1"


def key_for(record):
    return json.dumps([record["dataset"], record["pair_id"]], ensure_ascii=False, separators=(",", ":"))


class ReviewStore:
    def __init__(self, path, snapshot):
        self.path = Path(path)
        self.lock = threading.Lock()
        data = json.loads(Path(snapshot).read_text())
        self.cases = {key_for(c): c for c in data["queries"]["cases"]["rows"]}

    def read(self):
        if not self.path.exists():
            return {"schema": SCHEMA, "records": {}}
        return json.loads(self.path.read_text())

    def save(self, payload):
        record = payload.get("record", {})
        try:
            key = key_for(record)
            c = self.cases[key]
            selected = record["successful_models"]
            assert record["schema"] == SCHEMA and record["reviewed"] is True
            assert isinstance(selected, list) and len(selected) == len(set(selected))
            assert all(x in MODELS for x in selected)
            assert isinstance(record["fingerprint"], str) and len(record["fingerprint"]) < 5000
            assert isinstance(record["updated_at"], str) and len(record["updated_at"]) <= 40
            expected = payload["expected_revision"]
            assert type(expected) is int and expected >= 0
            for model in selected:
                t = c["models"].get(model, {}).get("translation")
                assert isinstance(t, list) and len(t) == 2 and all(isinstance(v, (int, float)) and math.isfinite(v) for v in t)
        except (KeyError, TypeError, AssertionError):
            return 400, {"error": "Invalid case or review payload"}
        with self.lock:
            data = self.read()
            old = data["records"].get(key)
            if expected != (old or {}).get("revision", 0):
                return 409, {"record": old}
            saved = {field: record[field] for field in ("schema", "dataset", "pair_id", "reviewed", "successful_models", "fingerprint", "updated_at")}
            saved.update(case_name=c["case_name"], label=int(c["label"]), revision=expected + 1)
            data["records"][key] = saved
            self.path.parent.mkdir(parents=True, exist_ok=True)
            fd, name = tempfile.mkstemp(dir=self.path.parent, prefix=".annotations-", suffix=".tmp")
            try:
                with os.fdopen(fd, "w") as out:
                    json.dump(data, out, ensure_ascii=False, indent=2)
                    out.flush()
                    os.fsync(out.fileno())
                os.replace(name, self.path)
            finally:
                if os.path.exists(name):
                    os.unlink(name)
            return 200, {"record": saved}


def make_handler(directory, store):
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(directory), **kwargs)

        def json_response(self, code, body):
            content = json.dumps(body, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(content)

        def do_GET(self):
            if urlsplit(self.path).path == "/api/layout-reviews":
                with store.lock:
                    self.json_response(200, store.read())
            else:
                super().do_GET()

        def do_POST(self):
            if urlsplit(self.path).path != "/api/layout-reviews":
                return self.json_response(404, {"error": "Unknown route"})
            origin = self.headers.get("Origin")
            if origin and origin != "http://" + self.headers.get("Host", ""):
                return self.json_response(403, {"error": "Same-origin requests only"})
            if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                return self.json_response(415, {"error": "JSON required"})
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 16384:
                    raise ValueError()
                payload = json.loads(self.rfile.read(size))
                if not isinstance(payload, dict):
                    raise ValueError()
                code, body = store.save(payload)
            except (ValueError, TypeError):
                return self.json_response(400, {"error": "Invalid JSON"})
            self.json_response(code, body)

    return Handler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8774)
    args = parser.parse_args()
    store = ReviewStore(args.store, args.project / "src/data.json")
    # Reject a damaged existing file rather than replace user annotations.
    store.read()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(args.project / "dist", store))
    print(f"Layout review listening on http://127.0.0.1:{args.port}/", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
