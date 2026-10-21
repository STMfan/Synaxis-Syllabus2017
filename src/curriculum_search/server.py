"""Small local HTTP API. One request thread avoids sharing a mutable tokenizer."""

import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, unquote, urlsplit

from .search import CurriculumSearch


def make_handler(search: CurriculumSearch):
    class Handler(BaseHTTPRequestHandler):
        def respond(self, status, value):
            body = json.dumps(value, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            parsed = urlsplit(self.path)
            if parsed.path == "/health":
                return self.respond(200, {"status": "ok", "subjects": search.subjects,
                                          "vectors_available": (search.data_dir / "vectors.json").is_file()})
            if parsed.path.startswith("/chunks/"):
                result = search.get(unquote(parsed.path.removeprefix("/chunks/")))
                return self.respond(200 if result else 404, result or {"error": "Chunk not found"})
            if parsed.path.startswith("/sections/"):
                result = search.get_section(unquote(parsed.path.removeprefix("/sections/")))
                return self.respond(200 if result else 404, result or {"error": "Section not found"})
            if parsed.path == "/search":
                query = {k: v[0] for k, v in parse_qs(parsed.query).items()}
                return self.run_search(query)
            return self.respond(404, {"error": "Use /health, /search, /chunks/{id}, /sections/{id}"})

        def do_POST(self):
            if urlsplit(self.path).path != "/search":
                return self.respond(404, {"error": "Route not found"})
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 65536:
                    raise ValueError("Request body must contain 1..65536 bytes")
                data = json.loads(self.rfile.read(length))
            except (ValueError, UnicodeDecodeError) as exc:
                return self.respond(400, {"error": str(exc)})
            return self.run_search(data)

        def run_search(self, data):
            try:
                if not isinstance(data, dict):
                    raise ValueError("Expected JSON object")
                results = search.search(data.get("query", ""), top_k=int(data.get("top_k", 5)),
                                        subject=data.get("subject"), course_category=data.get("course_category"),
                                        mode=data.get("mode", "hybrid"))
                self.respond(200, {"count": len(results), "results": results})
            except (ValueError, TypeError) as exc:
                self.respond(400, {"error": str(exc)})
            except (FileNotFoundError, ImportError) as exc:
                self.respond(503, {"error": str(exc)})
            except Exception as exc:
                self.respond(500, {"error": type(exc).__name__ + ": " + str(exc)})

    return Handler


def serve(search, host="127.0.0.1", port=8765):
    server = HTTPServer((host, port), make_handler(search))
    print(json.dumps({"listening": f"http://{host}:{port}"}), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
