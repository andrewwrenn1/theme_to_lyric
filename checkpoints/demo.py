"""Dependency-light, local browser GUI for the theme-to-lyric project.

Launch: python demo.py ; visit http://127.0.0.1:8501
The included small learned GRU checkpoint works offline. Optionally use a local
FLAN-T5 fine-tuned checkpoint produced by theme_to_lyric.py train.
"""
from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading

from offline_model import generate, load_checkpoint

ROOT = Path(__file__).resolve().parent
HTML = ROOT / 'web' / 'index.html'


def make_server(host: str, port: int, checkpoint: Path, t5_checkpoint: Path):
    import torch
    torch.set_num_threads(max(1, min(2, torch.get_num_threads())))
    model, bundle = load_checkpoint(checkpoint, 'cpu')
    lock = threading.Lock()
    t5_state = {'model': None, 'tokenizer': None}

    class Handler(BaseHTTPRequestHandler):
        def reply(self, status, payload):
            raw = json.dumps(payload, ensure_ascii=False).encode('utf-8')
            self.send_response(status)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(raw)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            if self.path in ('/', '/index.html'):
                raw = HTML.read_bytes()
                self.send_response(200)
                self.send_header('Content-Type','text/html; charset=utf-8')
                self.send_header('Content-Length',str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
            elif self.path == '/api/info':
                self.reply(200, {'themes':bundle['themes'], 'model':'Offline trained GRU',
                                 'model_ready':True, 't5_available':t5_checkpoint.is_dir(),
                                 'trained_epoch':bundle['epoch']})
            elif self.path == '/favicon.ico':
                self.send_response(204)
                self.end_headers()
            else:
                self.reply(404, {'error': 'Not found'})

        def do_POST(self):
            if self.path != '/api/generate':
                return self.reply(404, {'error': 'Not found'})
            length = int(self.headers.get('Content-Length','0'))
            if length < 1 or length > 4096:
                return self.reply(400, {'error':'Invalid request length'})
            try:
                body = json.loads(self.rfile.read(length))
                theme = str(body.get('theme','love'))
                count = int(body.get('count', 3))
                temperature = float(body.get('temperature', 0.9))
                seed = int(body.get('seed', 13))
                engine = str(body.get('engine','offline'))
                if theme not in bundle['themes']:
                    raise ValueError('Select a listed theme')
                if not (1 <= count <= 5 and 0.5 <= temperature <= 1.5 and 0 <= seed <= 2_147_483_647):
                    raise ValueError('Choose 1-5 lines, creativity 0.5-1.5, and a valid seed')
                with lock:
                    if engine == 'offline':
                        lines = [generate(model, bundle, theme, temperature=temperature, seed=seed+i*83)
                                 for i in range(count)]
                        used = 'Offline trained GRU'
                    elif engine == 't5':
                        if not t5_checkpoint.is_dir():
                            raise ValueError('Fine-tuned FLAN-T5 weights missing. Train first, then restart the demo.')
                        if t5_state['model'] is None:
                            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
                            t5_state['tokenizer'] = AutoTokenizer.from_pretrained(str(t5_checkpoint))
                            t5_state['model'] = AutoModelForSeq2SeqLM.from_pretrained(str(t5_checkpoint)).eval()
                        tok, tuned = t5_state['tokenizer'], t5_state['model']
                        with torch.inference_mode():
                            enc = tok(f'theme: {theme}', return_tensors='pt')
                            torch.manual_seed(seed)
                            ids = tuned.generate(**enc, do_sample=True, temperature=temperature,
                                                 top_p=0.9, max_new_tokens=40, num_return_sequences=count)
                            lines = [line.strip() for line in tok.batch_decode(ids, skip_special_tokens=True)]
                        used = 'Fine-tuned FLAN-T5 checkpoint'
                    else:
                        raise ValueError('Unknown model')
                # Refuse exact training-line reproduction when local user data is installed.
                data = ROOT / 'data' / 'train.jsonl'
                if data.is_file():
                    import hashlib
                    # Exact-match fingerprints only: do not display / log / distribute source lines.
                    with data.open(encoding='utf-8') as source_file:
                        source_hashes = {hashlib.sha256(json.loads(s)['text'].strip().lower().encode()).digest()
                                         for s in source_file if s.strip()}
                    lines = ['[Exact training-set match omitted; regenerate for a new line]'
                             if hashlib.sha256(line.strip().lower().encode()).digest() in source_hashes else line
                             for line in lines]
                self.reply(200, {'lines':lines, 'theme':theme, 'engine':used})
            except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
                self.reply(400, {'error':str(exc)})
            except Exception as exc:
                # Non-technical error, with diagnostic on the server console only.
                print('Generation failed:',repr(exc))
                self.reply(500, {'error':'Generation failed. See the terminal for details.'})

        def log_message(self, fmt, *args):
            print('[demo] ' + fmt % args)

    return ThreadingHTTPServer((host, port), Handler)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--port', type=int, default=8501)
    p.add_argument('--checkpoint', type=Path, default=ROOT/'checkpoints/offline_best.pt')
    p.add_argument('--t5-checkpoint', type=Path, default=ROOT/'runs/flan-t5-small/best')
    args = p.parse_args()
    server = make_server(args.host, args.port, args.checkpoint, args.t5_checkpoint)
    print(f'Open http://{args.host}:{args.port} in your browser (Ctrl+C to stop)', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
