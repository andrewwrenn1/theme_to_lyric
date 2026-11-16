"""Run from project root with python -m unittest discover -s tests -v."""
import json
import threading
import unittest
from pathlib import Path
from urllib.request import urlopen, Request

from offline_model import tokenize, detokenize, load_checkpoint, generate, encode_pair
from demo import make_server

ROOT=Path(__file__).resolve().parents[1]

class CoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model,cls.ck=load_checkpoint(ROOT/'checkpoints/offline_best.pt')

    def test_vocab_and_shapes(self):
        v=self.ck['vocab']
        src,tgt=encode_pair('love','A fresh original example.',v,16,False)
        self.assertEqual(len(src),len(tgt))
        self.assertEqual(src[1],v['<theme_love>'])
        self.assertEqual(tgt[0],v['<pad>'])

    def test_tokenization(self):
        self.assertEqual(tokenize('Light, and night!'), ['light',',','and','night','!'])
        self.assertEqual(detokenize(['lights',',','rise','.']), 'Lights, rise.')

    def test_model_inference(self):
        text=generate(self.model,self.ck,'love',seed=2026)
        self.assertIsInstance(text,str)
        self.assertGreater(len(text.split()),2)
        self.assertEqual(text, generate(self.model,self.ck,'love',seed=2026))

    def test_unsupported_theme(self):
        with self.assertRaises(ValueError):
            generate(self.model,self.ck,'not-a-theme')

    def test_metadata(self):
        obj=json.loads((ROOT/'data/meta.json').read_text())
        self.assertEqual(obj['split_counts'],{'train':10000,'val':1000,'test':1000})
        self.assertEqual(len(obj['themes']),10)

class DemoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server=make_server('127.0.0.1',0,ROOT/'checkpoints/offline_best.pt',ROOT/'runs/none/best')
        cls.port=cls.server.server_address[1]
        cls.thread=threading.Thread(target=cls.server.serve_forever,daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=3)

    def test_page(self):
        with urlopen(f'http://127.0.0.1:{self.port}/') as response:
            html=response.read().decode('utf-8')
            self.assertIn('Theme to Lyric',html)
            self.assertIn('Generate lyric ideas',html)

    def test_api(self):
        url=f'http://127.0.0.1:{self.port}'
        with urlopen(url+'/api/info') as resp:
            meta=json.load(resp)
            self.assertTrue(meta['model_ready'])
        req=Request(url+'/api/generate', data=json.dumps({'theme':'love','count':2,'seed':1223}).encode(),
                    headers={'Content-Type':'application/json'},method='POST')
        with urlopen(req) as resp:
            data=json.load(resp)
            self.assertEqual(len(data['lines']),2)
            self.assertTrue(all(isinstance(s,str) for s in data['lines']))

if __name__=='__main__':unittest.main()
