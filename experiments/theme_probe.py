"""Weak-label theme-alignment proxy: separate TF-IDF logistic classifier.

Train on the training-split theme labels, check its held-out ability to identify
at least one weak theme, then score generated lines from each GRU. A limited,
weak-supervision proxy only; NOT an independent human judgment.
Usage: python experiments/theme_probe.py --data-dir data
"""
from pathlib import Path
import argparse
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from offline_model import load_checkpoint, load_rows, generate


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',type=Path,default=Path('data'))
    p.add_argument('--checkpoints',type=Path,nargs='+',default=[Path('checkpoints/offline_best.pt'),Path('experiments/unconditional_best.pt')])
    p.add_argument('--count-per-theme',type=int,default=20)
    p.add_argument('--out',type=Path,default=Path('experiments/theme_probe.json'))
    a=p.parse_args()
    torch.set_num_threads(2)
    train=load_rows(a.data_dir,'train')
    test=load_rows(a.data_dir,'test')
    mlb=MultiLabelBinarizer()
    Y=mlb.fit_transform([r['themes'] for r in train])
    X=[r['text'] for r in train]
    vectorizer=TfidfVectorizer(ngram_range=(1,2),min_df=2,max_features=12000,sublinear_tf=True)
    Xt=vectorizer.fit_transform(X)
    classifier=OneVsRestClassifier(LogisticRegression(max_iter=250,random_state=13))
    classifier.fit(Xt,Y)
    probs=classifier.predict_proba(vectorizer.transform([r['text'] for r in test]))
    pred=probs.argmax(1)
    idx={theme:i for i,theme in enumerate(mlb.classes_)}
    test_hits=[int(mlb.classes_[p] in row['themes']) for p,row in zip(pred,test)]
    report={'judge':'TF-IDF (1-2 grams) + one-vs-rest logistic regression',
            'judge_training_lines':len(train),'judge_heldout_test_hit_at_1':round(float(np.mean(test_hits)),4),
            'num_themes':len(mlb.classes_),'samples_per_theme':a.count_per_theme,'models':{},
            'limitation':'Judge learns from the same noisy LLM-generated labels. Theme top-1 is a weak proxy, not independent human evaluation.'}
    for checkpoint in a.checkpoints:
        model,ck=load_checkpoint(checkpoint)
        strings=[];requested=[]
        for ti,theme in enumerate(ck['themes']):
            for n in range(a.count_per_theme):
                strings.append(generate(model,ck,theme,temperature=0.9,seed=1000+ti*10000+n*13))
                requested.append(theme)
        scores=classifier.predict_proba(vectorizer.transform(strings))
        ordered=np.argsort(-scores,axis=1)
        hit=np.mean([int(ordered[i,0]==idx[t]) for i,t in enumerate(requested)])
        mrr=np.mean([1.0/(np.where(ordered[i]==idx[t])[0][0]+1) for i,t in enumerate(requested)])
        tokens=[w.lower() for text in strings for w in text.split()]
        distinct1=len(set(tokens))/max(len(tokens),1)
        report['models'][checkpoint.stem]={'theme_hit_at_1':round(float(hit),4),'theme_mrr':round(float(mrr),4),
             'distinct_1':round(float(distinct1),4),'sample_count':len(strings),'unique_lines':len(set(strings))}
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
