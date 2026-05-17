#!/usr/bin/env python3
"""
Teacher generation orchestrator: retrieves context locally, sends to Modal 7B.

Usage:
  source /opt/homebrew/var/mtplx/venv-0.1.0rc3/bin/activate
  python3 scripts/teacher_gen.py --target 50 --out .repo-arch/training-data/teacher7b

Generates training targets using Qwen2.5-Coder-7B via Modal.
Each example: {cards + commits context, question, 7B answer}
"""

import argparse
import glob
import json
import math
import os
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict

random = __import__("random")
random.seed(42)
REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


# ── BM25 (same as hybrid_search) ────────────────────────────────


class BM25:
    def __init__(self, k1=1.5, b=0.75):
        self.k1=k1;self.b=b;self.doc=[];self.d2i={};self.i2d={};self.avgdl=0;self.N=0;self.df=Counter();self.tid=defaultdict(set)
    def tok(self,t): return re.findall(r'[a-zA-Z][a-zA-Z0-9_\-\./]{2,}',t.lower())
    def add(self,id_,text):
        i=len(self.doc);self.d2i[id_]=i;self.i2d[i]=id_
        t=self.tok(text);self.doc.append(t)
        for x in set(t): self.df[x]+=1;self.tid[x].add(i)
        self.N=len(self.doc);self.avgdl=sum(len(d)for d in self.doc)/max(1,self.N)
    def idf(self,t): n=self.df.get(t,0);return 0 if n==0 else math.log((self.N-n+0.5)/(n+0.5)+1)
    def search_all(self,q,k=200):
        t=self.tok(q);s=defaultdict(float)
        for x in t:
            i=self.idf(x)
            if i==0: continue
            for d in self.tid.get(x,set()):
                dl=len(self.doc[d]);tf=self.doc[d].count(x)
                denom=tf+self.k1*(1-self.b+self.b*dl/self.avgdl)
                s[d]+=i*tf*(self.k1+1)/max(denom,0.001)
        return {self.i2d.get(i,''): sc for i,sc in s.items()}


# ── Cosine + RRF ─────────────────────────────────────────────────


def cosine(a,b):
    dot=sum(x*y for x,y in zip(a,b))
    na=sum(x*x for x in a)**0.5;nb=sum(x*x for x in b)**0.5
    return dot/(na*nb+1e-10)


def rrf(bm25_s,vec_s,k=60,top=20):
    c=defaultdict(float)
    for i,(d,_) in enumerate(sorted(bm25_s.items(),key=lambda x:-x[1])): c[d]+=1/(k+i)
    for i,(d,_) in enumerate(sorted(vec_s.items(),key=lambda x:-x[1])): c[d]+=1/(k+i)
    return sorted(c.items(),key=lambda x:-x[1])[:top]


# ── Load indexes ─────────────────────────────────────────────────


def load_all(repo_dir):
    # Cards
    cfiles=glob.glob(os.path.join(repo_dir,'.repo-arch','cache','cards','*.json'))
    cards=[];cbm=BM25()
    if cfiles:
        with open(cfiles[0]) as f: d=json.load(f)
        cards=d.get('cards',[])
        rev=json.load(open(os.path.join(repo_dir,'.repo-arch','review-state.json')))
        for c in cards:
            c['status']=rev.get(c.get('id',''),{}).get('status','unreviewed')
            cbm.add(c['id'],f"{c.get('title','')} {c.get('suggestion','')} {' '.join(c.get('affectedFiles',[]))}")

    # Commits + hybrid
    idx_dir=os.path.join(repo_dir,'.repo-arch','index','hybrid')
    commits=[];co_bm=BM25();embeddings=None;doc_map=None
    doc_path=os.path.join(idx_dir,'docs.json')
    if os.path.exists(doc_path):
        with open(doc_path) as f: doc_list=json.load(f)['documents']
        for d in doc_list:
            co_bm.add(d['id'],f"{d.get('subject','')} {' '.join(d.get('paths',[]))} {' '.join(d.get('packages',[]))}")
        emb_path=os.path.join(idx_dir,'embeddings.npy')
        if os.path.exists(emb_path):
            import numpy as np
            embeddings=np.load(emb_path)
            with open(os.path.join(idx_dir,'doc_map.json')) as f: doc_map=json.load(f)
        commits=doc_list
    return cards,cbm,commits,co_bm,embeddings,doc_map


# ── Build context ────────────────────────────────────────────────


_EMBED_MODEL=None
def embed(q):
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        from sentence_transformers import SentenceTransformer
        _EMBED_MODEL=SentenceTransformer('all-MiniLM-L6-v2');_EMBED_MODEL.to('cpu')
    return _EMBED_MODEL.encode([q])[0]


def build_context(question,cards,cbm,commits,co_bm,embeddings,doc_map,max_c=4,max_k=10):
    parts=["--- REPO-ARCH CARDS ---"]
    cr=cbm.search_all(question,k=max_c*3) if cbm else {}
    seen_t=set()
    for cid,sc in sorted(cr.items(),key=lambda x:-x[1]):
        c=next((c for c in cards if c['id']==cid),None) if cards else None
        if not c or c.get('status')=='rejected' or c.get('type','') in seen_t: continue
        seen_t.add(c.get('type',''))
        aff=c.get('affectedFiles',[]);sug=c.get('suggestion','')[:300]
        parts.append(f'<CARD type="{c.get("type","")}" confidence="{c.get("confidence",0)}">\n  Files: {", ".join(aff[:3])}\n  Summary: {sug}\n</CARD>')
        if len(seen_t)>=max_c: break

    parts.append("\n--- COMMIT HISTORY ---")
    bm25_s=co_bm.search_all(question,k=200) if co_bm else {}
    if embeddings is not None:
        qe=embed(question);vs={}
        for did,ix_s in doc_map.items():
            ix=int(ix_s)
            if ix<len(embeddings): vs[did]=cosine(qe,embeddings[ix])
        fused=rrf(bm25_s,vs,k=60,top=max_k*2)
    else:
        fused=sorted(bm25_s.items(),key=lambda x:-x[1])[:max_k*2]

    seen_sha=set();seen_subj=set();by_pkg=defaultdict(list)
    for did,sc in fused:
        d=next((d for d in commits if d['id']==did),None) if commits else None
        if not d or d.get('sha','') in seen_sha: continue
        seen_sha.add(d['sha'])
        sj=d.get('subject','')[:80].lower().strip()
        if sj in seen_subj: continue
        seen_subj.add(sj)
        pkgs=d.get('packages',[]);by_pkg[pkgs[0] if pkgs else '<root>'].append(d)

    for pkg in sorted(by_pkg.keys()):
        items=by_pkg[pkg]
        parts.append(f"\n  Package: {pkg} ({len(items)} unique)")
        for d in items[:4]:
            sha=d.get('sha','')[:12];subj=d.get('subject','')[:150]
            dt=d.get('date','').split('T')[0] if d.get('date') else ''
            fls=', '.join(d.get('paths',[])[:3]) if d.get('paths') else ''
            parts.append(f"    {dt} {sha}: {subj}"+(f" files: {fls}" if fls else ""))
    return "\n".join(parts)


# ── Question generation ──────────────────────────────────────────


def generate_questions(cards,commits,target=200):
    qs=set()
    ac=[c for c in cards if c.get('status')=='accepted'] if cards else []
    for c in ac:
        for f in c.get('affectedFiles',[])[:2]:
            qs.add(f"What should I know before modifying {f}?")
            qs.add(f"What are the risks of changing {f}?")
            t=c.get('type','')
            if t=='repeated-fix': qs.add(f"How many times was {f} fixed?")
            elif t=='test-gap': qs.add(f"Is there test coverage for {f}?")
            elif t=='co-change': qs.add(f"What files co-change with {f}?")
            elif t=='revert-pattern': qs.add(f"Was {f} unstable historically?")
            elif t=='high-churn': qs.add(f"How often does {f} change?")

    if commits:
        fc=defaultdict(list)
        for d in commits[:2000]:
            for p in d.get('paths',[]):
                if 'CHANGELOG' not in p and '/index.' not in p:
                    fc[p].append(d)
        tf=sorted(fc.keys(),key=lambda f:-len(fc[f]))[:60]
        for f in tf:
            cl=fc[f]
            qs.add(f"How many commits touched {f}?")
            if len(cl)>5:
                qs.add(f"What is the commit history of {f}?")
                if any('fix' in c.get('subject','').lower() for c in cl): qs.add(f"What bugs were fixed in {f}?")
                if any('test' in c.get('subject','').lower() for c in cl): qs.add(f"What test changes affected {f}?")

    xcut=[
        "Which packages have the most bug fix commits?",
        "Which packages have test coverage gaps?",
        "What files co-change most frequently?",
        "Which files are most risky to modify?",
        "What is the most changed file in the coding-agent package?",
        "What is the most changed file in the ai package?",
        "What is the most changed file in the tui package?",
    ]
    for q in xcut: qs.add(q)
    return list(qs)[:target]


Q_SYSTEM="You are a repo-aware coding assistant for the pi monorepo. Answer using only the repo-arch cards and commit history provided. Be specific about file paths, commit SHAs, package names, and commit counts. If the context does not contain the answer, say what information is missing."


def call_modal(context,question,max_tok=1024,temp=0.3):
    """Call Modal 7B for a single question."""
    import subprocess as sp, json as _json
    prompt=f"### Context\n{context}\n\n### Question\n{question}\n\n### Answer"
    payload=_json.dumps({"context":context,"question":question,"max_tokens":max_tok,"temperature":temp})
    # Modal function call via subprocess
    cmd=["modal","run","scripts/modal_7b.py","--question",question]
    try:
        r=sp.run(cmd,capture_output=True,text=True,timeout=300)
        # Parse output for answer
        out=r.stdout+r.stderr
        return {"answer":out[-1000:] if out else "[no output]"}  # Parse more carefully in real usage
    except Exception as e:
        return {"answer":None,"error":str(e)}


# ── Main ──────────────────────────────────────────────────────────


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--target",type=int,default=200)
    parser.add_argument("--out",default=".repo-arch/training-data/teacher7b")
    parser.add_argument("--max-questions",type=int,default=50)
    args=parser.parse_args()

    out_dir=os.path.join(REPO_DIR,args.out);os.makedirs(out_dir,exist_ok=True)

    print("Loading indexes...",file=sys.stderr)
    cards,cbm,commits,co_bm,embeddings,doc_map=load_all(REPO_DIR)
    print(f"  Cards: {len(cards)}, Commits: {len(commits)}, Embeddings: {embeddings is not None}",file=sys.stderr)

    questions=generate_questions(cards,commits,target=args.target)
    random.shuffle(questions)
    selected=questions[:args.max_questions]
    with open(os.path.join(out_dir,"questions.json"),"w") as f: json.dump(selected,f,indent=2)
    print(f"Generated {len(selected)} questions",file=sys.stderr)

    print(f"\n{'='*60}",file=sys.stderr)
    print(f"  Run this on Modal:",file=sys.stderr)
    print(f"  modal run scripts/modal_7b.py --question \"{selected[0][:80]}...\"",file=sys.stderr)
    print(f"{'='*60}",file=sys.stderr)

    # Save context + questions for batch processing on Modal
    batch=[]
    for i,q in enumerate(selected):
        ctx=build_context(q,cards,cbm,commits,co_bm,embeddings,doc_map)
        batch.append({"id":i+1,"question":q,"context":ctx})
    with open(os.path.join(out_dir,"batch.json"),"w") as f: json.dump(batch,f,indent=2)
    print(f"Saved {len(batch)} batch items to {os.path.join(out_dir,'batch.json')}",file=sys.stderr)
    print(f"Context sizes: {[len(b['context']) for b in batch[:5]]}",file=sys.stderr)


if __name__=="__main__":
    main()
