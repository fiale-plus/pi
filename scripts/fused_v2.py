#!/usr/bin/env python3
"""
Fused retrieval v2: retrieval packer with dedup, package grouping, dynamic allocation.

Improvements over v1:
- Retrieve 50 candidates, pack down to diverse context (dedup by SHA/package/subject)
- Group commits by package, prefer diverse packages
- Dynamic allocation: pattern questions get more cards, change questions get more commits
- Deduplicate repeated subjects into summarized counts
- Order: high-score, diverse-package, recent-first within package

Usage:
  source /opt/homebrew/var/mtplx/venv-0.1.0rc3/bin/activate
  python3 scripts/fused_v2.py "What should I know about agent-session.ts?"
  python3 scripts/fused_v2.py --eval .repo-arch/eval/questions.jsonl --out results.jsonl
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
REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


# ── BM25 (same) ──────────────────────────────────────────────────


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
    def search(self,q,k=20):
        t=self.tok(q);s=defaultdict(float)
        for x in t:
            i=self.idf(x)
            if i==0: continue
            for d in self.tid.get(x,set()):
                dl=len(self.doc[d]);tf=self.doc[d].count(x)
                denom=tf+self.k1*(1-self.b+self.b*dl/self.avgdl)
                s[d]+=i*tf*(self.k1+1)/max(denom,0.001)
        return [(self.i2d.get(i,''),sc) for i,sc in sorted(s.items(),key=lambda x:-x[1])[:k]]


# ── Load ─────────────────────────────────────────────────────────


def load_all(repo_dir):
    cfiles=glob.glob(os.path.join(repo_dir,'.repo-arch','cache','cards','*.json'))
    cards=[];cbm=BM25()
    if cfiles:
        with open(cfiles[0]) as f: data=json.load(f)
        cards=data.get('cards',[])
        rev=json.load(open(os.path.join(repo_dir,'.repo-arch','review-state.json'))) if os.path.exists(os.path.join(repo_dir,'.repo-arch','review-state.json')) else {}
        for c in cards:
            cid=c.get('id','');c['status']=rev.get(cid,{}).get('status','unreviewed')
            cbm.add(cid,f"{c.get('title','')} {c.get('suggestion','')} {' '.join(c.get('affectedFiles',[]))}")

    idxp=os.path.join(repo_dir,'.repo-arch','index','full','index.json')
    commits=[];com_bm=BM25()
    if os.path.exists(idxp):
        with open(idxp) as f: data=json.load(f)
        docs=data.get('documents',[]);seen=set()
        for d in docs:
            if d.get('sha','') not in seen: seen.add(d['sha']);commits.append(d)
        for d in commits: com_bm.add(d['id'],d['text'])
    return cards,cbm,commits,com_bm


# ── Query type detection ────────────────────────────────────────


def detect_query_type(question):
    """Classify question for dynamic context allocation."""
    q=question.lower()
    # Pattern/design questions: need more cards
    if any(w in q for w in ['architecture','design','pattern','relationship',
                             'co-change','coupling','structure','boundary']):
        return 'pattern'
    # What-changed questions: need more commits
    if any(w in q for w in ['changed','changed repeatedly','history','evolution',
                             'how has','what was','when did']):
        return 'history'
    # Risk/safety questions: need both
    if any(w in q for w in ['risk','risky','fragile','break','dangerous',
                             'should i know','before modifying']):
        return 'risk'
    # Package-specific: target one package
    if any(w in q for w in ['tui package','ai package','coding-agent package',
                             'web-ui package','agent package']):
        return 'package'
    # Test coverage
    if any(w in q for w in ['test coverage','test gap','untested','without test']):
        return 'test'
    return 'general'


# ── Retrieval packer ─────────────────────────────────────────────


def pack_context(question,cards_ix,commits_ix,cards,commits,expand=50):
    """Retrieve broad, pack diverse, dynamically allocate."""
    qtype=detect_query_type(question)
    q=question.lower()

    # ── Allocate budget by query type ──
    budgets={
        'pattern':  (6, 6),    # cards, commits
        'history':  (2, 14),
        'risk':     (5, 10),
        'package':  (3, 12),
        'test':     (5, 8),
        'general':  (4, 10),
    }
    max_cards,max_commits=budgets.get(qtype,(4,10))

    # ── Retrieve candidates ──
    card_results=cards_ix.search(question,k=expand) if cards_ix else []
    commit_results=commits_ix.search(question,k=expand) if commits_ix else []

    # ── Pack cards ──
    packed_cards=[]
    seen_types=set()
    for cid,sc in card_results:
        c=next((c for c in cards if c['id']==cid),None) if cards else None
        if not c or c.get('status')=='rejected' or c.get('type','') in seen_types: continue
        seen_types.add(c.get('type',''))
        packed_cards.append(c)
        if len(packed_cards)>=max_cards: break

    # ── Pack commits with dedup + package diversity ──
    seen_sha=set()
    seen_subjects=set()
    seen_packages=set()
    packed_commits=[]

    # First pass: one per package
    for did,sc in commit_results:
        d=next((d for d in commits if d['id']==did),None) if commits else None
        if not d or d['sha'] in seen_sha: continue
        pkgs=d.get('packages',[])

        # Check if this adds a new package
        new_pkg=any(p not in seen_packages for p in pkgs)
        if new_pkg or not seen_packages:
            seen_sha.add(d['sha'])
            # Dedup similar subjects
            subj=d.get('subject','')[:80]
            subj_key=subj.lower().strip()
            if subj_key not in seen_subjects:
                seen_subjects.add(subj_key)
                for p in pkgs: seen_packages.add(p)
                packed_commits.append(d)
            if len(packed_commits)>=max_commits: break

    # Second pass: fill remaining with any diverse commits
    if len(packed_commits)<max_commits:
        for did,sc in commit_results:
            d=next((d for d in commits if d['id']==did),None) if commits else None
            if not d or d['sha'] in seen_sha: continue
            subj_key=d.get('subject','')[:80].lower().strip()
            if subj_key in seen_subjects: continue
            seen_sha.add(d['sha']);seen_subjects.add(subj_key)
            for p in d.get('packages',[]): seen_packages.add(p)
            packed_commits.append(d)
            if len(packed_commits)>=max_commits: break

    # Third pass: if still need more, allow subject dedup but add different SHAs
    if len(packed_commits)<max_commits:
        for did,sc in commit_results:
            if len(packed_commits)>=max_commits: break
            d=next((d for d in commits if d['id']==did),None) if commits else None
            if not d or d['sha'] in seen_sha: continue
            seen_sha.add(d['sha']);packed_commits.append(d)

    return packed_cards,packed_commits,qtype


def format_packed(cards,commits):
    """Format packed context."""
    parts=["--- REPO-ARCH CARDS ---"]
    for c in cards:
        aff=c.get('affectedFiles',[]);sug=c.get('suggestion','')[:300]
        parts.append(f'<CARD type="{c.get("type","")}" status="{c.get("status","")}" confidence="{c.get("confidence",0)}">\n  Files: {", ".join(aff[:3])}\n  Summary: {sug}\n</CARD>')

    # Group commits by package
    parts.append("\n--- COMMIT HISTORY ---")
    by_pkg=defaultdict(list)
    for d in commits:
        pkgs=d.get('packages',[]) or ['<root>']
        for p in pkgs:
            by_pkg[p].append(d)
            break  # only first package

    for pkg in sorted(by_pkg.keys()):
        items=by_pkg[pkg]
        parts.append(f"\n  Package: {pkg} ({len(items)} commits)")
        for d in items[:4]:  # max 4 per package
            sha=d.get('sha','')[:12];subj=d.get('subject','')[:150];dt=d.get('date','').split('T')[0] if d.get('date') else ''
            fls=' files: '+', '.join(d.get('paths',[])[:3]) if d.get('paths') else ''
            parts.append(f"    {dt} {sha}: {subj}{fls}")
        if len(items)>4:
            parts.append(f"    ... and {len(items)-4} more commits in {pkg}")

    return "\n".join(parts)


SYSTEM="You are a repo-aware coding assistant for the pi monorepo. Answer using only the repo-arch cards and commit history provided. Be specific about file paths, commit SHAs, package names, and commit counts. If the context does not contain the answer, say what information is missing."


def answer(context,q,adapter=None,max_tok=384,temp=0.3):
    up=f"### Context\n{context}\n\n### Question\n{q}\n\n### Answer"
    cmd=[sys.executable,'-m','mlx_lm','generate','--model','Qwen/Qwen2.5-Coder-1.5B-Instruct','--max-tokens',str(max_tok),'--temp',str(temp),'--system-prompt',SYSTEM,'--prompt',up,'--use-default-chat-template']
    if adapter: cmd.extend(['--adapter-path',adapter])
    try:
        start=time.time();result=subprocess.run(cmd,capture_output=True,text=True,timeout=120)
        el=time.time()-start;o=result.stdout.strip();p=o.split('==========')
        a=p[-2].strip() if len(p)>=3 and p[-2].strip() else p[-3].strip() if len(p)>=3 else o
        for l in a.split('\n'):
            if 'tokens-per-sec' in l or 'Peak memory' in l or 'Prompt:' in l: a=a.replace(l,'')
        return{"answer":a.strip(),"latency":round(el,2),"error":None}
    except subprocess.TimeoutExpired: return{"answer":None,"latency":120,"error":"timeout"}
    except Exception as e: return{"answer":None,"latency":0,"error":str(e)}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("question",nargs="?")
    parser.add_argument("--question","-q",dest="qflag")
    parser.add_argument("--eval");parser.add_argument("--out",default=".repo-arch/eval/runs/fused-v2-45.jsonl")
    parser.add_argument("--json",action="store_true");parser.add_argument("--adapter")
    args=parser.parse_args()

    print("Loading indexes...",file=sys.stderr)
    cards,cbm,commits,com_bm=load_all(REPO_DIR)
    print(f"  Cards: {len(cards)}, Commits: {len(commits)}",file=sys.stderr)

    if args.eval:
        qs=[json.loads(l) for l in open(os.path.join(REPO_DIR,args.eval))]
        out_path=os.path.join(REPO_DIR,args.out);os.makedirs(os.path.dirname(out_path),exist_ok=True)
        with open(out_path,"w") as f:
            for i,q in enumerate(qs):
                qid,question=q["id"],q["question"]
                print(f"[{i+1}/{len(qs)}] Q{qid} (query type: {detect_query_type(question)}) ... ",end="",file=sys.stderr,flush=True)
                pack_cards,pack_commits,qtype=pack_context(question,cbm,com_bm,cards,commits)
                ctx=format_packed(pack_cards,pack_commits)
                r=answer(ctx,question,adapter=args.adapter)
                f.write(json.dumps({
                    "question_id":qid,"question":question,"mode":"fused-v2",
                    "answer":r.get("answer"),"latency":r.get("latency"),
                    "query_type":qtype,"cards_used":len(pack_cards),"commits_used":len(pack_commits),
                    "error":r.get("error"),
                })+"\n")
                print(f"ok ({r['latency']}s, {len(pack_cards)}c/{len(pack_commits)}k)",file=sys.stderr)
        print(f"\nDone: {out_path}",file=sys.stderr)
        return

    question=args.question or args.qflag
    if not question: parser.print_help();sys.exit(1)
    pack_cards,pack_commits,qtype=pack_context(question,cbm,com_bm,cards,commits)
    ctx=format_packed(pack_cards,pack_commits)
    r=answer(ctx,question,adapter=args.adapter)
    if args.json:
        print(json.dumps({"question":question,"answer":r.get("answer"),"latency":r.get("latency"),"query_type":qtype,"cards_used":len(pack_cards),"commits_used":len(pack_commits)},indent=2))
    else:
        print(f"\n{'='*60}\n  [{qtype}] {question} ({len(pack_cards)}c, {len(pack_commits)}k)\n{'='*60}\n\n  {r.get('answer',r.get('error'))}")


if __name__=="__main__":
    main()
