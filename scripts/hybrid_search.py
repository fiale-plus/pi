#!/usr/bin/env python3
"""
Hybrid search: BM25 + embedding with RRF fusion for answer pipeline.

Usage:
  source /opt/homebrew/var/mtplx/venv-0.1.0rc3/bin/activate
  python3 scripts/hybrid_search.py "What should I know about agent-session.ts?" --top-k 10
  python3 scripts/hybrid_search.py --eval .repo-arch/eval/questions.jsonl --out results.jsonl
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

REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


# ── BM25 ─────────────────────────────────────────────────────────


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
        r=self.search_all(q,k)
        return sorted(r.items(),key=lambda x:-x[1])[:k]
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


# ── Cosine ────────────────────────────────────────────────────────


def cosine(a, b):
    dot = sum(x*y for x, y in zip(a, b))
    na = math.sqrt(sum(x*x for x in a))
    nb = math.sqrt(sum(x*x for x in b))
    return dot / (na * nb + 1e-10)


# ── RRF ───────────────────────────────────────────────────────────


def rrf(bm25_scores, vec_scores, k=60, top_k=20):
    combined = defaultdict(float)
    for i, (did, _) in enumerate(sorted(bm25_scores.items(), key=lambda x: -x[1])):
        combined[did] += 1.0 / (k + i)
    for i, (did, _) in enumerate(sorted(vec_scores.items(), key=lambda x: -x[1])):
        combined[did] += 1.0 / (k + i)
    return sorted(combined.items(), key=lambda x: -x[1])[:top_k]


# ── Load index ────────────────────────────────────────────────────


def load_index(repo_dir):
    idx_dir = os.path.join(repo_dir, ".repo-arch", "index", "hybrid")

    # BM25
    docs_path = os.path.join(idx_dir, "docs.json")
    if not os.path.exists(docs_path):
        return None, None, None, None, None

    with open(docs_path) as f:
        doc_list = json.load(f)["documents"]

    bm25 = BM25()
    for d in doc_list:
        text = f"{d.get('subject','')} {' '.join(d.get('paths',[]))} {' '.join(d.get('packages',[]))}"
        bm25.add(d["id"], text)

    # Embeddings
    import numpy as np
    emb_path = os.path.join(idx_dir, "embeddings.npy")
    if not os.path.exists(emb_path):
        return doc_list, bm25, None, None, None

    embeddings = np.load(emb_path)

    with open(os.path.join(idx_dir, "doc_map.json")) as f:
        doc_map = json.load(f)

    # Build reverse index
    id_to_idx = {v: k for k, v in doc_map.items()}

    return doc_list, bm25, embeddings, doc_map, id_to_idx


_EMBED_MODEL = None

def embed_query(query, model_name="all-MiniLM-L6-v2"):
    """Embed a single query with caching."""
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        from sentence_transformers import SentenceTransformer
        _EMBED_MODEL = SentenceTransformer(model_name)
        _EMBED_MODEL.to("cpu")
    return _EMBED_MODEL.encode([query])[0]


# ── Cards ─────────────────────────────────────────────────────────


def load_cards(repo_dir):
    cfiles = glob.glob(os.path.join(repo_dir, ".repo-arch", "cache", "cards", "*.json"))
    if not cfiles: return [], None
    with open(cfiles[0]) as f: data = json.load(f)
    cards = data.get("cards", [])
    rev = json.load(open(os.path.join(repo_dir, ".repo-arch", "review-state.json"))) if os.path.exists(os.path.join(repo_dir, ".repo-arch", "review-state.json")) else {}
    for c in cards:
        c["status"] = rev.get(c.get("id",""), {}).get("status", "unreviewed")
    return cards, None


# ── Format ────────────────────────────────────────────────────────


SYSTEM = "You are a repo-aware coding assistant for the pi monorepo. Answer using only the repo-arch cards and commit history provided. Be specific about file paths, commit SHAs, package names, and commit counts."


def build_context(question, doc_list, bm25, embeddings, doc_map, id_to_idx, cards, max_cards=4, max_commits=10):
    """Build hybrid context with BM25 + embedding RRF fusion."""
    parts = ["--- REPO-ARCH CARDS ---"]

    # Cards (BM25 only, small set)
    if cards:
        card_bm25 = BM25()
        for c in cards:
            if c.get("status") == "accepted":
                card_bm25.add(c["id"], f"{c.get('title','')} {c.get('suggestion','')} {' '.join(c.get('affectedFiles',[]))}")

        card_results = card_bm25.search_all(question, k=max_cards * 3)
        seen_types = set()
        for cid, score in sorted(card_results.items(), key=lambda x:-x[1])[:max_cards*3]:
            c = next((c for c in cards if c["id"] == cid), None)
            if not c or c.get("type","") in seen_types or c.get("status") == "rejected": continue
            seen_types.add(c.get("type",""))
            aff = c.get("affectedFiles", []); sug = c.get("suggestion", "")[:300]
            parts.append(f'<CARD type="{c.get("type","")}" confidence="{c.get("confidence",0)}">\n  Files: {", ".join(aff[:3])}\n  Summary: {sug}\n</CARD>')
            if len(seen_types) >= max_cards: break

    parts.append("\n--- COMMIT HISTORY ---")

    if embeddings is not None and doc_map is not None and id_to_idx is not None:
        # Hybrid: BM25 + embedding RRF
        bm25_scores = bm25.search_all(question, k=200) if bm25 else {}
        q_emb = embed_query(question)
        vec_scores = {}
        for did, idx_str in doc_map.items():
            idx = int(idx_str)
            if idx < len(embeddings):
                vec_scores[did] = cosine(q_emb, embeddings[idx])
        fused = rrf(bm25_scores, vec_scores, k=60, top_k=max_commits * 2)
    else:
        # BM25 only fallback
        bm25_scores = bm25.search_all(question, k=200) if bm25 else {}
        fused = sorted(bm25_scores.items(), key=lambda x: -x[1])[:max_commits * 2]

    # Format commits with dedup
    seen_sha = set(); seen_subj = set()
    by_pkg = defaultdict(list)
    for did, score in fused:
        doc = next((d for d in doc_list if d["id"] == did), None)
        if not doc or doc.get("sha","") in seen_sha: continue
        seen_sha.add(doc["sha"])
        subj = doc.get("subject","")[:80].lower().strip()
        if subj in seen_subj: continue
        seen_subj.add(subj)
        pkgs = doc.get("packages", [])
        pkg_key = pkgs[0] if pkgs else "<root>"
        by_pkg[pkg_key].append(doc)

    for pkg in sorted(by_pkg.keys()):
        items = by_pkg[pkg]
        parts.append(f"\n  Package: {pkg} ({len(items)} unique commits)")
        for d in items[:4]:
            sha = d.get("sha","")[:12]; subj = d.get("subject","")[:150]
            dt = d.get("date","").split("T")[0] if d.get("date") else ""
            fls = " files: " + ", ".join(d.get("paths",[])[:3]) if d.get("paths") else ""
            parts.append(f"    {dt} {sha}: {subj}{fls}")

    return "\n".join(parts)


def answer(context, question, max_tok=384, temp=0.3):
    up = f"### Context\n{context}\n\n### Question\n{question}\n\n### Answer"
    cmd = [sys.executable, "-m", "mlx_lm", "generate",
           "--model", "Qwen/Qwen2.5-Coder-1.5B-Instruct",
           "--max-tokens", str(max_tok), "--temp", str(temp),
           "--system-prompt", SYSTEM, "--prompt", up, "--use-default-chat-template"]
    try:
        start = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        el = time.time() - start
        output = result.stdout.strip()
        parts = output.split("==========")
        a = parts[-2].strip() if len(parts) >= 3 and parts[-2].strip() else parts[-3].strip() if len(parts) >= 3 else output
        for l in a.split("\n"):
            if "tokens-per-sec" in l or "Peak memory" in l or "Prompt:" in l:
                a = a.replace(l, "")
        return {"answer": a.strip(), "latency": round(el, 2), "error": None}
    except subprocess.TimeoutExpired:
        return {"answer": None, "latency": 120, "error": "timeout"}
    except Exception as e:
        return {"answer": None, "latency": 0, "error": str(e)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("question", nargs="?")
    parser.add_argument("--question", "-q", dest="qflag")
    parser.add_argument("--eval")
    parser.add_argument("--out", default=".repo-arch/eval/runs/hybrid-45.jsonl")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    print("Loading hybrid index...", file=sys.stderr)
    doc_list, bm25, embeddings, doc_map, id_to_idx = load_index(REPO_DIR)
    if doc_list is None:
        print("No hybrid index. Run scripts/build_hybrid_index.py first.", file=sys.stderr)
        sys.exit(1)

    cards, _ = load_cards(REPO_DIR)
    has_embeddings = embeddings is not None
    print(f"  Docs: {len(doc_list)}, Embeddings: {'yes' if has_embeddings else 'no'}, Cards: {len(cards)}", file=sys.stderr)

    if args.eval:
        qs = [json.loads(l) for l in open(os.path.join(REPO_DIR, args.eval))]
        out_path = os.path.join(REPO_DIR, args.out)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            for i, q in enumerate(qs):
                qid, question = q["id"], q["question"]
                mode = "hybrid-search" if has_embeddings else "bm25-only"
                print(f"[{i+1}/{len(qs)}] Q{qid} ({mode}) ... ", end="", file=sys.stderr, flush=True)
                ctx = build_context(question, doc_list, bm25, embeddings, doc_map, id_to_idx, cards)
                r = answer(ctx, question)
                f.write(json.dumps({
                    "question_id": qid, "question": question,
                    "mode": mode, "answer": r.get("answer"),
                    "latency": r.get("latency"), "error": r.get("error"),
                }) + "\n")
                print(f"ok ({r['latency']}s)", file=sys.stderr)
        print(f"\nDone: {out_path}", file=sys.stderr)
        return

    question = args.question or args.qflag
    if not question:
        parser.print_help()
        sys.exit(1)

    ctx = build_context(question, doc_list, bm25, embeddings, doc_map, id_to_idx, cards)
    r = answer(ctx, question)

    if args.json:
        print(json.dumps({"question": question, "answer": r.get("answer"), "latency": r.get("latency")}, indent=2))
    else:
        print(f"\n{'='*60}\n  {question} ({'hybrid' if has_embeddings else 'bm25'})\n{'='*60}\n\n  {r.get('answer', r.get('error'))}")


if __name__ == "__main__":
    main()
