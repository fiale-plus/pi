"""
Batch teacher generation on deployed Modal 7B.

Usage:
  modal run scripts/teacher_batch.py --limit 10
"""

import json
import modal

app = modal.App("pi-teacher-batch")

BATCH_FILE = ".repo-arch/training-data/teacher7b/batch.json"
OUT_FILE = ".repo-arch/training-data/teacher7b/targets.jsonl"

TEACHER = modal.Function.from_name("pi-7b-teacher", "generate")


@app.local_entrypoint()
def main(limit: int = 50):
    import os
    repo_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    batch_path = os.path.join(repo_dir, BATCH_FILE)
    out_path = os.path.join(repo_dir, OUT_FILE)

    with open(batch_path) as f:
        batch = json.load(f)

    selected = batch[:limit]
    print(f"Generating {len(selected)} teacher targets via Modal 7B...")

    results = []
    for i, item in enumerate(selected):
        qid = item["id"]
        question = item["question"]
        context = item["context"]

        print(f"[{i+1}/{len(selected)}] Q{qid} ({len(context)} chars)...", end=" ", flush=True)
        try:
            r = TEACHER.remote(
                context=context,
                question=question,
                max_tokens=1024,
                temperature=0.3,
            )
            results.append({
                "id": qid,
                "question": question,
                "context_len": len(context),
                "answer": r["answer"],
                "tokens": r["tokens"],
                "model": r.get("model", "Qwen2.5-Coder-7B"),
            })
            print(f"ok ({r['tokens']} tok)")
        except Exception as e:
            print(f"ERR: {e}")
            results.append({
                "id": qid, "question": question, "context_len": len(context),
                "answer": None, "error": str(e),
            })

    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"\nDone. {len(results)} results -> {out_path}")
    print(f"  Success: {sum(1 for r in results if r.get('answer'))}")
    print(f"  Failed:  {sum(1 for r in results if r.get('error'))}")
