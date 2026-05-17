"""
Parallel batch teacher generation on Modal 7B.

Uses concurrent.futures for parallel Modal calls (10x faster).

Usage:
  modal run scripts/teacher_batch.py --limit 200 --parallel 8
"""

import json
import modal
from concurrent.futures import ThreadPoolExecutor, as_completed

app = modal.App("pi-teacher-batch")

BATCH_FILE = ".repo-arch/training-data/teacher7b/batch.json"
OUT_FILE = ".repo-arch/training-data/teacher7b/targets.jsonl"

TEACHER = modal.Function.from_name("pi-7b-teacher", "generate")


def call_teacher(item):
    qid = item["id"]
    question = item["question"]
    context = item["context"]
    try:
        r = TEACHER.remote(context=context, question=question, max_tokens=1024, temperature=0.3)
        return {"id": qid, "question": question, "context_len": len(context),
                "answer": r["answer"], "tokens": r["tokens"],
                "model": r.get("model", "Qwen2.5-Coder-7B")}
    except Exception as e:
        return {"id": qid, "question": question, "context_len": len(context),
                "answer": None, "error": str(e)}


@app.local_entrypoint()
def main(limit: int = 200, parallel: int = 6, start_from: int = 0):
    import os
    repo_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    batch_path = os.path.join(repo_dir, BATCH_FILE)
    out_path = os.path.join(repo_dir, OUT_FILE)

    with open(batch_path) as f:
        batch = json.load(f)

    # Support resuming by skipping already-generated IDs
    existing_ids = set()
    if os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                r = json.loads(line)
                existing_ids.add(r.get("id"))

    selected = [item for item in batch[start_from:] if item["id"] not in existing_ids][:limit]
    print(f"Generating {len(selected)} teacher targets (parallel={parallel}, skipping {len(existing_ids)} existing)...")

    results = []
    with ThreadPoolExecutor(max_workers=parallel) as executor:
        futures = {executor.submit(call_teacher, item): item["id"] for item in selected}
        done = 0
        for future in as_completed(futures):
            r = future.result()
            results.append(r)
            done += 1
            status = "ok" if r.get("answer") else f"ERR: {r.get('error','')[:50]}"
            print(f"[{done}/{len(selected)}] Q{r['id']} ({r.get('tokens',0)} tok) {status}")

    # Append to existing file
    with open(out_path, "a") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    all_results = []
    with open(out_path) as f:
        for line in f:
            all_results.append(json.loads(line))

    print(f"\nDone. {len(results)} new results appended (total: {len(all_results)})")
    print(f"  Success: {sum(1 for r in all_results if r.get('answer'))}")
    print(f"  Failed:  {sum(1 for r in all_results if r.get('error'))}")
