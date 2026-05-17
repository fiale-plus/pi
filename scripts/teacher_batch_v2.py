"""
Fast batch teacher generation — warms container first, then parallel batch.
Runs as a direct Modal app for minimal overhead.

Usage:
  modal run scripts/teacher_batch_v2.py --limit 150
"""

import json
import modal
import time

app = modal.App("pi-teacher-batch-v2")

BATCH_FILE = ".repo-arch/training-data/teacher7b/batch.json"
OUT_FILE = ".repo-arch/training-data/teacher7b/targets.jsonl"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch>=2.1.0", "transformers>=4.44.0", "accelerate>=0.33.0")
)

MODEL_NAME = "Qwen/Qwen2.5-Coder-7B-Instruct"

Q_SYSTEM = (
    "You are a repo-aware coding assistant for the pi monorepo. "
    "Answer using only the repo-arch cards and commit history provided as context. "
    "Be specific about file paths, commit SHAs, package names, and commit counts. "
)


@app.function(image=image, gpu="A10G", timeout=60 * 30, scaledown_window=600)
def warm_generate(context: str, question: str, max_tokens: int = 1024, temperature: float = 0.3) -> dict:
    """Warm container, then generate. Loads model once per container lifetime."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Load model once (cached in container)
    if not hasattr(warm_generate, "_model"):
        print("Loading model...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            device_map="auto",
            trust_remote_code=True,
        ).eval()
        warm_generate._model = model
        warm_generate._tokenizer = tokenizer
        print(f"Model loaded on {model.device}")

    tokenizer = warm_generate._tokenizer
    model = warm_generate._model

    prompt = f"### Context\n{context}\n\n### Question\n{question}\n\n### Answer"
    messages = [{"role": "system", "content": Q_SYSTEM}, {"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=max_tokens, temperature=temperature,
            top_p=0.9, do_sample=temperature > 0, pad_token_id=tokenizer.eos_token_id,
        )

    generated = outputs[0][inputs["input_ids"].shape[1]:]
    answer = tokenizer.decode(generated, skip_special_tokens=True)
    return {"answer": answer.strip(), "tokens": len(generated), "model": MODEL_NAME}


def _call(item):
    r = warm_generate.remote(item["context"], item["question"], 1024, 0.3)
    return {
        "id": item["id"],
        "question": item["question"],
        "context_len": len(item["context"]),
        "answer": r["answer"],
        "tokens": r["tokens"],
        "model": r.get("model", MODEL_NAME),
    }


@app.local_entrypoint()
def main(limit: int = 150, parallel: int = 4):
    import os
    from concurrent.futures import ThreadPoolExecutor, as_completed

    repo_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    with open(os.path.join(repo_dir, BATCH_FILE)) as f:
        batch = json.load(f)

    out_path = os.path.join(repo_dir, OUT_FILE)
    existing_ids = set()
    if os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                existing_ids.add(json.loads(line).get("id"))

    selected = [item for item in batch if item["id"] not in existing_ids][:limit]
    print(f"Generating {len(selected)} teacher targets (parallel={parallel}, skipping {len(existing_ids)} existing)...")

    print("Warming container...")
    warm_generate.remote("test", "warmup", max_tokens=5)

    results = []
    with ThreadPoolExecutor(max_workers=parallel) as ex:
        futures = {ex.submit(_call, item): item["id"] for item in selected}
        done = 0
        for future in as_completed(futures):
            r = future.result()
            results.append(r)
            done += 1
            status = "ok" if r.get("answer") else "ERR"
            print(f"[{done}/{len(selected)}] Q{r['id']} ({r.get('tokens', 0)} tok) {status}")

    with open(out_path, "a") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    total = 0
    with open(out_path) as f:
        for line in f:
            total += 1

    print(f"\nDone. {len(results)} new results appended (total: {total})")
    print(f"  Success: {sum(1 for r in results if r.get('answer'))}")
    print(f"  Failed:  {sum(1 for r in results if not r.get('answer'))}")
