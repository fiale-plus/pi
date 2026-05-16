"""
Modal 7B — simplified teacher generation pipeline.

Usage:
  modal run scripts/modal_7b.py  (test)
  modal run scripts/modal_7b.py --question "your question here"
"""

import modal

MODEL_NAME = "Qwen/Qwen2.5-Coder-7B-Instruct"

app = modal.App("pi-7b-teacher")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch>=2.1.0", "transformers>=4.44.0", "accelerate>=0.33.0")
)

Q_SYSTEM = (
    "You are a repo-aware coding assistant for the pi monorepo. "
    "Answer using only the repo-arch cards and commit history provided as context. "
    "Be specific about file paths, commit SHAs, package names, and commit counts. "
    "If the context does not contain the answer, say what information is missing."
)


@app.function(image=image, gpu="A10G", timeout=60 * 15)
def generate(context: str, question: str, max_tokens: int = 1024, temperature: float = 0.3) -> dict:
    """Generate a 7B answer from context + question."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading {MODEL_NAME}...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto",
        trust_remote_code=True,
    ).eval()

    print(f"Model loaded on {model.device}")

    prompt = f"### Context\n{context}\n\n### Question\n{question}\n\n### Answer"
    messages = [
        {"role": "system", "content": Q_SYSTEM},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=0.9,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = outputs[0][inputs["input_ids"].shape[1]:]
    answer = tokenizer.decode(generated, skip_special_tokens=True)

    return {
        "answer": answer.strip(),
        "tokens": len(generated),
        "model": MODEL_NAME,
    }


@app.local_entrypoint()
def main(question: str = "What should I know about agent-session.ts?"):
    """Test run."""
    # Build context from local retrieval
    import subprocess, json

    result = subprocess.run(
        ["python3", "scripts/fused_answer.py", "--json", question],
        capture_output=True, text=True, timeout=30,
    )
    if result.returncode == 0:
        local = json.loads(result.stdout)
        context = local.get("answer", "")[:200]
        print(f"Local answer: {context[:100]}...")
    else:
        context = "No local context available."

    print(f"Sending to Modal 7B...")
    r = generate.remote(context, question)
    print(f"\n7B answer ({r['tokens']} tokens):\n{r['answer']}")
